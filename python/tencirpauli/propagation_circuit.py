"""Unified Python facade for deterministic Pauli propagation circuits."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Type, TypeVar, Union, cast

import numpy as np

from .circuit import (
    Angle,
    Parameter,
    ParameterExpr,
    _coerce_parameters,
    _evaluate_angle,
    _evaluate_angle_value,
    _slot_set,
)
from .hamiltonian import DEFAULT_MAX_BYTES, _validate_max_bytes
from .pauli import PauliOperator
from .propagation import (
    ComputationalBasisState,
    GateTape,
    ProductBlochState,
    ProfiledExpectation,
    PropagationEngine,
    PropagationValueAndGradient,
    ZeroState,
)


PropagationState = Union[ZeroState, ComputationalBasisState, ProductBlochState, str]
_WIRE_SENTINEL = 2 * sys.maxsize + 1
_USE_DEFAULT_MAX_BYTES = object()
_CircuitT = TypeVar("_CircuitT", bound="_CircuitBuilder")


@dataclass(frozen=True)
class _PropagationOperation:
    name: str
    wires: tuple[int, ...]
    theta: Optional[Angle] = None
    payload: Optional[tuple[float, ...]] = None


_FIXED_GATES = {"x", "y", "z", "h", "s", "sdg", "cnot", "cz", "swap"}
_ROTATION_GATES = {"rx", "ry", "rz", "rxx", "ryy", "rzz"}
_ONE_QUBIT_GATES = {"x", "y", "z", "h", "s", "sdg", "rx", "ry", "rz"}
_TWO_QUBIT_GATES = {"cnot", "cz", "swap", "rxx", "ryy", "rzz"}
_KIND_TO_NAME = {
    0: "x",
    1: "y",
    2: "z",
    3: "h",
    4: "s",
    5: "sdg",
    6: "cnot",
    7: "cz",
    8: "swap",
    9: "rx",
    10: "ry",
    11: "rz",
    12: "rxx",
    13: "ryy",
    14: "rzz",
}


def _validate_wires(nqubits: int, wires: Sequence[int], arity: int) -> tuple[int, ...]:
    normalized = tuple(wires)
    if len(normalized) != arity:
        raise ValueError(f"gate requires exactly {arity} wires")
    if any(
        not isinstance(wire, int)
        or isinstance(wire, bool)
        or wire < 0
        or wire >= nqubits
        for wire in normalized
    ):
        raise ValueError("gate wire is outside the circuit")
    if len(set(normalized)) != len(normalized):
        raise ValueError("gate wires must be distinct")
    return normalized


def _operation_from_tape(
    operation: tuple[int, int, int, int, float, tuple[float, ...]],
) -> _PropagationOperation:
    kind, wire0, wire1, parameter, angle, payload = operation
    if kind == 15:
        wires: tuple[int, ...] = (wire0,) if wire1 == _WIRE_SENTINEL else (wire0, wire1)
        dimension = 4 if len(wires) == 1 else 16
        if len(payload) != dimension * dimension:
            raise ValueError("invalid PTM payload in gate tape")
        return _PropagationOperation(
            "ptm", wires, payload=tuple(float(value) for value in payload)
        )
    name = _KIND_TO_NAME.get(kind)
    if name is None:
        raise ValueError(f"unsupported gate kind {kind}")
    arity = 1 if name in _ONE_QUBIT_GATES else 2
    wires = _validate_wires(
        max(wire0 + 1, wire1 + 1 if arity == 2 else wire0 + 1),
        (wire0,) if arity == 1 else (wire0, wire1),
        arity,
    )
    theta: Optional[Angle]
    if name in _ROTATION_GATES:
        theta = Parameter(parameter) if parameter >= 0 else float(angle)
    else:
        theta = None
    return _PropagationOperation(name, wires, theta=theta)


class PropagationCircuitPlan:
    """Immutable compiled facade for a :class:`PropagationCircuit`.

    Dynamic symbolic angles are evaluated once per call and their Jacobian is
    chained into the native parameter gradient. The plan is invalidated when
    the source circuit is mutated and should be reused for repeated parameter
    evaluations.
    """

    __slots__ = (
        "_dynamic_angles",
        "_engine",
        "_locked",
        "is_exact",
        "max_weight",
        "nparameters",
        "nqubits",
    )

    def __init__(
        self,
        engine: PropagationEngine,
        dynamic_angles: tuple[Angle, ...],
        nparameters: int,
    ) -> None:
        self._engine = engine
        self._dynamic_angles = dynamic_angles
        self.nqubits = engine.nqubits
        self.nparameters = nparameters
        self.max_weight = engine.max_weight
        self.is_exact = engine.is_exact
        self._locked = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("PropagationCircuitPlan is immutable")
        object.__setattr__(self, name, value)

    def _native_parameters(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        values = _coerce_parameters(parameters, self.nparameters)
        native: np.ndarray[Any, Any] = np.empty(
            len(self._dynamic_angles), dtype=np.float64
        )
        jacobian: np.ndarray[Any, Any] = np.empty(
            (len(self._dynamic_angles), self.nparameters), dtype=np.float64
        )
        for index, angle in enumerate(self._dynamic_angles):
            native[index], jacobian[index] = _evaluate_angle(
                angle, values, self.nparameters
            )
        return native, jacobian

    def _native_values(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]]
    ) -> np.ndarray[Any, Any]:
        """Evaluate dynamic angles without allocating discarded Jacobians."""
        values = _coerce_parameters(parameters, self.nparameters)
        native: np.ndarray[Any, Any] = np.empty(
            len(self._dynamic_angles), dtype=np.float64
        )
        for index, angle in enumerate(self._dynamic_angles):
            native[index] = _evaluate_angle_value(angle, values, self.nparameters)
        return native

    def expectation(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None
    ) -> float:
        """Evaluate and return one real expectation.

        Raises ``ValueError`` when the compiled observable is not exactly
        Hermitian.
        """
        native = self._native_values(parameters)
        return self._engine.expectation(native)

    def value_and_grad(
        self,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        checkpoint_interval: Optional[int] = None,
    ) -> PropagationValueAndGradient:
        """Return the value and gradient for an exactly Hermitian observable.

        Raises ``ValueError`` when the compiled observable is not exactly
        Hermitian.
        """
        native, jacobian = self._native_parameters(parameters)
        result = self._engine.value_and_grad(
            native, checkpoint_interval=checkpoint_interval
        )
        gradient = np.ascontiguousarray(jacobian.T @ result.gradient)
        gradient.flags.writeable = False
        return PropagationValueAndGradient(result.value, gradient)

    def propagate_operator(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None
    ) -> PauliOperator:
        """Return the canonical propagated operator for the supplied parameters."""
        native = self._native_values(parameters)
        return self._engine.propagate_operator(native)

    def profile(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None
    ) -> ProfiledExpectation:
        """Return a profiled real expectation for an exactly Hermitian observable.

        Raises ``ValueError`` when the compiled observable is not exactly
        Hermitian.
        """
        native = self._native_values(parameters)
        return self._engine.profile(native)


class _CircuitBuilder:
    """Shared TensorCircuit-style circuit builder mechanics.

    Gates are appended in execution order with zero-based wires. Rotation
    angles may be constants, :class:`Parameter` objects, or arithmetic
    :class:`ParameterExpr` values. Call :meth:`compile` for repeated use or
    the convenience evaluation methods for one-off calls.

    Examples:
        >>> import tencirpauli as tcp
        >>> circuit = tcp.PropagationCircuit(1)
        >>> circuit.h(0)
        >>> observable = tcp.PauliOperator.from_terms(1, [("Z", 1.0)])
        >>> plan = circuit.compile(observable)
        >>> plan.nqubits
        1
    """

    def __init__(
        self,
        nqubits: int,
        *,
        initial_state: PropagationState = "zero",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("nqubits must be a non-negative integer")
        _validate_max_bytes(max_bytes)
        self.nqubits = nqubits
        self.initial_state = initial_state
        self.max_bytes = max_bytes
        self._operations: list[_PropagationOperation] = []
        self._generation = 0
        self._cached_plan: Optional[tuple[Any, ...]] = None

    _supports_ptm = False

    def __len__(self) -> int:
        return len(self._operations)

    @property
    def nparameters(self) -> int:
        """Return one plus the largest parameter slot used by the circuit."""
        slots = {
            slot
            for operation in self._operations
            if operation.theta is not None
            for slot in _slot_set(operation.theta)
        }
        return max(slots, default=-1) + 1

    def _append(
        self,
        name: str,
        wires: Sequence[int],
        theta: Optional[Angle] = None,
        payload: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> None:
        normalized_name = name.lower()
        if normalized_name not in _FIXED_GATES | _ROTATION_GATES | {"ptm"}:
            raise ValueError(f"unsupported propagation gate {name!r}")
        if normalized_name == "ptm" and not self._supports_ptm:
            raise ValueError("SPPSCircuit does not support PTM gates")
        arity = 1 if normalized_name in _ONE_QUBIT_GATES else 2
        if normalized_name == "ptm":
            normalized_wires = tuple(wires)
            if len(normalized_wires) not in (1, 2):
                raise ValueError("PTM wires must contain one or two distinct wires")
            normalized_wires = _validate_wires(
                self.nqubits, normalized_wires, len(normalized_wires)
            )
            array = np.asarray(payload, dtype=np.float64)
            dimension = 4 if len(normalized_wires) == 1 else 16
            if array.shape != (dimension, dimension) or not np.isfinite(array).all():
                raise ValueError(
                    f"PTM matrix must have shape ({dimension}, {dimension})"
                )
            self._operations.append(
                _PropagationOperation(
                    "ptm",
                    normalized_wires,
                    payload=tuple(float(value) for value in array.ravel()),
                )
            )
        else:
            normalized_wires = _validate_wires(self.nqubits, wires, arity)
            normalized_theta = None
            if normalized_name in _ROTATION_GATES:
                normalized_theta = 0.0 if theta is None else theta
                if not isinstance(normalized_theta, (Parameter, ParameterExpr)):
                    normalized_theta = float(normalized_theta)
                if isinstance(normalized_theta, float) and not np.isfinite(
                    normalized_theta
                ):
                    raise ValueError("gate angle must be finite")
            operation = _PropagationOperation(
                normalized_name, normalized_wires, normalized_theta
            )
            candidate_operations = (*self._operations, operation)
            slots = {
                slot
                for candidate in candidate_operations
                if candidate.theta is not None
                for slot in _slot_set(candidate.theta)
            }
            expected = set(range(max(slots, default=-1) + 1))
            if slots != expected:
                raise ValueError("parameter slots must cover 0..nparameters-1")
            self._operations.append(operation)
        self._generation += 1
        self._cached_plan = None

    def x(self, wire: int) -> None:
        """Append an X gate on ``wire``."""
        self._append("x", (wire,))

    def y(self, wire: int) -> None:
        """Append a Y gate on ``wire``."""
        self._append("y", (wire,))

    def z(self, wire: int) -> None:
        """Append a Z gate on ``wire``."""
        self._append("z", (wire,))

    def h(self, wire: int) -> None:
        """Append a Hadamard gate on ``wire``."""
        self._append("h", (wire,))

    def s(self, wire: int) -> None:
        """Append an S gate on ``wire``."""
        self._append("s", (wire,))

    def sdg(self, wire: int) -> None:
        """Append an inverse-S gate on ``wire``."""
        self._append("sdg", (wire,))

    def cnot(self, control: int, target: int) -> None:
        """Append a directed CNOT from ``control`` to ``target``."""
        self._append("cnot", (control, target))

    def cz(self, wire0: int, wire1: int) -> None:
        """Append a controlled-Z gate on two distinct wires."""
        self._append("cz", (wire0, wire1))

    def swap(self, wire0: int, wire1: int) -> None:
        """Append a SWAP gate on two distinct wires."""
        self._append("swap", (wire0, wire1))

    def rx(self, wire: int, theta: Angle = 0.0) -> None:
        """Append an X rotation; ``theta`` is in radians or symbolic form."""
        self._append("rx", (wire,), theta)

    def ry(self, wire: int, theta: Angle = 0.0) -> None:
        """Append a Y rotation; ``theta`` is in radians or symbolic form."""
        self._append("ry", (wire,), theta)

    def rz(self, wire: int, theta: Angle = 0.0) -> None:
        """Append a Z rotation; ``theta`` is in radians or symbolic form."""
        self._append("rz", (wire,), theta)

    def rxx(self, wire0: int, wire1: int, theta: Angle = 0.0) -> None:
        """Append an X-X rotation; ``theta`` is in radians or symbolic form."""
        self._append("rxx", (wire0, wire1), theta)

    def ryy(self, wire0: int, wire1: int, theta: Angle = 0.0) -> None:
        """Append a Y-Y rotation; ``theta`` is in radians or symbolic form."""
        self._append("ryy", (wire0, wire1), theta)

    def rzz(self, wire0: int, wire1: int, theta: Angle = 0.0) -> None:
        """Append a Z-Z rotation; ``theta`` is in radians or symbolic form."""
        self._append("rzz", (wire0, wire1), theta)

    def _native_tape(self) -> tuple[GateTape, tuple[Angle, ...]]:
        tape = GateTape(self.nqubits)
        dynamic: list[Angle] = []
        for operation in self._operations:
            name = operation.name
            if name == "ptm":
                if not self._supports_ptm:
                    raise ValueError("SPPSCircuit does not support PTM gates")
                assert operation.payload is not None
                dimension = 4 if len(operation.wires) == 1 else 16
                tape.ptm(
                    operation.wires,
                    np.asarray(operation.payload, dtype=np.float64).reshape(
                        (dimension, dimension)
                    ),
                )
                continue
            if name in _FIXED_GATES:
                if len(operation.wires) == 1:
                    getattr(tape, name)(operation.wires[0])
                else:
                    getattr(tape, name)(*operation.wires)
                continue
            assert operation.theta is not None
            if isinstance(operation.theta, (Parameter, ParameterExpr)):
                slot = len(dynamic)
                dynamic.append(operation.theta)
                getattr(tape, name)(*operation.wires, parameter=slot)
            else:
                getattr(tape, name)(*operation.wires, angle=float(operation.theta))
        return tape, tuple(dynamic)

    def _plan_key(
        self,
        observable: PauliOperator,
        initial_state: PropagationState,
        max_weight: Optional[int],
        max_bytes: Optional[int],
    ) -> tuple[Any, ...]:
        return (
            self._generation,
            id(observable),
            id(initial_state),
            max_weight,
            max_bytes,
        )

    def to_qir(self) -> list[dict[str, object]]:
        """Serialize the circuit to deterministic JSON-like gate records."""
        result: list[dict[str, object]] = []
        for operation in self._operations:
            item: dict[str, object] = {
                "name": operation.name,
                "index": operation.wires,
            }
            if operation.theta is not None:
                item["parameters"] = {"theta": operation.theta}
            if operation.payload is not None:
                dimension = 4 if len(operation.wires) == 1 else 16
                item["matrix"] = tuple(
                    tuple(operation.payload[row * dimension : (row + 1) * dimension])
                    for row in range(dimension)
                )
            result.append(item)
        return result

    @classmethod
    def from_circuit(
        cls: Type[_CircuitT],
        circuit: Any,
        *,
        parameter_order: Optional[Sequence[Any]] = None,
        initial_state: PropagationState = "zero",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> _CircuitT:
        """Convert a supported TensorCircuit circuit into this facade.

        ``tensorcircuit-ng`` must be installed. ``parameter_order`` fixes the
        runtime slot order for symbolic circuit parameters; without it, the
        integration's deterministic order is used.
        """
        from .integrations.tensorcircuit import gate_tape_from_circuit

        converted = gate_tape_from_circuit(circuit, parameter_order=parameter_order)
        if not getattr(cls, "_supports_ptm", False) and any(
            operation[0] == 15 for operation in converted.tape._operations
        ):
            raise ValueError("SPPSCircuit conversion does not support PTM gates")
        result = cls(
            converted.tape.nqubits,
            initial_state=initial_state,
            max_bytes=max_bytes,
        )
        result._operations = [
            _operation_from_tape(operation) for operation in converted.tape._operations
        ]
        result._generation = len(result._operations)
        result._cached_plan = None
        return result

    @classmethod
    def from_qir(
        cls: Type[_CircuitT],
        qir: Sequence[Mapping[str, object]],
        circuit_params: Mapping[str, object],
        *,
        parameter_order: Optional[Sequence[Any]] = None,
        initial_state: PropagationState = "zero",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> _CircuitT:
        """Restore a propagation circuit from QIR gate records.

        ``circuit_params`` must contain ``nqubits`` and may contain a
        ``parameter_order`` sequence used to resolve symbolic angle values.
        Unsupported gates, malformed wires, and inconsistent parameter slots
        raise ``ValueError``.
        """
        nqubits = circuit_params.get("nqubits")
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("circuit_params nqubits must be a non-negative integer")
        result = cls(nqubits, initial_state=initial_state, max_bytes=max_bytes)

        ordered_symbols = list(
            parameter_order
            if parameter_order is not None
            else cast(Sequence[Any], circuit_params.get("parameter_order", ()))
        )
        for index, symbol in enumerate(ordered_symbols):
            if any(bool(symbol == previous) for previous in ordered_symbols[:index]):
                raise ValueError("parameter_order must not contain duplicates")
        seen_symbols: list[Any] = []

        def symbol_slot(value: object) -> Optional[int]:
            for index, symbol in enumerate(ordered_symbols):
                try:
                    equal = bool(value == symbol)
                except Exception:
                    equal = False
                if equal:
                    if symbol not in seen_symbols:
                        seen_symbols.append(symbol)
                    return index
            return None

        def angle(value: object, default: float) -> Angle:
            if value is None:
                return default
            if isinstance(value, (Parameter, ParameterExpr)):
                return value
            slot = symbol_slot(value)
            if slot is not None:
                return Parameter(slot)
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    "QIR angles must be finite real values or direct symbols"
                )
            try:
                numeric = float(cast(Any, value))
            except (TypeError, ValueError) as error:
                if parameter_order is None and not isinstance(
                    value, (str, bytes, list, tuple, dict)
                ):
                    for index, symbol in enumerate(seen_symbols):
                        try:
                            if bool(value == symbol):
                                return Parameter(index)
                        except Exception:
                            continue
                    seen_symbols.append(value)
                    return Parameter(len(seen_symbols) - 1)
                raise TypeError(
                    "QIR angles must be finite real values or direct symbols"
                ) from error
            if not np.isfinite(numeric):
                raise ValueError("QIR angles must be finite")
            return numeric

        operations: list[_PropagationOperation] = []
        fixed = _FIXED_GATES
        rotations = _ROTATION_GATES
        for item in qir:
            if not isinstance(item, Mapping):
                raise ValueError("QIR item must be a mapping")
            name_value = item.get("name", item.get("gate"))
            if not isinstance(name_value, str):
                raise ValueError("QIR item must contain a gate name")
            name = name_value.lower()
            wires_value = item.get("index")
            if not isinstance(wires_value, Sequence) or isinstance(
                wires_value, (str, bytes)
            ):
                raise ValueError("QIR item must contain an index sequence")
            arity = 1 if name in _ONE_QUBIT_GATES else 2
            if name == "ptm":
                if not getattr(cls, "_supports_ptm", False):
                    raise ValueError(
                        "SPPSCircuit QIR conversion does not support PTM gates"
                    )
                if len(wires_value) not in (1, 2):
                    raise ValueError("PTM wires must contain one or two distinct wires")
                wires = _validate_wires(
                    nqubits,
                    tuple(cast(Any, wire) for wire in wires_value),
                    len(wires_value),
                )
                dimension = 4 if len(wires) == 1 else 16
                matrix = np.asarray(item.get("matrix"), dtype=np.float64)
                if matrix.ndim == 1 and matrix.shape == (dimension * dimension,):
                    matrix = matrix.reshape((dimension, dimension))
                if (
                    matrix.shape != (dimension, dimension)
                    or not np.isfinite(matrix).all()
                ):
                    raise ValueError(
                        f"PTM matrix must have shape ({dimension}, {dimension})"
                    )
                operations.append(
                    _PropagationOperation(
                        "ptm",
                        wires,
                        payload=tuple(float(value) for value in matrix.ravel()),
                    )
                )
                continue
            if name not in fixed | rotations:
                raise ValueError(f"unsupported propagation QIR gate {name!r}")
            wires = _validate_wires(
                nqubits,
                tuple(cast(Any, wire) for wire in wires_value),
                arity,
            )
            parameters = item.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise ValueError("QIR parameters must be a mapping")
            if name in fixed:
                if parameters:
                    raise ValueError(f"fixed gate {name} cannot have parameters")
                operations.append(_PropagationOperation(name, wires))
            else:
                theta = angle(parameters.get("theta"), 0.0)
                operations.append(_PropagationOperation(name, wires, theta=theta))

        symbols_in_order = {
            id(symbol) for symbol in seen_symbols if symbol in ordered_symbols
        }
        if parameter_order is not None and symbols_in_order != {
            id(symbol) for symbol in ordered_symbols
        }:
            raise ValueError("parameter_order must exactly cover direct QIR symbols")
        slots = {
            slot
            for operation in operations
            if operation.theta is not None
            for slot in _slot_set(operation.theta)
        }
        if slots != set(range(max(slots, default=-1) + 1)):
            raise ValueError("parameter slots must cover 0..nparameters-1")
        result._operations = operations
        result._generation = len(operations)
        result._cached_plan = None
        return result


class PropagationCircuit(_CircuitBuilder):
    """Deterministic Pauli-propagation circuit facade."""

    _supports_ptm = True

    def ptm(
        self,
        wires: Sequence[int],
        matrix: np.ndarray[Any, Any],
        *,
        name: str | None = None,
    ) -> None:
        """Append a finite one- or two-qubit real Pauli-transfer matrix."""
        del name
        self._append("ptm", wires, payload=np.asarray(matrix, dtype=np.float64))

    def compile(
        self,
        observable: PauliOperator,
        *,
        initial_state: Optional[PropagationState] = None,
        max_weight: Optional[int] = None,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> PropagationCircuitPlan:
        """Compile a reusable deterministic plan.

        Construction accepts non-Hermitian observables so that
        :meth:`PropagationCircuitPlan.propagate_operator` remains available.
        Scalar expectation, gradient, and profile terminals require an exactly
        Hermitian observable and raise ``ValueError`` otherwise.
        """
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        state = self.initial_state if initial_state is None else initial_state
        budget = (
            self.max_bytes
            if max_bytes is _USE_DEFAULT_MAX_BYTES
            else cast(Optional[int], max_bytes)
        )
        _validate_max_bytes(budget)
        key = self._plan_key(observable, state, max_weight, budget)
        if self._cached_plan is not None and self._cached_plan[:5] == key:
            return cast(PropagationCircuitPlan, self._cached_plan[5])
        tape, dynamic = self._native_tape()
        engine = PropagationEngine(
            tape,
            observable,
            initial_state=state,
            max_weight=max_weight,
            max_bytes=budget,
        )
        plan = PropagationCircuitPlan(engine, dynamic, self.nparameters)
        # Retain the key objects as well as their ids; otherwise CPython may
        # reuse an id after garbage collection and return a stale native plan.
        self._cached_plan = (*key, plan, observable, state)
        return plan

    def expectation(
        self,
        observable: PauliOperator,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        initial_state: Optional[PropagationState] = None,
        max_weight: Optional[int] = None,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> float:
        """Return a real expectation; the observable must be exactly Hermitian.

        Raises ``ValueError`` for a non-Hermitian observable.
        """
        return self.compile(
            observable,
            initial_state=initial_state,
            max_weight=max_weight,
            max_bytes=max_bytes,
        ).expectation(parameters)

    def value_and_grad(
        self,
        observable: PauliOperator,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        initial_state: Optional[PropagationState] = None,
        max_weight: Optional[int] = None,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
        checkpoint_interval: Optional[int] = None,
    ) -> PropagationValueAndGradient:
        """Return a real value and gradient for an exactly Hermitian observable.

        Raises ``ValueError`` for a non-Hermitian observable.
        """
        return self.compile(
            observable,
            initial_state=initial_state,
            max_weight=max_weight,
            max_bytes=max_bytes,
        ).value_and_grad(parameters, checkpoint_interval=checkpoint_interval)

    def propagate_operator(
        self,
        observable: PauliOperator,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        initial_state: Optional[PropagationState] = None,
        max_weight: Optional[int] = None,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> PauliOperator:
        """Return the canonical propagated operator without a Hermiticity requirement."""
        return self.compile(
            observable,
            initial_state=initial_state,
            max_weight=max_weight,
            max_bytes=max_bytes,
        ).propagate_operator(parameters)

    def profile(
        self,
        observable: PauliOperator,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        initial_state: Optional[PropagationState] = None,
        max_weight: Optional[int] = None,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> ProfiledExpectation:
        """Return a profiled real expectation for an exactly Hermitian observable.

        Raises ``ValueError`` for a non-Hermitian observable.
        """
        return self.compile(
            observable,
            initial_state=initial_state,
            max_weight=max_weight,
            max_bytes=max_bytes,
        ).profile(parameters)


__all__ = ["PropagationCircuit", "PropagationCircuitPlan"]
