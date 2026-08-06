"""Public deterministic circuit facade with occurrence-space gradients."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Type, TypeVar, Union, cast

import numpy as np

from .circuit import Angle, _angle, _concrete_angle, _concrete_angles, _is_jax_value
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

_FIXED_GATES = {"x", "y", "z", "h", "s", "sdg", "cnot", "cz", "swap"}
_ROTATION_GATES = {"rx", "ry", "rz", "rxx", "ryy", "rzz"}
_ONE_QUBIT_GATES = {"x", "y", "z", "h", "s", "sdg", "rx", "ry", "rz"}
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


@dataclass(frozen=True)
class _PropagationOperation:
    name: str
    wires: tuple[int, ...]
    theta: Optional[Angle] = None
    payload: Optional[tuple[float, ...]] = None


@dataclass(frozen=True)
class _PropagationObjective:
    engine: PropagationEngine
    angles: tuple[Angle, ...]
    gradient: bool


def _operation_from_tape(
    operation: tuple[int, int, int, int, float, tuple[float, ...]],
) -> _PropagationOperation:
    kind, wire0, wire1, parameter, angle, payload = operation
    if kind == 15:
        ptm_wires = (wire0,) if wire1 == _WIRE_SENTINEL else (wire0, wire1)
        dimension = 4 if len(ptm_wires) == 1 else 16
        if len(payload) != dimension * dimension:
            raise ValueError("invalid PTM payload in gate tape")
        return _PropagationOperation("ptm", ptm_wires, payload=tuple(payload))
    name = _KIND_TO_NAME.get(kind)
    if name is None:
        raise ValueError(f"unsupported gate kind {kind}")
    arity = 1 if name in _ONE_QUBIT_GATES else 2
    wires: tuple[int, ...] = _validate_wires(
        max(wire0 + 1, wire1 + 1 if arity == 2 else wire0 + 1),
        (wire0,) if arity == 1 else (wire0, wire1),
        arity,
    )
    if name in _ROTATION_GATES:
        if parameter >= 0:
            raise ValueError("TensorCircuit conversion must contain concrete angles")
        return _PropagationOperation(name, wires, theta=_angle(angle))
    return _PropagationOperation(name, wires)


class _CircuitBuilder:
    """Shared circuit construction and private native lowering."""

    _supports_ptm = False

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
        self._forward_tape_cache: Optional[tuple[int, GateTape]] = None
        self._gradient_tape_cache: Optional[tuple[int, GateTape]] = None
        self._objective_cache: Optional[
            tuple[tuple[Any, ...], _PropagationObjective]
        ] = None

    def __len__(self) -> int:
        return len(self._operations)

    @property
    def angle_count(self) -> int:
        """Number of gradient-supported gate-angle occurrences."""
        return sum(operation.theta is not None for operation in self._operations)

    def _append(
        self,
        name: str,
        wires: Sequence[int],
        theta: Optional[object] = None,
        payload: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> None:
        normalized_name = name.lower()
        if normalized_name not in _FIXED_GATES | _ROTATION_GATES | {"ptm"}:
            raise ValueError(f"unsupported propagation gate {name!r}")
        if normalized_name == "ptm":
            if not self._supports_ptm:
                raise ValueError("SPPSCircuit does not support PTM gates")
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
            operation = _PropagationOperation(
                "ptm",
                normalized_wires,
                payload=tuple(float(value) for value in array.ravel()),
            )
        else:
            arity = 1 if normalized_name in _ONE_QUBIT_GATES else 2
            normalized_wires = _validate_wires(self.nqubits, wires, arity)
            operation = _PropagationOperation(
                normalized_name,
                normalized_wires,
                (
                    None
                    if normalized_name not in _ROTATION_GATES
                    else _angle(0.0 if theta is None else theta)
                ),
            )
        self._operations.append(operation)
        self._generation += 1
        self._forward_tape_cache = None
        self._gradient_tape_cache = None
        self._objective_cache = None

    def x(self, wire: int) -> None:
        self._append("x", (wire,))

    def y(self, wire: int) -> None:
        self._append("y", (wire,))

    def z(self, wire: int) -> None:
        self._append("z", (wire,))

    def h(self, wire: int) -> None:
        self._append("h", (wire,))

    def s(self, wire: int) -> None:
        self._append("s", (wire,))

    def sdg(self, wire: int) -> None:
        self._append("sdg", (wire,))

    def cnot(self, control: int, target: int) -> None:
        self._append("cnot", (control, target))

    def cz(self, wire0: int, wire1: int) -> None:
        self._append("cz", (wire0, wire1))

    def swap(self, wire0: int, wire1: int) -> None:
        self._append("swap", (wire0, wire1))

    def rx(self, wire: int, theta: object = 0.0) -> None:
        self._append("rx", (wire,), theta)

    def ry(self, wire: int, theta: object = 0.0) -> None:
        self._append("ry", (wire,), theta)

    def rz(self, wire: int, theta: object = 0.0) -> None:
        self._append("rz", (wire,), theta)

    def rxx(self, wire0: int, wire1: int, theta: object = 0.0) -> None:
        self._append("rxx", (wire0, wire1), theta)

    def ryy(self, wire0: int, wire1: int, theta: object = 0.0) -> None:
        self._append("ryy", (wire0, wire1), theta)

    def rzz(self, wire0: int, wire1: int, theta: object = 0.0) -> None:
        self._append("rzz", (wire0, wire1), theta)

    def _build_tape(self, gradient: bool) -> GateTape:
        tape = GateTape(self.nqubits)
        angle_slot = 0
        for operation in self._operations:
            if operation.name == "ptm":
                assert operation.payload is not None
                dimension = 4 if len(operation.wires) == 1 else 16
                tape.ptm(
                    operation.wires,
                    np.asarray(operation.payload, dtype=np.float64).reshape(
                        (dimension, dimension)
                    ),
                )
            elif operation.name in _FIXED_GATES:
                getattr(tape, operation.name)(*operation.wires)
            else:
                assert operation.theta is not None
                if gradient:
                    getattr(tape, operation.name)(
                        *operation.wires, parameter=angle_slot
                    )
                else:
                    getattr(tape, operation.name)(
                        *operation.wires, angle=_concrete_angle(operation.theta)
                    )
                angle_slot += 1
        return tape

    def _native_tape(self, gradient: bool) -> GateTape:
        cache = self._gradient_tape_cache if gradient else self._forward_tape_cache
        if cache is not None and cache[0] == self._generation:
            return cache[1]
        tape = self._build_tape(gradient)
        if gradient:
            self._gradient_tape_cache = (self._generation, tape)
        else:
            self._forward_tape_cache = (self._generation, tape)
        return tape

    def _angle_values(self) -> np.ndarray[Any, Any]:
        return _concrete_angles(
            operation.theta
            for operation in self._operations
            if operation.theta is not None
        )

    def _objective(
        self,
        observable: PauliOperator,
        *,
        initial_state: PropagationState,
        max_weight: Optional[int],
        max_bytes: Optional[int],
        gradient: bool,
    ) -> _PropagationObjective:
        key = (
            self._generation,
            id(observable),
            id(initial_state),
            max_weight,
            max_bytes,
            gradient,
        )
        if self._objective_cache is not None and self._objective_cache[0] == key:
            return self._objective_cache[1]
        tape = self._native_tape(gradient)
        engine = PropagationEngine(
            tape,
            observable,
            initial_state=initial_state,
            max_weight=max_weight,
            max_bytes=max_bytes,
        )
        angles = tuple(
            operation.theta
            for operation in self._operations
            if operation.theta is not None
        )
        objective = _PropagationObjective(engine, angles, gradient)
        self._objective_cache = (key, objective)
        return objective

    def to_qir(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for operation in self._operations:
            item: dict[str, object] = {"name": operation.name, "index": operation.wires}
            if operation.theta is not None:
                item["parameters"] = {"theta": _concrete_angle(operation.theta)}
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
        initial_state: PropagationState = "zero",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> _CircuitT:
        from .integrations.tensorcircuit import gate_tape_from_circuit

        converted = gate_tape_from_circuit(circuit)
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
        return result

    @classmethod
    def from_qir(
        cls: Type[_CircuitT],
        qir: Sequence[Mapping[str, object]],
        circuit_params: Mapping[str, object],
        *,
        initial_state: PropagationState = "zero",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> _CircuitT:
        nqubits = circuit_params.get("nqubits")
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("circuit_params nqubits must be a non-negative integer")
        result = cls(nqubits, initial_state=initial_state, max_bytes=max_bytes)
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
            if name == "ptm":
                if not getattr(cls, "_supports_ptm", False):
                    raise ValueError(
                        "SPPSCircuit QIR conversion does not support PTM gates"
                    )
                wires = _validate_wires(nqubits, tuple(wires_value), len(wires_value))
                matrix = np.asarray(item.get("matrix"), dtype=np.float64)
                dimension = 4 if len(wires) == 1 else 16
                if matrix.ndim == 1 and matrix.shape == (dimension * dimension,):
                    matrix = matrix.reshape((dimension, dimension))
                if (
                    matrix.shape != (dimension, dimension)
                    or not np.isfinite(matrix).all()
                ):
                    raise ValueError(
                        f"PTM matrix must have shape ({dimension}, {dimension})"
                    )
                result._append("ptm", wires, payload=matrix)
                continue
            if name not in _FIXED_GATES | _ROTATION_GATES:
                raise ValueError(f"unsupported propagation QIR gate {name!r}")
            arity = 1 if name in _ONE_QUBIT_GATES else 2
            wires = _validate_wires(nqubits, tuple(wires_value), arity)
            parameters = item.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise ValueError("QIR parameters must be a mapping")
            if name in _FIXED_GATES:
                if parameters:
                    raise ValueError(f"fixed gate {name} cannot have parameters")
                getattr(result, name)(*wires)
            else:
                theta = parameters.get("theta", 0.0)
                if _is_jax_value(theta):
                    raise TypeError("QIR angles must be concrete finite real values")
                getattr(result, name)(*wires, theta=_concrete_angle(theta))
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
        del name
        self._append("ptm", wires, payload=matrix)

    def _options(self, max_bytes: object) -> Optional[int]:
        budget = self.max_bytes if max_bytes is _USE_DEFAULT_MAX_BYTES else max_bytes
        _validate_max_bytes(cast(Optional[int], budget))
        return cast(Optional[int], budget)

    def expectation(
        self,
        observable: PauliOperator,
        *,
        initial_state: Optional[PropagationState] = None,
        max_weight: Optional[int] = None,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> float:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        objective = self._objective(
            observable,
            initial_state=(
                self.initial_state if initial_state is None else initial_state
            ),
            max_weight=max_weight,
            max_bytes=self._options(max_bytes),
            gradient=False,
        )
        return objective.engine.expectation(np.empty(0, dtype=np.float64))

    def value_and_grad(
        self,
        observable: PauliOperator,
        *,
        initial_state: Optional[PropagationState] = None,
        max_weight: Optional[int] = None,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
        checkpoint_interval: Optional[int] = None,
    ) -> PropagationValueAndGradient:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        objective = self._objective(
            observable,
            initial_state=(
                self.initial_state if initial_state is None else initial_state
            ),
            max_weight=max_weight,
            max_bytes=self._options(max_bytes),
            gradient=True,
        )
        return objective.engine.value_and_grad(
            self._angle_values(), checkpoint_interval=checkpoint_interval
        )

    def expectation_jax(
        self,
        observable: PauliOperator,
        *,
        initial_state: Optional[PropagationState] = None,
        max_weight: Optional[int] = None,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
        checkpoint_interval: Optional[int] = None,
    ) -> Any:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        objective = self._objective(
            observable,
            initial_state=(
                self.initial_state if initial_state is None else initial_state
            ),
            max_weight=max_weight,
            max_bytes=self._options(max_bytes),
            gradient=True,
        )
        from .jax_support import native_expectation_jax

        return native_expectation_jax(
            tuple(objective.angles),
            objective.engine,
            checkpoint_interval=checkpoint_interval,
        )

    def propagate_operator(
        self,
        observable: PauliOperator,
        *,
        initial_state: Optional[PropagationState] = None,
        max_weight: Optional[int] = None,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> PauliOperator:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        objective = self._objective(
            observable,
            initial_state=(
                self.initial_state if initial_state is None else initial_state
            ),
            max_weight=max_weight,
            max_bytes=self._options(max_bytes),
            gradient=False,
        )
        return objective.engine.propagate_operator(np.empty(0, dtype=np.float64))

    def profile(
        self,
        observable: PauliOperator,
        *,
        initial_state: Optional[PropagationState] = None,
        max_weight: Optional[int] = None,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> ProfiledExpectation:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        objective = self._objective(
            observable,
            initial_state=(
                self.initial_state if initial_state is None else initial_state
            ),
            max_weight=max_weight,
            max_bytes=self._options(max_bytes),
            gradient=False,
        )
        return objective.engine.profile(np.empty(0, dtype=np.float64))


__all__ = ["PropagationCircuit"]
