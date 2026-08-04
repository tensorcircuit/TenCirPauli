"""Lazy TensorCircuit-semantics-compatible Rust-native U(1) circuit."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as operator_index
from typing import Any, Mapping, Optional, Sequence, cast

import numpy as np

from . import _native
from ._validation import normalize_pauli_code
from .circuit import (
    Angle,
    Parameter,
    ParameterExpr,
    _CircuitProgram,
    _gate,
    _LogicalGate,
)
from .hamiltonian import (
    DEFAULT_MAX_BYTES,
    _check_allocation,
    _effective_max_bytes,
    _validate_max_bytes,
)
from .pauli import PauliOperator, PauliWord
from .symmetry import U1Sector


def _readonly(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    result = np.ascontiguousarray(array)
    result.flags.writeable = False
    return cast(np.ndarray[Any, Any], result)


def _parameter_array(
    parameters: Optional[Sequence[float] | np.ndarray[Any, Any]], nparameters: int
) -> np.ndarray[Any, Any]:
    if parameters is None:
        if nparameters != 0:
            raise ValueError("parameters are required for a parameterized circuit")
        values: np.ndarray[Any, Any] = np.empty(0, dtype=np.float64)
    else:
        values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != nparameters:
        raise ValueError(
            f"parameters must have shape ({nparameters},), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("parameters must be finite")
    return cast(np.ndarray[Any, Any], np.ascontiguousarray(values))


def _state_array(
    state: Sequence[complex] | np.ndarray[Any, Any], dimension: int
) -> np.ndarray[Any, Any]:
    values = np.asarray(state, dtype=np.complex128)
    if values.ndim != 1 or values.shape[0] != dimension:
        raise ValueError(
            f"initial_state must have shape ({dimension},), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("initial_state must contain finite values")
    return cast(np.ndarray[Any, Any], np.ascontiguousarray(values))


def _encode_program(
    program: _CircuitProgram,
) -> tuple[
    list[tuple[int, int, int, float]],
    list[tuple[int, int, int, int, list[int], list[float], list[float]]],
]:
    nodes: list[tuple[int, int, int, float]] = []
    node_indices: dict[object, int] = {}

    def visit(value: object) -> int:
        if isinstance(value, Parameter):
            key: object = value
            if key not in node_indices:
                node_indices[key] = len(nodes)
                nodes.append((1, value.slot, 0, 0.0))
            return node_indices[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            constant = float(value)
            key = ("constant", constant)
            if key not in node_indices:
                node_indices[key] = len(nodes)
                nodes.append((0, 0, 0, constant))
            return node_indices[key]
        if not isinstance(value, ParameterExpr):
            raise TypeError(
                "gate angle must be a real scalar, Parameter, or ParameterExpr"
            )
        child_indices = tuple(visit(child) for child in value.operands)
        opcode = {"neg": 2, "add": 3, "sub": 4, "mul": 5, "div": 6}[value.operation]
        key = (value.operation, child_indices)
        if key not in node_indices:
            left = child_indices[0]
            right = child_indices[1] if len(child_indices) == 2 else 0
            node_indices[key] = len(nodes)
            nodes.append((opcode, left, right, 0.0))
        return node_indices[key]

    encoded_gates = []
    for operation in program.operations:
        angle = 0 if operation.angle is None else visit(operation.angle)
        opcode = {
            "rz": 0,
            "rzz": 1,
            "cz": 2,
            "cphase": 3,
            "swap": 4,
            "iswap": 5,
            "diagonal": 6,
        }[operation.name]
        payload = operation.payload or ()
        encoded_gates.append(
            (
                opcode,
                operation.wires[0],
                operation.wires[1] if len(operation.wires) > 1 else 0,
                angle,
                list(operation.wires) if operation.name == "diagonal" else [],
                [value.real for value in payload],
                [value.imag for value in payload],
            )
        )
    return nodes, encoded_gates


def _pauli_codes(
    nqubits: int,
    x: Optional[Sequence[int]],
    y: Optional[Sequence[int]],
    z: Optional[Sequence[int]],
    ps: Optional[Sequence[int]],
) -> tuple[int, ...]:
    if ps is not None:
        if x is not None or y is not None or z is not None:
            raise ValueError("ps cannot be combined with x, y, or z")
        codes = tuple(ps)
        if len(codes) != nqubits:
            raise ValueError("ps must contain exactly nqubits codes")
        return tuple(normalize_pauli_code(code) for code in codes)
    codes_list: list[int] = [0] * nqubits
    for code, indices in ((1, x or ()), (2, y or ()), (3, z or ())):
        seen: set[int] = set()
        for index in indices:
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= nqubits
            ):
                raise ValueError(f"Pauli index {index!r} is outside the circuit")
            if index in seen:
                raise ValueError("Pauli index sequences must not contain duplicates")
            seen.add(index)
            if codes_list[index] != 0 and codes_list[index] != code:
                raise ValueError("Pauli x/y/z index sets overlap")
            codes_list[index] = code
    return tuple(codes_list)


def _normalize_ps(value: object, nqubits: int) -> tuple[int, ...]:
    if isinstance(value, PauliWord):
        if value.nqubits != nqubits:
            raise ValueError("observable has the wrong nqubits")
        return value.to_codes()
    if isinstance(value, str):
        word = PauliWord.from_string(value)
        if word.nqubits != nqubits:
            raise ValueError("observable has the wrong nqubits")
        return word.to_codes()
    if isinstance(value, Mapping):
        return _pauli_codes(
            nqubits,
            cast(Optional[Sequence[int]], value.get("x")),
            cast(Optional[Sequence[int]], value.get("y")),
            cast(Optional[Sequence[int]], value.get("z")),
            None,
        )
    return _pauli_codes(nqubits, None, None, None, cast(Sequence[int], value))


@dataclass(frozen=True)
class U1CircuitValueAndGradient:
    """Real expectation value and exact native adjoint gradient."""

    value: float
    gradient: np.ndarray[Any, Any]


@dataclass
class _FinalStateCache:
    generation: int
    parameter_bits: bytes
    native: Any
    state: Optional[np.ndarray[Any, Any]] = None


@dataclass(frozen=True, init=False)
class U1CircuitPlan:
    """Immutable compiled execution plan in a fixed-particle-number sector.

    The plan acts on vectors of length ``sector.dimension`` and never leaves
    the selected U(1) sector. Returned arrays are read-only NumPy arrays.
    """

    sector: U1Sector
    dimension: int
    nparameters: int
    _initial_state: Optional[np.ndarray[Any, Any]]
    _native: Any

    def __init__(
        self,
        sector: U1Sector,
        native_plan: Any,
        initial_state: Optional[np.ndarray[Any, Any]] = None,
    ) -> None:
        object.__setattr__(self, "sector", sector)
        object.__setattr__(self, "dimension", int(native_plan.dimension))
        object.__setattr__(self, "nparameters", int(native_plan.nparameters))
        if initial_state is not None:
            initial_state = _readonly(np.asarray(initial_state, dtype=np.complex128))
        object.__setattr__(self, "_initial_state", initial_state)
        object.__setattr__(self, "_native", native_plan)

    def _params(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]]
    ) -> np.ndarray[Any, Any]:
        return _parameter_array(parameters, self.nparameters)

    def _state(
        self, initial_state: Optional[Sequence[complex] | np.ndarray[Any, Any]]
    ) -> np.ndarray[Any, Any]:
        selected = self._initial_state if initial_state is None else initial_state
        if selected is None:
            raise TypeError(
                "initial_state must be provided for a standalone U1CircuitPlan"
            )
        return _state_array(selected, self.dimension)

    def run(
        self,
        initial_state: Optional[Sequence[complex] | np.ndarray[Any, Any]] = None,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> np.ndarray[Any, Any]:
        """Apply the circuit to a restricted-sector state vector.

        ``initial_state`` must have length ``dimension`` and parameters must
        match ``nparameters``. The returned complex128 vector remains in the
        same sector ordering.
        """
        result = self._native.run(self._state(initial_state), self._params(parameters))
        return _readonly(np.asarray(result, dtype=np.complex128))

    def probability(
        self,
        initial_state: Optional[Sequence[complex] | np.ndarray[Any, Any]] = None,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> np.ndarray[Any, Any]:
        """Return a read-only ``float64`` vector of restricted-sector probabilities.

        The shape is ``(dimension,)`` and rows follow the sector's
        qubit-zero-is-MSB ordering.
        """
        result = self._native.probability(
            self._state(initial_state),
            self._params(parameters),
        )
        return _readonly(np.asarray(result, dtype=np.float64))

    def state_full(
        self,
        initial_state: Optional[Sequence[complex] | np.ndarray[Any, Any]] = None,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> np.ndarray[Any, Any]:
        """Return an owned read-only full computational-basis ``complex128`` vector.

        The result has shape ``(2**nqubits,)`` in qubit-zero-is-MSB ordering;
        amplitudes outside the selected particle-number sector are zero.
        """
        result = self._native.to_dense(
            self._state(initial_state),
            self._params(parameters),
        )
        return _readonly(np.asarray(result, dtype=np.complex128))

    def probability_full(
        self,
        initial_state: Optional[Sequence[complex] | np.ndarray[Any, Any]] = None,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> np.ndarray[Any, Any]:
        """Return a read-only full computational-basis ``float64`` vector.

        The shape is ``(2**nqubits,)`` in qubit-zero-is-MSB ordering. This
        expands the restricted result and can therefore be much larger than
        :meth:`probability`.
        """
        result = self._native.probability_full(
            self._state(initial_state),
            self._params(parameters),
        )
        return _readonly(np.asarray(result, dtype=np.float64))

    def expectation(
        self,
        initial_state: Optional[Sequence[complex] | np.ndarray[Any, Any]],
        observable: PauliOperator,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> complex:
        """Return the complex expectation of a Pauli observable."""
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        selected_state = self._initial_state if initial_state is None else initial_state
        if selected_state is None:
            raise TypeError(
                "initial_state must be provided for a standalone U1CircuitPlan"
            )
        real, imaginary = self._native.expectation(
            _state_array(selected_state, self.dimension),
            *observable._arrays(),
            self._params(parameters),
        )
        return complex(float(real), float(imaginary))

    def value_and_grad(
        self,
        initial_state: Optional[Sequence[complex] | np.ndarray[Any, Any]],
        observable: PauliOperator,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> U1CircuitValueAndGradient:
        """Return a real value and exact gradient for an exactly Hermitian observable.

        The result is a read-only ``float64`` vector in parameter-slot order.
        ``ValueError`` is raised for a non-Hermitian observable.
        """
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        if not observable._exact_hermitian_value():
            raise ValueError("value_and_grad requires an exactly Hermitian observable")
        selected_state = self._initial_state if initial_state is None else initial_state
        if selected_state is None:
            raise TypeError(
                "initial_state must be provided for a standalone U1CircuitPlan"
            )
        value, gradient = self._native.value_and_grad(
            _state_array(selected_state, self.dimension),
            *observable._arrays(),
            self._params(parameters),
        )
        return U1CircuitValueAndGradient(
            float(value), _readonly(np.asarray(gradient, dtype=np.float64))
        )


class U1Circuit:
    """Lazy circuit that preserves a fixed particle-number sector.

    Construct with ``particle_number`` and optionally an ``occupied`` basis
    initialization or an explicit restricted-sector ``initial_state``. Gates
    are diagonal or particle-number preserving, and
    execution is deferred until a state, probability, expectation, or compiled
    plan is requested.

    Examples:
        >>> import tencirpauli as tcp
        >>> circuit = tcp.U1Circuit(2, particle_number=1)
        >>> circuit.rz(0, 0.2)
        >>> circuit.probability().shape
        (2,)
    """

    def __init__(
        self,
        nqubits: int,
        particle_number: Optional[int] = None,
        *,
        occupied: Optional[Sequence[int]] = None,
        initial_state: Optional[Sequence[complex] | np.ndarray[Any, Any]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("nqubits must be a non-negative integer")
        _validate_max_bytes(max_bytes)
        if occupied is not None and initial_state is not None:
            raise ValueError("occupied and initial_state are mutually exclusive")
        normalized_occupied = None
        if occupied is not None:
            try:
                normalized_occupied = tuple(operator_index(index) for index in occupied)
            except TypeError as error:
                raise TypeError(
                    "occupied must contain integer qubit indices"
                ) from error
            if any(index < 0 or index >= nqubits for index in normalized_occupied):
                raise ValueError("occupied indices must be in range 0..nqubits")
            if len(set(normalized_occupied)) != len(normalized_occupied):
                raise ValueError("occupied indices must be distinct")
            normalized_occupied = tuple(sorted(normalized_occupied))
            if (
                particle_number is not None
                and len(normalized_occupied) != particle_number
            ):
                raise ValueError("particle_number must equal len(occupied)")
            particle_number = len(normalized_occupied)
        if particle_number is None:
            if initial_state is not None:
                raise ValueError("particle_number is required with initial_state")
            raise ValueError("particle_number or occupied must be provided")
        if (
            isinstance(particle_number, bool)
            or not isinstance(particle_number, int)
            or not 0 <= particle_number <= nqubits
        ):
            raise ValueError("particle_number must be between 0 and nqubits")
        if normalized_occupied is None:
            normalized_occupied = tuple(range(particle_number))
        self.nqubits = nqubits
        self.particle_number = particle_number
        self.max_bytes = max_bytes
        self.sector = U1Sector(nqubits, particle_number)
        dimension = self.sector.dimension
        _check_allocation(
            dimension * np.dtype(np.complex128).itemsize,
            max_bytes,
            "U1 circuit initial state",
        )
        if initial_state is None:
            basis_value = sum(
                1 << (nqubits - 1 - index) for index in normalized_occupied
            )
            initial_index = self.sector.rank(basis_value)
            initial: np.ndarray[Any, Any] = np.zeros(dimension, dtype=np.complex128)
            initial[initial_index] = 1.0
        else:
            initial = _state_array(initial_state, dimension).copy()
        initial.flags.writeable = False
        self._initial_state = initial
        self._program = _CircuitProgram(nqubits)
        self._native_plan: Optional[U1CircuitPlan] = None
        self._state_cache: Optional[_FinalStateCache] = None
        self._generation = 0

    @classmethod
    def _from_program(
        cls,
        other: "U1Circuit",
        program: _CircuitProgram,
    ) -> "U1Circuit":
        result = cls.__new__(cls)
        result.nqubits = other.nqubits
        result.particle_number = other.particle_number
        result.max_bytes = other.max_bytes
        result.sector = other.sector
        result._initial_state = other._initial_state.copy()
        result._initial_state.flags.writeable = False
        result._program = program
        result._native_plan = None
        result._state_cache = None
        result._generation = 0
        return result

    @property
    def nparameters(self) -> int:
        """Return the number of symbolic parameter slots in the circuit."""
        return self._program.nparameters

    @property
    def dimension(self) -> int:
        """Return ``C(nqubits, particle_number)``, the sector dimension."""
        return self.sector.dimension

    def _append(self, operation: _LogicalGate) -> None:
        self._program = self._program.with_operations(
            (*self._program.operations, operation)
        )
        self._native_plan = None
        self._state_cache = None
        self._generation += 1

    def rz(self, i: int, theta: Angle = 0.0) -> None:
        """Append an RZ gate; ``theta`` is measured in radians or symbolic form."""
        self._append(_gate("rz", (i,), theta))

    def rzz(self, i: int, j: int, theta: Angle = 0.0) -> None:
        """Append an RZZ gate; ``theta`` is measured in radians or symbolic form."""
        self._append(_gate("rzz", (i, j), theta))

    def cz(self, i: int, j: int) -> None:
        """Append a controlled-Z gate on two distinct qubits."""
        self._append(_gate("cz", (i, j)))

    def cphase(self, i: int, j: int, theta: Angle = 0.0) -> None:
        """Append a controlled-phase gate with a radian or symbolic angle."""
        self._append(_gate("cphase", (i, j), theta))

    def swap(self, i: int, j: int) -> None:
        """Append a SWAP gate on two distinct qubits."""
        self._append(_gate("swap", (i, j)))

    def iswap(self, i: int, j: int, theta: Angle = 1.0) -> None:
        """Append an iSWAP interpolation using the normalized angle convention."""
        self._append(_gate("iswap", (i, j), theta))

    def diagonal(
        self,
        *indices: int,
        diagonal: Optional[Sequence[complex] | np.ndarray[Any, Any]] = None,
        diag: Optional[Sequence[complex] | np.ndarray[Any, Any]] = None,
    ) -> None:
        """Append a static diagonal gate on the selected qubits.

        The payload must contain exactly ``2**len(indices)`` finite complex
        values. ``diagonal`` and its compatibility alias ``diag`` are mutually
        exclusive.
        """
        if diagonal is not None and diag is not None:
            raise ValueError("provide only one of diagonal or diag")
        payload = diagonal if diagonal is not None else diag
        if payload is None:
            raise TypeError("diagonal requires a diagonal payload")
        values = np.asarray(payload, dtype=np.complex128).reshape(-1)
        expected = 1 << len(indices)
        if values.shape != (expected,):
            raise ValueError(
                f"diagonal must have shape ({expected},), got {values.shape}"
            )
        self._append(_gate("diagonal", indices, payload=values.tolist()))

    def compile(self) -> U1CircuitPlan:
        """Compile and cache the native fixed-sector execution plan."""
        if self._native_plan is None:
            expression_nodes, gates = _encode_program(self._program)
            native = _native.u1_circuit_plan(
                self.nqubits,
                self.particle_number,
                1,
                self.nparameters,
                expression_nodes,
                gates,
                _effective_max_bytes(self.max_bytes),
            )
            self._native_plan = U1CircuitPlan(self.sector, native, self._initial_state)
        return self._native_plan

    def _cached_final(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]]
    ) -> _FinalStateCache:
        values = _parameter_array(parameters, self.nparameters)
        parameter_bits = values.tobytes()
        cache = self._state_cache
        if (
            cache is None
            or cache.generation != self._generation
            or cache.parameter_bits != parameter_bits
        ):
            native = self.compile()._native.run_cached(self._initial_state, values)
            cache = _FinalStateCache(self._generation, parameter_bits, native)
            self._state_cache = cache
        return cache

    def state(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None
    ) -> np.ndarray[Any, Any]:
        """Return the final state in restricted-sector ordering."""
        cache = self._cached_final(parameters)
        if cache.state is None:
            cache.state = _readonly(
                np.asarray(cache.native.state_array(), dtype=np.complex128)
            )
        return cache.state

    def probability(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None
    ) -> np.ndarray[Any, Any]:
        """Return probabilities in restricted-sector ordering."""
        return _readonly(
            np.asarray(self._cached_final(parameters).native.probability())
        )

    def state_full(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None
    ) -> np.ndarray[Any, Any]:
        """Return the final state expanded to the full computational basis."""
        return _readonly(np.asarray(self._cached_final(parameters).native.to_dense()))

    def probability_full(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None
    ) -> np.ndarray[Any, Any]:
        """Expand and return probabilities over the full computational basis."""
        return _readonly(
            np.asarray(self._cached_final(parameters).native.probability_full())
        )

    def expectation(
        self,
        observable: PauliOperator,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> complex:
        """Return a complex expectation for an arbitrary Pauli observable."""
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        real, imaginary = self._cached_final(parameters).native.expectation(
            *observable._arrays()
        )
        return complex(float(real), float(imaginary))

    def value_and_grad(
        self,
        observable: PauliOperator,
        *,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> U1CircuitValueAndGradient:
        """Return a real value and exact gradient for an exactly Hermitian observable.

        The returned owned array is read-only ``float64`` with one entry per
        parameter slot. ``ValueError`` is raised for a non-Hermitian observable.
        """
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        if not observable._exact_hermitian_value():
            raise ValueError("value_and_grad requires an exactly Hermitian observable")
        value, gradient = self._cached_final(parameters).native.value_and_grad(
            *observable._arrays()
        )
        return U1CircuitValueAndGradient(
            float(value), _readonly(np.asarray(gradient, dtype=np.float64))
        )

    @classmethod
    def from_circuit(
        cls,
        circuit: Any,
        *,
        parameter_order: Optional[Sequence[Any]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "U1Circuit":
        """Convert a supported TensorCircuit U(1) circuit."""
        from .integrations.tensorcircuit import u1_circuit_from_tensorcircuit

        conversion = u1_circuit_from_tensorcircuit(
            circuit, parameter_order=parameter_order, max_bytes=max_bytes
        )
        return conversion.circuit

    def bind_parameters(self, values: Mapping[int, float]) -> "U1Circuit":
        """Return a copy with selected parameter slots replaced by constants."""
        return self._from_program(self, self._program.bind(values))

    def remap_parameters(self, mapping: Mapping[int, int]) -> "U1Circuit":
        """Return a copy with parameter slots renamed according to ``mapping``."""
        return self._from_program(self, self._program.remap(mapping))

    def inverse(self) -> "U1Circuit":
        """Return a copy with the gate sequence inverted."""
        return self._from_program(self, self._program.inverse())

    def append(
        self,
        other: "U1Circuit",
        *,
        parameter_map: Optional[Mapping[int, int]] = None,
    ) -> "U1Circuit":
        """Return the concatenation of two compatible U1 circuits.

        ``parameter_map`` optionally remaps the appended circuit's parameter
        slots before concatenation.
        """
        if not isinstance(other, U1Circuit):
            raise TypeError("other must be a U1Circuit")
        return self._from_program(
            self, self._program.append(other._program, parameter_map)
        )

    def to_qir(self) -> list[dict[str, object]]:
        """Serialize the circuit to deterministic JSON-like gate records."""
        return self._program.to_qir()

    @classmethod
    def from_qir(
        cls,
        qir: Sequence[Mapping[str, object]],
        circuit_params: Mapping[str, object],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "U1Circuit":
        """Restore supported QIR gates; iSWAP angles use the normalized convention."""
        if "nqubits" not in circuit_params:
            raise ValueError("circuit_params must contain nqubits")
        nqubits_value = circuit_params["nqubits"]
        if not isinstance(nqubits_value, int) or isinstance(nqubits_value, bool):
            raise ValueError("circuit_params nqubits must be an integer")
        circuit = cls(
            nqubits_value,
            cast(Optional[int], circuit_params.get("particle_number")),
            occupied=cast(Optional[Sequence[int]], circuit_params.get("occupied")),
            initial_state=cast(Any, circuit_params.get("initial_state")),
            max_bytes=max_bytes,
        )
        parameter_order: tuple[object, ...] = tuple(
            cast(Sequence[object], circuit_params.get("parameter_order", ()))
        )
        symbols: dict[object, Parameter] = {}

        def angle(value: object, default: float) -> Angle:
            if value is None:
                return default
            if isinstance(value, (Parameter, ParameterExpr, int, float)):
                return value
            if parameter_order and value in parameter_order:
                if value not in symbols:
                    symbols[value] = Parameter(parameter_order.index(value))
                return symbols[value]
            raise TypeError(
                "QIR angles must be finite real values or direct parameters"
            )

        for item in qir:
            name_value = item.get("name", item.get("gate"))
            if not isinstance(name_value, str):
                raise ValueError("QIR item must contain a gate name")
            name = name_value.lower()
            wires_value = item.get("index")
            if not isinstance(wires_value, Sequence) or isinstance(
                wires_value, (str, bytes)
            ):
                raise ValueError("QIR item must contain an index sequence")
            wires = tuple(int(cast(Any, wire)) for wire in wires_value)
            parameters = item.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise ValueError("QIR parameters must be a mapping")
            theta = parameters.get("theta")
            if name == "rz":
                circuit.rz(wires[0], angle(theta, 0.0))
            elif name == "rzz":
                circuit.rzz(wires[0], wires[1], angle(theta, 0.0))
            elif name == "cz":
                circuit.cz(wires[0], wires[1])
            elif name == "cphase":
                circuit.cphase(wires[0], wires[1], angle(theta, 0.0))
            elif name == "swap":
                circuit.swap(wires[0], wires[1])
            elif name == "iswap":
                circuit.iswap(wires[0], wires[1], angle(theta, 1.0))
            elif name == "diagonal":
                payload = item.get("diagonal", item.get("diag"))
                if payload is None:
                    raise ValueError(
                        "QIR diagonal item must contain a static diagonal payload"
                    )
                circuit.diagonal(*wires, diagonal=cast(Sequence[complex], payload))
            else:
                raise ValueError(f"unsupported U1Circuit QIR gate {name!r}")
        return circuit


__all__ = ["U1Circuit", "U1CircuitPlan", "U1CircuitValueAndGradient"]
