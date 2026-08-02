"""Lazy TensorCircuit-semantics-compatible Rust-native U(1) circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, cast

import numpy as np

from . import _native
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
        if len(codes) != nqubits or any(
            not isinstance(code, int) or isinstance(code, bool) or code not in range(4)
            for code in codes
        ):
            raise ValueError("ps must contain exactly nqubits codes in 0..3")
        return codes
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
    """Immutable compiled U(1) execution plan."""

    sector: U1Sector
    dimension: int
    nparameters: int
    _native: Any

    def __init__(self, sector: U1Sector, native_plan: Any) -> None:
        object.__setattr__(self, "sector", sector)
        object.__setattr__(self, "dimension", int(native_plan.dimension))
        object.__setattr__(self, "nparameters", int(native_plan.nparameters))
        object.__setattr__(self, "_native", native_plan)

    def _params(
        self, parameters: Sequence[float] | np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        return _parameter_array(parameters, self.nparameters)

    def run(
        self,
        initial_state: Sequence[complex] | np.ndarray[Any, Any],
        parameters: Sequence[float] | np.ndarray[Any, Any] = (),
    ) -> np.ndarray[Any, Any]:
        result = self._native.run(
            _state_array(initial_state, self.dimension), self._params(parameters)
        )
        return _readonly(np.asarray(result, dtype=np.complex128))

    def probability(
        self,
        initial_state: Sequence[complex] | np.ndarray[Any, Any],
        parameters: Sequence[float] | np.ndarray[Any, Any] = (),
    ) -> np.ndarray[Any, Any]:
        result = self._native.probability(
            _state_array(initial_state, self.dimension), self._params(parameters)
        )
        return _readonly(np.asarray(result, dtype=np.float64))

    def to_dense(
        self,
        initial_state: Sequence[complex] | np.ndarray[Any, Any],
        parameters: Sequence[float] | np.ndarray[Any, Any] = (),
    ) -> np.ndarray[Any, Any]:
        result = self._native.to_dense(
            _state_array(initial_state, self.dimension), self._params(parameters)
        )
        return _readonly(np.asarray(result, dtype=np.complex128))

    def probability_full(
        self,
        initial_state: Sequence[complex] | np.ndarray[Any, Any],
        parameters: Sequence[float] | np.ndarray[Any, Any] = (),
    ) -> np.ndarray[Any, Any]:
        result = self._native.probability_full(
            _state_array(initial_state, self.dimension), self._params(parameters)
        )
        return _readonly(np.asarray(result, dtype=np.float64))

    def expectation(
        self,
        initial_state: Sequence[complex] | np.ndarray[Any, Any],
        observable: PauliOperator,
        parameters: Sequence[float] | np.ndarray[Any, Any] = (),
    ) -> complex:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        real, imaginary = self._native.expectation(
            _state_array(initial_state, self.dimension),
            *observable._arrays(),
            self._params(parameters),
        )
        return complex(float(real), float(imaginary))

    def value_and_grad(
        self,
        initial_state: Sequence[complex] | np.ndarray[Any, Any],
        observable: PauliOperator,
        parameters: Sequence[float] | np.ndarray[Any, Any],
    ) -> U1CircuitValueAndGradient:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        value, gradient = self._native.value_and_grad(
            _state_array(initial_state, self.dimension),
            *observable._arrays(),
            self._params(parameters),
        )
        return U1CircuitValueAndGradient(
            float(value), _readonly(np.asarray(gradient, dtype=np.float64))
        )


class U1Circuit:
    """Lazy, fixed-particle-number circuit executed by the Rust backend."""

    def __init__(
        self,
        nqubits: int,
        k: Optional[int] = None,
        filled: Optional[Sequence[int]] = None,
        inputs: Optional[Sequence[complex] | np.ndarray[Any, Any]] = None,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("nqubits must be a non-negative integer")
        _validate_max_bytes(max_bytes)
        if k is None and filled is None:
            raise ValueError("either k or filled must be provided")
        normalized_filled = None if filled is None else tuple(filled)
        if normalized_filled is not None:
            if any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= nqubits
                for index in normalized_filled
            ):
                raise ValueError("filled indices must be distinct in-range integers")
            if len(set(normalized_filled)) != len(normalized_filled):
                raise ValueError("filled indices must be distinct")
            if k is not None and len(normalized_filled) != k:
                raise ValueError("k must equal len(filled)")
            if k is None:
                k = len(normalized_filled)
        assert k is not None
        if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= nqubits:
            raise ValueError("k must be between 0 and nqubits")
        if normalized_filled is None:
            normalized_filled = tuple(range(k))
        self.nqubits = nqubits
        self.k = k
        self.max_bytes = max_bytes
        self.sector = U1Sector(nqubits, k)
        dimension = self.sector.dimension
        _check_allocation(
            dimension * np.dtype(np.complex128).itemsize,
            max_bytes,
            "U1 circuit initial state",
        )
        if inputs is None:
            basis_value = sum(1 << (nqubits - 1 - index) for index in normalized_filled)
            initial_index = self.sector.rank(basis_value)
            initial: np.ndarray[Any, Any] = np.zeros(dimension, dtype=np.complex128)
            initial[initial_index] = 1.0
        else:
            initial = _state_array(inputs, dimension).copy()
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
        result.k = other.k
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
        return self._program.nparameters

    @property
    def dimension(self) -> int:
        return self.sector.dimension

    def _append(self, operation: _LogicalGate) -> None:
        self._program = self._program.with_operations(
            (*self._program.operations, operation)
        )
        self._native_plan = None
        self._state_cache = None
        self._generation += 1

    def rz(self, i: int, theta: Angle = 0.0) -> None:
        self._append(_gate("rz", (i,), theta))

    def rzz(self, i: int, j: int, theta: Angle = 0.0) -> None:
        self._append(_gate("rzz", (i, j), theta))

    def cz(self, i: int, j: int) -> None:
        self._append(_gate("cz", (i, j)))

    def cphase(self, i: int, j: int, theta: Angle = 0.0) -> None:
        self._append(_gate("cphase", (i, j), theta))

    def swap(self, i: int, j: int) -> None:
        self._append(_gate("swap", (i, j)))

    def iswap(self, i: int, j: int, theta: Angle = 1.0) -> None:
        self._append(_gate("iswap", (i, j), theta))

    def diagonal(
        self, *indices: int, diag: Sequence[complex] | np.ndarray[Any, Any]
    ) -> None:
        values = np.asarray(diag, dtype=np.complex128).reshape(-1)
        expected = 1 << len(indices)
        if values.shape != (expected,):
            raise ValueError(f"diag must have shape ({expected},), got {values.shape}")
        self._append(_gate("diagonal", indices, payload=values.tolist()))

    def compile(self) -> U1CircuitPlan:
        if self._native_plan is None:
            expression_nodes, gates = _encode_program(self._program)
            native = _native.u1_circuit_plan(
                self.nqubits,
                self.k,
                1,
                self.nparameters,
                expression_nodes,
                gates,
                _effective_max_bytes(self.max_bytes),
            )
            self._native_plan = U1CircuitPlan(self.sector, native)
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
        cache = self._cached_final(parameters)
        if cache.state is None:
            cache.state = _readonly(
                np.asarray(cache.native.state_array(), dtype=np.complex128)
            )
        return cache.state

    wavefunction = state

    def probability(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None
    ) -> np.ndarray[Any, Any]:
        return _readonly(
            np.asarray(self._cached_final(parameters).native.probability())
        )

    def to_dense(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None
    ) -> np.ndarray[Any, Any]:
        return _readonly(np.asarray(self._cached_final(parameters).native.to_dense()))

    def probability_full(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None
    ) -> np.ndarray[Any, Any]:
        return _readonly(
            np.asarray(self._cached_final(parameters).native.probability_full())
        )

    def expectation_z(
        self,
        i: int,
        *,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> float:
        value = self.expectation_ps(z=(i,), parameters=parameters)
        return float(value.real)

    def expectation_ps(
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        ps: Optional[Sequence[int]] = None,
        *,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> complex:
        codes = _pauli_codes(self.nqubits, x, y, z, ps)
        observable = PauliOperator(self.nqubits, [(codes, 1.0)])
        real, imaginary = self._cached_final(parameters).native.expectation(
            *observable._arrays()
        )
        return complex(float(real), float(imaginary))

    def expectation_pss(
        self,
        ps_list: Sequence[object],
        coefficients: Sequence[complex] | np.ndarray[Any, Any],
        *,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
    ) -> complex:
        normalized = tuple(_normalize_ps(value, self.nqubits) for value in ps_list)
        values = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
        if len(normalized) != values.shape[0]:
            raise ValueError("ps_list and coefficients must have the same length")
        if not np.isfinite(values).all():
            raise ValueError("coefficients must be finite")
        observable = PauliOperator(self.nqubits, list(zip(normalized, values.tolist())))
        real, imaginary = self._cached_final(parameters).native.expectation(
            *observable._arrays()
        )
        return complex(float(real), float(imaginary))

    def value_and_grad(
        self,
        observable: PauliOperator,
        *,
        parameters: Sequence[float] | np.ndarray[Any, Any],
    ) -> U1CircuitValueAndGradient:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        value, gradient = self._cached_final(parameters).native.value_and_grad(
            *observable._arrays()
        )
        return U1CircuitValueAndGradient(
            float(value), _readonly(np.asarray(gradient, dtype=np.float64))
        )

    def bind_parameters(self, values: Mapping[int, float]) -> "U1Circuit":
        return self._from_program(self, self._program.bind(values))

    def remap_parameters(self, mapping: Mapping[int, int]) -> "U1Circuit":
        return self._from_program(self, self._program.remap(mapping))

    def inverse(self) -> "U1Circuit":
        return self._from_program(self, self._program.inverse())

    def append(
        self,
        other: "U1Circuit",
        *,
        parameter_map: Optional[Mapping[int, int]] = None,
    ) -> "U1Circuit":
        if not isinstance(other, U1Circuit):
            raise TypeError("other must be a U1Circuit")
        return self._from_program(
            self, self._program.append(other._program, parameter_map)
        )

    def to_qir(self) -> list[dict[str, object]]:
        return self._program.to_qir()

    @classmethod
    def from_qir(
        cls,
        qir: Sequence[Mapping[str, object]],
        circuit_params: Mapping[str, object],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "U1Circuit":
        """Restore the supported logical gates from TensorCircuit-style QIR."""
        if "nqubits" not in circuit_params:
            raise ValueError("circuit_params must contain nqubits")
        nqubits_value = circuit_params["nqubits"]
        if not isinstance(nqubits_value, int) or isinstance(nqubits_value, bool):
            raise ValueError("circuit_params nqubits must be an integer")
        circuit = cls(
            nqubits_value,
            cast(Optional[int], circuit_params.get("k")),
            cast(Optional[Sequence[int]], circuit_params.get("filled")),
            cast(Any, circuit_params.get("inputs")),
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
                if "diag" not in item:
                    raise ValueError(
                        "QIR diagonal item must contain a static diag payload"
                    )
                circuit.diagonal(*wires, diag=cast(Sequence[complex], item["diag"]))
            else:
                raise ValueError(f"unsupported U1Circuit QIR gate {name!r}")
        return circuit


__all__ = ["U1Circuit", "U1CircuitPlan", "U1CircuitValueAndGradient"]
