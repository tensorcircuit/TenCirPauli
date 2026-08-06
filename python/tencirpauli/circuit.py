"""Private, backend-neutral circuit records.

The public circuit facades store the actual angle supplied to each gate.  A
private gradient lowering pass assigns occurrence slots when a native
value-and-gradient terminal is requested; no symbolic parameter language is
part of this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Optional, Sequence, Tuple, Union, cast

import numpy as np


RealAngle = Union[float, int, np.floating[Any]]
Angle = Any


def _is_jax_value(value: object) -> bool:
    """Recognize scalar JAX arrays/tracers without importing JAX."""
    module = type(value).__module__
    return module.startswith("jax.") or module.startswith("jaxlib.")


def _angle(value: object) -> Angle:
    """Validate a concrete angle or retain a scalar JAX value for tracing."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("gate angle must be a finite real scalar")
    if isinstance(value, (int, float, np.integer, np.floating)):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("gate angle must be finite")
        return normalized
    if _is_jax_value(value):
        ndim = getattr(value, "ndim", None)
        dtype = getattr(value, "dtype", None)
        if ndim != 0 or dtype is None or np.issubdtype(dtype, np.complexfloating):
            raise TypeError("gate angle must be a scalar real value")
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("gate angle must be a scalar real value")
        return value
    raise TypeError("gate angle must be a finite real scalar")


def _concrete_angle(value: object) -> float:
    """Convert one stored angle at an ordinary native terminal boundary."""
    if _is_jax_value(value):
        raise TypeError(
            "JAX-valued angles require expectation_jax(); native terminals "
            "accept concrete finite real scalars"
        )
    normalized = _angle(value)
    if not isinstance(normalized, (int, float, np.floating)):
        raise TypeError("gate angle must be a concrete finite real scalar")
    return float(normalized)


def _concrete_angles(angles: Iterable[object]) -> np.ndarray[Any, Any]:
    """Return the occurrence-ordered concrete angle vector."""
    values = np.fromiter((_concrete_angle(angle) for angle in angles), dtype=np.float64)
    return cast(np.ndarray[Any, Any], np.ascontiguousarray(values))


def _coerce_parameters(
    parameters: Optional[Sequence[float] | np.ndarray[Any, Any]],
    nparameters: int,
) -> np.ndarray[Any, Any]:
    """Validate the flat numerical ABI used by the advanced raw engines."""
    if parameters is None:
        if nparameters != 0:
            raise ValueError("parameters are required for this native tape")
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


@dataclass(frozen=True)
class _LogicalGate:
    name: str
    wires: Tuple[int, ...]
    angle: Optional[Angle] = None
    payload: Optional[Tuple[complex, ...]] = None


def _gate(
    name: str,
    wires: Sequence[int],
    angle: Optional[object] = None,
    payload: Optional[Sequence[complex]] = None,
) -> _LogicalGate:
    return _LogicalGate(
        name,
        tuple(wires),
        None if angle is None else _angle(angle),
        None if payload is None else tuple(complex(value) for value in payload),
    )


class _CircuitProgram:
    """Small immutable U(1) logical tape containing concrete angles only."""

    _SUPPORTED: ClassVar[set[str]] = {
        "rz",
        "rzz",
        "cz",
        "cphase",
        "swap",
        "iswap",
        "diagonal",
    }

    def __init__(self, nqubits: int, operations: Iterable[_LogicalGate] = ()) -> None:
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("nqubits must be a non-negative integer")
        self.nqubits = nqubits
        self.operations = tuple(operations)
        self._validate()

    def _validate(self) -> None:
        for operation in self.operations:
            if operation.name not in self._SUPPORTED:
                raise ValueError(f"unsupported circuit gate {operation.name!r}")
            if not operation.wires or any(
                not isinstance(wire, int)
                or isinstance(wire, bool)
                or wire < 0
                or wire >= self.nqubits
                for wire in operation.wires
            ):
                raise ValueError("gate wire is outside the circuit")
            if len(set(operation.wires)) != len(operation.wires):
                raise ValueError("gate wires must be distinct")
            if (
                operation.name in {"rzz", "cz", "cphase", "swap", "iswap"}
                and len(operation.wires) != 2
            ):
                raise ValueError("two-qubit gate requires exactly two wires")
            if operation.name == "rz" and len(operation.wires) != 1:
                raise ValueError("rz requires exactly one wire")
            if operation.name == "diagonal":
                if operation.payload is None:
                    raise ValueError("diagonal requires a static payload")
                expected = 1 << len(operation.wires)
                if len(operation.payload) != expected:
                    raise ValueError("diagonal payload length does not match arity")
                if any(
                    not math.isfinite(value.real)
                    or not math.isfinite(value.imag)
                    or abs(abs(value) - 1.0) > 1e-12
                    for value in operation.payload
                ):
                    raise ValueError("diagonal payload must be finite and unit modulus")
            if operation.angle is not None:
                _angle(operation.angle)

    @property
    def angle_count(self) -> int:
        return sum(operation.angle is not None for operation in self.operations)

    def with_operations(self, operations: Iterable[_LogicalGate]) -> "_CircuitProgram":
        return _CircuitProgram(self.nqubits, operations)

    def inverse(self) -> "_CircuitProgram":
        operations: list[_LogicalGate] = []
        for operation in reversed(self.operations):
            angle = operation.angle
            payload = operation.payload
            if operation.name in {"rz", "rzz", "cphase", "iswap"}:
                assert angle is not None
                angle = -angle
            if operation.name == "diagonal":
                assert payload is not None
                payload = tuple(value.conjugate() for value in payload)
            operations.append(
                _LogicalGate(operation.name, operation.wires, angle, payload)
            )
        return self.with_operations(operations)

    def append(self, other: "_CircuitProgram") -> "_CircuitProgram":
        if self.nqubits != other.nqubits:
            raise ValueError("circuits must have the same nqubits")
        return self.with_operations(self.operations + other.operations)

    def to_qir(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for operation in self.operations:
            item: dict[str, object] = {"name": operation.name, "index": operation.wires}
            if operation.angle is not None:
                item["parameters"] = {"theta": _concrete_angle(operation.angle)}
            if operation.payload is not None:
                item["diagonal"] = operation.payload
            result.append(item)
        return result


__all__: list[str] = []
