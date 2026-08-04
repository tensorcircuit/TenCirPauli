"""Backend-neutral logical circuit and parameter-expression primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np


_Real = Union[float, int]
ExpressionOperand = Union[_Real, "Parameter", "ParameterExpr"]


def _real_constant(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("parameter expressions accept finite real scalars only")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("parameter expressions accept finite real scalars only")
    return normalized


def _operand(value: ExpressionOperand) -> Union[float, "Parameter", "ParameterExpr"]:
    if isinstance(value, (Parameter, ParameterExpr)):
        return value
    return _real_constant(value)


def _binary(
    operation: str, left: ExpressionOperand, right: ExpressionOperand
) -> "ParameterExpr":
    return ParameterExpr(operation, (_operand(left), _operand(right)))


@dataclass(frozen=True)
class Parameter:
    """An immutable, reusable non-negative parameter slot identity."""

    slot: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.slot, int)
            or isinstance(self.slot, bool)
            or self.slot < 0
        ):
            raise ValueError("parameter slot must be a non-negative integer")

    def __neg__(self) -> "ParameterExpr":
        return ParameterExpr("neg", (self,))

    def __add__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("add", self, other)

    def __radd__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("add", other, self)

    def __sub__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("sub", self, other)

    def __rsub__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("sub", other, self)

    def __mul__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("mul", self, other)

    def __rmul__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("mul", other, self)

    def __truediv__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("div", self, other)

    def __rtruediv__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("div", other, self)


@dataclass(frozen=True)
class ParameterExpr:
    """An immutable arithmetic DAG node over parameters and real constants."""

    operation: str
    operands: Tuple[Union[float, Parameter, "ParameterExpr"], ...]

    def __post_init__(self) -> None:
        arity = {"neg": 1, "add": 2, "sub": 2, "mul": 2, "div": 2}.get(self.operation)
        if arity is None or len(self.operands) != arity:
            raise ValueError("invalid parameter expression operation or arity")
        normalized = tuple(_operand(item) for item in self.operands)
        object.__setattr__(self, "operands", normalized)

    def __neg__(self) -> "ParameterExpr":
        return ParameterExpr("neg", (self,))

    def __add__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("add", self, other)

    def __radd__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("add", other, self)

    def __sub__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("sub", self, other)

    def __rsub__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("sub", other, self)

    def __mul__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("mul", self, other)

    def __rmul__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("mul", other, self)

    def __truediv__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("div", self, other)

    def __rtruediv__(self, other: ExpressionOperand) -> "ParameterExpr":
        return _binary("div", other, self)


Angle = Union[float, Parameter, ParameterExpr]


@dataclass(frozen=True)
class _LogicalGate:
    name: str
    wires: Tuple[int, ...]
    angle: Optional[Angle] = None
    payload: Optional[Tuple[complex, ...]] = None


def _as_angle(value: Angle) -> Angle:
    if isinstance(value, (Parameter, ParameterExpr)):
        return value
    return _real_constant(value)


def _slot_set(value: Union[float, Parameter, ParameterExpr]) -> set[int]:
    if isinstance(value, Parameter):
        return {value.slot}
    if isinstance(value, ParameterExpr):
        result: set[int] = set()
        for operand in value.operands:
            if isinstance(operand, (Parameter, ParameterExpr)):
                result.update(_slot_set(operand))
        return result
    return set()


def _coerce_parameters(
    parameters: Optional[Sequence[float] | np.ndarray[Any, Any]],
    nparameters: int,
) -> np.ndarray[Any, Any]:
    """Convert a runtime parameter vector to a finite contiguous float64 array."""
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


def _evaluate_angle(
    value: Angle,
    parameters: np.ndarray[Any, Any],
    nparameters: int,
) -> tuple[float, np.ndarray[Any, Any]]:
    """Evaluate one angle and its public-slot Jacobian."""
    if isinstance(value, Parameter):
        if value.slot >= nparameters:
            raise ValueError(f"parameter slot {value.slot} is outside 0..{nparameters}")
        gradient: np.ndarray[Any, Any] = np.zeros(nparameters, dtype=np.float64)
        gradient[value.slot] = 1.0
        return float(parameters[value.slot]), gradient
    if not isinstance(value, ParameterExpr):
        return _real_constant(value), np.zeros(nparameters, dtype=np.float64)

    evaluated = [
        _evaluate_angle(operand, parameters, nparameters) for operand in value.operands
    ]
    left, left_gradient = evaluated[0]
    if value.operation == "neg":
        return -left, -left_gradient
    right, right_gradient = evaluated[1]
    if value.operation == "add":
        return left + right, left_gradient + right_gradient
    if value.operation == "sub":
        return left - right, left_gradient - right_gradient
    if value.operation == "mul":
        return left * right, right * left_gradient + left * right_gradient
    if right == 0.0:
        raise ValueError("parameter expression divides by zero")
    return (
        left / right,
        (right * left_gradient - left * right_gradient) / (right * right),
    )


def _evaluate_angle_value(
    value: Angle, parameters: np.ndarray[Any, Any], nparameters: int
) -> float:
    """Evaluate one angle without allocating its slot Jacobian."""
    if isinstance(value, Parameter):
        if value.slot >= nparameters:
            raise ValueError(f"parameter slot {value.slot} is outside 0..{nparameters}")
        return float(parameters[value.slot])
    if not isinstance(value, ParameterExpr):
        return _real_constant(value)

    left = _evaluate_angle_value(value.operands[0], parameters, nparameters)
    if value.operation == "neg":
        return -left
    right = _evaluate_angle_value(value.operands[1], parameters, nparameters)
    if value.operation == "add":
        return left + right
    if value.operation == "sub":
        return left - right
    if value.operation == "mul":
        return left * right
    if right == 0.0:
        raise ValueError("parameter expression divides by zero")
    return left / right


def _replace_expression(
    value: Union[float, Parameter, ParameterExpr],
    replacements: Mapping[int, Union[float, Parameter]],
) -> Union[float, Parameter, ParameterExpr]:
    if isinstance(value, Parameter):
        return replacements.get(value.slot, value)
    if isinstance(value, float):
        return value
    children = tuple(
        (
            _replace_expression(operand, replacements)
            if isinstance(operand, (Parameter, ParameterExpr))
            else operand
        )
        for operand in value.operands
    )
    if value.operation == "neg" and isinstance(children[0], float):
        return _real_constant(-children[0])
    if value.operation in {"add", "sub", "mul", "div"} and all(
        isinstance(child, float) for child in children
    ):
        left, right = children
        assert isinstance(left, float) and isinstance(right, float)
        if value.operation == "add":
            return _real_constant(left + right)
        if value.operation == "sub":
            return _real_constant(left - right)
        if value.operation == "mul":
            return _real_constant(left * right)
        if right == 0.0:
            raise ValueError("parameter expression divides by zero")
        return _real_constant(left / right)
    return ParameterExpr(value.operation, children)


class _CircuitProgram:
    """Immutable logical tape shared by the public U1 facade."""

    def __init__(self, nqubits: int, operations: Iterable[_LogicalGate] = ()) -> None:
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("nqubits must be a non-negative integer")
        self.nqubits = nqubits
        self.operations = tuple(operations)
        self._validate_operations()

    def _validate_operations(self) -> None:
        slots: set[int] = set()
        for operation in self.operations:
            if operation.name not in {
                "rz",
                "rzz",
                "cz",
                "cphase",
                "swap",
                "iswap",
                "diagonal",
            }:
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
                _as_angle(operation.angle)
                slots.update(_slot_set(operation.angle))
        if slots != set(range(max(slots, default=-1) + 1)):
            raise ValueError(
                "parameter slots must cover 0..nparameters-1 without holes"
            )

    @property
    def nparameters(self) -> int:
        slots: set[int] = set()
        for operation in self.operations:
            if operation.angle is not None:
                slots.update(_slot_set(operation.angle))
        return max(slots, default=-1) + 1

    def with_operations(self, operations: Iterable[_LogicalGate]) -> "_CircuitProgram":
        return _CircuitProgram(self.nqubits, operations)

    def bind(self, values: Mapping[int, float]) -> "_CircuitProgram":
        normalized: dict[int, float] = {}
        for slot, value in values.items():
            if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
                raise ValueError("parameter binding keys must be non-negative integers")
            normalized[slot] = _real_constant(value)
        replacements: dict[int, Union[float, Parameter]] = {
            slot: value for slot, value in normalized.items()
        }
        replaced = []
        for operation in self.operations:
            angle = (
                None
                if operation.angle is None
                else _replace_expression(operation.angle, replacements)
            )
            replaced.append(
                _LogicalGate(operation.name, operation.wires, angle, operation.payload)
            )
        remaining = sorted(
            slot
            for operation in replaced
            if operation.angle is not None
            for slot in _slot_set(operation.angle)
        )
        compact = {
            slot: Parameter(index)
            for index, slot in enumerate(dict.fromkeys(remaining))
        }
        return self.with_operations(
            _LogicalGate(
                operation.name,
                operation.wires,
                (
                    None
                    if operation.angle is None
                    else _replace_expression(operation.angle, compact)
                ),
                operation.payload,
            )
            for operation in replaced
        )

    def remap(self, mapping: Mapping[int, int]) -> "_CircuitProgram":
        normalized: dict[int, Parameter] = {}
        for old, new in mapping.items():
            if (
                not isinstance(old, int)
                or isinstance(old, bool)
                or old < 0
                or not isinstance(new, int)
                or isinstance(new, bool)
                or new < 0
            ):
                raise ValueError("parameter remapping must use non-negative integers")
            normalized[old] = Parameter(new)
        operations = []
        for operation in self.operations:
            angle = (
                None
                if operation.angle is None
                else _replace_expression(operation.angle, normalized)
            )
            operations.append(
                _LogicalGate(operation.name, operation.wires, angle, operation.payload)
            )
        slots = {
            slot
            for operation in operations
            if operation.angle is not None
            for slot in _slot_set(operation.angle)
        }
        if slots != set(range(max(slots, default=-1) + 1)):
            raise ValueError("remapped parameter slots must be contiguous")
        return self.with_operations(operations)

    def inverse(self) -> "_CircuitProgram":
        operations = []
        for operation in reversed(self.operations):
            angle = operation.angle
            payload = operation.payload
            if operation.name in {"rz", "rzz", "cphase", "iswap"}:
                assert angle is not None
                angle = (
                    -angle
                    if isinstance(angle, (Parameter, ParameterExpr))
                    else -float(angle)
                )
            if operation.name == "diagonal":
                assert payload is not None
                payload = tuple(value.conjugate() for value in payload)
            operations.append(
                _LogicalGate(operation.name, operation.wires, angle, payload)
            )
        return self.with_operations(operations)

    def append(
        self,
        other: "_CircuitProgram",
        parameter_map: Optional[Mapping[int, int]] = None,
    ) -> "_CircuitProgram":
        if self.nqubits != other.nqubits:
            raise ValueError("circuits must have the same nqubits")
        mapping = parameter_map or {}
        remapped = other.remap(mapping) if mapping else other
        operations = self.operations + remapped.operations
        result = self.with_operations(operations)
        slots = {
            slot
            for operation in result.operations
            if operation.angle is not None
            for slot in _slot_set(operation.angle)
        }
        if slots != set(range(max(slots, default=-1) + 1)):
            raise ValueError("appended parameter slots must be contiguous")
        return result

    def to_qir(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for operation in self.operations:
            item: dict[str, object] = {
                "name": operation.name,
                "index": operation.wires,
            }
            if operation.angle is not None:
                item["parameters"] = {"theta": operation.angle}
            if operation.payload is not None:
                item["diagonal"] = operation.payload
            result.append(item)
        return result


def _gate(
    name: str,
    wires: Sequence[int],
    angle: Optional[Angle] = None,
    payload: Optional[Sequence[complex]] = None,
) -> _LogicalGate:
    return _LogicalGate(
        name,
        tuple(wires),
        None if angle is None else _as_angle(angle),
        None if payload is None else tuple(complex(value) for value in payload),
    )


__all__ = ["Parameter", "ParameterExpr"]
