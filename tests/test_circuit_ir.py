"""Backend-neutral parameter and logical circuit contract tests."""

from __future__ import annotations

import pytest

import tencirpauli as tcp
from tencirpauli import _native


def test_parameter_expression_is_structural_and_reusable() -> None:
    p0 = tcp.Parameter(0)
    p1 = tcp.Parameter(1)
    expression = 2.0 * p0 - p1 / 3.0
    assert isinstance(expression, tcp.ParameterExpr)
    assert expression == 2.0 * p0 - p1 / 3.0
    assert p0 == tcp.Parameter(0)
    assert p0 != p1


def test_parameter_rejects_bool_and_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        tcp.Parameter(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tcp.ParameterExpr("add", (1.0, float("inf")))


def test_parameter_arithmetic_rejects_complex_operands() -> None:
    with pytest.raises(TypeError):
        _ = tcp.Parameter(0) + (1.0 + 2.0j)  # type: ignore[operator]


def test_expression_nodes_preserve_operand_order() -> None:
    p0 = tcp.Parameter(0)
    p1 = tcp.Parameter(1)
    left = p0 - p1
    right = p1 - p0
    assert left != right
    assert left.operands != right.operands


def test_native_ir_rejects_schema_mismatch_and_parameter_holes() -> None:
    with pytest.raises(ValueError, match="schema"):
        _native.u1_circuit_plan(2, 1, 999, 0, [], [], 1 << 30)
    with pytest.raises(ValueError, match="holes"):
        _native.u1_circuit_plan(
            2,
            1,
            1,
            3,
            [(1, 2, 0, 0.0)],
            [(5, 0, 1, 0, [], [], [])],
            1 << 30,
        )
