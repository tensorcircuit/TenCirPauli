"""Backend-neutral parameter and logical circuit contract tests."""

from __future__ import annotations

import pytest

import tencirpauli as tcp


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
