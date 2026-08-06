"""Tests for the numerical backend-neutral circuit IR."""

from __future__ import annotations

import pytest

from tencirpauli import _native


def test_symbolic_parameter_exports_are_removed() -> None:
    import tencirpauli as tcp

    assert not hasattr(tcp, "Parameter")
    assert not hasattr(tcp, "ParameterExpr")


def test_native_ir_rejects_schema_mismatch_and_parameter_holes() -> None:
    with pytest.raises(ValueError, match="schema"):
        _native.u1_circuit_plan(2, 1, 999, 0, [], [], 1 << 30)
    with pytest.raises(ValueError, match="holes"):
        _native.u1_circuit_plan(
            2,
            1,
            1,
            3,
            [(0, 0.0), (1, 0.0)],
            [],
            1 << 30,
        )
