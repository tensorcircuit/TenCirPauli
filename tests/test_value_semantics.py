"""Equality, hashing, and residency contracts for native-backed values."""

from __future__ import annotations

import math
from typing import Any, Callable

import pytest

import tencirpauli as tcp
from tencirpauli.structured import _StructuredOperator


def _constructors() -> tuple[Callable[[complex], object], ...]:
    space = tcp.OperatorSpace(qubits=1)
    return (
        lambda coefficient: tcp.PauliOperator.from_terms(1, [("X", coefficient)]),
        lambda coefficient: tcp.MajoranaOperator.from_terms(1, [((0,), coefficient)]),
        lambda coefficient: tcp.FermionOperator.from_terms(
            1, [(((0, "create"), (0, "annihilate")), coefficient)]
        ),
        lambda coefficient: tcp.BosonOperator.from_terms(
            1, [(((0, "create"), (0, "annihilate")), coefficient)]
        ),
        lambda coefficient: tcp.QuditWeylOperator.from_terms(
            3, [(((0, 0, 0),), coefficient)], n_sites=1
        ),
        lambda coefficient: space.qubit.z(0).scale(coefficient),
    )


def test_equal_native_values_hash_equal_for_signed_zero_and_complex_coefficients() -> (
    None
):
    for construct in _constructors():
        signed_plus = construct(complex(0.75, 0.0))
        signed_minus = construct(complex(0.75, -0.0))
        ordinary_left = construct(complex(0.75, 0.25))
        ordinary_right = construct(complex(0.75, 0.25))
        unequal = construct(complex(0.75, 0.5))

        assert signed_plus == signed_minus
        assert hash(signed_plus) == hash(signed_minus)
        assert ordinary_left == ordinary_right
        assert hash(ordinary_left) == hash(ordinary_right)
        assert ordinary_left != unequal


def test_real_native_adjoint_hashes_equal_across_operator_families() -> None:
    space = tcp.OperatorSpace(qubits=1)
    operators: tuple[Any, ...] = (
        tcp.PauliOperator.from_terms(1, [("X", 1.25)]),
        tcp.MajoranaOperator.from_terms(1, [((0,), 1.25)]),
        tcp.FermionOperator.from_terms(1, [(((0, "create"), (0, "annihilate")), 1.25)]),
        tcp.BosonOperator.from_terms(1, [(((0, "create"), (0, "annihilate")), 1.25)]),
        tcp.QuditWeylOperator.from_terms(3, [(((0, 0, 0),), 1.25)], n_sites=1),
        space.qubit.z(0).scale(1.25),
    )
    for operator in operators:
        adjoint = operator.adjoint()
        assert operator == adjoint
        assert hash(operator) == hash(adjoint)


def test_hash_and_equality_do_not_materialize_native_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_materialization(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("value semantics must stay on native storage")

    monkeypatch.setattr(tcp.PauliOperator, "_arrays", forbidden_materialization)
    monkeypatch.setattr(
        _StructuredOperator, "_materialized_terms", forbidden_materialization
    )
    monkeypatch.setattr(
        tcp.MajoranaOperator, "terms", property(forbidden_materialization)
    )
    for construct in _constructors():
        left = construct(complex(0.5, 0.25))
        right = construct(complex(0.5, 0.25))
        assert left == right
        assert hash(left) == hash(right)


def test_nonfinite_coefficients_are_rejected_consistently() -> None:
    for construct in _constructors():
        with pytest.raises(ValueError, match="finite"):
            construct(complex(math.inf, 0.0))
        with pytest.raises(ValueError, match="finite"):
            construct(complex(0.0, math.nan))
