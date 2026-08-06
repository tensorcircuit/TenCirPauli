"""Numerical tests for the canonical spin-orbital integral constructor."""

from __future__ import annotations

import numpy as np
import pytest

import tencirpauli as tcp


def test_from_integrals_applies_one_half_and_keeps_nuclear_constant() -> None:
    one_body = np.array([[1.0, 0.2], [0.2, -0.4]], dtype=np.float64)
    two_body = np.zeros((2, 2, 2, 2), dtype=np.float64)
    two_body[0, 1, 0, 1] = 2.0
    operator = tcp.FermionOperator.from_integrals(one_body, two_body, constant=1.25)
    expected = tcp.FermionOperator.from_terms(
        2,
        (
            ((), 1.25),
            (((0, "create"), (0, "annihilate")), 1.0),
            (((0, "create"), (1, "annihilate")), 0.2),
            (((1, "create"), (0, "annihilate")), 0.2),
            (((1, "create"), (1, "annihilate")), -0.4),
            (
                (
                    (0, "create"),
                    (1, "create"),
                    (1, "annihilate"),
                    (0, "annihilate"),
                ),
                1.0,
            ),
        ),
    )
    np.testing.assert_allclose(operator.compile("dense"), expected.compile("dense"))
    assert operator.terms[0].coefficient == 1.25


def test_from_integrals_normalizes_hermitian_roundoff_without_cutoff() -> None:
    one_body = np.array([[1.0, 0.2 + 1.0e-13j], [0.2, 2.0]], dtype=np.complex128)
    two_body = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    two_body[0, 1, 0, 1] = 2.0 + 1.0e-13j
    operator = tcp.FermionOperator.from_integrals(one_body, two_body)
    assert operator.is_hermitian()
    off_diagonal = next(
        term
        for term in operator.terms
        if term.word.creation_modes == (0,) and term.word.annihilation_modes == (1,)
    )
    assert off_diagonal.coefficient == pytest.approx(0.2 + 0.5e-13j)


@pytest.mark.parametrize(
    "one_body, two_body, message",
    [
        (np.ones((2, 2), dtype=np.float32), np.zeros((2, 2, 2, 2)), "dtype"),
        (np.ones((2, 2), order="F"), np.zeros((2, 2, 2, 2)), "C-contiguous"),
        (
            np.eye(2),
            np.arange(16, dtype=np.float64).reshape((2, 2, 2, 2)),
            "Hermitian pair symmetry",
        ),
    ],
)
def test_from_integrals_rejects_ambiguous_input_contract(
    one_body: np.ndarray, two_body: np.ndarray, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        tcp.FermionOperator.from_integrals(one_body, two_body)


def test_from_integrals_checks_constant_and_memory_budget() -> None:
    zeros = np.zeros((1, 1), dtype=np.float64)
    two_body = np.zeros((1, 1, 1, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="constant must be real"):
        tcp.FermionOperator.from_integrals(zeros, two_body, constant=1.0 + 1.0j)
    with pytest.raises(MemoryError):
        tcp.FermionOperator.from_integrals(zeros, two_body, constant=1.0, max_bytes=1)


def test_from_integrals_rejects_nonfinite_values_in_native_validation() -> None:
    one_body = np.array([[np.nan]], dtype=np.float64)
    two_body = np.zeros((1, 1, 1, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        tcp.FermionOperator.from_integrals(one_body, two_body)
