"""P2 PauliOperator canonicalization and algebra differential tests."""

from __future__ import annotations

import numpy as np
import pytest
from reference import dense_operator

from tencirpauli import PauliOperator, PauliWord


def as_dense(operator: PauliOperator) -> np.ndarray:
    return dense_operator(
        operator.nqubits,
        (term.word.to_codes() for term in operator.terms),
        tuple(term.coefficient for term in operator.terms),
    )


def test_canonical_order_duplicate_aggregation_and_exact_cancellation() -> None:
    operator = PauliOperator.from_terms(
        2,
        (("ZX", 1.0), ("II", 2.0), ("ZX", -0.25j), ("II", -2.0)),
    )
    assert tuple(term.word.to_string() for term in operator.terms) == ("ZX",)
    assert operator.terms[0].coefficient == 1.0 - 0.25j
    assert PauliOperator.empty(2).terms == ()


def test_random_operator_algebra_matches_numpy_dense_reference() -> None:
    rng = np.random.default_rng(20260801)
    for nqubits in range(4):
        structures_left = rng.integers(0, 4, size=(6, nqubits)).tolist()
        structures_right = rng.integers(0, 4, size=(5, nqubits)).tolist()
        coefficients_left = rng.normal(size=6) + 1j * rng.normal(size=6)
        coefficients_right = rng.normal(size=5) + 1j * rng.normal(size=5)
        left = PauliOperator.from_terms(
            nqubits, zip(structures_left, coefficients_left.tolist())
        )
        right = PauliOperator.from_terms(
            nqubits, zip(structures_right, coefficients_right.tolist())
        )
        left_dense = dense_operator(nqubits, structures_left, coefficients_left)
        right_dense = dense_operator(nqubits, structures_right, coefficients_right)
        np.testing.assert_allclose(as_dense(left), left_dense, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(as_dense(right), right_dense, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            as_dense(left + right), left_dense + right_dense, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            as_dense(left.multiply(right)),
            left_dense @ right_dense,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            as_dense(left.commutator(right)),
            left_dense @ right_dense - right_dense @ left_dense,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            as_dense(left.anticommutator(right)),
            left_dense @ right_dense + right_dense @ left_dense,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            as_dense(left.adjoint()), left_dense.conj().T, rtol=1e-12, atol=1e-12
        )


def test_complex_scaling_and_hermiticity_validation() -> None:
    operator = PauliOperator.from_terms(1, (("X", 2.0), ("Y", -1.5j)))
    np.testing.assert_allclose(
        as_dense(operator.scale(0.5 + 0.25j)), (0.5 + 0.25j) * as_dense(operator)
    )
    assert operator.is_hermitian() is False
    hermitian = PauliOperator.from_terms(2, (("XX", 1.0), ("IZ", -2.0)))
    assert hermitian.is_hermitian()
    with pytest.raises(ValueError, match="finite"):
        operator.scale(complex(float("nan"), 0.0))
    with pytest.raises(ValueError, match="tolerance"):
        hermitian.is_hermitian(-1.0)


def test_operator_input_and_output_qubit_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="expected 2 qubits"):
        PauliOperator.from_terms(2, (("X", 1.0),))
    with pytest.raises(ValueError, match="incompatible qubit counts"):
        PauliOperator.empty(1).add(PauliOperator.empty(2))
    with pytest.raises(ValueError, match="finite"):
        PauliOperator.from_terms(1, ((PauliWord.from_string("X"), float("inf")),))
