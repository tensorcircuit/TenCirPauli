"""fixed-vector fixed vectors and randomized checks for the independent dense oracle."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from reference import (
    PAULI_MATRICES,
    codes_to_dense,
    codes_to_symplectic,
    commutes,
    dense_operator,
    multiply_codes,
    product_single,
    support,
    symplectic_to_codes,
)


def test_complete_single_qubit_multiplication_table() -> None:
    for left, right in product(range(4), repeat=2):
        result, phase = product_single(left, right)
        expected = PAULI_MATRICES[left] @ PAULI_MATRICES[right]
        np.testing.assert_allclose(
            phase * PAULI_MATRICES[result], expected, rtol=0.0, atol=0.0
        )


def test_xy_and_yx_have_opposite_exact_phases() -> None:
    assert product_single(1, 2) == (3, 1j)
    assert product_single(2, 1) == (3, -1j)


def test_adjoint_commutation_weight_and_support_vectors() -> None:
    codes = (1, 2, 0, 3)
    matrix = codes_to_dense(codes)
    np.testing.assert_array_equal(matrix.conj().T, matrix)
    assert len(support(codes)) == 3
    x_left, z_left = codes_to_symplectic(codes)
    x_right, z_right = codes_to_symplectic((1, 0, 0, 3))
    assert tuple(index for index, code in enumerate(codes) if code) == (0, 1, 3)
    assert commutes(codes, (1, 0, 0, 3))
    assert (
        bin(x_left & z_right).count("1") + bin(z_left & x_right).count("1")
    ) % 2 == 0


def test_qubit_zero_is_msb_while_reference_packed_bit_zero_is_lsb() -> None:
    np.testing.assert_array_equal(
        codes_to_dense((1, 0)),
        np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),
    )
    assert codes_to_symplectic((1, 0)) == (1, 0)
    assert codes_to_symplectic((0, 1)) == (2, 0)
    assert symplectic_to_codes(1, 0, 2) == (1, 0)
    assert symplectic_to_codes(2, 0, 2) == (0, 1)


@pytest.mark.parametrize("code", (1, 2, 3))
def test_first_and_last_qubit_matrix_vectors(code: int) -> None:
    first = codes_to_dense((code, 0, 0))
    last = codes_to_dense((0, 0, code))
    assert not np.array_equal(first, last)
    np.testing.assert_allclose(first.conj().T, first, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(last.conj().T, last, atol=0.0, rtol=0.0)


def test_empty_identity_and_duplicate_exact_cancellation() -> None:
    np.testing.assert_array_equal(
        codes_to_dense(()), np.ones((1, 1), dtype=np.complex128)
    )
    np.testing.assert_array_equal(
        dense_operator(2, ((0, 0), (1, 2), (1, 2)), (1, 1.5, -1.5)),
        codes_to_dense((0, 0)),
    )
    np.testing.assert_array_equal(
        dense_operator(1, ((1,), (1,)), (1.0, -1.0)),
        np.zeros((2, 2), dtype=np.complex128),
    )


@pytest.mark.parametrize("bad", ((4,), (-1,), (0, 5)))
def test_invalid_codes_are_rejected_by_reference(bad: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        codes_to_dense(bad)


def test_shape_nqubits_and_overflow_edges_are_rejected() -> None:
    with pytest.raises(ValueError):
        dense_operator(2, ((0,),), (1,))
    with pytest.raises(ValueError):
        symplectic_to_codes(4, 0, 2)
    with pytest.raises(ValueError):
        symplectic_to_codes(0, 0, -1)


def test_random_small_dense_reference_is_deterministic() -> None:
    rng = np.random.default_rng(20260801)
    for nqubits in range(7):
        for _ in range(12):
            structures = rng.integers(0, 4, size=(9, nqubits), dtype=np.int8)
            coefficients = rng.normal(size=9) + 1j * rng.normal(size=9)
            first = dense_operator(nqubits, structures.tolist(), coefficients)
            second = dense_operator(nqubits, structures.tolist(), coefficients)
            np.testing.assert_array_equal(first, second)
            for structure in structures[:3]:
                product_result, phase = multiply_codes(structure, structure)
                assert product_result == (0,) * nqubits
                assert phase == 1.0 + 0.0j
