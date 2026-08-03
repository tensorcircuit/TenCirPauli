"""Independent small-system checks for the Phase 7 structured algebra slice."""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

import tencirpauli as tcp


def _fermion_matrix(n_modes: int, factors: tuple[tuple[int, str], ...]) -> np.ndarray:
    dimension = 1 << n_modes
    result = np.eye(dimension, dtype=np.complex128)
    for mode, action in factors:
        local = np.zeros((dimension, dimension), dtype=np.complex128)
        for column in range(dimension):
            occupied = (column >> (n_modes - 1 - mode)) & 1
            if action == "annihilate":
                if not occupied:
                    continue
                row = column & ~(1 << (n_modes - 1 - mode))
            else:
                if occupied:
                    continue
                row = column | (1 << (n_modes - 1 - mode))
            lower_mask = 0
            for lower_mode in range(mode):
                lower_mask |= 1 << (n_modes - 1 - lower_mode)
            parity = (column & lower_mask).bit_count() & 1
            local[row, column] = -1.0 if parity else 1.0
        result = result @ local
    return result


def _boson_matrix(cutoff: int, factors: tuple[tuple[int, str], ...]) -> np.ndarray:
    result = np.eye(cutoff + 1, dtype=np.complex128)
    for _, action in factors:
        local = np.zeros_like(result)
        for column in range(cutoff + 1):
            if action == "annihilate" and column:
                local[column - 1, column] = math.sqrt(column)
            if action == "create" and column < cutoff:
                local[column + 1, column] = math.sqrt(column + 1)
        result = result @ local
    return result


def _projected_boson_monomial(
    cutoff: int, creation_power: int, annihilation_power: int
) -> np.ndarray:
    result = np.zeros((cutoff + 1, cutoff + 1), dtype=np.complex128)
    for column in range(cutoff + 1):
        if column < annihilation_power:
            continue
        destination = column - annihilation_power + creation_power
        if destination > cutoff:
            continue
        amplitude = 1.0
        for offset in range(annihilation_power):
            amplitude *= math.sqrt(column - offset)
        remaining = column - annihilation_power
        for offset in range(creation_power):
            amplitude *= math.sqrt(remaining + offset + 1)
        result[destination, column] = amplitude
    if creation_power == 0 and annihilation_power == 0:
        np.fill_diagonal(result, 1.0)
    return result


def _weyl_matrix(dimension: int, a: int, b: int) -> np.ndarray:
    omega = cmath.exp(2j * math.pi / dimension)
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        result[(column + a) % dimension, column] = omega ** (b * column)
    return result


def test_fermion_car_and_jordan_wigner_reference() -> None:
    annihilate = tcp.FermionOperator.from_terms(2, [(((0, "annihilate"),), 1.0)])
    create = tcp.FermionOperator.from_terms(2, [(((0, "create"),), 1.0)])
    aa = annihilate.compile("dense")
    ad = create.compile("dense")
    identity = np.eye(4, dtype=np.complex128)
    np.testing.assert_allclose(aa @ ad + ad @ aa, identity)
    np.testing.assert_allclose(aa, _fermion_matrix(2, ((0, "annihilate"),)))
    np.testing.assert_allclose(ad, _fermion_matrix(2, ((0, "create"),)))
    assert not (annihilate * annihilate).terms


def test_boson_closed_form_and_projected_boundary() -> None:
    for m in range(4):
        for n in range(4):
            operator = tcp.BosonOperator.from_terms(
                1,
                [(((0, "annihilate"),) * m + ((0, "create"),) * n, 1.0)],
            )
            actual = operator.compile("dense", boson_cutoffs={0: 5})
            raw_factors = ((0, "annihilate"),) * m + ((0, "create"),) * n
            large = _boson_matrix(8, raw_factors)
            reference = large[:6, :6]
            np.testing.assert_allclose(actual, reference)
    raising = tcp.BosonOperator.from_terms(1, [(((0, "create"),), 1.0)])
    assert raising.compile("dense", boson_cutoffs={0: 0})[0, 0] == 0
    lower = tcp.BosonOperator.from_terms(1, [(((0, "annihilate"),), 1.0)])
    with pytest.raises(ValueError, match="exactly one"):
        lower.compile("dense", boson_cutoffs={})


@pytest.mark.parametrize("dimension", [3, 4, 5, 6])
def test_direct_weyl_reference_and_targets(dimension: int) -> None:
    operator = tcp.QuditWeylOperator.from_terms(
        dimension, [(((0, 1, 2),), 0.75 - 0.2j)]
    )
    expected = (0.75 - 0.2j) * _weyl_matrix(dimension, 1, 2)
    np.testing.assert_allclose(operator.compile("dense"), expected)
    coo = operator.compile("coo")
    csr = operator.compile("csr")
    np.testing.assert_allclose(
        np.asarray(csr.data),
        expected[np.nonzero(expected)],
    )
    state = np.arange(dimension, dtype=np.complex128)
    np.testing.assert_allclose(
        operator.compile("native_mvp").apply(state), expected @ state
    )
    assert coo.shape == (dimension, dimension)


def test_hybrid_targets_and_mixed_radix_ordering() -> None:
    space = tcp.OperatorSpace(fermions=1, bosons=1, qubits=1, qudits=(3,))
    operator = (
        space.fermion.create(0) * space.fermion.annihilate(0)
        + 2.0 * space.boson.create(0) * space.boson.annihilate(0)
        + space.qubit.z(0)
        + space.qudit.weyl(0, 1, 0)
    )
    dense = operator.compile("dense", boson_cutoffs={0: 1})
    coo = operator.compile("coo", boson_cutoffs={0: 1})
    csr = operator.compile("csr", boson_cutoffs={0: 1})
    plan = operator.compile("native_mvp", boson_cutoffs={0: 1})
    reconstructed = np.zeros_like(dense)
    reconstructed[coo.row, coo.column] = coo.data
    np.testing.assert_array_equal(reconstructed, dense)
    np.testing.assert_allclose(
        csr.data,
        dense[
            np.repeat(np.arange(dense.shape[0]), np.diff(csr.indptr).astype(int)),
            csr.indices,
        ],
    )
    state = np.arange(dense.shape[0], dtype=np.complex128)
    np.testing.assert_allclose(plan.apply(state), dense @ state)
    assert plan.local_dimensions == (2, 2, 2, 3)


def test_tensor_product_grading_and_layout_compatibility() -> None:
    left = tcp.OperatorSpace(fermions=1).fermion.annihilate(0)
    right = tcp.OperatorSpace(fermions=1).fermion.create(0)
    tensor = left.tensor_product(right)
    assert tensor.space.axes == (("fermion", 0, 2), ("fermion", 1, 2))
    term = tensor.terms[0]
    assert term.fermion.creation_modes == (1,)
    assert term.fermion.annihilation_modes == (0,)
    assert term.coefficient == -1
    with pytest.raises(ValueError, match="incompatible"):
        _ = left + tcp.OperatorSpace(fermions=2).fermion.annihilate(0)


def test_phase7_memory_and_target_errors() -> None:
    assert tcp.DEFAULT_MAX_BYTES == 16 * 1024**3
    operator = tcp.BosonOperator.from_terms(1, [(((0, "create"),), 1.0)])
    with pytest.raises(MemoryError):
        operator.compile("dense", boson_cutoffs={0: 3}, max_bytes=1)
    with pytest.raises(NotImplementedError):
        operator.compile("backend_mvp", boson_cutoffs={0: 3})
    fermion = tcp.FermionOperator.from_terms(1, [(((0, "annihilate"),), 1.0)])
    with pytest.raises(NotImplementedError):
        fermion.compile("dense", mapping="bravyi_kitaev")
