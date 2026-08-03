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


def test_hybrid_symbolic_multiply_and_jordan_wigner_use_batch_semantics() -> None:
    space = tcp.OperatorSpace(fermions=1, bosons=1, qubits=1)
    left = space.fermion.annihilate(0) * space.boson.create(0) * space.qubit.x(0)
    right = space.fermion.create(0) * space.boson.annihilate(0) * space.qubit.y(0)
    product = left * right
    expected = np.kron(
        np.kron(
            _fermion_matrix(1, ((0, "annihilate"), (0, "create"))),
            _boson_matrix(1, ((0, "create"), (0, "annihilate"))),
        ),
        np.asarray([[1j, 0], [0, -1j]], dtype=np.complex128),
    )
    np.testing.assert_allclose(product.compile("dense", boson_cutoffs={0: 1}), expected)

    fermion_qubit = tcp.OperatorSpace(fermions=1, qubits=1)
    mapped = (fermion_qubit.fermion.create(0) * fermion_qubit.qubit.z(0)).map_fermions()
    assert isinstance(mapped, tcp.PauliOperator)
    assert mapped.compile("dense").shape == (4, 4)


def test_operator_builder_batch_canonicalization() -> None:
    space = tcp.OperatorSpace(fermions=2, bosons=1, qubits=1, qudits=(3,))
    builder = space.builder()
    builder.add_product(
        fermions=((0, "annihilate"), (0, "create")),
        bosons=((0, "annihilate"), (0, "create")),
        qubits=((0, "X"),),
        qudits=((0, 1, 0),),
    )
    builder.add_product(2, fermions=((1, "create"),), bosons=((0, "create"),))
    actual = builder.finish()
    expected = space.fermion.annihilate(0) * space.fermion.create(
        0
    ) * space.boson.annihilate(0) * space.boson.create(0) * space.qubit.x(
        0
    ) * space.qudit.weyl(
        0, 1, 0
    ) + 2 * space.fermion.create(
        1
    ) * space.boson.create(
        0
    )
    np.testing.assert_allclose(
        actual.compile("dense", boson_cutoffs={0: 2}),
        expected.compile("dense", boson_cutoffs={0: 2}),
    )


def test_adaptive_rust_sparse_targets_and_native_mvp() -> None:
    space = tcp.OperatorSpace(bosons=2)
    operator = space.boson.create(0)
    for term in (
        space.boson.create(1),
        space.boson.annihilate(0),
        space.boson.annihilate(1),
        space.boson.create(0) * space.boson.create(1),
        space.boson.create(0) * space.boson.annihilate(1),
        space.boson.annihilate(0) * space.boson.create(1),
        space.boson.annihilate(0) * space.boson.annihilate(1),
    ):
        operator = operator + term
    cutoffs = {0: 7, 1: 7}
    dense = operator.compile("dense", boson_cutoffs=cutoffs)
    coo = operator.compile("coo", boson_cutoffs=cutoffs)
    csr = operator.compile("csr", boson_cutoffs=cutoffs)
    plan = operator.compile("native_mvp", boson_cutoffs=cutoffs)
    assert plan.strategy == "structured_mvp_native"
    reconstructed_coo = np.zeros_like(dense)
    reconstructed_coo[coo.row, coo.column] = coo.data
    np.testing.assert_array_equal(reconstructed_coo, dense)
    reconstructed_csr = np.zeros_like(dense)
    for row in range(64):
        start, stop = int(csr.indptr[row]), int(csr.indptr[row + 1])
        reconstructed_csr[row, csr.indices[start:stop]] = csr.data[start:stop]
    np.testing.assert_array_equal(reconstructed_csr, dense)
    state = np.arange(64, dtype=np.complex128)
    np.testing.assert_allclose(plan.apply(state), dense @ state)

    small = tcp.BosonOperator.from_terms(1, [(((0, "create"),), 1.0)])
    assert (
        small.compile("native_mvp", boson_cutoffs={0: 1}).strategy
        == "structured_mvp_native"
    )


def test_adaptive_rust_sparse_weyl_targets_match_dense() -> None:
    terms = [
        (((0, a, b),), 1.0 + 0.01j * (a * 4 + b)) for a in range(4) for b in range(4)
    ]
    operator = tcp.QuditWeylOperator.from_terms(5, terms)
    dense = operator.compile("dense")
    expected = sum(
        (coefficient * _weyl_matrix(5, a, b) for ((_, a, b),), coefficient in terms),
        start=np.zeros((5, 5), dtype=np.complex128),
    )
    np.testing.assert_allclose(dense, expected)
    coo = operator.compile("coo")
    plan = operator.compile("native_mvp")
    assert plan.strategy == "structured_mvp_native"
    reconstructed = np.zeros_like(dense)
    reconstructed[coo.row, coo.column] = coo.data
    np.testing.assert_array_equal(reconstructed, dense)
    state = np.arange(5, dtype=np.complex128)
    np.testing.assert_allclose(plan.apply(state), dense @ state)


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


def test_partial_mapping_preserves_existing_pauli_factors() -> None:
    space = tcp.OperatorSpace(fermions=1, bosons=1)
    create = space.fermion.create(0)
    annihilate = space.fermion.annihilate(0)
    create_boson = create * space.boson.create(0)
    partially_mapped = (create_boson.map_fermions() * annihilate).map_fermions()
    mapped_after_product = create_boson.multiply(annihilate).map_fermions()
    np.testing.assert_allclose(
        partially_mapped.compile("dense", boson_cutoffs={0: 2}),
        mapped_after_product.compile("dense", boson_cutoffs={0: 2}),
    )


def test_embedding_validates_permutations_dimensions_and_fermion_signs() -> None:
    source = tcp.OperatorSpace(fermions=2)
    target = tcp.OperatorSpace(fermions=2)
    operator = source.fermion.create(0) * source.fermion.annihilate(1)
    embedded = target.embed(operator, fermions={0: 1, 1: 0})
    expected = target.fermion.create(1) * target.fermion.annihilate(0)
    np.testing.assert_allclose(embedded.compile("dense"), expected.compile("dense"))
    with pytest.raises(ValueError, match="injective"):
        target.embed(operator, fermions={0: 0, 1: 0})
    with pytest.raises(ValueError, match="integers"):
        target.embed(operator, fermions={0: "1", 1: 0})
    qudit4 = tcp.QuditWeylOperator.from_terms(4, [(((0, 0, 3),), 1.0)], n_sites=1)
    with pytest.raises(ValueError, match="equal source and target dimensions"):
        tcp.OperatorSpace(qudits=(3,)).embed(qudit4, qudits={0: 0})


def test_builder_multiplies_repeated_qubit_factors() -> None:
    space = tcp.OperatorSpace(qubits=1)
    actual = space.builder().add_product(qubits=((0, "X"), (0, "Y"))).finish()
    expected = 1j * space.qubit.z(0)
    np.testing.assert_allclose(actual.compile("dense"), expected.compile("dense"))


def test_finite_boson_large_ladder_factor_is_target_consistent() -> None:
    operator = tcp.BosonOperator.from_terms(1, [(((0, "create"),) * 171, 1.0)])
    dense = operator.compile("dense", boson_cutoffs={0: 171})
    coo = operator.compile("coo", boson_cutoffs={0: 171})
    assert np.isfinite(dense).all()
    assert dense[171, 0] == pytest.approx(3.522808638313566e154)
    assert coo.data[-1] == pytest.approx(dense[171, 0])


def test_uniform_qudit_backend_mvp_uses_direct_weyl_plan() -> None:
    import tensorcircuit as tc

    tc.set_backend("numpy")
    operator = tcp.QuditWeylOperator.from_terms(
        3,
        [(((0, 1, 2),), 0.7 - 0.2j), (((1, 2, 1),), 0.3 + 0.1j)],
        n_sites=2,
    )
    plan = operator.compile("backend_mvp")
    assert plan.plan_kind == "direct_weyl"
    state = np.arange(9, dtype=np.complex128)
    expected = operator.compile("dense") @ state
    np.testing.assert_allclose(tcp.backend_mvp(plan)(state), expected)
    np.testing.assert_allclose(
        tcp.backend_mvp(plan)(state.reshape(3, 3)), expected.reshape(3, 3)
    )
    with pytest.raises(NotImplementedError):
        tcp.OperatorSpace(qubits=1, qudits=(3,)).qubit.z(0).compile("backend_mvp")
