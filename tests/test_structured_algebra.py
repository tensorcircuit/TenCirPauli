"""Independent small-system checks for the structured algebra slice."""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

import tencirpauli as tcp
from tencirpauli import _native, advanced
from tencirpauli.structured import _jordan_wigner_word, _StructuredOperator


def _bit_count(value: int) -> int:
    return bin(value).count("1")


def test_structured_operators_keep_native_storage_until_terms_are_requested() -> None:
    fermion = tcp.FermionOperator.from_terms(
        2, [(((0, "create"), (1, "annihilate")), 1.0)]
    )
    boson = tcp.BosonOperator.from_terms(1, [(((0, "create"), (0, "annihilate")), 1.0)])
    qudit = tcp.QuditWeylOperator.from_terms(3, [(((0, 1, 2),), 1.0)], n_sites=1)
    hybrid_space = tcp.OperatorSpace(fermions=1, bosons=1, qubits=1, qudits=(3,))
    hybrid = (
        hybrid_space.fermion.create(0)
        * hybrid_space.boson.create(0)
        * hybrid_space.qubit.x(0)
        * hybrid_space.qudit.weyl(0, 1, 2)
    )

    for operator in (fermion, boson, qudit, hybrid):
        assert operator._terms is None
        assert operator.term_count == 1
        assert operator.to_dict()
        assert operator._terms is None

    assert isinstance(fermion._native_handle, _native.NativeFermionOperatorHandle)
    assert isinstance(boson._native_handle, _native.NativeBosonOperatorHandle)
    assert isinstance(qudit._native_handle, _native.NativeHybridOperatorHandle)
    assert isinstance(hybrid._native_handle, _native.NativeHybridOperatorHandle)
    assert not hasattr(fermion, "_native_data")
    assert not hasattr(boson, "_native_data")
    assert not hasattr(qudit, "_native_data")
    assert not hasattr(hybrid, "_native_data")

    assert fermion.multiply(fermion)._terms is None
    assert boson.add(boson)._terms is None
    assert qudit.adjoint()._terms is None
    assert hybrid.commutator(hybrid)._terms is None

    assert fermion.terms
    assert boson.terms
    assert qudit.terms
    assert hybrid.terms


def test_compatible_specialized_and_hybrid_addition_promotes_natively() -> None:
    fermion = tcp.FermionOperator.from_terms(1, [(((0, "create"),), 1.0)])
    fermion_hybrid = tcp.OperatorSpace(fermions=1).fermion.create(0)
    assert isinstance(fermion + fermion_hybrid, tcp.HybridOperator)
    assert list((fermion + fermion_hybrid).to_dict().values()) == [2.0]
    assert isinstance(fermion_hybrid + fermion, tcp.HybridOperator)

    boson = tcp.BosonOperator.from_terms(1, [(((0, "create"),), 1.0)])
    boson_hybrid = tcp.OperatorSpace(bosons=1).boson.create(0)
    assert isinstance(boson + boson_hybrid, tcp.HybridOperator)
    assert isinstance(boson_hybrid + boson, tcp.HybridOperator)

    qudit = tcp.QuditWeylOperator.from_terms(3, [(((0, 1, 1),), 1.0)], n_sites=1)
    qudit_hybrid = tcp.OperatorSpace(qudits=(3,)).qudit.weyl(0, 1, 1)
    assert isinstance(qudit + qudit_hybrid, tcp.HybridOperator)
    assert isinstance(qudit_hybrid + qudit, tcp.HybridOperator)
    with pytest.raises(ValueError, match="incompatible"):
        _ = fermion + tcp.OperatorSpace(fermions=2).fermion.create(0)


def test_fused_structured_commutators_match_composed_native_reference() -> None:
    spaces_and_pairs = (
        (
            tcp.OperatorSpace(fermions=2),
            lambda space: (
                space.fermion.create(0) + 2 * space.fermion.annihilate(1),
                space.fermion.create(1) + 3 * space.fermion.annihilate(0),
            ),
        ),
        (
            tcp.OperatorSpace(bosons=1),
            lambda space: (
                space.boson.create(0) + space.boson.annihilate(0),
                2 * space.boson.create(0) + space.boson.annihilate(0),
            ),
        ),
        (
            tcp.OperatorSpace(fermions=1, bosons=1, qubits=1),
            lambda space: (
                space.fermion.create(0) * space.boson.create(0) + space.qubit.x(0),
                space.fermion.annihilate(0) * space.boson.annihilate(0)
                + space.qubit.z(0),
            ),
        ),
    )
    for space, factory in spaces_and_pairs:
        left, right = factory(space)
        composed = left.multiply(right).add(
            right.multiply(left).scale(-1),
        )
        assert left.commutator(right).to_dict() == composed.to_dict()


def test_fused_structured_operations_match_independent_dense_references() -> None:
    fermion_left = tcp.FermionOperator.from_terms(
        1, [(((0, "create"),), 1.0), (((0, "annihilate"),), 0.3)]
    )
    fermion_right = tcp.FermionOperator.from_terms(
        1, [(((0, "create"),), 0.7), (((0, "annihilate"),), -1.0)]
    )

    boson_left = tcp.BosonOperator.from_terms(
        1, [(((0, "create"),), 1.0), (((0, "annihilate"),), 1.0)]
    )
    boson_right = tcp.BosonOperator.from_terms(
        1, [(((0, "create"),), 0.4), (((0, "annihilate"),), -1.0)]
    )

    qudit_left = tcp.QuditWeylOperator.from_terms(
        3, [(((0, 1, 1),), 0.6 - 0.1j)], n_sites=1
    )
    qudit_right = tcp.QuditWeylOperator.from_terms(
        3, [(((0, 2, 1),), -0.2 + 0.5j)], n_sites=1
    )

    hybrid_space = tcp.OperatorSpace(fermions=1, qubits=1)
    hybrid_left = hybrid_space.fermion.create(0) * hybrid_space.qubit.x(
        0
    ) + hybrid_space.qubit.z(0)
    hybrid_right = hybrid_space.fermion.annihilate(0) * hybrid_space.qubit.z(
        0
    ) + 0.5 * hybrid_space.qubit.x(0)

    cases = (
        (fermion_left, fermion_right, {}, "fermion"),
        (boson_left, boson_right, {"boson_cutoffs": {0: 3}}, "boson"),
        (qudit_left, qudit_right, {}, "qudit"),
        (hybrid_left, hybrid_right, {}, "hybrid"),
    )
    for left, right, compile_kwargs, _family in cases:
        if _family == "boson":
            product = left.multiply(right)
            commutator = left.commutator(right)
            anticommutator = left.anticommutator(right)
            expected_product = _boson_operator_dense(product, 3)
            expected_commutator = _boson_operator_dense(commutator, 3)
            expected_anticommutator = _boson_operator_dense(anticommutator, 3)
        else:
            left_dense = left.compile("dense", **compile_kwargs)
            right_dense = right.compile("dense", **compile_kwargs)
            expected_product = left_dense @ right_dense
            expected_commutator = expected_product - right_dense @ left_dense
            expected_anticommutator = expected_product + right_dense @ left_dense
        np.testing.assert_allclose(
            left.multiply(right).compile("dense", **compile_kwargs), expected_product
        )
        np.testing.assert_allclose(
            left.commutator(right).compile("dense", **compile_kwargs),
            expected_commutator,
        )
        np.testing.assert_allclose(
            left.anticommutator(right).compile("dense", **compile_kwargs),
            expected_anticommutator,
        )

    assert isinstance(qudit_left + qudit_right, tcp.QuditWeylOperator)
    assert isinstance(qudit_left.multiply(qudit_right), tcp.QuditWeylOperator)
    assert isinstance(qudit_left.commutator(qudit_right), tcp.QuditWeylOperator)
    assert isinstance(qudit_left.anticommutator(qudit_right), tcp.QuditWeylOperator)
    mixed = tcp.OperatorSpace(qudits=(3,)).qudit.weyl(0, 1, 1)
    assert isinstance(qudit_left + mixed, tcp.HybridOperator)


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
            parity = _bit_count(column & lower_mask) & 1
            local[row, column] = -1.0 if parity else 1.0
        result = result @ local
    return result


def _hubbard_terms(
    n_sites: int,
) -> tuple[tuple[tuple[tuple[int, str], ...], complex], ...]:
    """Return a small spinful Hubbard Hamiltonian in raw ladder form."""
    terms: list[tuple[tuple[tuple[int, str], ...], complex]] = []
    for site in range(n_sites):
        up, down = 2 * site, 2 * site + 1
        terms.append(
            (
                (
                    (up, "create"),
                    (up, "annihilate"),
                    (down, "create"),
                    (down, "annihilate"),
                ),
                1.25,
            )
        )
    for site in range(n_sites - 1):
        for spin in (0, 1):
            left, right = 2 * site + spin, 2 * (site + 1) + spin
            terms.extend(
                (
                    (((left, "create"), (right, "annihilate")), -0.7),
                    (((right, "create"), (left, "annihilate")), -0.7),
                )
            )
    return tuple(terms)


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


def _boson_operator_dense(operator: tcp.BosonOperator, cutoff: int) -> np.ndarray:
    result = np.zeros((cutoff + 1, cutoff + 1), dtype=np.complex128)
    for term in operator.terms:
        monomial = np.eye(cutoff + 1, dtype=np.complex128)
        for mode, creation, annihilation in term.word.blocks:
            assert mode == 0
            monomial = monomial @ _projected_boson_monomial(
                cutoff, creation, annihilation
            )
        result += term.coefficient * monomial
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


def test_fermion_car_identities_across_modes() -> None:
    n_modes = 3
    identity = np.eye(1 << n_modes, dtype=np.complex128)
    annihilators = [
        tcp.FermionOperator.from_terms(n_modes, [(((mode, "annihilate"),), 1.0)])
        for mode in range(n_modes)
    ]
    creators = [
        tcp.FermionOperator.from_terms(n_modes, [(((mode, "create"),), 1.0)])
        for mode in range(n_modes)
    ]
    for left_mode in range(n_modes):
        for right_mode in range(n_modes):
            aa = (
                annihilators[left_mode] * annihilators[right_mode]
                + annihilators[right_mode] * annihilators[left_mode]
            ).compile("dense")
            dd = (
                creators[left_mode] * creators[right_mode]
                + creators[right_mode] * creators[left_mode]
            ).compile("dense")
            ad = (
                annihilators[left_mode] * creators[right_mode]
                + creators[right_mode] * annihilators[left_mode]
            ).compile("dense")
            np.testing.assert_allclose(aa, 0.0)
            np.testing.assert_allclose(dd, 0.0)
            np.testing.assert_allclose(ad, identity if left_mode == right_mode else 0.0)


def test_hubbard_quartic_fermion_dense_and_mvp_reference() -> None:
    n_sites = 2
    raw_terms = _hubbard_terms(n_sites)
    operator = tcp.FermionOperator.from_terms(2 * n_sites, raw_terms)
    expected = sum(
        (
            coefficient * _fermion_matrix(2 * n_sites, factors)
            for factors, coefficient in raw_terms
        ),
        start=np.zeros((1 << (2 * n_sites), 1 << (2 * n_sites)), dtype=np.complex128),
    )
    assert max(len(term.word.factors) for term in operator.terms) == 4
    np.testing.assert_allclose(operator.compile("dense"), expected)
    np.testing.assert_allclose(operator.map_fermions().compile("dense"), expected)
    state = np.arange(1 << (2 * n_sites), dtype=np.complex128)
    np.testing.assert_allclose(
        operator.compile("native_mvp").apply(state), expected @ state
    )


@pytest.mark.parametrize("dimension", [3, 5])
@pytest.mark.parametrize("target", ["coo", "csr"])
def test_qudit_weyl_sparse_targets_match_dense(dimension: int, target: str) -> None:
    operator = tcp.QuditWeylOperator.from_terms(
        dimension,
        [
            (((0, 1, 2), (1, 2, 1)), 0.7 - 0.2j),
            (((1, 2, 3),), -0.3 + 0.1j),
            (((0, 2, 1),), 0.2),
        ],
        n_sites=2,
    )
    dense = operator.compile("dense")
    sparse = operator.compile(target)
    reconstructed = np.zeros_like(dense)
    if target == "coo":
        np.add.at(reconstructed, (sparse.row, sparse.column), sparse.data)
    else:
        for row in range(dense.shape[0]):
            start, stop = int(sparse.indptr[row]), int(sparse.indptr[row + 1])
            reconstructed[row, sparse.indices[start:stop]] += sparse.data[start:stop]
    np.testing.assert_allclose(reconstructed, dense)


def test_qudit_weyl_native_mvp_matches_dense() -> None:
    operator = tcp.QuditWeylOperator.from_terms(
        4,
        [
            (((0, 1, 2),), 0.7 - 0.2j),
            (((1, 2, 1),), -0.3 + 0.1j),
            (((0, 3, 1), (1, 1, 2)), 0.2),
        ],
        n_sites=2,
    )
    dense = operator.compile("dense")
    state = np.arange(dense.shape[0], dtype=np.complex128)
    np.testing.assert_allclose(
        operator.compile("native_mvp").apply(state), dense @ state
    )


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "bravyi_kitaev"])
@pytest.mark.parametrize("target", ["coo", "csr", "native_mvp"])
def test_fermion_mapping_sparse_and_mvp_match_dense(mapping: str, target: str) -> None:
    operator = tcp.FermionOperator.from_terms(4, _hubbard_terms(2))
    dense = operator.compile("dense", mapping=mapping)
    compiled = operator.compile(target, mapping=mapping)
    if target == "coo":
        reconstructed = np.zeros_like(dense)
        np.add.at(reconstructed, (compiled.row, compiled.column), compiled.data)
        np.testing.assert_allclose(reconstructed, dense)
    elif target == "csr":
        reconstructed = np.zeros_like(dense)
        for row in range(dense.shape[0]):
            start, stop = int(compiled.indptr[row]), int(compiled.indptr[row + 1])
            reconstructed[row, compiled.indices[start:stop]] += compiled.data[
                start:stop
            ]
        np.testing.assert_allclose(reconstructed, dense)
    else:
        state = np.arange(dense.shape[0], dtype=np.complex128)
        np.testing.assert_allclose(compiled.apply(state), dense @ state)


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


def test_boson_plan_reuse_across_cutoffs_and_unbounded_guard() -> None:
    factors = ((0, "annihilate"), (0, "create"))
    operator = tcp.BosonOperator.from_terms(1, [(factors, 1.0)])
    for cutoff in (1, 3):
        actual = operator.compile("dense", boson_cutoffs={0: cutoff}, max_bytes=None)
        large = _boson_matrix(cutoff + 1, factors)
        np.testing.assert_allclose(actual, large[: cutoff + 1, : cutoff + 1])


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


def test_native_weyl_plan_metadata_and_unbounded_guard() -> None:
    operator = tcp.QuditWeylOperator.from_terms(5, [(((0, 1, 2),), 1.0)], n_sites=2)
    plan = operator.compile("native_mvp", max_bytes=None)
    assert plan.basis_ordering == "operator_space_axis0_msb_mixed_radix"
    assert plan.nqubits == 0
    assert plan.qudit_dimension == 5
    assert plan.weyl_convention == "X^a Z^b"
    assert plan.boson_cutoffs == ()
    state = np.arange(25, dtype=np.complex128)
    np.testing.assert_allclose(
        plan.apply(state, max_bytes=None), operator.compile("dense") @ state
    )


def test_structured_plan_metadata_matches_the_physical_basis() -> None:
    pauli = tcp.PauliOperator.from_terms(2, [("XZ", 1.0)]).compile("native_mvp")
    assert pauli.basis_ordering == "qubit0_msb_matrix"
    assert pauli.nqubits == 2
    assert pauli.local_dimensions == (2, 2)

    fermion = tcp.FermionOperator.from_terms(
        2, [(((0, "create"), (1, "annihilate")), 1.0)]
    ).compile("native_mvp")
    assert fermion.basis_ordering == "qubit0_msb_matrix"
    assert fermion.mapping == "jordan_wigner"
    assert fermion.nqubits == 2

    boson_space = tcp.OperatorSpace(bosons=1)
    boson = boson_space.boson.create(0).compile("native_mvp", boson_cutoffs={0: 2})
    assert boson.basis_ordering == "operator_space_axis0_msb_mixed_radix"
    assert boson.nqubits == 0
    assert boson.local_dimensions == (3,)

    hybrid_space = tcp.OperatorSpace(fermions=1, bosons=1, qubits=1)
    hybrid = (hybrid_space.fermion.create(0) * hybrid_space.boson.create(0)).compile(
        "native_mvp", boson_cutoffs={0: 1}
    )
    assert hybrid.basis_ordering == "operator_space_axis0_msb_mixed_radix"
    assert hybrid.nqubits == 0
    assert hybrid.local_dimensions == (2, 2, 2)

    backend = tcp.QuditWeylOperator.from_terms(
        3, [(((0, 1, 1),), 1.0)], n_sites=2
    ).compile("backend_mvp")
    assert backend.basis_ordering == "qudit0_msb_matrix"
    assert backend.nqubits == 0
    assert backend.word_count == 0
    assert backend.local_dimensions == (3, 3)


def test_structured_holstein_independent_dense_sparse_and_mvp_differential() -> None:
    cutoff = 2
    space = tcp.OperatorSpace(fermions=1, bosons=1)
    hamiltonian = (
        0.7 * space.fermion.create(0) * space.fermion.annihilate(0)
        + 0.5 * space.boson.create(0) * space.boson.annihilate(0)
        + 0.3
        * (space.fermion.create(0) + space.fermion.annihilate(0))
        * (space.boson.create(0) + space.boson.annihilate(0))
    )
    fermion_number = _fermion_matrix(1, ((0, "create"), (0, "annihilate")))
    boson_number = _boson_matrix(cutoff, ((0, "create"), (0, "annihilate")))
    fermion_x = _fermion_matrix(1, ((0, "create"),)) + _fermion_matrix(
        1, ((0, "annihilate"),)
    )
    boson_x = _boson_matrix(cutoff, ((0, "create"),)) + _boson_matrix(
        cutoff, ((0, "annihilate"),)
    )
    independent = 0.7 * np.kron(fermion_number, np.eye(cutoff + 1))
    independent += 0.5 * np.kron(np.eye(2), boson_number)
    independent += 0.3 * np.kron(fermion_x, boson_x)

    dense = hamiltonian.compile("dense", boson_cutoffs={0: cutoff})
    np.testing.assert_allclose(dense, independent)
    coo = hamiltonian.compile("coo", boson_cutoffs={0: cutoff})
    reconstructed = np.zeros_like(dense)
    reconstructed[coo.row, coo.column] = coo.data
    np.testing.assert_array_equal(reconstructed, independent)
    csr = hamiltonian.compile("csr", boson_cutoffs={0: cutoff})
    csr_reconstructed = np.zeros_like(dense)
    for row in range(dense.shape[0]):
        start, stop = int(csr.indptr[row]), int(csr.indptr[row + 1])
        csr_reconstructed[row, csr.indices[start:stop]] = csr.data[start:stop]
    np.testing.assert_array_equal(csr_reconstructed, independent)
    state = np.arange(dense.shape[0], dtype=np.complex128)
    plan = hamiltonian.compile("native_mvp", boson_cutoffs={0: cutoff})
    np.testing.assert_allclose(plan.apply(state), independent @ state)


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


def test_structured_memory_and_target_errors() -> None:
    assert tcp.DEFAULT_MAX_BYTES == 16 * 1024**3
    operator = tcp.BosonOperator.from_terms(1, [(((0, "create"),), 1.0)])
    with pytest.raises(MemoryError):
        operator.compile("dense", boson_cutoffs={0: 3}, max_bytes=1)
    with pytest.raises(NotImplementedError):
        operator.compile("backend_mvp", boson_cutoffs={0: 3})
    fermion = tcp.FermionOperator.from_terms(1, [(((0, "annihilate"),), 1.0)])
    np.testing.assert_allclose(
        fermion.compile("dense", mapping="bravyi_kitaev"),
        fermion.compile("dense", mapping="jordan_wigner"),
    )


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
    assert isinstance(embedded, tcp.FermionOperator)
    expected = target.fermion.create(1) * target.fermion.annihilate(0)
    np.testing.assert_allclose(embedded.compile("dense"), expected.compile("dense"))

    annihilation_source = (
        source.fermion.create(1)
        * source.fermion.annihilate(1)
        * source.fermion.annihilate(0)
    )
    annihilation_embedded = target.embed(annihilation_source, fermions={0: 1, 1: 0})
    annihilation_expected = -(
        target.fermion.create(0)
        * target.fermion.annihilate(1)
        * target.fermion.annihilate(0)
    )
    np.testing.assert_allclose(
        annihilation_embedded.compile("dense"), annihilation_expected.compile("dense")
    )
    with pytest.raises(ValueError, match="injective"):
        target.embed(operator, fermions={0: 0, 1: 0})
    with pytest.raises(ValueError, match="integers"):
        target.embed(operator, fermions={0: "1", 1: 0})
    qudit4 = tcp.QuditWeylOperator.from_terms(4, [(((0, 0, 3),), 1.0)], n_sites=1)
    with pytest.raises(ValueError, match="equal source and target dimensions"):
        tcp.OperatorSpace(qudits=(3,)).embed(qudit4, qudits={0: 0})


def test_embedding_full_domain_permutation_reference() -> None:
    source = tcp.OperatorSpace(fermions=2, bosons=2, qubits=2, qudits=(3, 3))
    target = tcp.OperatorSpace(fermions=2, bosons=2, qubits=2, qudits=(3, 3))
    operator = (
        source.fermion.create(0)
        * source.fermion.annihilate(1)
        * source.boson.create(0)
        * source.boson.annihilate(1)
        * source.qubit.x(0)
        * source.qudit.weyl(0, 1, 2)
    )
    embedded = target.embed(
        operator,
        fermions={0: 1, 1: 0},
        bosons={0: 1, 1: 0},
        qubits={0: 1, 1: 0},
        qudits={0: 1, 1: 0},
    )
    expected = (
        target.fermion.create(1)
        * target.fermion.annihilate(0)
        * target.boson.create(1)
        * target.boson.annihilate(0)
        * target.qubit.x(1)
        * target.qudit.weyl(1, 1, 2)
    )
    np.testing.assert_allclose(
        embedded.compile("dense", boson_cutoffs={0: 1, 1: 1}),
        expected.compile("dense", boson_cutoffs={0: 1, 1: 1}),
    )


def test_embedding_preserves_boson_qudit_and_mixed_facades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_materialization(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("embedding must stay on native structured handles")

    monkeypatch.setattr(
        _StructuredOperator, "_materialized_terms", forbidden_materialization
    )
    boson_space = tcp.OperatorSpace(bosons=1)
    boson = boson_space.boson.create(0) * boson_space.boson.annihilate(0)
    embedded_boson = tcp.OperatorSpace(bosons=2).embed(boson, bosons={0: 1})
    assert isinstance(embedded_boson, tcp.BosonOperator)

    qudit_space = tcp.OperatorSpace(qudits=(3,))
    qudit = qudit_space.qudit.weyl(0, 1, 2)
    embedded_qudit = qudit_space.embed(qudit, qudits={0: 0})
    assert isinstance(embedded_qudit, tcp.QuditWeylOperator)

    mixed_space = tcp.OperatorSpace(fermions=1, qubits=1)
    mixed = mixed_space.fermion.create(0) * mixed_space.qubit.z(0)
    embedded_mixed = tcp.OperatorSpace(fermions=2, qubits=2).embed(
        mixed, fermions={0: 1}, qubits={0: 0}
    )
    assert isinstance(embedded_mixed, tcp.HybridOperator)
    repeated = [
        tcp.OperatorSpace(fermions=2, qubits=2).embed(
            mixed, fermions={0: 1}, qubits={0: 0}
        )
        for _ in range(3)
    ]
    assert all(value._native_handle is not None for value in repeated)


def test_structured_mapping_replay_is_deterministic() -> None:
    space = tcp.OperatorSpace(fermions=3, bosons=1, qubits=1, qudits=(3,))
    operator = (
        space.fermion.create(0) * space.fermion.annihilate(2)
        + 0.3 * space.fermion.create(1) * space.boson.create(0)
        + 0.2j * space.qubit.y(0) * space.qudit.weyl(0, 2, 1)
    )
    mapped = [operator.map_fermions() for _ in range(4)]
    assert all(value.terms == mapped[0].terms for value in mapped[1:])
    dense = [value.compile("dense", boson_cutoffs={0: 2}) for value in mapped]
    for value in dense[1:]:
        np.testing.assert_array_equal(value, dense[0])


def test_embedding_mixed_domain_collisions_have_deterministic_order() -> None:
    source = tcp.OperatorSpace(fermions=2, bosons=1, qubits=1, qudits=(3,))
    target = tcp.OperatorSpace(fermions=3, bosons=2, qubits=2, qudits=(3, 3))
    operator = (
        source.fermion.create(0)
        * source.fermion.annihilate(1)
        * source.boson.create(0)
        * source.qubit.y(0)
        * source.qudit.weyl(0, 1, 2)
    )
    maps = {
        "fermions": {0: 2, 1: 0},
        "bosons": {0: 0},
        "qubits": {0: 0},
        "qudits": {0: 1},
    }
    embedded = target.embed(operator, **maps)
    expected = (
        target.fermion.create(2)
        * target.fermion.annihilate(0)
        * target.boson.create(0)
        * target.qubit.y(0)
        * target.qudit.weyl(1, 1, 2)
    )
    np.testing.assert_allclose(
        embedded.compile("dense", boson_cutoffs={0: 1, 1: 0}),
        expected.compile("dense", boson_cutoffs={0: 1, 1: 0}),
    )
    assert embedded.terms == target.embed(operator, **maps).terms


def test_builder_multiplies_repeated_qubit_factors() -> None:
    space = tcp.OperatorSpace(qubits=1)
    actual = space.builder().add_product(qubits=((0, "x"), (0, "y"))).finish()
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
    tc = pytest.importorskip("tensorcircuit")

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


@pytest.mark.parametrize("dimension", [3, 4, 5, 6])
def test_weyl_structural_phases_adjoint_commutation_and_hermiticity(
    dimension: int,
) -> None:
    left = tcp.QuditWeylWord(dimension, ((0, 1, 2), (1, 2, 1)))
    right = tcp.QuditWeylWord(dimension, ((0, 2, 1), (1, 1, 3)))
    product = left.multiply(right)
    expected_exponent = (2 * 2 + 1 * 1) % dimension
    assert product.phase_exponent == expected_exponent
    expected_word = tcp.QuditWeylWord(
        dimension,
        (
            (0, (1 + 2) % dimension, (2 + 1) % dimension),
            (1, (2 + 1) % dimension, (1 + 3) % dimension),
        ),
    )
    assert product.word == expected_word
    left_matrix = np.kron(_weyl_matrix(dimension, 1, 2), _weyl_matrix(dimension, 2, 1))
    right_matrix = np.kron(_weyl_matrix(dimension, 2, 1), _weyl_matrix(dimension, 1, 3))
    product_matrix = np.kron(
        _weyl_matrix(dimension, (1 + 2) % dimension, (2 + 1) % dimension),
        _weyl_matrix(dimension, (2 + 1) % dimension, (1 + 3) % dimension),
    )
    np.testing.assert_allclose(
        left_matrix @ right_matrix,
        cmath.exp(2j * math.pi * expected_exponent / dimension) * product_matrix,
    )
    assert not left.commutes_with(right)
    assert left.commutes_with(left)
    adjoint = left.adjoint()
    adjoint_matrix = np.eye(1, dtype=np.complex128)
    for _, a, b in adjoint.word.triples:
        adjoint_matrix = np.kron(adjoint_matrix, _weyl_matrix(dimension, a, b))
    left_matrix = np.kron(_weyl_matrix(dimension, 1, 2), _weyl_matrix(dimension, 2, 1))
    phase = cmath.exp(2j * math.pi * adjoint.phase_exponent / dimension)
    np.testing.assert_allclose(phase * adjoint_matrix, left_matrix.conj().T)

    operator = tcp.QuditWeylOperator.from_terms(
        dimension,
        [(((0, 1, 2),), 1.0), (((0, 1, 2),), -0.25)],
        n_sites=1,
    )
    assert operator.term_count == 1
    hermitian = operator + operator.adjoint()
    assert hermitian.is_hermitian(1e-12)
    sparse = hermitian.compile("coo")
    reconstructed = np.zeros((dimension, dimension), dtype=np.complex128)
    reconstructed[sparse.row, sparse.column] = sparse.data
    np.testing.assert_allclose(hermitian.compile("dense"), reconstructed)


def test_fermion_products_adjoint_commutator_and_dense_reference() -> None:
    space = tcp.OperatorSpace(fermions=2)
    create = space.fermion.create(0)
    annihilate = space.fermion.annihilate(0)
    number = create * annihilate
    identity = np.eye(4, dtype=np.complex128)
    np.testing.assert_allclose(
        number.compile("dense"), _fermion_matrix(2, ((0, "create"), (0, "annihilate")))
    )
    np.testing.assert_allclose(
        (annihilate.commutator(create)).compile("dense"),
        identity - 2 * number.compile("dense"),
    )
    np.testing.assert_allclose(
        (create * space.fermion.annihilate(1)).adjoint().compile("dense"),
        (create * space.fermion.annihilate(1)).compile("dense").conj().T,
    )
    assert number.is_hermitian()


@pytest.mark.parametrize("count", [40, 128])
def test_fermion_canonical_fast_path_accepts_long_words(count: int) -> None:
    factors = tuple((mode, "create") for mode in range(count))
    operator = tcp.FermionOperator.from_terms(
        count, [(factors, 1.0)], max_bytes=count * 192 + 192
    )
    assert operator.term_count == 1
    assert operator.terms[0].word.creation_modes == tuple(range(count))


def test_fermion_inversion_nilpotency_and_running_guard() -> None:
    factors = tuple((mode, "create") for mode in range(127, -1, -1))
    inverted = tcp.FermionOperator.from_terms(
        128, [(factors, 1.0)], max_bytes=128 * 192 + 192
    )
    assert inverted.term_count == 1
    assert inverted.terms[0].coefficient == 1.0
    for action in ("create", "annihilate"):
        duplicate = tcp.FermionOperator.from_terms(
            3,
            [(((0, action), (1, action), (0, action)), 1.0)],
            max_bytes=1024,
        )
        assert duplicate.term_count == 0
    with pytest.raises(MemoryError):
        tcp.FermionOperator.from_terms(
            3,
            [
                (
                    (
                        (0, "annihilate"),
                        (1, "annihilate"),
                        (2, "annihilate"),
                        (0, "create"),
                        (1, "create"),
                        (2, "create"),
                    ),
                    1.0,
                )
            ],
            max_bytes=1400,
        )


def test_partial_mapping_both_orders_nested_and_tensor_product() -> None:
    space = tcp.OperatorSpace(fermions=1, bosons=1)
    create = space.fermion.create(0)
    annihilate = space.fermion.annihilate(0)
    boson_create = space.boson.create(0)
    left_raw = create * boson_create
    right_raw = annihilate * space.boson.annihilate(0)
    for actual, expected in (
        (
            (annihilate * left_raw.map_fermions()).map_fermions(),
            (annihilate * left_raw).map_fermions(),
        ),
        (
            (left_raw.map_fermions() * annihilate).map_fermions(),
            (left_raw * annihilate).map_fermions(),
        ),
        (
            (annihilate * (left_raw.map_fermions() * right_raw)).map_fermions(),
            (annihilate * (left_raw * right_raw)).map_fermions(),
        ),
    ):
        np.testing.assert_allclose(
            actual.compile("dense", boson_cutoffs={0: 1}),
            expected.compile("dense", boson_cutoffs={0: 1}),
        )

    other_space = tcp.OperatorSpace(fermions=1, bosons=1)
    other_raw = other_space.fermion.create(0) * other_space.boson.create(0)
    expected_tensor = left_raw.tensor_product(other_raw).map_fermions()
    for actual in (
        left_raw.map_fermions().tensor_product(other_raw),
        left_raw.tensor_product(other_raw.map_fermions()),
        left_raw.map_fermions().tensor_product(other_raw.map_fermions()),
    ):
        np.testing.assert_allclose(
            actual.compile("dense", boson_cutoffs={0: 1, 1: 1}),
            expected_tensor.compile("dense", boson_cutoffs={0: 1, 1: 1}),
        )

    for raw_operator, mapped_operator in (
        (left_raw, left_raw.map_fermions()),
        (right_raw, right_raw.map_fermions()),
    ):
        np.testing.assert_allclose(
            mapped_operator.adjoint().compile("dense", boson_cutoffs={0: 1}),
            raw_operator.adjoint()
            .map_fermions()
            .compile("dense", boson_cutoffs={0: 1}),
        )
    np.testing.assert_allclose(
        (left_raw.map_fermions().commutator(annihilate)).compile(
            "dense", boson_cutoffs={0: 1}
        ),
        left_raw.commutator(annihilate)
        .map_fermions()
        .compile("dense", boson_cutoffs={0: 1}),
    )


@pytest.mark.parametrize(
    ("creation", "annihilation"),
    [((0,), ()), ((2,), (1,)), ((0, 2), (2, 0)), ((1, 3), (2, 0))],
)
def test_tensor_product_jordan_wigner_adapter_matches_native(
    creation: tuple[int, ...], annihilation: tuple[int, ...]
) -> None:
    word = tcp.FermionWord(4, creation, annihilation)
    operator = tcp.FermionOperator.from_terms(
        word.n_modes, [((word.factors), 1.0)]
    ).map_fermions()
    expected = {
        "".join("IXYZ"[code] for code in structure): coefficient
        for structure, coefficient in _jordan_wigner_word(word)
    }
    assert operator.to_dict() == expected


@pytest.mark.parametrize("dimension", [3, 4, 5, 6])
@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_uniform_weyl_backend_numpy_jax_matrix(
    dimension: int, backend_name: str
) -> None:
    tc = pytest.importorskip("tensorcircuit")

    if backend_name == "jax":
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        state = jnp.arange(dimension**2, dtype=jnp.complex128)
        override = jnp.asarray([0.25 - 0.1j, -0.4 + 0.3j])
    else:
        state = np.arange(dimension**2, dtype=np.complex128)
        override = np.asarray([0.25 - 0.1j, -0.4 + 0.3j])
    operator = tcp.QuditWeylOperator.from_terms(
        dimension,
        [(((0, 1, 2),), 0.7 - 0.2j), (((1, 2, 1),), 0.3 + 0.1j)],
        n_sites=2,
    )
    plan = operator.compile("backend_mvp")
    assert plan.local_dimensions == (dimension, dimension)
    assert plan.required_operations == (
        "broadcast_phase",
        "cyclic_shift",
        "multiply",
        "add",
    )
    tc.set_backend(backend_name)
    try:
        apply = tcp.backend_mvp(plan, coefficients=override)
        expected_operator = tcp.QuditWeylOperator.from_terms(
            dimension,
            [
                (((0, 1, 2),), complex(override[0])),
                (((1, 2, 1),), complex(override[1])),
            ],
            n_sites=2,
        )
        expected = expected_operator.compile("dense") @ np.asarray(state)
        actual = np.asarray(apply(state))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        rank_state = state.reshape((dimension, dimension))
        np.testing.assert_allclose(
            np.asarray(apply(rank_state)),
            expected.reshape((dimension, dimension)),
            rtol=1e-10,
            atol=1e-10,
        )
        if backend_name == "jax" and dimension == 3:
            np.testing.assert_allclose(
                np.asarray(jax.jit(apply)(state)), expected, rtol=1e-10, atol=1e-10
            )
    finally:
        tc.set_backend("numpy")


def test_plan_metadata_and_checked_weyl_dimension() -> None:
    boson = tcp.BosonOperator.from_terms(1, [(((0, "create"),), 1.0)])
    native = boson.compile("native_mvp", boson_cutoffs={0: 2})
    assert native.target == "native_mvp"
    assert native.schema_version == 1
    assert native.boson_cutoffs == ((0, 2),)
    assert native.boson_boundary == "projected_fock"
    assert native.dimension == 3
    fermion = tcp.FermionOperator.from_terms(1, [(((0, "create"),), 1.0)])
    fermion_plan = fermion.compile("native_mvp")
    assert fermion_plan.mapping == "jordan_wigner"
    assert fermion_plan.source_term_count == 1
    with pytest.raises(OverflowError):
        tcp.QuditWeylOperator.from_terms(3, [(((0, 1, 0),), 1.0)], n_sites=50).compile(
            "backend_mvp"
        )
    with pytest.raises(TypeError, match="factory"):
        advanced.BackendMVPPlan(
            2,
            0,
            1,
            np.empty((1, 0), dtype=np.uint64),
            np.empty((1, 0), dtype=np.uint64),
            np.ones(1, dtype=np.complex128),
            local_dimensions=(3, 3),
            plan_kind="direct_weyl",
            qudit_dimension=3,
            a_exponents=np.full((1, 2), 3, dtype=np.uint32),
            b_exponents=np.zeros((1, 2), dtype=np.uint32),
            required_operations=("broadcast_phase", "cyclic_shift", "multiply", "add"),
            weyl_convention="X^a Z^b",
        )
    with pytest.raises(TypeError, match="factory"):
        advanced.BackendMVPPlan(
            2,
            0,
            0,
            np.empty((0, 0), dtype=np.uint64),
            np.empty((0, 0), dtype=np.uint64),
            np.empty(0, dtype=np.complex128),
            local_dimensions=(0, 3),
        )
