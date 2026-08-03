"""Independent P0/P1 checks for Phase 7.5 Majorana and mapping contracts."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import tencirpauli as tcp


def _fermion_matrix(n_modes: int, factors: tuple[tuple[int, str], ...]) -> np.ndarray:
    dimension = 1 << n_modes
    result = np.eye(dimension, dtype=np.complex128)
    for mode, action in factors:
        local = np.zeros_like(result)
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
            lower_mask = sum(1 << (n_modes - 1 - index) for index in range(mode))
            sign = -1.0 if (column & lower_mask).bit_count() & 1 else 1.0
            local[row, column] = sign
        result = result @ local
    return result


def _majorana_matrices(n_modes: int) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for mode in range(n_modes):
        annihilate = _fermion_matrix(n_modes, ((mode, "annihilate"),))
        create = _fermion_matrix(n_modes, ((mode, "create"),))
        result.extend((create + annihilate, 1j * (create - annihilate)))
    return tuple(result)


def _majorana_reference(n_modes: int, indices: tuple[int, ...]) -> np.ndarray:
    result = np.eye(1 << n_modes, dtype=np.complex128)
    generators = _majorana_matrices(n_modes)
    for index in indices:
        result = result @ generators[index]
    return result


def _encoded_basis_permutation(plan: tcp.FermionQubitMapping) -> np.ndarray:
    dimension = 1 << plan.n_modes
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for occupation_integer in range(dimension):
        occupation = tuple(
            (occupation_integer >> (plan.n_modes - 1 - index)) & 1
            for index in range(plan.n_modes)
        )
        encoded = plan.encode_occupation(occupation)
        encoded_integer = 0
        for bit in encoded:
            encoded_integer = (encoded_integer << 1) | bit
        result[encoded_integer, occupation_integer] = 1.0
    return result


def test_majorana_word_signs_and_canonical_constructor() -> None:
    with pytest.raises(ValueError, match="sorted"):
        tcp.MajoranaWord(2, (1, 0))
    with pytest.raises(ValueError, match="unique"):
        tcp.MajoranaWord(2, (1, 1))
    product = tcp.MajoranaWord.from_indices(2, (1, 0, 1))
    assert product.word == tcp.MajoranaWord(2, (0,))
    assert product.sign == -1

    for left_indices in itertools.chain.from_iterable(
        itertools.combinations(range(4), degree) for degree in range(5)
    ):
        left = tcp.MajoranaWord(2, left_indices)
        for right_indices in itertools.chain.from_iterable(
            itertools.combinations(range(4), degree) for degree in range(5)
        ):
            right = tcp.MajoranaWord(2, right_indices)
            product = left.multiply(right)
            actual = _majorana_reference(2, left_indices) @ _majorana_reference(
                2, right_indices
            )
            expected = product.sign * _majorana_reference(2, product.word.indices)
            np.testing.assert_allclose(actual, expected)


def test_majorana_anticommutation_adjoint_and_fock_differential() -> None:
    generators = _majorana_matrices(2)
    identity = np.eye(4, dtype=np.complex128)
    for left in range(4):
        for right in range(4):
            expected = 2 * identity if left == right else np.zeros_like(identity)
            np.testing.assert_allclose(
                generators[left] @ generators[right]
                + generators[right] @ generators[left],
                expected,
            )
    for indices in itertools.chain.from_iterable(
        itertools.combinations(range(4), degree) for degree in range(5)
    ):
        operator = tcp.MajoranaOperator.from_terms(2, [(indices, 1.0)])
        np.testing.assert_allclose(
            operator.to_fermion().compile("dense"),
            _majorana_reference(2, indices),
        )
        round_trip = operator.to_fermion().to_majorana()
        np.testing.assert_allclose(
            round_trip.to_fermion().compile("dense"),
            _majorana_reference(2, indices),
        )
    assert tcp.MajoranaOperator.from_terms(2, [((0, 1), 1.0)]).is_hermitian() is False
    assert tcp.MajoranaOperator.from_terms(2, [((0,), 1.0)]).is_hermitian()


def test_native_majorana_fermion_expansion_matches_python_reference() -> None:
    operator = tcp.MajoranaOperator.from_terms(
        3,
        [
            ((0, 1, 4, 5), 0.3 - 0.2j),
            ((1, 2, 3), -0.4j),
            ((), 0.7),
        ],
    )
    actual = operator.to_fermion().compile("dense")
    expected = sum(
        coefficient * _majorana_reference(3, indices)
        for indices, coefficient in (
            ((0, 1, 4, 5), 0.3 - 0.2j),
            ((1, 2, 3), -0.4j),
            ((), 0.7),
        )
    )
    np.testing.assert_allclose(actual, expected)


def test_majorana_operator_aggregation_and_expansion_guard() -> None:
    operator = tcp.MajoranaOperator.from_terms(2, [((1, 0, 1), 1.0), ((0,), 1.0)])
    assert operator.term_count == 0
    with pytest.raises(MemoryError):
        tcp.MajoranaOperator.from_terms(5, [(tuple(range(10)), 1.0)]).to_fermion(
            max_bytes=100
        )


@pytest.mark.parametrize("name", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_frozen_mapping_matrices_cnot_provenance_and_dense_differential(
    name: str,
) -> None:
    plan = tcp.FermionQubitMapping.from_name(name, 4)
    matrix = np.asarray(plan.encoding_matrix, dtype=np.uint8)
    inverse = np.asarray(plan.inverse_encoding_matrix, dtype=np.uint8)
    np.testing.assert_array_equal((matrix @ inverse) % 2, np.eye(4, dtype=np.uint8))
    assert not plan.encoding_matrix.flags.writeable
    assert not plan.inverse_encoding_matrix.flags.writeable
    for occupation in itertools.product((0, 1), repeat=4):
        bits = list(occupation)
        for control, target in plan.cnot_operations:
            bits[target] ^= bits[control]
        assert tuple(bits) == plan.encode_occupation(occupation)

    raw_terms = [
        (((0, "create"),), 0.7),
        (((1, "create"), (2, "annihilate")), -0.3j),
        (
            (
                (0, "create"),
                (1, "create"),
                (3, "annihilate"),
                (2, "annihilate"),
            ),
            0.2,
        ),
    ]
    operator = tcp.FermionOperator.from_terms(4, raw_terms)
    encoded_basis = _encoded_basis_permutation(plan)
    expected = (
        encoded_basis
        @ operator.map_fermions("jordan_wigner").compile("dense")
        @ encoded_basis.conj().T
    )
    actual = operator.map_fermions(plan).compile("dense")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", ["parity", "bravyi_kitaev"])
def test_native_mapping_batch_matches_python_clifford_reference(name: str) -> None:
    plan = tcp.FermionQubitMapping.from_name(name, 8)
    operator = tcp.PauliOperator.from_terms(
        8,
        [
            ((1, 0, 2, 3, 0, 1, 0, 2), 0.4 - 0.1j),
            ((3, 1, 0, 2, 1, 0, 3, 0), -0.2j),
            ((0, 0, 0, 0, 0, 0, 0, 0), 0.7),
        ],
    )
    expected = tcp.PauliOperator.from_terms(
        8,
        (
            (transformed, term.coefficient * phase)
            for term in operator.terms
            for transformed, phase in (
                plan._transform_codes_with_phase(term.word.to_codes()),
            )
        ),
    )
    actual = plan.map_pauli(operator)
    assert actual.terms == expected.terms


def test_majorana_mapping_and_reusable_plan_metadata() -> None:
    majorana = tcp.MajoranaOperator.from_terms(2, [((0, 3), 0.5), ((1,), -0.2j)])
    plan = tcp.FermionQubitMapping.bravyi_kitaev(2)
    mapped = majorana.map_fermions(plan)
    expected = plan.map_fermion_operator(majorana.to_fermion())
    np.testing.assert_array_equal(mapped.terms, expected.terms)
    compiled = majorana.compile("native_mvp", mapping=plan)
    assert compiled.mapping == "bravyi_kitaev"
    assert compiled.nqubits == 2
    assert compiled.basis_ordering == "qubit0_msb_matrix"


def test_mapping_plan_and_mapped_output_honor_memory_limits() -> None:
    with pytest.raises(MemoryError):
        tcp.FermionQubitMapping.parity(8, max_bytes=1)
    plan = tcp.FermionQubitMapping.bravyi_kitaev(2)
    operator = tcp.FermionOperator.from_terms(2, [(((0, "create"),), 1.0)])
    with pytest.raises(MemoryError):
        plan.map_fermion_operator(operator, max_bytes=1)


def test_hybrid_mapping_replaces_only_fermion_axes() -> None:
    space = tcp.OperatorSpace(fermions=2, bosons=1, qubits=1)
    operator = (
        (
            space.fermion.create(0) * space.fermion.annihilate(1)
            + space.fermion.create(1) * space.fermion.annihilate(0)
        )
        * space.boson.create(0)
        * space.qubit.z(0)
    )
    plan = tcp.FermionQubitMapping.parity(2)
    mapped = operator.map_fermions(plan)
    assert isinstance(mapped, tcp.HybridOperator)
    encoded = _encoded_basis_permutation(plan)
    expected = np.kron(
        np.kron(encoded, np.eye(2, dtype=np.complex128)),
        np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
    )
    raw = operator.compile("dense", boson_cutoffs={0: 1})
    mapped_dense = mapped.compile("dense", boson_cutoffs={0: 1})
    np.testing.assert_allclose(mapped_dense, expected @ raw @ expected.conj().T)
