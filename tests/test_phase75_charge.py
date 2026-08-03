"""Independent charge, sector, and restricted-basis checks for Phase 7.5."""

from __future__ import annotations

import math

import numpy as np
import pytest

import tencirpauli as tcp


def _selected_basis(sector: tcp.ChargeSector) -> list[tuple[int, ...]]:
    return [sector.unrank(index) for index in range(sector.dimension)]


def _independent_projected(
    matrix: np.ndarray, basis: list[tuple[int, ...]]
) -> np.ndarray:
    del basis
    return matrix


def _qubit_matrix(codes: tuple[int, ...]) -> np.ndarray:
    matrices = (
        np.eye(2, dtype=np.complex128),
        np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
        np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
    )
    result = matrices[codes[0]]
    for code in codes[1:]:
        result = np.kron(result, matrices[code])
    return result


def test_charge_equality_ignores_name_and_sector_rank_unrank() -> None:
    space = tcp.OperatorSpace(fermions=3)
    first = tcp.AdditiveCharge(
        space, name="particle_number", fermions={0: 1, 1: 1, 2: 1}
    )
    second = tcp.AdditiveCharge(space, name="N", fermions={0: 1, 1: 1, 2: 1})
    assert first == second
    assert hash(first) == hash(second)
    sector = first.sector(1)
    assert sector.dimension == 3
    states = _selected_basis(sector)
    assert states == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    for index, state in enumerate(states):
        assert sector.rank(state) == index
    assert sector.basis_ordering == "operator_space_axis0_msb_mixed_radix"


def test_inferred_boson_sector_and_zero_charge_qudit_spectator() -> None:
    boson_space = tcp.OperatorSpace(bosons=2)
    total_boson = tcp.AdditiveCharge(boson_space, bosons={0: 1, 1: 1})
    sector = total_boson.sector(2)
    assert sector.boson_cutoffs == ((0, 2), (1, 2))
    assert _selected_basis(sector) == [(0, 2), (1, 1), (2, 0)]

    spectator_space = tcp.OperatorSpace(fermions=2, qudits=(3,))
    number = tcp.AdditiveCharge(spectator_space, fermions={0: 1, 1: 1})
    spectator_sector = number.sector(1)
    assert spectator_sector.local_dimensions == (2, 2, 3)
    assert spectator_sector.dimension == 6
    assert all(state[2] in range(3) for state in _selected_basis(spectator_sector))

    unbounded = tcp.AdditiveCharge(tcp.OperatorSpace(bosons=1))
    with pytest.raises(ValueError, match="finiteness"):
        unbounded.sector(0)


def test_simultaneous_charge_constraints_and_layout_validation() -> None:
    space = tcp.OperatorSpace(fermions=2)
    particle_number = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1})
    spin = tcp.AdditiveCharge(space, name="2Sz", fermions={0: 1, 1: -1})
    sector = tcp.ChargeSector(((particle_number, 1), (spin, 1)))
    assert sector.dimension == 1
    assert sector.unrank(0) == (1, 0)
    with pytest.raises(ValueError, match="repeat"):
        tcp.ChargeSector(((particle_number, 1), (particle_number, 1)))
    with pytest.raises(ValueError, match="same OperatorSpace"):
        tcp.ChargeSector(
            (
                (particle_number, 1),
                (tcp.AdditiveCharge(tcp.OperatorSpace(fermions=1), fermions={0: 1}), 1),
            )
        )


def test_charge_analysis_uses_aggregated_commutator_cancellation() -> None:
    space = tcp.OperatorSpace(qubits=2)
    number = tcp.AdditiveCharge(space, name="N", qubits={0: (0, 1), 1: (0, 1)})
    hopping = space.qubit.x(0) * space.qubit.x(1) + space.qubit.y(0) * space.qubit.y(1)
    analysis = hopping.analyze_charge(number)
    assert analysis.is_conserved
    assert analysis.commutator_term_count == 0
    assert hopping.conserves(number)
    with pytest.raises(ValueError, match="outside"):
        tcp.AdditiveCharge(space, qubits={0: (0, 1)}).sector(0, boson_cutoffs={0: 0})
    broken = space.qubit.x(0)
    assert not broken.conserves(number)


def test_restricted_qubit_targets_match_independent_projector() -> None:
    space = tcp.OperatorSpace(qubits=2)
    number = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
    hopping = space.qubit.x(0) * space.qubit.x(1) + space.qubit.y(0) * space.qubit.y(1)
    sector = number.sector(1)
    restricted = hopping.restrict_charge(sector)
    assert restricted.dimension == 2
    basis = _selected_basis(sector)
    independent_full = _qubit_matrix((1, 1)) + _qubit_matrix((2, 2))
    independent = np.asarray(
        [
            [
                independent_full[row[0] * 2 + row[1], column[0] * 2 + column[1]]
                for column in basis
            ]
            for row in basis
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(restricted.dense(), independent)
    coo = restricted.coo()
    reconstructed = np.zeros_like(independent)
    reconstructed[coo.row, coo.column] = coo.data
    np.testing.assert_array_equal(reconstructed, independent)
    csr = restricted.csr()
    csr_reconstructed = np.zeros_like(independent)
    for row in range(restricted.dimension):
        start, stop = int(csr.indptr[row]), int(csr.indptr[row + 1])
        csr_reconstructed[row, csr.indices[start:stop]] = csr.data[start:stop]
    np.testing.assert_array_equal(csr_reconstructed, independent)
    state = np.asarray([0.5 - 0.2j, -0.3 + 0.7j])
    np.testing.assert_allclose(restricted.apply(state), independent @ state)
    assert restricted.mvp_plan().transition_count == 2


def test_restricted_fermion_and_boson_transitions_are_exact() -> None:
    fermion_space = tcp.OperatorSpace(fermions=2)
    number = tcp.AdditiveCharge(fermion_space, fermions={0: 1, 1: 1})
    hopping = fermion_space.fermion.create(0) * fermion_space.fermion.annihilate(
        1
    ) + fermion_space.fermion.create(1) * fermion_space.fermion.annihilate(0)
    sector = number.sector(1)
    restricted = hopping.restrict_charge(sector)
    np.testing.assert_allclose(
        restricted.dense(), np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    )

    boson_space = tcp.OperatorSpace(bosons=2)
    total = tcp.AdditiveCharge(boson_space, bosons={0: 1, 1: 1})
    hopping_boson = boson_space.boson.create(0) * boson_space.boson.annihilate(
        1
    ) + boson_space.boson.create(1) * boson_space.boson.annihilate(0)
    boson_sector = total.sector(2)
    actual = hopping_boson.restrict_charge(boson_sector).dense()
    expected = np.asarray(
        [[0, math.sqrt(2), 0], [math.sqrt(2), 0, math.sqrt(2)], [0, math.sqrt(2), 0]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(actual, expected)


def test_qudit_spectator_is_retained_in_restricted_execution() -> None:
    space = tcp.OperatorSpace(fermions=2, qudits=(3,))
    number = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1})
    operator = (
        space.fermion.create(0)
        * space.fermion.annihilate(0)
        * space.qudit.weyl(0, 1, 0)
    )
    sector = number.sector(1)
    restricted = operator.restrict_charge(sector)
    assert restricted.dimension == 6
    expected = np.kron(np.diag([0, 1]), np.roll(np.eye(3), 1, axis=0))
    np.testing.assert_allclose(restricted.dense(), expected)


def test_restriction_rejects_leakage_and_memory_guards() -> None:
    space = tcp.OperatorSpace(qubits=2)
    number = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
    sector = number.sector(1)
    with pytest.raises(ValueError, match="conserved"):
        space.qubit.x(0).restrict_charge(sector)
    conserved = space.qubit.z(0)
    with pytest.raises(MemoryError):
        conserved.restrict_charge(sector, max_bytes=1)
    with pytest.raises(ValueError, match="shape"):
        conserved.restrict_charge(sector).apply(np.zeros(3))
