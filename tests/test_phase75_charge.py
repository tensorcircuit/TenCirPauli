"""Independent charge, sector, and restricted-basis checks for Phase 7.5."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import tencirpauli as tcp
import tencirpauli.pauli as pauli_module


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


def _independent_charge_value(
    charge: tcp.AdditiveCharge, occupations: tuple[int, ...]
) -> int:
    total = charge.offset
    for (domain, index, _), value in zip(charge.space.axes, occupations):
        if domain == "fermion":
            total += charge.fermion_weights[index] * value
        elif domain == "boson":
            total += charge.boson_weights[index] * value
        elif domain == "qubit":
            total += charge.qubit_levels[index][value]
    return total


def _independent_sector_basis(
    sector: tcp.ChargeSector,
) -> list[tuple[int, ...]]:
    constraints = sector.constraints
    return [
        occupations
        for occupations in itertools.product(
            *(range(dimension) for dimension in sector.local_dimensions)
        )
        if all(
            _independent_charge_value(charge, occupations) == target
            for charge, target in constraints
        )
    ]


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


@pytest.mark.parametrize("n_modes", [62, 63, 64, 65])
def test_wide_small_generic_charge_sector_avoids_full_space_index_limit(
    n_modes: int,
) -> None:
    space = tcp.OperatorSpace(fermions=n_modes)
    number = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(n_modes)})
    sector = number.sector(1)
    assert sector.dimension == n_modes
    assert sector.rank(sector.unrank(n_modes - 1)) == n_modes - 1

    marked = tcp.AdditiveCharge(space, fermions={0: 1})
    simultaneous = tcp.ChargeSector(((number, 1), (marked, 1)))
    assert simultaneous.dimension == 1
    assert simultaneous.unrank(0) == (1,) + (0,) * (n_modes - 1)

    if n_modes == 65:
        hopping = space.fermion.create(64) * space.fermion.annihilate(
            0
        ) + space.fermion.create(0) * space.fermion.annihilate(64)
        generic = hopping.restrict_charge(sector).dense()
        specialized = (
            hopping.map_fermions("jordan_wigner")
            .restrict_u1(tcp.U1Sector(n_modes, 1))
            .dense()
        )
        np.testing.assert_allclose(generic, specialized)
    assert sector.basis_ordering == "operator_space_axis0_msb_mixed_radix"


def test_native_charge_plan_matches_independent_mixed_basis_reference() -> None:
    space = tcp.OperatorSpace(fermions=2, bosons=1, qubits=1, qudits=(3,))
    total = tcp.AdditiveCharge(
        space,
        fermions={0: 1, 1: 1},
        bosons={0: 1},
        qubits={0: (0, 1)},
    )
    balance = tcp.AdditiveCharge(
        space,
        fermions={0: 1, 1: -1},
        qubits={0: (0, 1)},
    )
    sector = tcp.ChargeSector(
        ((total, 2), (balance, 0)),
        boson_cutoffs={0: 3},
    )
    expected = _independent_sector_basis(sector)
    actual = [tuple(int(value) for value in row) for row in sector.basis_states()]
    assert actual == expected
    assert sector.dimension == len(expected)
    for index, occupations in enumerate(expected):
        assert sector.rank(occupations) == index
        assert sector.unrank(index) == occupations


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


def test_pauli_charge_analysis_uses_cached_code_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = tcp.OperatorSpace(qubits=2)
    number = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
    operator = tcp.PauliOperator.from_terms(2, [("XX", 1.0), ("YY", 1.0)])

    def reject_per_term_conversion(self: object) -> object:
        del self
        raise AssertionError("charge analysis converted one Pauli word at a time")

    monkeypatch.setattr(pauli_module.PauliWord, "to_codes", reject_per_term_conversion)
    assert operator.analyze_charge(number).is_conserved


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
    assert restricted.estimated_bytes == restricted.mvp_plan().estimated_bytes


def test_restricted_compiler_aggregates_leaking_terms_before_sector_check() -> None:
    space = tcp.OperatorSpace(qubits=2)
    number = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
    parity_cancellation = space.qubit.x(0) * space.qubit.x(1) + space.qubit.y(
        0
    ) * space.qubit.y(1)
    sector = number.sector(0)
    restricted = parity_cancellation.restrict_charge(sector)
    np.testing.assert_array_equal(restricted.dense(), np.zeros((1, 1)))
    assert restricted.mvp_plan().transition_count == 0


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


def test_sector_zero_negative_weights_overflow_and_memory_boundaries() -> None:
    zero_weight = tcp.AdditiveCharge(tcp.OperatorSpace(bosons=1), bosons={0: 0})
    with pytest.raises(ValueError, match="finiteness"):
        zero_weight.sector(0)
    finite_zero_weight = zero_weight.sector(0, boson_cutoffs={0: 2})
    assert finite_zero_weight.dimension == 3

    negative = tcp.AdditiveCharge(tcp.OperatorSpace(bosons=1), bosons={0: -1})
    with pytest.raises(ValueError, match="finiteness"):
        negative.sector(0)
    assert negative.sector(0, boson_cutoffs={0: 2}).dimension == 1

    with pytest.raises(OverflowError):
        tcp.AdditiveCharge(tcp.OperatorSpace(qudits=(3,) * 50)).sector(0)
    with pytest.raises(MemoryError):
        tcp.AdditiveCharge(tcp.OperatorSpace(fermions=4), fermions={0: 1}).sector(
            0, max_bytes=1
        )


@pytest.mark.parametrize("base", [2**53, 2**63, 2**127, -(2**127)])
def test_large_integer_charge_weights_never_use_lossy_float_selection_rules(
    base: int,
) -> None:
    space = tcp.OperatorSpace(fermions=2)
    hopping = tcp.FermionOperator.from_terms(
        2, [(((0, "create"), (1, "annihilate")), 1.0)]
    )
    broken = tcp.AdditiveCharge(space, fermions={0: base, 1: base + 1})
    equal = tcp.AdditiveCharge(space, fermions={0: base, 1: base}, offset=base)
    assert not hopping.conserves(broken)
    assert hopping.analyze_charge(broken).commutator_term_count == 1
    assert hopping.conserves(equal)


def test_charge_generator_rejects_unrepresentable_integer_coefficients() -> None:
    charge = tcp.AdditiveCharge(tcp.OperatorSpace(fermions=1), fermions={0: 2**53 + 1})
    with pytest.raises(ValueError, match="representable exactly"):
        charge.as_operator()


def test_charge_analysis_and_sector_preflight_share_low_memory_policy() -> None:
    qubit_space = tcp.OperatorSpace(qubits=2)
    charge = tcp.AdditiveCharge(qubit_space, qubits={0: (0, 1), 1: (0, 1)})
    operator = tcp.PauliOperator.from_terms(2, [((1, 1), 1.0)])
    with pytest.raises(MemoryError):
        operator.analyze_charge(charge, max_bytes=1)

    boson_space = tcp.OperatorSpace(bosons=1)
    boson_charge = tcp.AdditiveCharge(boson_space, bosons={0: 1})
    with pytest.raises(MemoryError, match="preflight"):
        boson_charge.sector(0, boson_cutoffs={0: 200_000}, max_bytes=1)

    bounded = boson_charge.sector(0, boson_cutoffs={0: 2}, max_bytes=None)
    larger = boson_charge.sector(0, boson_cutoffs={0: 3}, max_bytes=None)
    assert bounded.dimension == larger.dimension == 1
    assert larger.estimated_bytes > bounded.estimated_bytes


def test_native_restricted_compiler_uses_plan_rank_unrank_without_basis_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = tcp.OperatorSpace(fermions=4)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(4)})
    sector = charge.sector(2)
    operator = space.fermion.create(0) * space.fermion.annihilate(
        1
    ) + space.fermion.create(1) * space.fermion.annihilate(0)

    def forbidden_basis(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("native restricted compilation must not materialize basis")

    monkeypatch.setattr(tcp.ChargeSector, "basis_states", forbidden_basis)
    restricted = operator.restrict_charge(sector)
    assert restricted.dimension == 6
    assert restricted.mvp_plan().transition_count == 4
