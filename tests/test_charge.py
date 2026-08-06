"""Independent charge, sector, and restricted-basis checks."""

from __future__ import annotations

import itertools
import math
import threading
from typing import Any

import numpy as np
import pytest

import tencirpauli as tcp
import tencirpauli.pauli as pauli_module
from tencirpauli.structured import _StructuredOperator


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
    eager_plan = restricted.mvp_plan(storage="eager")
    assert eager_plan.transition_count == 2
    assert restricted.estimated_bytes >= eager_plan.estimated_bytes


def test_restricted_compiler_aggregates_leaking_terms_before_sector_check() -> None:
    space = tcp.OperatorSpace(qubits=2)
    number = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
    parity_cancellation = space.qubit.x(0) * space.qubit.x(1) + space.qubit.y(
        0
    ) * space.qubit.y(1)
    sector = number.sector(0)
    restricted = parity_cancellation.restrict_charge(sector)
    np.testing.assert_array_equal(restricted.dense(), np.zeros((1, 1)))
    assert restricted.mvp_plan(storage="eager").transition_count == 0

    lazy = parity_cancellation.restrict_charge(sector, storage="lazy")
    assert lazy.storage == "lazy"
    assert lazy.mvp_plan().storage == "lazy"
    np.testing.assert_array_equal(lazy.apply(np.asarray([1.0 + 0j])), np.zeros(1))
    np.testing.assert_array_equal(lazy.dense(), np.zeros((1, 1)))


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
    lazy = hopping.restrict_charge(sector, storage="lazy")
    state = np.asarray([0.5 - 0.2j, -0.3 + 0.7j])
    np.testing.assert_allclose(lazy.apply(state), restricted.apply(state))

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


def _spinful_terms(
    sites: int,
) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    hopping = 0.7 + 0.2j
    terms = [
        (
            (
                (0, "create"),
                (0, "annihilate"),
                (sites, "create"),
                (sites, "annihilate"),
            ),
            1.3 + 0.0j,
        ),
        (((0, "create"), (1, "annihilate")), hopping),
        (((1, "create"), (0, "annihilate")), hopping.conjugate()),
        (((sites, "create"), (sites + 1, "annihilate")), -hopping),
        (((sites + 1, "create"), (sites, "annihilate")), -hopping.conjugate()),
        (
            (
                (0, "create"),
                (sites + 1, "create"),
                (1, "annihilate"),
                (sites, "annihilate"),
            ),
            0.0 + 0.35j,
        ),
        (
            (
                (sites, "create"),
                (1, "create"),
                (sites + 1, "annihilate"),
                (0, "annihilate"),
            ),
            0.0 - 0.35j,
        ),
        (((0, "create"), (0, "annihilate")), 0.25 + 0.0j),
        (((0, "create"), (0, "annihilate")), -0.25 + 0.0j),
        (((1, "create"), (1, "annihilate")), 0.0 + 0.0j),
    ]
    return terms


def _operator_from_ladder_terms(
    space: tcp.OperatorSpace,
    terms: list[tuple[tuple[tuple[int, str], ...], complex]],
) -> Any:
    operator: Any = None
    for operations, coefficient in terms:
        term: Any = None
        for mode, action in operations:
            factor = getattr(space.fermion, action)(mode)
            term = factor if term is None else term * factor
        term = term * coefficient
        operator = term if operator is None else operator + term
    return operator


def _spinful_sector(
    space: tcp.OperatorSpace, sites: int, particles: int, *, boson: bool = False
) -> tcp.ChargeSector:
    total = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(2 * sites)})
    balance = tcp.AdditiveCharge(
        space,
        fermions={index: (1 if index < sites else -1) for index in range(2 * sites)},
    )
    return tcp.ChargeSector(
        ((total, 2 * particles), (balance, 0)),
        boson_cutoffs={0: 0} if boson else None,
    )


@pytest.mark.parametrize(
    ("sites", "particles"),
    [(sites, particles) for sites in range(2, 7) for particles in range(1, sites + 1)],
)
def test_spinful_fermion_fast_path_matches_generic_and_eager_across_fillings(
    sites: int, particles: int
) -> None:
    terms = _spinful_terms(sites)
    space = tcp.OperatorSpace(fermions=2 * sites)
    operator = tcp.FermionOperator.from_terms(2 * sites, terms)
    sector = _spinful_sector(space, sites, particles)
    fast = operator.restrict_charge(sector, storage="lazy")
    eager = operator.restrict_charge(sector, storage="eager")

    # A zero-cutoff boson spectator preserves the basis while making the
    # spinful shortcut ineligible, yielding a durable generic differential.
    mixed_space = tcp.OperatorSpace(fermions=2 * sites, bosons=1)
    mixed_operator = _operator_from_ladder_terms(mixed_space, terms)
    generic = mixed_operator.restrict_charge(
        _spinful_sector(mixed_space, sites, particles, boson=True)
    )
    state = np.arange(fast.dimension, dtype=np.float64).astype(np.complex128)
    state += 0.125j * state[::-1]
    output = np.empty_like(state)
    fast.mvp_plan().apply_into(state, output, max_bytes=0)
    np.testing.assert_allclose(output, generic.apply(state))
    np.testing.assert_allclose(output, eager.apply(state))


def test_spinful_fermion_combinatorial_rank_fallback_matches_generic() -> None:
    # More than 20 sites deterministically disables the dense rank table while
    # a one-particle-per-spin sector keeps the differential compact.
    sites = 21
    particles = 1
    terms = _spinful_terms(sites)
    space = tcp.OperatorSpace(fermions=2 * sites)
    operator = tcp.FermionOperator.from_terms(2 * sites, terms)
    fast = operator.restrict_charge(_spinful_sector(space, sites, particles))
    mixed_space = tcp.OperatorSpace(fermions=2 * sites, bosons=1)
    generic = _operator_from_ladder_terms(mixed_space, terms).restrict_charge(
        _spinful_sector(mixed_space, sites, particles, boson=True)
    )
    state = np.linspace(-0.3, 0.8, fast.dimension).astype(np.complex128)
    output = np.empty_like(state)
    fast.mvp_plan().apply_into(state, output, max_bytes=0)
    np.testing.assert_allclose(output, generic.apply(state))


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
    with pytest.raises(ValueError, match="storage"):
        space.qubit.z(0).restrict_charge(sector, storage="auto")
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


def test_small_integer_charge_weights_use_float_selection_rules() -> None:
    base = 7
    space = tcp.OperatorSpace(fermions=2)
    hopping = tcp.FermionOperator.from_terms(
        2, [(((0, "create"), (1, "annihilate")), 1.0)]
    )
    broken = tcp.AdditiveCharge(space, fermions={0: base, 1: base + 1})
    equal = tcp.AdditiveCharge(space, fermions={0: base, 1: base}, offset=base)
    assert not hopping.conserves(broken)
    assert hopping.analyze_charge(broken).commutator_term_count == 1
    assert hopping.conserves(equal)
    assert hopping.analyze_charge(equal).method == "native_float_selection_rules"


def test_charge_generator_uses_ordinary_float64_coefficients() -> None:
    charge = tcp.AdditiveCharge(tcp.OperatorSpace(fermions=1), fermions={0: 2**53 + 1})
    assert charge.as_operator().terms[0].coefficient == float(2**53 + 1)


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
    assert restricted.mvp_plan(storage="eager").transition_count == 4


def test_native_charge_restriction_stays_on_handles_for_all_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_materialization(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("charge restriction must not materialize Python terms")

    monkeypatch.setattr(tcp.PauliOperator, "_arrays", forbidden_materialization)
    monkeypatch.setattr(
        _StructuredOperator, "_materialized_terms", forbidden_materialization
    )

    pauli_space = tcp.OperatorSpace(qubits=2)
    pauli_charge = tcp.AdditiveCharge(pauli_space, qubits={0: (0, 2), 1: (0, 4)})
    pauli_sector = pauli_charge.sector(2)
    pauli = tcp.PauliOperator.from_terms(2, [("ZZ", 1.0)])
    assert pauli.restrict_charge(pauli_sector).mvp_plan(storage="eager").dimension == 1

    fermion_space = tcp.OperatorSpace(fermions=2)
    fermion_charge = tcp.AdditiveCharge(fermion_space, fermions={0: 1, 1: 1})
    fermion_sector = fermion_charge.sector(1)
    fermion = fermion_space.fermion.create(0) * fermion_space.fermion.annihilate(
        1
    ) + fermion_space.fermion.create(1) * fermion_space.fermion.annihilate(0)
    assert (
        fermion.restrict_charge(fermion_sector).mvp_plan(storage="eager").dimension == 2
    )

    boson_space = tcp.OperatorSpace(bosons=2)
    boson_charge = tcp.AdditiveCharge(boson_space, bosons={0: 1, 1: 1})
    boson_sector = boson_charge.sector(1)
    boson = boson_space.boson.create(0) * boson_space.boson.annihilate(
        1
    ) + boson_space.boson.create(1) * boson_space.boson.annihilate(0)
    assert boson.restrict_charge(boson_sector).mvp_plan(storage="eager").dimension == 2

    hybrid_space = tcp.OperatorSpace(fermions=1, bosons=1)
    hybrid_charge = tcp.AdditiveCharge(hybrid_space, fermions={0: 1}, bosons={0: 1})
    hybrid_sector = hybrid_charge.sector(1)
    hybrid = hybrid_space.fermion.create(0) * hybrid_space.boson.annihilate(
        0
    ) + hybrid_space.fermion.annihilate(0) * hybrid_space.boson.create(0)
    assert (
        hybrid.restrict_charge(hybrid_sector).mvp_plan(storage="eager").dimension == 2
    )


def test_native_charge_plan_construction_releases_gil() -> None:
    n_modes = 120
    space = tcp.OperatorSpace(fermions=n_modes)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(n_modes)})
    sector = charge.sector(1)
    terms = [
        (
            ((index, "create"), ((index + 1) % n_modes, "annihilate")),
            1.0,
        )
        for index in range(n_modes)
    ]
    operator = tcp.FermionOperator.from_terms(n_modes, terms)
    progress = [0]
    ready = threading.Event()
    stop = threading.Event()

    def observe() -> None:
        ready.set()
        while not stop.is_set():
            progress[0] += 1

    observer = threading.Thread(target=observe)
    observer.start()
    assert ready.wait(2.0)
    try:
        restricted = operator.restrict_charge(sector)
        assert restricted.dimension == n_modes
    finally:
        stop.set()
        observer.join(2.0)
    assert not observer.is_alive()
    assert progress[0] > 0
