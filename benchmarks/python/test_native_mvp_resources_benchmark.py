"""Release-mode native MVP storage and reusable-buffer benchmarks."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


def _record_plan_metadata(
    benchmark: BenchmarkFixture,
    plan: Any,
    state: np.ndarray[Any, Any],
    *,
    allocation_mode: str,
    max_abs_error: float,
) -> None:
    """Record stable workload and resource metadata beside each timing."""
    benchmark.extra_info.update(
        {
            "dimension": int(plan.dimension),
            "term_count": int(getattr(plan, "term_count", -1)),
            "transition_count": int(getattr(plan, "transition_count", -1)),
            "storage": str(plan.storage),
            "strategy": str(plan.strategy),
            "state_bytes": int(state.nbytes),
            "output_bytes": int(state.nbytes),
            "retained_plan_bytes": int(plan.estimated_bytes),
            "allocation_mode": allocation_mode,
            "max_abs_error": float(max_abs_error),
            "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "unset"),
        }
    )


def _pauli_operator() -> tcp.PauliOperator:
    return tcp.PauliOperator.from_terms(
        10,
        [
            (tuple((index + qubit) % 4 for qubit in range(10)), 1.0 + 0.01j * index)
            for index in range(32)
        ],
    )


def test_lazy_plan_construction_and_apply(benchmark: BenchmarkFixture) -> None:
    operator = _pauli_operator()
    plan = benchmark(operator.compile, "native_mvp", storage="lazy")
    assert plan.storage == "lazy"


def test_eager_plan_construction(benchmark: BenchmarkFixture) -> None:
    operator = _pauli_operator()
    plan = benchmark(operator.compile, "native_mvp", storage="eager")
    assert plan.storage == "eager"


def test_lazy_plan_first_apply(benchmark: BenchmarkFixture) -> None:
    plan = _pauli_operator().compile("native_mvp", storage="lazy")
    state = np.arange(plan.dimension, dtype=np.complex128)
    result = benchmark.pedantic(plan.apply, args=(state,), rounds=1, iterations=1)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="one_owned_output",
        max_abs_error=float(np.max(np.abs(result - plan.apply(state)))),
    )
    assert result.shape == state.shape


def test_eager_plan_steady_apply(benchmark: BenchmarkFixture) -> None:
    plan = _pauli_operator().compile("native_mvp", storage="eager")
    state = np.arange(plan.dimension, dtype=np.complex128)
    expected = plan.apply(state)
    result = benchmark(plan.apply, state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="one_owned_output",
        max_abs_error=float(np.max(np.abs(result - expected))),
    )
    assert result.shape == state.shape


def test_pauli_apply_into(benchmark: BenchmarkFixture) -> None:
    plan = _pauli_operator().compile("native_mvp")
    state = np.arange(plan.dimension, dtype=np.complex128)
    output = np.empty_like(state)
    benchmark(plan.apply_into, state, output)
    expected = plan.apply(state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="caller_owned_output",
        max_abs_error=float(np.max(np.abs(output - expected))),
    )


def test_pauli_repeated_two_buffer_apply_into(benchmark: BenchmarkFixture) -> None:
    plan = _pauli_operator().compile("native_mvp")
    state = np.arange(plan.dimension, dtype=np.complex128)
    first = np.empty_like(state)
    second = np.empty_like(state)

    def run() -> None:
        plan.apply_into(state, first)
        plan.apply_into(state, second)

    benchmark.pedantic(run, rounds=5, iterations=1)
    expected = plan.apply(state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="alternating_caller_owned_outputs",
        max_abs_error=float(
            max(np.max(np.abs(first - expected)), np.max(np.abs(second - expected)))
        ),
    )


def test_lazy_plan_apply(benchmark: BenchmarkFixture) -> None:
    plan = _pauli_operator().compile("native_mvp")
    state = np.arange(plan.dimension, dtype=np.complex128)
    result = benchmark(plan.apply, state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="one_owned_output",
        max_abs_error=float(np.max(np.abs(result - plan.apply(state)))),
    )


def test_charge_cache_and_apply_into(benchmark: BenchmarkFixture) -> None:
    space = tcp.OperatorSpace(fermions=8)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(8)})
    operator = space.fermion.create(0) * space.fermion.annihilate(1)
    restricted = operator.restrict_charge(charge.sector(1))
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.complex128)
    output = np.empty_like(state)
    benchmark(plan.apply_into, state, output)
    expected = plan.apply(state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="caller_owned_output",
        max_abs_error=float(np.max(np.abs(output - expected))),
    )


def test_u1_lazy_and_eager_apply(benchmark: BenchmarkFixture) -> None:
    operator = tcp.PauliOperator.from_terms(
        12,
        [("XX" + "I" * 10, 1.0), ("YY" + "I" * 10, 1.0)],
    )
    restricted = operator.restrict_charge(tcp.U1Sector(12, 1))
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.complex128)
    result = benchmark(plan.apply, state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="one_owned_output",
        max_abs_error=float(np.max(np.abs(result - plan.apply(state)))),
    )


def test_u1_eager_steady_apply(benchmark: BenchmarkFixture) -> None:
    operator = tcp.PauliOperator.from_terms(
        12,
        [("XX" + "I" * 10, 1.0), ("YY" + "I" * 10, 1.0)],
    )
    plan = operator.restrict_charge(tcp.U1Sector(12, 1)).mvp_plan(storage="eager")
    state = np.arange(plan.dimension, dtype=np.complex128)
    expected = plan.apply(state)
    result = benchmark(plan.apply, state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="one_owned_output",
        max_abs_error=float(np.max(np.abs(result - expected))),
    )


def _hubbard_terms(
    rows: int, columns: int, hopping: float = 1.0, interaction: float = 4.0
) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    sites = rows * columns
    terms: list[tuple[tuple[tuple[int, str], ...], complex]] = []
    for site in range(sites):
        terms.append(
            (
                (
                    (site, "create"),
                    (site, "annihilate"),
                    (sites + site, "create"),
                    (sites + site, "annihilate"),
                ),
                complex(interaction),
            )
        )
    bonds = []
    for row in range(rows):
        for column in range(columns):
            site = row * columns + column
            if column + 1 < columns:
                bonds.append((site, site + 1))
            if row + 1 < rows:
                bonds.append((site, site + columns))
    for left, right in bonds:
        for spin in (0, 1):
            left_mode = left + spin * sites
            right_mode = right + spin * sites
            terms.extend(
                [
                    (((left_mode, "create"), (right_mode, "annihilate")), -hopping),
                    (((right_mode, "create"), (left_mode, "annihilate")), -hopping),
                ]
            )
    return terms


def _hubbard_restricted(rows: int, columns: int) -> Any:
    sites = rows * columns
    space = tcp.OperatorSpace(fermions=2 * sites)
    total = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(2 * sites)})
    balance = tcp.AdditiveCharge(
        space,
        fermions={index: (1 if index < sites else -1) for index in range(2 * sites)},
    )
    sector = tcp.ChargeSector(((total, sites), (balance, 0)))
    operator = tcp.FermionOperator.from_terms(2 * sites, _hubbard_terms(rows, columns))
    return operator.restrict_charge(sector)


def _all_to_all_charge_restricted(nfermions: int) -> Any:
    space = tcp.OperatorSpace(fermions=nfermions, bosons=1)
    charge = tcp.AdditiveCharge(
        space, fermions={index: 1 for index in range(nfermions)}
    )
    operator: Any = None
    for left in range(nfermions):
        for right in range(left + 1, nfermions):
            term = space.fermion.create(left) * space.fermion.annihilate(right)
            term = term + space.fermion.create(right) * space.fermion.annihilate(left)
            operator = term if operator is None else operator + term
    sector = tcp.ChargeSector(((charge, nfermions // 2),), boson_cutoffs={0: 0})
    return operator.restrict_charge(sector)


@pytest.mark.parametrize("storage", ["lazy", "eager"])
def test_structured_storage_apply(benchmark: BenchmarkFixture, storage: str) -> None:
    space = tcp.OperatorSpace(bosons=4)
    operator = space.boson.create(0) * space.boson.annihilate(1)
    plan = operator.compile(
        "native_mvp",
        storage=storage,
        boson_cutoffs={0: 7, 1: 7, 2: 7, 3: 7},
    )
    state = np.arange(plan.dimension, dtype=np.complex128)
    result = benchmark(plan.apply, state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="one_owned_output",
        max_abs_error=float(np.max(np.abs(result - plan.apply(state)))),
    )
    assert plan.estimated_bytes > 0


def test_generic_charge_aggregation_steady_apply(benchmark: BenchmarkFixture) -> None:
    # The boson spectator deliberately keeps this off the packed U1 shortcut.
    # XX and YY must still cancel their nonconserving contributions before the
    # restricted transition graph is accepted.
    space = tcp.OperatorSpace(bosons=1, qubits=10)
    number = tcp.AdditiveCharge(
        space,
        bosons={0: 0},
        qubits={index: (0, 1) for index in range(space.qubits)},
    )
    operator = space.qubit.x(0) * space.qubit.x(1) + space.qubit.y(0) * space.qubit.y(1)
    restricted = operator.restrict_charge(number.sector(5, boson_cutoffs={0: 0}))
    plan = restricted.mvp_plan()
    assert plan.strategy == "term_direct"
    state = np.arange(plan.dimension, dtype=np.complex128)
    result = benchmark(plan.apply, state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="one_owned_output",
        max_abs_error=float(np.max(np.abs(result - plan.apply(state)))),
    )
    assert result.shape == state.shape


def test_generic_charge_prepared_layout_apply_into(
    benchmark: BenchmarkFixture,
) -> None:
    # Keep the sector small while using enough canonical descriptors to make
    # any repeated O(T) plan validation visible in steady execution.
    space = tcp.OperatorSpace(bosons=1, qubits=6)
    number = tcp.AdditiveCharge(
        space,
        bosons={0: 0},
        qubits={index: (0, 1) for index in range(space.qubits)},
    )
    operator: Any = None
    for mask in range(1, 1 << space.qubits):
        term: Any = None
        for qubit in range(space.qubits):
            if mask & (1 << qubit):
                factor = space.qubit.z(qubit)
                term = factor if term is None else term * factor
        term = term * (1.0 / mask)
        operator = term if operator is None else operator + term
    restricted = operator.restrict_charge(number.sector(3, boson_cutoffs={0: 0}))
    plan = restricted.mvp_plan()
    assert plan.strategy == "term_direct"
    state = np.linspace(-0.5, 0.75, plan.dimension).astype(np.complex128)
    output = np.empty_like(state)
    benchmark(plan.apply_into, state, output)
    expected = plan.apply(state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="caller_owned_output",
        max_abs_error=float(np.max(np.abs(output - expected))),
    )
    benchmark.extra_info.update(
        {
            "native_layout_conversions_per_apply": 0,
            "native_term_validation_passes_per_apply": 0,
        }
    )


def test_generic_charge_eager_construction(benchmark: BenchmarkFixture) -> None:
    plan = benchmark(lambda: _hubbard_restricted(1, 4).mvp_plan(storage="eager"))
    assert plan.strategy == "destination_major_csr"
    assert plan.transition_count > 0


def test_generic_charge_eager_first_apply(benchmark: BenchmarkFixture) -> None:
    plan = _hubbard_restricted(1, 4).mvp_plan(storage="eager")
    state = np.arange(plan.dimension, dtype=np.complex128)
    result = benchmark.pedantic(plan.apply, args=(state,), rounds=1, iterations=1)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="one_owned_output",
        max_abs_error=float(np.max(np.abs(result - plan.apply(state)))),
    )


def test_generic_charge_eager_steady_apply(benchmark: BenchmarkFixture) -> None:
    restricted = _hubbard_restricted(1, 4)
    plan = restricted.mvp_plan(storage="eager")
    state = np.arange(plan.dimension, dtype=np.complex128)
    result = benchmark(plan.apply, state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="one_owned_output",
        max_abs_error=float(np.max(np.abs(result - plan.apply(state)))),
    )
    assert plan.strategy == "destination_major_csr"
    assert plan.transition_count > 0


def test_generic_charge_csr_materialization_after_cache(
    benchmark: BenchmarkFixture,
) -> None:
    restricted = _hubbard_restricted(1, 4)
    restricted.mvp_plan(storage="eager")
    matrix = benchmark(restricted.csr)
    assert matrix.shape == (restricted.dimension, restricted.dimension)


def test_spinful_hubbard_2x4_cached_lazy_apply(benchmark: BenchmarkFixture) -> None:
    restricted = _hubbard_restricted(2, 4)
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.complex128)
    output = np.empty_like(state)
    benchmark(plan.apply_into, state, output)
    expected = plan.apply(state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="caller_owned_output",
        max_abs_error=float(np.max(np.abs(output - expected))),
    )
    assert plan.dimension == 4900
    assert plan.estimated_bytes < 2 * 1024 * 1024


@pytest.mark.performance_large
def test_spinful_hubbard_4x3_cached_lazy_apply(benchmark: BenchmarkFixture) -> None:
    restricted = _hubbard_restricted(4, 3)
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.complex128)
    output = np.empty_like(state)
    benchmark(plan.apply_into, state, output)
    expected = plan.apply(state)
    _record_plan_metadata(
        benchmark,
        plan,
        state,
        allocation_mode="caller_owned_output",
        max_abs_error=float(np.max(np.abs(output - expected))),
    )
    assert plan.dimension == 853776
    assert plan.estimated_bytes < 4 * 1024 * 1024


@pytest.mark.performance_large
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_generic_charge_large_csr_gather_ab(
    benchmark: BenchmarkFixture, parallel: bool
) -> None:
    restricted = _all_to_all_charge_restricted(16)
    plan = restricted.mvp_plan(storage="eager")
    state = np.arange(plan.dimension, dtype=np.complex128)
    result = benchmark(
        plan._native_plan.apply_with_parallelism,
        state,
        2**63 - 1,
        parallel,
    )
    assert result.shape == state.shape
    assert plan.transition_count == 823680
