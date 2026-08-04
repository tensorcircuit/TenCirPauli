"""Release-mode Phase 8.5 MVP storage and reusable-buffer benchmarks."""

from __future__ import annotations

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


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


def test_eager_plan_construction_and_apply(benchmark: BenchmarkFixture) -> None:
    operator = _pauli_operator()
    plan = benchmark(operator.compile, "native_mvp", storage="eager")
    assert plan.storage == "eager"


def test_pauli_apply_into(benchmark: BenchmarkFixture) -> None:
    plan = _pauli_operator().compile("native_mvp")
    state = np.arange(plan.dimension, dtype=np.complex128)
    output = np.empty_like(state)
    benchmark(plan.apply_into, state, output)


def test_lazy_plan_apply(benchmark: BenchmarkFixture) -> None:
    plan = _pauli_operator().compile("native_mvp")
    state = np.arange(plan.dimension, dtype=np.complex128)
    benchmark(plan.apply, state)


def test_charge_cache_and_apply_into(benchmark: BenchmarkFixture) -> None:
    space = tcp.OperatorSpace(fermions=8)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(8)})
    operator = space.fermion.create(0) * space.fermion.annihilate(1)
    restricted = operator.restrict_charge(charge.sector(1))
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.complex128)
    output = np.empty_like(state)
    benchmark(plan.apply_into, state, output)


def test_u1_lazy_and_eager_apply(benchmark: BenchmarkFixture) -> None:
    operator = tcp.PauliOperator.from_terms(
        12,
        [("XX" + "I" * 10, 1.0), ("YY" + "I" * 10, 1.0)],
    )
    restricted = operator.restrict_charge(tcp.U1Sector(12, 1))
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.complex128)
    benchmark(plan.apply, state)
