"""Release benchmarks for Phase 7 symbolic construction and finite kernels."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


def fermion_workload(n_modes: int = 12) -> tcp.FermionOperator:
    """Build sparse hopping and density-density terms."""
    terms = []
    for mode in range(n_modes - 1):
        terms.extend(
            [
                (((mode, "create"), (mode + 1, "annihilate")), 1.0),
                (((mode + 1, "create"), (mode, "annihilate")), 1.0),
            ]
        )
    for mode in range(n_modes - 1):
        terms.append(
            (
                (
                    (mode, "create"),
                    (mode, "annihilate"),
                    (mode + 1, "create"),
                    (mode + 1, "annihilate"),
                ),
                0.5,
            )
        )
    return tcp.FermionOperator.from_terms(n_modes, terms)


def fermion_terms(
    n_modes: int = 12,
) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    """Return the raw sparse fermion workload for construction benchmarks."""
    terms = []
    for mode in range(n_modes - 1):
        terms.extend(
            [
                (((mode, "create"), (mode + 1, "annihilate")), 1.0),
                (((mode + 1, "create"), (mode, "annihilate")), 1.0),
            ]
        )
    for mode in range(n_modes - 1):
        terms.append(
            (
                (
                    (mode, "create"),
                    (mode, "annihilate"),
                    (mode + 1, "create"),
                    (mode + 1, "annihilate"),
                ),
                0.5,
            )
        )
    return terms


def boson_workload() -> Tuple[tcp.HybridOperator, dict[int, int]]:
    """Build a low-degree two-mode finite-Fock workload."""
    space = tcp.OperatorSpace(bosons=2, qubits=1)
    operator = 0.7 * space.boson.create(0) * space.boson.annihilate(0)
    operator = operator + 0.4 * space.boson.create(1) * space.boson.annihilate(1)
    operator = operator + 0.2 * space.boson.create(0) * space.boson.create(1)
    operator = operator + 0.2 * space.boson.annihilate(0) * space.boson.annihilate(1)
    operator = operator + 0.3 * space.qubit.z(0)
    return operator, {0: 3, 1: 3}


def test_phase7_fermion_jordan_wigner(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure full input construction plus common one-/two-body JW mapping."""
    operator = fermion_workload()
    expected = operator.compile("native_mvp")
    result = benchmark(operator.compile, "native_mvp")
    state = np.ones(1 << 12, dtype=np.complex128)
    np.testing.assert_allclose(result.apply(state), expected.apply(state))


def test_phase7_fermion_native_construction(benchmark: BenchmarkFixture) -> None:
    """Measure raw Python input conversion plus Rust CAR canonicalization."""
    terms = fermion_terms()
    expected = tcp.FermionOperator.from_terms(12, terms)
    result = benchmark(tcp.FermionOperator.from_terms, 12, terms)
    assert result.term_count == expected.term_count


def test_phase7_fermion_native_mapping(benchmark: BenchmarkFixture) -> None:
    """Measure batched Rust Jordan-Wigner expansion and Pauli aggregation."""
    operator = fermion_workload()
    expected = operator.map_fermions()
    result = benchmark(operator.map_fermions)
    np.testing.assert_allclose(result.compile("dense"), expected.compile("dense"))


def test_phase7_boson_native_dense(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure Python conversion plus the Rust mixed-radix dense kernel."""
    operator, cutoffs = boson_workload()
    expected = operator.compile("dense", boson_cutoffs=cutoffs)
    result = benchmark(operator.compile, "dense", boson_cutoffs=cutoffs)
    np.testing.assert_allclose(result, expected)


def test_phase7_boson_native_mvp(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure reusable finite-plan apply on a mixed local-dimension state."""
    operator, cutoffs = boson_workload()
    plan = operator.compile("native_mvp", boson_cutoffs=cutoffs)
    state = np.random.default_rng(20260803).normal(size=32).astype(np.complex128)
    expected = plan.apply(state)
    result = benchmark(plan.apply, state)
    np.testing.assert_allclose(result, expected)


def test_phase7_hybrid_native_multiply(benchmark: BenchmarkFixture) -> None:
    """Measure one coarse PyO3 call for mixed-domain symbolic multiplication."""
    space = tcp.OperatorSpace(fermions=4, bosons=2, qubits=2, qudits=(3,))
    left = (
        space.fermion.annihilate(0)
        * space.boson.create(0)
        * space.qubit.x(0)
        * space.qudit.weyl(0, 1, 2)
    )
    right = (
        space.fermion.create(0)
        * space.boson.annihilate(0)
        * space.qubit.y(0)
        * space.qudit.weyl(0, 2, 1)
    )
    expected = left * right
    result = benchmark(left.multiply, right)
    np.testing.assert_allclose(
        result.compile("dense", boson_cutoffs={0: 2, 1: 2}),
        expected.compile("dense", boson_cutoffs={0: 2, 1: 2}),
    )


def test_phase7_hybrid_native_mapping(benchmark: BenchmarkFixture) -> None:
    """Measure batched mixed-domain Jordan-Wigner mapping."""
    space = tcp.OperatorSpace(fermions=4, bosons=2, qubits=1)
    operator = (
        space.fermion.create(0) * space.boson.annihilate(0)
        + space.fermion.annihilate(1) * space.boson.create(1)
        + space.qubit.z(0)
    )
    expected = operator.map_fermions()
    result = benchmark(operator.map_fermions)
    np.testing.assert_allclose(
        result.compile("dense", boson_cutoffs={0: 2, 1: 2}),
        expected.compile("dense", boson_cutoffs={0: 2, 1: 2}),
    )


def test_phase7_hybrid_native_builder(benchmark: BenchmarkFixture) -> None:
    """Measure batched Rust canonicalization from raw builder products."""
    space = tcp.OperatorSpace(fermions=8, bosons=2, qubits=2)
    builder = space.builder()
    for mode in range(7):
        builder.add_product(
            fermions=((mode, "create"), (mode + 1, "annihilate")),
            bosons=((0, "create"), (1, "annihilate")),
            qubits=((mode % 2, "Z"),),
        )
    expected = builder.finish()
    result = benchmark(builder.finish)
    assert result.term_count == expected.term_count
