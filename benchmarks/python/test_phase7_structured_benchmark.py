"""Release benchmarks for Phase 7 symbolic construction and finite kernels."""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pytest
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


def structured_sparse_operator(term_count: int) -> tcp.BosonOperator:
    """Build a pure-boson workload with a controlled finite work size."""
    local_monomials = ((0, 0), (1, 0), (0, 1), (1, 1))
    terms: List[Tuple[Tuple[Tuple[int, str], ...], complex]] = []
    for index, ((create0, annihilate0), (create1, annihilate1)) in enumerate(
        product(local_monomials, repeat=2)
    ):
        factors = (
            ((0, "create"),) * create0
            + ((0, "annihilate"),) * annihilate0
            + ((1, "create"),) * create1
            + ((1, "annihilate"),) * annihilate1
        )
        terms.append((factors, 1.0 + 0.01j * index))
        if len(terms) == term_count:
            break
    return tcp.BosonOperator.from_terms(2, terms)


def direct_weyl_workload(dimension: int = 5, n_sites: int = 3) -> tcp.QuditWeylOperator:
    """Build a uniform-Weyl chain for direct backend MVP measurements."""
    terms = []
    for site in range(n_sites):
        terms.append((((site, 1, 2),), 0.5 + 0.01j * site))
        terms.append((((site, 2, 1),), -0.25 + 0.02j * site))
    return tcp.QuditWeylOperator.from_terms(dimension, terms, n_sites=n_sites)


STRUCTURED_SPARSE_CASES = (
    (
        "small_python",
        tcp.BosonOperator.from_terms(1, [(((0, "create"),), 1.0)]),
        {0: 1},
    ),
    ("threshold_64", structured_sparse_operator(8), {0: 1, 1: 3}),
    ("medium_rust", structured_sparse_operator(16), {0: 3, 1: 3}),
    ("large_rust", structured_sparse_operator(8), {0: 7, 1: 7}),
)
STRUCTURED_SPARSE_CASE_IDS = tuple(case[0] for case in STRUCTURED_SPARSE_CASES)


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


@pytest.mark.parametrize(
    ("case", "operator", "cutoffs"),
    STRUCTURED_SPARSE_CASES,
    ids=STRUCTURED_SPARSE_CASE_IDS,
)
@pytest.mark.parametrize("target", ("coo", "csr"))
def test_phase7_structured_sparse_scaling(
    benchmark: BenchmarkFixture,
    case: str,
    operator: tcp.BosonOperator,
    cutoffs: Dict[int, int],
    target: str,
) -> None:
    """Measure adaptive Python/Rust sparse compilation over several scales."""
    dense = operator.compile("dense", boson_cutoffs=cutoffs)
    expected = operator.compile(target, boson_cutoffs=cutoffs)
    result = benchmark(operator.compile, target, boson_cutoffs=cutoffs)
    if target == "coo":
        reconstructed = np.zeros_like(dense)
        reconstructed[result.row, result.column] = result.data
        np.testing.assert_array_equal(reconstructed, dense)
        assert result.row.shape == expected.row.shape
    else:
        reconstructed = np.zeros_like(dense)
        for row in range(dense.shape[0]):
            start, stop = int(result.indptr[row]), int(result.indptr[row + 1])
            reconstructed[row, result.indices[start:stop]] = result.data[start:stop]
        np.testing.assert_array_equal(reconstructed, dense)
        assert result.indices.shape == expected.indices.shape
    benchmark.extra_info.update({"case": case, "target": target})


@pytest.mark.parametrize(
    ("case", "operator", "cutoffs"),
    STRUCTURED_SPARSE_CASES,
    ids=STRUCTURED_SPARSE_CASE_IDS,
)
def test_phase7_structured_mvp_construction(
    benchmark: BenchmarkFixture,
    case: str,
    operator: tcp.BosonOperator,
    cutoffs: Dict[int, int],
) -> None:
    """Measure plan construction, including the adaptive dispatch decision."""
    expected = operator.compile("native_mvp", boson_cutoffs=cutoffs)
    result = benchmark(operator.compile, "native_mvp", boson_cutoffs=cutoffs)
    assert result.strategy == expected.strategy
    benchmark.extra_info.update(
        {
            "case": case,
            "strategy": result.strategy,
            "dimension": result.dimension,
            "term_count": result.term_count,
            "plan_bytes": result.estimated_bytes,
        }
    )


@pytest.mark.parametrize(
    ("case", "operator", "cutoffs"),
    (
        STRUCTURED_SPARSE_CASES[0],
        STRUCTURED_SPARSE_CASES[2],
        STRUCTURED_SPARSE_CASES[3],
    ),
    ids=("small_python", "medium_rust", "large_rust"),
)
def test_phase7_structured_mvp_apply(
    benchmark: BenchmarkFixture,
    case: str,
    operator: tcp.BosonOperator,
    cutoffs: Dict[int, int],
) -> None:
    """Measure steady reusable MVP apply for Python and Rust plans."""
    plan = operator.compile("native_mvp", boson_cutoffs=cutoffs)
    state = (
        np.random.default_rng(20260803)
        .normal(size=plan.dimension)
        .astype(np.complex128)
    )
    expected = plan.apply(state)
    result = benchmark(plan.apply, state)
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update({"case": case, "strategy": plan.strategy})


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


def test_phase7_uniform_weyl_backend_mvp(benchmark: BenchmarkFixture) -> None:
    """Measure factorized uniform-qudit backend execution end to end."""
    operator = direct_weyl_workload()
    plan = operator.compile("backend_mvp")
    state = (
        np.random.default_rng(20260803)
        .normal(size=plan.dimension)
        .astype(np.complex128)
    )
    expected = operator.compile("dense") @ state
    result = benchmark(tcp.backend_mvp(plan), state)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    benchmark.extra_info.update(
        {
            "dimension": plan.dimension,
            "term_count": plan.term_count,
            "plan_bytes": plan.estimated_bytes,
            "local_dimension": plan.qudit_dimension,
        }
    )


def test_phase7_expansion_guard_smoke(benchmark: BenchmarkFixture) -> None:
    """Keep the recursive-expansion memory guard executable and bounded."""
    factors = (((0, "annihilate"),) * 5) + (((0, "create"),) * 5)

    def guarded() -> None:
        with pytest.raises(MemoryError):
            tcp.BosonOperator.from_terms(1, [(factors, 1.0)], max_bytes=1)

    benchmark(guarded)
