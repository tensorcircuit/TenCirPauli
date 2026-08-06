"""Release benchmarks for structured symbolic construction and finite kernels."""

from __future__ import annotations

import os
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


def _record_metadata(
    benchmark: BenchmarkFixture,
    *,
    input_terms: int,
    canonical_terms: int,
    generated_contributions: int,
    dimension: int,
    nonzeros_or_transitions: int,
    plan_bytes: int = 0,
    output_bytes: int = 0,
    numerical_error: float = 0.0,
    workload: str,
) -> None:
    """Record the frozen structured workload fields on every benchmark case."""
    stats = benchmark.stats or {}
    mean = float(stats.get("mean", 0.0))
    throughput = generated_contributions / mean if mean > 0.0 else 0.0
    benchmark.extra_info.update(
        {
            "workload": workload,
            "input_terms": input_terms,
            "canonical_terms": canonical_terms,
            "generated_contributions": generated_contributions,
            "dimension": dimension,
            "nonzeros_or_transitions": nonzeros_or_transitions,
            "plan_bytes": plan_bytes,
            "output_bytes": output_bytes,
            "thread_count": int(
                os.environ.get("RAYON_NUM_THREADS", str(os.cpu_count() or 1))
            ),
            "throughput_contributions_per_second": throughput,
            "numerical_error": numerical_error,
        }
    )


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


def hubbard_quartic_terms(
    n_sites: int = 4,
) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    """Return a spinful Hubbard chain with explicit quartic onsite terms."""
    terms = []
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
    return terms


def hubbard_quartic_workload() -> tuple[
    tcp.FermionOperator,
    list[tuple[tuple[tuple[int, str], ...], complex]],
]:
    """Build a bounded eight-mode Hubbard chain for quartic JW timing."""
    terms = hubbard_quartic_terms()
    return tcp.FermionOperator.from_terms(8, terms), terms


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


def duplicate_fermion_terms(
    n_modes: int = 12, repetitions: int = 24
) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    """Return a duplicate-heavy canonicalization workload."""
    return fermion_terms(n_modes) * repetitions


def holstein_workload() -> Tuple[tcp.HybridOperator, dict[int, int]]:
    """Build a bounded mixed fermion-boson/qubit Holstein-style workload."""
    space = tcp.OperatorSpace(fermions=6, bosons=2, qubits=2)
    displacement = space.boson.create(0) + space.boson.annihilate(0)
    displacement = displacement + space.boson.create(1) + space.boson.annihilate(1)
    operator = 0.5 * space.fermion.create(0) * space.fermion.annihilate(0)
    for mode in range(1, 6):
        density = space.fermion.create(mode) * space.fermion.annihilate(mode)
        operator = operator + 0.1 * density * displacement
    operator = operator + 0.25 * space.qubit.z(0) + 0.15 * space.qubit.x(1)
    return operator, {0: 3, 1: 3}


STRUCTURED_SPARSE_CASES = (
    (
        "small_native",
        tcp.BosonOperator.from_terms(1, [(((0, "create"),), 1.0)]),
        {0: 1},
    ),
    ("threshold_64", structured_sparse_operator(8), {0: 1, 1: 3}),
    ("medium_native", structured_sparse_operator(16), {0: 3, 1: 3}),
    ("large_native", structured_sparse_operator(8), {0: 7, 1: 7}),
)
STRUCTURED_SPARSE_CASE_IDS = tuple(case[0] for case in STRUCTURED_SPARSE_CASES)


def test_structured_fermion_jordan_wigner(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure full input construction plus common one-/two-body JW mapping."""
    operator = fermion_workload()
    expected = operator.compile("native_mvp")
    result = benchmark(operator.compile, "native_mvp")
    state = np.ones(1 << 12, dtype=np.complex128)
    np.testing.assert_allclose(result.apply(state), expected.apply(state))
    _record_metadata(
        benchmark,
        input_terms=len(fermion_terms()),
        canonical_terms=operator.term_count,
        generated_contributions=len(operator.map_fermions().terms),
        dimension=1 << 12,
        nonzeros_or_transitions=operator.term_count * (1 << 12),
        plan_bytes=result.estimated_bytes,
        output_bytes=state.nbytes,
        workload="fermion_jordan_wigner_native_mvp",
    )


def test_structured_fermion_native_construction(benchmark: BenchmarkFixture) -> None:
    """Measure raw Python input conversion plus Rust CAR canonicalization."""
    terms = fermion_terms()
    expected = tcp.FermionOperator.from_terms(12, terms)
    result = benchmark(tcp.FermionOperator.from_terms, 12, terms)
    assert result.term_count == expected.term_count
    _record_metadata(
        benchmark,
        input_terms=len(terms),
        canonical_terms=result.term_count,
        generated_contributions=len(terms),
        dimension=1 << 12,
        nonzeros_or_transitions=result.term_count,
        workload="fermion_sparse_construction",
    )


def test_structured_fermion_native_mapping(benchmark: BenchmarkFixture) -> None:
    """Measure batched Rust Jordan-Wigner expansion and Pauli aggregation."""
    operator = fermion_workload()
    expected = operator.map_fermions()
    result = benchmark(operator.map_fermions)
    np.testing.assert_allclose(result.compile("dense"), expected.compile("dense"))
    _record_metadata(
        benchmark,
        input_terms=len(fermion_terms()),
        canonical_terms=operator.term_count,
        generated_contributions=len(expected.terms),
        dimension=1 << 12,
        nonzeros_or_transitions=len(expected.terms),
        output_bytes=expected.compile("dense").nbytes,
        workload="fermion_jordan_wigner_mapping",
    )


def test_structured_hubbard_quartic_mapping(benchmark: BenchmarkFixture) -> None:
    """Measure JW mapping of explicit Hubbard quartic interactions."""
    operator, raw_terms = hubbard_quartic_workload()
    expected = operator.map_fermions()
    result = benchmark(operator.map_fermions)
    np.testing.assert_allclose(result.compile("dense"), expected.compile("dense"))
    generated = sum(
        1 << (len(term.word.creation_modes) + len(term.word.annihilation_modes))
        for term in operator.terms
    )
    _record_metadata(
        benchmark,
        input_terms=len(raw_terms),
        canonical_terms=operator.term_count,
        generated_contributions=generated,
        dimension=1 << 8,
        nonzeros_or_transitions=len(expected.terms),
        output_bytes=expected.compile("dense").nbytes,
        workload="hubbard_quartic_jordan_wigner_mapping",
    )


def test_structured_fermion_duplicate_canonicalization(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure aggregation-heavy duplicate canonicalization."""
    terms = duplicate_fermion_terms()
    expected = tcp.FermionOperator.from_terms(12, terms)
    result = benchmark(tcp.FermionOperator.from_terms, 12, terms)
    assert result.term_count == expected.term_count
    _record_metadata(
        benchmark,
        input_terms=len(terms),
        canonical_terms=result.term_count,
        generated_contributions=len(terms),
        dimension=1 << 12,
        nonzeros_or_transitions=result.term_count,
        workload="fermion_duplicate_aggregation",
    )


def test_structured_boson_native_dense(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure Python conversion plus the Rust mixed-radix dense kernel."""
    operator, cutoffs = boson_workload()
    expected = operator.compile("dense", boson_cutoffs=cutoffs)
    result = benchmark(operator.compile, "dense", boson_cutoffs=cutoffs)
    np.testing.assert_allclose(result, expected)
    _record_metadata(
        benchmark,
        input_terms=operator.term_count,
        canonical_terms=operator.term_count,
        generated_contributions=operator.term_count,
        dimension=result.shape[0],
        nonzeros_or_transitions=int(np.count_nonzero(result)),
        output_bytes=result.nbytes,
        numerical_error=float(np.max(np.abs(result - expected))),
        workload="hybrid_boson_dense",
    )


def test_structured_boson_native_mvp(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure reusable finite-plan apply on a mixed local-dimension state."""
    operator, cutoffs = boson_workload()
    plan = operator.compile("native_mvp", boson_cutoffs=cutoffs)
    state = np.random.default_rng(20260803).normal(size=32).astype(np.complex128)
    expected = plan.apply(state)
    result = benchmark(plan.apply, state)
    np.testing.assert_allclose(result, expected)
    _record_metadata(
        benchmark,
        input_terms=operator.term_count,
        canonical_terms=operator.term_count,
        generated_contributions=operator.term_count,
        dimension=plan.dimension,
        nonzeros_or_transitions=operator.term_count,
        plan_bytes=plan.estimated_bytes,
        output_bytes=result.nbytes,
        numerical_error=float(np.max(np.abs(result - expected))),
        workload="hybrid_boson_native_mvp_apply",
    )


@pytest.mark.parametrize(
    ("case", "operator", "cutoffs"),
    STRUCTURED_SPARSE_CASES,
    ids=STRUCTURED_SPARSE_CASE_IDS,
)
@pytest.mark.parametrize("target", ("coo", "csr"))
def test_structured_structured_sparse_scaling(
    benchmark: BenchmarkFixture,
    case: str,
    operator: tcp.BosonOperator,
    cutoffs: Dict[int, int],
    target: str,
) -> None:
    """Measure native sparse compilation over several finite scales."""
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
    nonzeros = len(result.row) if target == "coo" else len(result.indices)
    _record_metadata(
        benchmark,
        input_terms=operator.term_count,
        canonical_terms=operator.term_count,
        generated_contributions=operator.term_count * dense.shape[0],
        dimension=dense.shape[0],
        nonzeros_or_transitions=nonzeros,
        output_bytes=(
            int(result.row.nbytes + result.column.nbytes + result.data.nbytes)
            if target == "coo"
            else int(result.indptr.nbytes + result.indices.nbytes + result.data.nbytes)
        ),
        numerical_error=float(np.max(np.abs(reconstructed - dense))),
        workload=f"{case}_{target}_native",
    )


@pytest.mark.parametrize(
    ("case", "operator", "cutoffs"),
    STRUCTURED_SPARSE_CASES,
    ids=STRUCTURED_SPARSE_CASE_IDS,
)
def test_structured_structured_mvp_construction(
    benchmark: BenchmarkFixture,
    case: str,
    operator: tcp.BosonOperator,
    cutoffs: Dict[int, int],
) -> None:
    """Measure reusable native plan construction."""
    expected = operator.compile("native_mvp", boson_cutoffs=cutoffs)
    result = benchmark(operator.compile, "native_mvp", boson_cutoffs=cutoffs)
    assert result.strategy == expected.strategy
    _record_metadata(
        benchmark,
        input_terms=operator.term_count,
        canonical_terms=operator.term_count,
        generated_contributions=operator.term_count,
        dimension=result.dimension,
        nonzeros_or_transitions=operator.term_count,
        plan_bytes=result.estimated_bytes,
        workload=f"{case}_native_mvp_construction",
    )


@pytest.mark.parametrize(
    ("case", "operator", "cutoffs"),
    (
        STRUCTURED_SPARSE_CASES[0],
        STRUCTURED_SPARSE_CASES[2],
        STRUCTURED_SPARSE_CASES[3],
    ),
    ids=("small_native", "medium_native", "large_native"),
)
def test_structured_structured_mvp_apply(
    benchmark: BenchmarkFixture,
    case: str,
    operator: tcp.BosonOperator,
    cutoffs: Dict[int, int],
) -> None:
    """Measure steady reusable native MVP apply."""
    plan = operator.compile("native_mvp", boson_cutoffs=cutoffs)
    state = (
        np.random.default_rng(20260803)
        .normal(size=plan.dimension)
        .astype(np.complex128)
    )
    expected = plan.apply(state)
    result = benchmark(plan.apply, state)
    np.testing.assert_allclose(result, expected)
    _record_metadata(
        benchmark,
        input_terms=operator.term_count,
        canonical_terms=operator.term_count,
        generated_contributions=operator.term_count,
        dimension=plan.dimension,
        nonzeros_or_transitions=operator.term_count,
        plan_bytes=plan.estimated_bytes,
        output_bytes=result.nbytes,
        numerical_error=float(np.max(np.abs(result - expected))),
        workload=f"{case}_native_mvp_apply",
    )


def test_structured_hybrid_native_multiply(benchmark: BenchmarkFixture) -> None:
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
    _record_metadata(
        benchmark,
        input_terms=2,
        canonical_terms=result.term_count,
        generated_contributions=1,
        dimension=2**4 * 3**2 * 2**2 * 3,
        nonzeros_or_transitions=result.term_count,
        workload="hybrid_coarse_multiply",
    )


def test_structured_hybrid_native_mapping(benchmark: BenchmarkFixture) -> None:
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
    _record_metadata(
        benchmark,
        input_terms=operator.term_count,
        canonical_terms=operator.term_count,
        generated_contributions=len(expected.terms),
        dimension=2**4 * 3**2 * 2,
        nonzeros_or_transitions=len(expected.terms),
        workload="hybrid_jordan_wigner_mapping",
    )


def test_structured_hybrid_native_builder(benchmark: BenchmarkFixture) -> None:
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
    _record_metadata(
        benchmark,
        input_terms=7,
        canonical_terms=result.term_count,
        generated_contributions=7,
        dimension=2**8 * 3**2 * 2**2,
        nonzeros_or_transitions=result.term_count,
        workload="hybrid_builder_batch_canonicalization",
    )


def test_structured_holstein_native_mvp(benchmark: BenchmarkFixture) -> None:
    """Measure bounded Holstein-style mixed native plan construction and apply."""
    operator, cutoffs = holstein_workload()
    plan = operator.compile("native_mvp", boson_cutoffs=cutoffs)
    state = (
        np.random.default_rng(20260803)
        .normal(size=plan.dimension)
        .astype(np.complex128)
    )
    expected = plan.apply(state)
    result = benchmark(plan.apply, state)
    error = float(np.max(np.abs(result - expected)))
    np.testing.assert_allclose(result, expected)
    _record_metadata(
        benchmark,
        input_terms=operator.term_count,
        canonical_terms=operator.term_count,
        generated_contributions=operator.term_count,
        dimension=plan.dimension,
        nonzeros_or_transitions=operator.term_count,
        plan_bytes=plan.estimated_bytes,
        output_bytes=result.nbytes,
        numerical_error=error,
        workload="holstein_mixed_native_mvp_apply",
    )


def test_structured_holstein_native_mvp_first_apply(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure the first caller-visible apply after native plan construction."""
    operator, cutoffs = holstein_workload()
    plan = operator.compile("native_mvp", boson_cutoffs=cutoffs)
    state = (
        np.random.default_rng(20260803)
        .normal(size=plan.dimension)
        .astype(np.complex128)
    )
    result = benchmark.pedantic(
        plan.apply, args=(state,), rounds=1, iterations=1, warmup_rounds=0
    )
    expected = plan.apply(state)
    error = float(np.max(np.abs(result - expected)))
    np.testing.assert_allclose(result, expected)
    _record_metadata(
        benchmark,
        input_terms=operator.term_count,
        canonical_terms=operator.term_count,
        generated_contributions=operator.term_count,
        dimension=plan.dimension,
        nonzeros_or_transitions=operator.term_count,
        plan_bytes=plan.estimated_bytes,
        output_bytes=result.nbytes,
        numerical_error=error,
        workload="holstein_mixed_native_mvp_first_apply",
    )


@pytest.mark.parametrize("dimension", [3, 4, 5, 6])
def test_structured_uniform_weyl_backend_mvp(
    benchmark: BenchmarkFixture, dimension: int
) -> None:
    """Measure factorized uniform-qudit backend execution end to end."""
    operator = direct_weyl_workload(dimension)
    plan = operator.compile("backend_mvp")
    state = (
        np.random.default_rng(20260803)
        .normal(size=plan.dimension)
        .astype(np.complex128)
    )
    expected = operator.compile("dense") @ state
    result = benchmark(tcp.backend_mvp(plan), state)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    _record_metadata(
        benchmark,
        input_terms=operator.term_count,
        canonical_terms=operator.term_count,
        generated_contributions=operator.term_count,
        dimension=plan.dimension,
        nonzeros_or_transitions=operator.term_count * plan.dimension,
        plan_bytes=plan.estimated_bytes,
        output_bytes=result.nbytes,
        numerical_error=float(np.max(np.abs(result - expected))),
        workload=f"uniform_weyl_d{dimension}_backend_mvp",
    )


@pytest.mark.parametrize(
    ("dimension", "target"),
    tuple(product((3, 5, 7), ("coo", "csr", "native_mvp"))),
    ids=lambda value: str(value),
)
def test_structured_uniform_weyl_sparse_and_mvp(
    benchmark: BenchmarkFixture, dimension: int, target: str
) -> None:
    """Measure direct qudit Hamiltonian sparse construction and native MVP."""
    operator = direct_weyl_workload(dimension, n_sites=3)
    dense = operator.compile("dense")
    expected = operator.compile(target)
    state = (
        np.random.default_rng(20260804 + dimension)
        .normal(size=dense.shape[0])
        .astype(np.complex128)
    )
    result = benchmark(operator.compile, target)
    if target == "coo":
        reconstructed = np.zeros_like(dense)
        np.add.at(reconstructed, (result.row, result.column), result.data)
        numerical_error = float(np.max(np.abs(reconstructed - dense)))
        np.testing.assert_allclose(reconstructed, dense)
        nonzeros = len(result.data)
        output_bytes = int(
            result.row.nbytes + result.column.nbytes + result.data.nbytes
        )
        plan_bytes = 0
    elif target == "csr":
        reconstructed = np.zeros_like(dense)
        for row in range(dense.shape[0]):
            start, stop = int(result.indptr[row]), int(result.indptr[row + 1])
            reconstructed[row, result.indices[start:stop]] += result.data[start:stop]
        numerical_error = float(np.max(np.abs(reconstructed - dense)))
        np.testing.assert_allclose(reconstructed, dense)
        nonzeros = len(result.data)
        output_bytes = int(
            result.indptr.nbytes + result.indices.nbytes + result.data.nbytes
        )
        plan_bytes = 0
    else:
        assert target == "native_mvp"
        expected_apply = expected.apply(state)
        actual_apply = result.apply(state)
        numerical_error = float(np.max(np.abs(actual_apply - expected_apply)))
        np.testing.assert_allclose(actual_apply, expected_apply)
        nonzeros = operator.term_count * dense.shape[0]
        output_bytes = int(actual_apply.nbytes)
        plan_bytes = int(result.estimated_bytes)
    _record_metadata(
        benchmark,
        input_terms=operator.term_count,
        canonical_terms=operator.term_count,
        generated_contributions=operator.term_count * dense.shape[0],
        dimension=dense.shape[0],
        nonzeros_or_transitions=nonzeros,
        plan_bytes=plan_bytes,
        output_bytes=output_bytes,
        numerical_error=numerical_error,
        workload=f"uniform_weyl_d{dimension}_{target}",
    )


@pytest.mark.parametrize(
    ("mapping", "target"),
    tuple(
        product(
            ("jordan_wigner", "parity", "bravyi_kitaev"),
            ("coo", "csr", "native_mvp"),
        )
    ),
    ids=lambda value: str(value),
)
def test_structured_fermion_mapping_sparse_and_mvp(
    benchmark: BenchmarkFixture, mapping: str, target: str
) -> None:
    """Compare mapping-specific sparse/MVP compilation end to end."""
    operator, raw_terms = hubbard_quartic_workload()
    dense = operator.compile("dense", mapping=mapping)
    expected = operator.compile(target, mapping=mapping)
    state = (
        np.random.default_rng(20260804)
        .normal(size=dense.shape[0])
        .astype(np.complex128)
    )
    result = benchmark(operator.compile, target, mapping=mapping)
    if target == "coo":
        reconstructed = np.zeros_like(dense)
        np.add.at(reconstructed, (result.row, result.column), result.data)
        numerical_error = float(np.max(np.abs(reconstructed - dense)))
        np.testing.assert_allclose(reconstructed, dense)
        nonzeros = len(result.data)
        output_bytes = int(
            result.row.nbytes + result.column.nbytes + result.data.nbytes
        )
        plan_bytes = 0
    elif target == "csr":
        reconstructed = np.zeros_like(dense)
        for row in range(dense.shape[0]):
            start, stop = int(result.indptr[row]), int(result.indptr[row + 1])
            reconstructed[row, result.indices[start:stop]] += result.data[start:stop]
        numerical_error = float(np.max(np.abs(reconstructed - dense)))
        np.testing.assert_allclose(reconstructed, dense)
        nonzeros = len(result.data)
        output_bytes = int(
            result.indptr.nbytes + result.indices.nbytes + result.data.nbytes
        )
        plan_bytes = 0
    else:
        assert target == "native_mvp"
        expected_apply = expected.apply(state)
        actual_apply = result.apply(state)
        numerical_error = float(np.max(np.abs(actual_apply - expected_apply)))
        np.testing.assert_allclose(actual_apply, expected_apply)
        nonzeros = int(np.count_nonzero(dense))
        output_bytes = int(actual_apply.nbytes)
        plan_bytes = int(result.estimated_bytes)
    _record_metadata(
        benchmark,
        input_terms=len(raw_terms),
        canonical_terms=operator.term_count,
        generated_contributions=int(np.count_nonzero(dense)),
        dimension=dense.shape[0],
        nonzeros_or_transitions=nonzeros,
        plan_bytes=plan_bytes,
        output_bytes=output_bytes,
        numerical_error=numerical_error,
        workload=f"fermion_{mapping}_{target}",
    )


def test_structured_expansion_guard_smoke(benchmark: BenchmarkFixture) -> None:
    """Keep the recursive-expansion memory guard executable and bounded."""
    factors = (
        (0, "annihilate"),
        (1, "annihilate"),
        (2, "annihilate"),
        (0, "create"),
        (1, "create"),
        (2, "create"),
    )

    def guarded() -> None:
        with pytest.raises(MemoryError):
            tcp.FermionOperator.from_terms(3, [(factors, 1.0)], max_bytes=1400)

    benchmark(guarded)
    _record_metadata(
        benchmark,
        input_terms=1,
        canonical_terms=0,
        generated_contributions=6,
        dimension=1 << 3,
        nonzeros_or_transitions=0,
        workload="fermion_contraction_expansion_guard",
    )


def test_structured_native_embedding_permutation(benchmark: BenchmarkFixture) -> None:
    source = tcp.OperatorSpace(fermions=4, bosons=1, qubits=2, qudits=(3, 3))
    target = tcp.OperatorSpace(fermions=4, bosons=1, qubits=2, qudits=(3, 3))
    operator = (
        source.fermion.create(0)
        * source.fermion.annihilate(3)
        * source.boson.create(0)
        * source.qubit.x(0)
        * source.qudit.weyl(0, 1, 2)
    )
    maps = {
        "fermions": {0: 3, 1: 2, 2: 1, 3: 0},
        "bosons": {0: 0},
        "qubits": {0: 1, 1: 0},
        "qudits": {0: 1, 1: 0},
    }
    expected = target.embed(operator, **maps)
    result = benchmark(target.embed, operator, **maps)
    assert result.term_count == expected.term_count
    benchmark.extra_info.update(
        {
            "source_term_count": operator.term_count,
            "output_term_count": result.term_count,
            "source_layout_width": len(source.axes),
            "target_layout_width": len(target.axes),
            "fermion_permutation": (3, 2, 1, 0),
            "materialized": False,
        }
    )
