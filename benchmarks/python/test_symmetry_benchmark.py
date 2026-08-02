"""Phase 2 symmetry setup and restricted-sector benchmarks."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tencirpauli import PauliOperator, U1Sector


def make_tfim() -> PauliOperator:
    terms = [("X" * 8, 0.25)]
    terms.extend(("I" * index + "ZZ" + "I" * (6 - index), -1.0) for index in range(7))
    terms.extend(("I" * index + "X" + "I" * (7 - index), -0.2) for index in range(8))
    return PauliOperator.from_terms(8, terms)


def make_hopping(nqubits: int = 12) -> PauliOperator:
    terms = []
    for index in range(nqubits - 1):
        prefix = "I" * index
        suffix = "I" * (nqubits - 2 - index)
        terms.extend(((prefix + "XX" + suffix, 0.5), (prefix + "YY" + suffix, 0.5)))
    return PauliOperator.from_terms(nqubits, terms)


def make_wide_hopping(nqubits: int) -> PauliOperator:
    """Build the fixed Phase 5 nearest-neighbor hopping workload."""
    terms = []
    for index in range(nqubits - 1):
        prefix = "I" * index
        suffix = "I" * (nqubits - 2 - index)
        terms.extend(((prefix + "XX" + suffix, 0.5), (prefix + "YY" + suffix, 0.5)))
    return PauliOperator.from_terms(nqubits, terms)


def make_large_diagonal_operator() -> PauliOperator:
    """Make a 26-qubit diagonal workload without materializing its matrix."""
    identity = (0,) * 26
    first_z = (3,) + (0,) * 25
    last_z = (0,) * 25 + (3,)
    return PauliOperator.from_code_arrays(
        (identity, first_z, last_z), (0.25, 0.5, -0.2)
    )


def test_z2_analysis_setup(benchmark: BenchmarkFixture) -> None:
    operator = make_tfim()
    expected = operator.find_z2_symmetries()
    result = benchmark(operator.find_z2_symmetries)
    assert result.generators == expected.generators


def test_z2_tapering_setup_and_transform(benchmark: BenchmarkFixture) -> None:
    operator = make_tfim()
    analysis = operator.find_z2_symmetries()
    plan = analysis.tapering_plan((1,) * analysis.rank)
    expected = plan.transform_operator(operator)
    result = benchmark(plan.transform_operator, operator)
    assert result == expected


def test_u1_restriction_setup(benchmark: BenchmarkFixture) -> None:
    operator = make_hopping()
    sector = U1Sector(12, 2)
    expected = operator.restrict_u1(sector)
    result = benchmark(operator.restrict_u1, sector)
    assert result.dimension == expected.dimension


def test_u1_restricted_mvp_apply(benchmark: BenchmarkFixture) -> None:
    operator = make_hopping()
    restricted = operator.restrict_u1(U1Sector(12, 2))
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.float64) + 1j * np.arange(plan.dimension)
    expected = plan.apply(state)
    result = benchmark(plan.apply, state)
    np.testing.assert_allclose(result, expected)


def test_u1_restricted_csr(benchmark: BenchmarkFixture) -> None:
    operator = make_hopping()
    restricted = operator.restrict_u1(U1Sector(12, 2))
    expected = restricted.csr()
    result = benchmark(restricted.csr)
    np.testing.assert_array_equal(result.indptr, expected.indptr)
    np.testing.assert_array_equal(result.indices, expected.indices)
    np.testing.assert_allclose(result.data, expected.data)


def test_u1_restricted_dense(benchmark: BenchmarkFixture) -> None:
    operator = make_hopping()
    restricted = operator.restrict_u1(U1Sector(12, 2))
    expected = restricted.dense()
    result = benchmark(restricted.dense)
    np.testing.assert_allclose(result, expected)


def test_u1_restricted_coo(benchmark: BenchmarkFixture) -> None:
    operator = make_hopping()
    restricted = operator.restrict_u1(U1Sector(12, 2))
    expected = restricted.coo()
    result = benchmark(restricted.coo)
    np.testing.assert_array_equal(result.row, expected.row)
    np.testing.assert_array_equal(result.column, expected.column)
    np.testing.assert_allclose(result.data, expected.data)


@pytest.mark.performance_large
@pytest.mark.parametrize("target", ["coo", "csr"])
def test_phase5_128q_k2_sparse_materialization(
    benchmark: BenchmarkFixture, target: str
) -> None:
    """Measure 128-qubit k=2 sparse output without allocating a dense target."""
    operator = make_wide_hopping(128)
    restricted = operator.restrict_u1(U1Sector(128, 2))
    expected = getattr(restricted, target)()
    result = benchmark.pedantic(
        getattr(restricted, target), rounds=7, iterations=1, warmup_rounds=1
    )
    if target == "coo":
        np.testing.assert_array_equal(result.row, expected.row)
        np.testing.assert_array_equal(result.column, expected.column)
        output_bytes = result.row.nbytes + result.column.nbytes + result.data.nbytes
    else:
        np.testing.assert_array_equal(result.indptr, expected.indptr)
        np.testing.assert_array_equal(result.indices, expected.indices)
        output_bytes = result.indptr.nbytes + result.indices.nbytes + result.data.nbytes
    np.testing.assert_allclose(result.data, expected.data)
    plan_bytes = (
        (result.shape[0] + 1) * np.dtype(np.intp).itemsize
        + result.data.size * np.dtype(np.intp).itemsize
        + result.data.size * np.dtype(np.complex128).itemsize
    )
    benchmark.extra_info.update(
        {
            "target": target,
            "dimension": result.shape[0],
            "nnz": int(result.data.size),
            "output_bytes": output_bytes,
            "steady_plan_bytes": plan_bytes,
            "steady_plan_plus_output_bytes": plan_bytes + output_bytes,
        }
    )


@pytest.mark.performance_large
@pytest.mark.parametrize(
    ("nqubits", "particle_number"),
    [
        (63, 2),
        (64, 2),
        (65, 2),
        (128, 2),
        (129, 2),
        (128, 126),
        (256, 1),
        (256, 2),
    ],
)
def test_phase5_wide_u1_setup_and_mvp(
    benchmark: BenchmarkFixture, nqubits: int, particle_number: int
) -> None:
    operator = make_wide_hopping(nqubits)
    sector = U1Sector(nqubits, particle_number)
    restricted = operator.restrict_u1(sector)
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.float64) + 1j * np.arange(plan.dimension)
    expected = plan.apply(state)
    benchmark.extra_info.update(
        {
            "nqubits": nqubits,
            "particle_number": particle_number,
            "word_count": (nqubits + 63) // 64,
            "dimension": plan.dimension,
            "term_count": len(operator.terms),
            "nnz": int(restricted.csr().data.size),
        }
    )
    result = benchmark.pedantic(
        operator.restrict_u1, args=(sector,), rounds=3, iterations=1
    )
    np.testing.assert_allclose(plan.apply(state), expected)
    assert result.dimension == plan.dimension


@pytest.mark.performance_large
@pytest.mark.parametrize(
    ("nqubits", "particle_number"),
    [(63, 2), (64, 2), (65, 2), (128, 2), (129, 2), (128, 126), (256, 1), (256, 2)],
)
def test_phase5_wide_u1_steady_mvp(
    benchmark: BenchmarkFixture, nqubits: int, particle_number: int
) -> None:
    operator = make_wide_hopping(nqubits)
    restricted = operator.restrict_u1(U1Sector(nqubits, particle_number))
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.float64) + 1j * np.arange(plan.dimension)
    expected = plan.apply(state)
    result = benchmark.pedantic(plan.apply, args=(state,), rounds=5, iterations=1)
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update(
        {
            "nqubits": nqubits,
            "particle_number": particle_number,
            "word_count": (nqubits + 63) // 64,
            "dimension": plan.dimension,
            "nnz": int(restricted.csr().data.size),
        }
    )


@pytest.mark.performance_large
def test_u1_central_sector_mvp(benchmark: BenchmarkFixture) -> None:
    """Measure a central fixed-weight sector rather than only low-k scaling."""
    operator = make_hopping(16)
    restricted = operator.restrict_u1(U1Sector(16, 8))
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.float64) + 1j * np.arange(plan.dimension)
    expected = plan.apply(state)
    benchmark.extra_info["dimension"] = plan.dimension
    result = benchmark.pedantic(plan.apply, args=(state,), rounds=5, iterations=1)
    np.testing.assert_allclose(result, expected)


@pytest.mark.performance_large
def test_u1_central_sector_csr_storage(benchmark: BenchmarkFixture) -> None:
    """Record canonical CSR output size for a representative central sector."""
    operator = make_hopping(16)
    restricted = operator.restrict_u1(U1Sector(16, 8))
    expected = restricted.csr()
    result = benchmark(restricted.csr)
    output_bytes = (
        expected.indptr.nbytes + expected.indices.nbytes + expected.data.nbytes
    )
    benchmark.extra_info["dimension"] = expected.shape[0]
    benchmark.extra_info["nnz"] = int(expected.data.size)
    benchmark.extra_info["output_bytes"] = output_bytes
    np.testing.assert_array_equal(result.indptr, expected.indptr)
    np.testing.assert_array_equal(result.indices, expected.indices)
    np.testing.assert_allclose(result.data, expected.data)


@pytest.mark.performance_large
def test_u1_restricted_mvp_apply_26q(benchmark: BenchmarkFixture) -> None:
    """Measure a 26-qubit fixed-k MVP without allocating the 2**26 state space."""
    operator = make_hopping(26)
    restricted = operator.restrict_u1(U1Sector(26, 2))
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.float64) + 1j * np.arange(plan.dimension)
    expected = plan.apply(state)
    result = benchmark.pedantic(plan.apply, args=(state,), rounds=5, iterations=1)
    np.testing.assert_allclose(result, expected)


@pytest.mark.performance_large
def test_native_26q_fullspace_mvp_warm(benchmark: BenchmarkFixture) -> None:
    """Measure reusable native MVP over a real 26-qubit statevector."""
    max_bytes = 4 * 1024**3
    operator = make_large_diagonal_operator()
    plan = operator.native_mvp_plan(max_bytes=max_bytes)
    state = np.ones(1 << 26, dtype=np.complex128)
    expected = plan.apply(state, max_bytes=max_bytes)
    result = benchmark.pedantic(
        plan.apply,
        args=(state,),
        kwargs={"max_bytes": max_bytes},
        rounds=3,
        iterations=1,
        warmup_rounds=1,
    )
    np.testing.assert_allclose(result, expected)
