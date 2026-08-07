"""symmetry symmetry setup and restricted-sector benchmarks."""

from __future__ import annotations

import os

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
    """Build the fixed wide-sector nearest-neighbor hopping workload."""
    terms = []
    for index in range(nqubits - 1):
        prefix = "I" * index
        suffix = "I" * (nqubits - 2 - index)
        terms.extend(((prefix + "XX" + suffix, 0.5), (prefix + "YY" + suffix, 0.5)))
    return PauliOperator.from_terms(nqubits, terms)


def make_long_range_duplicate_x(nqubits: int = 129) -> PauliOperator:
    """Build a cross-limb diagonal/long-range workload with repeated X masks."""
    structures: list[tuple[int, ...]] = []
    coefficients: list[float] = []
    for index in range(nqubits):
        z = [0] * nqubits
        z[index] = 3
        structures.append(tuple(z))
        coefficients.append(0.01)
    for left in range(nqubits):
        for right in range(left + 1, nqubits):
            zz = [0] * nqubits
            zz[left] = 3
            zz[right] = 3
            structures.append(tuple(zz))
            coefficients.append(0.001)
    for left, right in ((0, nqubits // 2), (1, nqubits - 2), (2, nqubits - 1)):
        xx = [0] * nqubits
        yy = [0] * nqubits
        xx[left] = xx[right] = 1
        yy[left] = yy[right] = 2
        structures.extend((tuple(xx), tuple(yy)))
        coefficients.extend((0.5, 0.5))
    return PauliOperator.from_code_arrays(structures, coefficients)


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
def test_wide_sector_128q_k2_sparse_materialization(
    benchmark: BenchmarkFixture, target: str
) -> None:
    """Measure 128-qubit k=2 sparse output without allocating a dense target."""
    operator = make_wide_hopping(128)

    def materialize():
        restricted = operator.restrict_charge(
            U1Sector(128, 2), storage="lazy", max_bytes=None
        )
        return getattr(restricted, target)(max_bytes=None)

    expected = materialize()
    result = benchmark.pedantic(materialize, rounds=7, iterations=1, warmup_rounds=1)
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
def test_wide_sector_wide_u1_setup_and_mvp(
    benchmark: BenchmarkFixture, nqubits: int, particle_number: int
) -> None:
    operator = make_wide_hopping(nqubits)
    sector = U1Sector(nqubits, particle_number)
    restricted = operator.restrict_charge(sector, storage="lazy")
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.float64) + 1j * np.arange(plan.dimension)
    expected = plan.apply(state)
    csr = restricted.csr()
    group_count = len({tuple(term.word.x_words) for term in operator.terms})
    plan_bytes = csr.indptr.nbytes + csr.indices.nbytes + csr.data.nbytes
    benchmark.extra_info.update(
        {
            "nqubits": nqubits,
            "particle_number": particle_number,
            "word_count": (nqubits + 63) // 64,
            "dimension": plan.dimension,
            "term_count": len(operator.terms),
            "distinct_x_groups": group_count,
            "nnz": int(csr.data.size),
            "thread_count": 1,
            "plan_bytes": plan_bytes,
            "output_bytes": expected.nbytes,
        }
    )
    result = benchmark.pedantic(
        operator.restrict_charge,
        args=(sector,),
        kwargs={"storage": "lazy"},
        rounds=3,
        iterations=1,
    )
    np.testing.assert_allclose(plan.apply(state), expected)
    assert result.dimension == plan.dimension
    result_csr = result.csr()
    np.testing.assert_array_equal(result_csr.indptr, csr.indptr)
    np.testing.assert_array_equal(result_csr.indices, csr.indices)
    np.testing.assert_allclose(result_csr.data, csr.data)
    benchmark.extra_info["numerical_error"] = 0.0


@pytest.mark.performance_large
def test_wide_sector_long_range_duplicate_x_setup(benchmark: BenchmarkFixture) -> None:
    """Measure cross-limb Z-group aggregation and long-range hopping setup."""
    operator = make_long_range_duplicate_x()
    sector = U1Sector(129, 2)
    expected = operator.restrict_charge(sector, storage="lazy")
    expected_csr = expected.csr()
    result = benchmark.pedantic(
        operator.restrict_charge,
        args=(sector,),
        kwargs={"storage": "lazy"},
        rounds=3,
        iterations=1,
    )
    result_csr = result.csr()
    np.testing.assert_array_equal(result_csr.indptr, expected_csr.indptr)
    np.testing.assert_array_equal(result_csr.indices, expected_csr.indices)
    np.testing.assert_allclose(result_csr.data, expected_csr.data)
    distinct_x_groups = len({tuple(term.word.x_words) for term in operator.terms})
    plan_bytes = (
        result_csr.indptr.nbytes + result_csr.indices.nbytes + result_csr.data.nbytes
    )
    output_bytes = plan_bytes
    benchmark.extra_info.update(
        {
            "nqubits": 129,
            "particle_number": 2,
            "word_count": 3,
            "dimension": result.dimension,
            "term_count": len(operator.terms),
            "distinct_x_groups": distinct_x_groups,
            "nnz": int(result_csr.data.size),
            "thread_count": 1,
            "plan_bytes": plan_bytes,
            "output_bytes": output_bytes,
            "numerical_error": 0.0,
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
def test_wide_sector_wide_u1_steady_mvp(
    benchmark: BenchmarkFixture, nqubits: int, particle_number: int
) -> None:
    operator = make_wide_hopping(nqubits)
    restricted = operator.restrict_charge(
        U1Sector(nqubits, particle_number), storage="lazy"
    )
    plan = restricted.mvp_plan()
    state = np.arange(plan.dimension, dtype=np.float64) + 1j * np.arange(plan.dimension)
    expected = plan.apply(state)
    csr = restricted.csr()
    group_count = len({tuple(term.word.x_words) for term in operator.terms})
    plan_bytes = csr.indptr.nbytes + csr.indices.nbytes + csr.data.nbytes
    result = benchmark.pedantic(plan.apply, args=(state,), rounds=5, iterations=1)
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update(
        {
            "nqubits": nqubits,
            "particle_number": particle_number,
            "word_count": (nqubits + 63) // 64,
            "dimension": plan.dimension,
            "distinct_x_groups": group_count,
            "nnz": int(csr.data.size),
            "thread_count": int(
                os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1)
            ),
            "plan_bytes": plan_bytes,
            "output_bytes": result.nbytes,
            "numerical_error": float(np.max(np.abs(result - expected), initial=0.0)),
        }
    )


@pytest.mark.performance_large
@pytest.mark.parametrize(("nqubits", "particle_number"), [(16, 8), (18, 9), (20, 10)])
def test_u1_medium_csr(
    benchmark: BenchmarkFixture, nqubits: int, particle_number: int
) -> None:
    """Measure full eager CSR construction at representative medium sizes."""
    operator = make_wide_hopping(nqubits)
    sector = U1Sector(nqubits, particle_number)

    def materialize():
        restricted = operator.restrict_charge(sector, storage="eager", max_bytes=None)
        return restricted.csr(max_bytes=None)

    expected = materialize()
    result = benchmark.pedantic(materialize, rounds=3, iterations=1)
    np.testing.assert_array_equal(result.indptr, expected.indptr)
    np.testing.assert_array_equal(result.indices, expected.indices)
    np.testing.assert_allclose(result.data, expected.data)
    benchmark.extra_info.update(
        {
            "nqubits": nqubits,
            "particle_number": particle_number,
            "dimension": result.shape[0],
            "nnz": int(result.data.size),
            "output_bytes": result.indptr.nbytes
            + result.indices.nbytes
            + result.data.nbytes,
            "numerical_error": 0.0,
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
