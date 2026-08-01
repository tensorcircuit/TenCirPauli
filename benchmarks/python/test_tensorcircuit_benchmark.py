"""Optional same-workload TensorCircuit/JAX comparison benchmarks."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tencirpauli import PauliOperator


def make_workload(
    nqubits: int, count: int
) -> Tuple[PauliOperator, Tuple[Tuple[int, ...], ...], np.ndarray]:
    """Build unique, deterministic terms for cross-implementation timing."""
    structures = tuple(
        tuple((index // (4**qubit)) % 4 for qubit in range(nqubits))
        for index in range(count)
    )
    weights = tuple(1.0 + index / 100.0 for index in range(count))
    operator = PauliOperator.from_terms(nqubits, tuple(zip(structures, weights)))
    state = np.random.default_rng(20260801 + nqubits).normal(
        size=1 << nqubits
    ) + 1j * np.random.default_rng(7 + nqubits).normal(size=1 << nqubits)
    state = (state / np.linalg.norm(state)).astype(np.complex128)
    return operator, structures, state


def _sync(value: Any) -> None:
    """Synchronize a backend result when the selected backend is asynchronous."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _sync_sparse(value: Any) -> None:
    """Synchronize a JAX sparse value and its asynchronously computed buffers."""
    _sync(value)
    for name in ("data", "indices"):
        component = getattr(value, name, None)
        if component is not None:
            _sync(component)


def _make_tensorcircuit_jax_sparse(
    structures: Tuple[Tuple[int, ...], ...], weights: Tuple[float, ...]
) -> Any:
    """Construct TensorCircuit's JAX BCOO sparse matrix for one workload."""
    import tensorcircuit as tc

    return tc.quantum.PauliStringSum2COO(structures, weights)


def _make_tensorcircuit_jax_sparse_synced(
    structures: Tuple[Tuple[int, ...], ...], weights: Tuple[float, ...]
) -> Any:
    """Construct and synchronize a JAX BCOO value inside the timed call."""
    result = _make_tensorcircuit_jax_sparse(structures, weights)
    _sync_sparse(result)
    return result


def _sum_duplicates_synced(value: Any) -> Any:
    """Canonicalize and synchronize a JAX BCOO value inside the timed call."""
    result = value.sum_duplicates()
    _sync_sparse(result)
    return result


def _assert_jax_sparse_metadata(
    raw: Any, canonical: Any, raw_nse: int, canonical_nnz: int
) -> None:
    """Check raw/canonical BCOO sizes and the duplicate-index contract."""
    assert int(raw.nse) == raw_nse
    assert raw.data.shape == (raw_nse,)
    assert raw.indices.shape == (raw_nse, 2)
    assert not bool(raw.unique_indices)
    assert not bool(raw.indices_sorted)

    canonical_nse = int(canonical.nse)
    assert canonical_nse >= canonical_nnz
    assert canonical.data.shape == (canonical_nse,)
    assert canonical.indices.shape == (canonical_nse, 2)
    exact_nnz = int(np.count_nonzero(np.asarray(canonical.data)))
    significant_nnz = int(np.count_nonzero(np.abs(np.asarray(canonical.data)) > 1e-12))
    assert canonical_nse < raw_nse
    assert 0 < significant_nnz <= exact_nnz <= canonical_nse
    assert bool(canonical.unique_indices)
    assert bool(canonical.indices_sorted)


def _assert_jax_raw_metadata(raw: Any, raw_nse: int) -> None:
    """Check raw BCOO storage before duplicate canonicalization."""
    assert int(raw.nse) == raw_nse
    assert raw.data.shape == (raw_nse,)
    assert raw.indices.shape == (raw_nse, 2)
    assert not bool(raw.unique_indices)
    assert not bool(raw.indices_sorted)


def _sparse_storage_bytes(value: Any) -> int:
    """Return the host-visible values plus indices storage of a sparse object."""
    return int(np.asarray(value.data).nbytes + np.asarray(value.indices).nbytes)


@pytest.mark.parametrize(("nqubits", "count"), ((10, 64), (16, 256)))
def test_native_reusable_mvp_warm(
    benchmark: BenchmarkFixture, nqubits: int, count: int
) -> None:
    """Measure reusable native apply without plan construction in the timed loop."""
    operator, _, state = make_workload(nqubits, count)
    plan = operator.native_mvp_plan()
    expected = plan.apply(state)
    result = benchmark(plan.apply, state)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("nqubits", "count"), ((10, 64), (16, 256)))
def test_tensorcircuit_jax_mvp_warm(
    benchmark: BenchmarkFixture, nqubits: int, count: int
) -> None:
    """Measure TensorCircuit's compiled MVP after one synchronized compile call."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    _, structures, state = make_workload(nqubits, count)
    weights = tuple(1.0 + index / 100.0 for index in range(count))
    mvp = tc.quantum.PauliStringSum2MVP(structures, weights)
    compiled = tc.backend.jit(mvp)
    backend_state = tc.backend.convert_to_tensor(state)
    expected = compiled(backend_state)
    _sync(expected)
    result = benchmark(compiled, backend_state)
    _sync(result)
    np.testing.assert_allclose(
        tc.backend.numpy(result), tc.backend.numpy(expected), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(("nqubits", "count"), ((8, 32), (10, 64), (12, 64)))
def test_tensorcircuit_jax_sparse_construction_first(
    benchmark: BenchmarkFixture, nqubits: int, count: int
) -> None:
    """Measure first JAX BCOO construction, including the shape-specialized compile."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    _, structures, _ = make_workload(nqubits, count)
    weights = tuple(1.0 + index / 100.0 for index in range(count))
    result = benchmark.pedantic(
        _make_tensorcircuit_jax_sparse_synced,
        args=(structures, weights),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    _sync_sparse(result)
    _assert_jax_raw_metadata(result, count * (1 << nqubits))
    assert _sparse_storage_bytes(result) > 0


@pytest.mark.parametrize(("nqubits", "count"), ((8, 32), (10, 64), (12, 64)))
def test_tensorcircuit_jax_sparse_sum_duplicates_first(
    benchmark: BenchmarkFixture, nqubits: int, count: int
) -> None:
    """Measure first JAX duplicate canonicalization after raw BCOO construction."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    operator, structures, _ = make_workload(nqubits, count)
    weights = tuple(1.0 + index / 100.0 for index in range(count))
    raw = _make_tensorcircuit_jax_sparse(structures, weights)
    _sync_sparse(raw)
    canonical_nnz = int(operator.coo().data.size)
    result = benchmark.pedantic(
        _sum_duplicates_synced,
        args=(raw,),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    _sync_sparse(result)
    _assert_jax_sparse_metadata(raw, result, count * (1 << nqubits), canonical_nnz)


@pytest.mark.parametrize(("nqubits", "count"), ((8, 32), (10, 64), (12, 64)))
def test_tensorcircuit_jax_sparse_construction_warm(
    benchmark: BenchmarkFixture, nqubits: int, count: int
) -> None:
    """Measure warm raw JAX BCOO construction without duplicate canonicalization."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    operator, structures, _ = make_workload(nqubits, count)
    weights = tuple(1.0 + index / 100.0 for index in range(count))
    expected = _make_tensorcircuit_jax_sparse(structures, weights)
    _sync_sparse(expected)
    canonical_nnz = int(operator.coo().data.size)
    result = benchmark(_make_tensorcircuit_jax_sparse_synced, structures, weights)
    _sync_sparse(result)
    canonical = result.sum_duplicates()
    _sync_sparse(canonical)
    _assert_jax_sparse_metadata(
        result, canonical, count * (1 << nqubits), canonical_nnz
    )
    assert _sparse_storage_bytes(result) >= _sparse_storage_bytes(canonical)


@pytest.mark.parametrize(("nqubits", "count"), ((8, 32), (10, 64), (12, 64)))
def test_tensorcircuit_jax_sparse_sum_duplicates_warm(
    benchmark: BenchmarkFixture, nqubits: int, count: int
) -> None:
    """Measure warm JAX canonicalization separately from raw BCOO construction."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    operator, structures, _ = make_workload(nqubits, count)
    weights = tuple(1.0 + index / 100.0 for index in range(count))
    raw = _make_tensorcircuit_jax_sparse(structures, weights)
    _sync_sparse(raw)
    expected = raw.sum_duplicates()
    _sync_sparse(expected)
    canonical_nnz = int(operator.coo().data.size)
    result = benchmark(_sum_duplicates_synced, raw)
    _sync_sparse(result)
    _assert_jax_sparse_metadata(raw, result, count * (1 << nqubits), canonical_nnz)
    np.testing.assert_allclose(
        np.asarray(result.data), np.asarray(expected.data), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(("nqubits", "count"), ((8, 32), (10, 64), (12, 64)))
def test_native_sparse_construction_warm(
    benchmark: BenchmarkFixture, nqubits: int, count: int
) -> None:
    """Measure duplicate-aggregated COO construction through the public API."""
    operator, _, _ = make_workload(nqubits, count)
    expected = operator.coo()
    result = benchmark(operator.coo)
    assert result.data.shape == expected.data.shape
    np.testing.assert_array_equal(result.row, expected.row)
    np.testing.assert_allclose(result.data, expected.data, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("nqubits", "count"), ((8, 32), (10, 64), (12, 64)))
def test_tensorcircuit_numpy_sparse_construction_warm(
    benchmark: BenchmarkFixture, nqubits: int, count: int
) -> None:
    """Measure TensorCircuit's SciPy COO construction on the same workload."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("scipy")
    tc.set_backend("numpy")
    tc.set_dtype("complex128")
    _, structures, _ = make_workload(nqubits, count)
    weights = tuple(1.0 + index / 100.0 for index in range(count))
    expected = tc.quantum.PauliStringSum2COO(structures, weights, numpy=True)
    result = benchmark(tc.quantum.PauliStringSum2COO, structures, weights, numpy=True)
    assert result.shape == expected.shape
    np.testing.assert_allclose(
        result.toarray(), expected.toarray(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(("nqubits", "count"), ((8, 32), (10, 64)))
def test_tensorcircuit_jax_sparse_matvec_warm(
    benchmark: BenchmarkFixture, nqubits: int, count: int
) -> None:
    """Measure JAX BCOO warm matvec separately from sparse-plan construction."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    _, structures, state = make_workload(nqubits, count)
    weights = tuple(1.0 + index / 100.0 for index in range(count))
    sparse = tc.quantum.PauliStringSum2COO(structures, weights)
    backend_state = tc.backend.convert_to_tensor(state)
    expected = sparse @ backend_state
    _sync(expected)

    def apply() -> Any:
        result = sparse @ backend_state
        _sync(result)
        return result

    result = benchmark(apply)
    np.testing.assert_allclose(
        tc.backend.numpy(result), tc.backend.numpy(expected), rtol=1e-12, atol=1e-12
    )
