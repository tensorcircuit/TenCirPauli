"""Release-mode large-workload Hamiltonian and TensorCircuit comparisons."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tencirpauli import PauliOperator


Terms = Tuple[Tuple[int, ...], ...]
Weights = Tuple[float, ...]


def make_terms(nqubits: int, count: int, seed: int) -> Tuple[Terms, Weights]:
    """Build a deterministic full-width random Pauli workload."""
    rng = np.random.default_rng(seed)
    structures = tuple(
        tuple(int(code) for code in row)
        for row in rng.integers(0, 4, size=(count, nqubits))
    )
    weights = tuple(1.0 + index / 100.0 for index in range(count))
    return structures, weights


def make_operator(
    nqubits: int, count: int, seed: int
) -> Tuple[PauliOperator, Terms, Weights, np.ndarray[Any, Any]]:
    """Build a full-width operator and a normalized complex128 state."""
    structures, weights = make_terms(nqubits, count, seed)
    operator = PauliOperator.from_terms(nqubits, tuple(zip(structures, weights)))
    rng = np.random.default_rng(seed + 1)
    state = rng.normal(size=1 << nqubits) + 1j * rng.normal(size=1 << nqubits)
    state = (state / np.linalg.norm(state)).astype(np.complex128)
    return operator, structures, weights, state


def _sync(value: Any) -> None:
    """Synchronize a JAX value when the selected backend is asynchronous."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _sync_sparse(value: Any) -> None:
    """Synchronize both buffers of a JAX BCOO value."""
    _sync(value)
    for name in ("data", "indices"):
        component = getattr(value, name, None)
        if component is not None:
            _sync(component)


def _apply_jax_synced(compiled: Callable[[Any], Any], state: Any) -> Any:
    """Apply a compiled JAX function and synchronize inside the timed call."""
    result = compiled(state)
    _sync(result)
    return result


def _make_jax_mvp(
    structures: Terms, weights: Weights, state: np.ndarray[Any, Any]
) -> Tuple[Callable[[Any], Any], Any]:
    """Create a compiled TensorCircuit MVP and its backend state."""
    import tensorcircuit as tc

    mvp = tc.quantum.PauliStringSum2MVP(structures, weights)
    compiled = tc.backend.jit(mvp)
    return compiled, tc.backend.convert_to_tensor(state)


def _make_jax_sparse(structures: Terms, weights: Weights) -> Any:
    """Construct TensorCircuit's raw JAX BCOO sparse representation."""
    import tensorcircuit as tc

    return tc.quantum.PauliStringSum2COO(structures, weights)


def _make_jax_sparse_synced(structures: Terms, weights: Weights) -> Any:
    """Construct and synchronize a JAX BCOO value inside the timed call."""
    result = _make_jax_sparse(structures, weights)
    _sync_sparse(result)
    return result


def _sum_duplicates_synced(value: Any) -> Any:
    """Canonicalize and synchronize a JAX BCOO value inside the timed call."""
    result = value.sum_duplicates()
    _sync_sparse(result)
    return result


def _assert_jax_sparse_shape(
    value: Any, nqubits: int, count: int, *, canonical: bool
) -> None:
    """Validate large BCOO entry count and canonical metadata."""
    nse = count * (1 << nqubits)
    assert int(value.nse) == nse
    assert value.data.shape == (nse,)
    assert value.indices.shape == (nse, 2)
    assert bool(value.unique_indices) is canonical
    assert bool(value.indices_sorted) is canonical


@pytest.mark.parametrize("target", ("coo", "csr"))
def test_native_20q_sparse_memory_guard(
    benchmark: BenchmarkFixture, target: str
) -> None:
    """Measure explicit refusal of an oversized 20q/64-term matrix target."""
    operator, _, _, _ = make_operator(20, 64, 20260820)
    compile_target = getattr(operator, target)

    def reject_oversized_target() -> str:
        try:
            compile_target()
        except MemoryError as error:
            return str(error)
        raise AssertionError(f"20q/64-term {target} target unexpectedly succeeded")

    message = benchmark(reject_oversized_target)
    assert "exceeds memory limit" in message


def test_native_20q_coo_warm(benchmark: BenchmarkFixture) -> None:
    """Measure bounded 20q/3-term canonical COO construction."""
    operator, _, _, _ = make_operator(20, 3, 20260821)
    expected = operator.coo()
    result = benchmark.pedantic(operator.coo, rounds=5, iterations=1, warmup_rounds=1)
    assert result.data.size == expected.data.size == 3 * (1 << 20)
    np.testing.assert_array_equal(result.row, expected.row)
    np.testing.assert_allclose(result.data, expected.data, rtol=1e-12, atol=1e-12)


def test_native_20q_csr_warm(benchmark: BenchmarkFixture) -> None:
    """Measure bounded 20q/3-term canonical CSR construction."""
    operator, _, _, _ = make_operator(20, 3, 20260821)
    expected = operator.csr()
    result = benchmark.pedantic(operator.csr, rounds=5, iterations=1, warmup_rounds=1)
    assert result.data.size == expected.data.size == 3 * (1 << 20)
    np.testing.assert_array_equal(result.indptr, expected.indptr)
    np.testing.assert_array_equal(result.indices, expected.indices)
    np.testing.assert_allclose(result.data, expected.data, rtol=1e-12, atol=1e-12)


def test_native_20q_mvp_plan_construction(benchmark: BenchmarkFixture) -> None:
    """Measure 20q/64-term reusable native MVP plan construction."""
    operator, _, _, _ = make_operator(20, 64, 20260820)
    expected = operator.native_mvp_plan()
    result = benchmark(operator.native_mvp_plan)
    assert result.strategy == expected.strategy == "term_direct"
    assert result.term_count == expected.term_count == 64


def test_native_20q_mvp_warm(benchmark: BenchmarkFixture) -> None:
    """Measure 20q/64-term reusable native MVP application."""
    operator, _, _, state = make_operator(20, 64, 20260820)
    plan = operator.native_mvp_plan()
    expected = plan.apply(state)
    result = benchmark.pedantic(plan.apply, args=(state,), rounds=5, iterations=1)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_tensorcircuit_jax_20q_mvp_first(benchmark: BenchmarkFixture) -> None:
    """Measure first 20q/64-term JAX MVP call, including XLA compilation."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    operator, structures, weights, state = make_operator(20, 64, 20260820)
    expected = operator.mvp(state)
    compiled, backend_state = _make_jax_mvp(structures, weights, state)
    result = benchmark.pedantic(
        _apply_jax_synced,
        args=(compiled, backend_state),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    _sync(result)
    np.testing.assert_allclose(
        tc.backend.numpy(result), expected, rtol=1e-12, atol=1e-12
    )


def test_tensorcircuit_jax_20q_mvp_warm(benchmark: BenchmarkFixture) -> None:
    """Measure warm 20q/64-term JAX MVP after synchronized compilation."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    _, structures, weights, state = make_operator(20, 64, 20260820)
    compiled, backend_state = _make_jax_mvp(structures, weights, state)
    expected = compiled(backend_state)
    _sync(expected)
    result = benchmark.pedantic(
        _apply_jax_synced, args=(compiled, backend_state), rounds=5, iterations=1
    )
    _sync(result)
    np.testing.assert_allclose(
        tc.backend.numpy(result), tc.backend.numpy(expected), rtol=1e-12, atol=1e-12
    )


def test_tensorcircuit_jax_20q_sparse_first(benchmark: BenchmarkFixture) -> None:
    """Measure first 20q/3-term raw JAX BCOO construction."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    _, structures, weights, _ = make_operator(20, 3, 20260821)
    result = benchmark.pedantic(
        _make_jax_sparse_synced,
        args=(structures, weights),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    _sync_sparse(result)
    _assert_jax_sparse_shape(result, 20, 3, canonical=False)


def test_tensorcircuit_jax_20q_sparse_sum_duplicates_first(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure first 20q/3-term JAX BCOO canonicalization."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    _, structures, weights, _ = make_operator(20, 3, 20260821)
    raw = _make_jax_sparse(structures, weights)
    _sync_sparse(raw)
    result = benchmark.pedantic(
        _sum_duplicates_synced,
        args=(raw,),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    _sync_sparse(result)
    _assert_jax_sparse_shape(result, 20, 3, canonical=True)


def test_tensorcircuit_jax_20q_sparse_warm(benchmark: BenchmarkFixture) -> None:
    """Measure warm 20q/3-term raw JAX BCOO construction."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    _, structures, weights, _ = make_operator(20, 3, 20260821)
    expected = _make_jax_sparse(structures, weights)
    _sync_sparse(expected)
    result = benchmark.pedantic(
        _make_jax_sparse_synced,
        args=(structures, weights),
        rounds=5,
        iterations=1,
    )
    _sync_sparse(result)
    _assert_jax_sparse_shape(result, 20, 3, canonical=False)
    np.testing.assert_allclose(
        np.asarray(result.data), np.asarray(expected.data), rtol=1e-12, atol=1e-12
    )


def test_tensorcircuit_jax_20q_sparse_sum_duplicates_warm(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure warm 20q/3-term JAX BCOO canonicalization."""
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    _, structures, weights, _ = make_operator(20, 3, 20260821)
    raw = _make_jax_sparse(structures, weights)
    _sync_sparse(raw)
    expected = raw.sum_duplicates()
    _sync_sparse(expected)
    result = benchmark.pedantic(
        _sum_duplicates_synced, args=(raw,), rounds=5, iterations=1
    )
    _sync_sparse(result)
    _assert_jax_sparse_shape(result, 20, 3, canonical=True)
    np.testing.assert_allclose(
        np.asarray(result.data), np.asarray(expected.data), rtol=1e-12, atol=1e-12
    )
