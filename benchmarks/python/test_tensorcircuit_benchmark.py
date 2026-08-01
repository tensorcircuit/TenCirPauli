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
