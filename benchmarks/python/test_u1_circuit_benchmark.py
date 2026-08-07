"""Release-mode U1Circuit comparisons against TensorCircuit with JAX JIT."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


@dataclass(frozen=True)
class U1Workload:
    """A compressed-state workload shared by all head-to-head benchmarks."""

    name: str
    nqubits: int
    particles: int
    layers: int


WORKLOADS = (
    U1Workload("20q-k2", nqubits=20, particles=2, layers=8),
    U1Workload("24q-k3", nqubits=24, particles=3, layers=6),
    U1Workload("32q-k3", nqubits=32, particles=3, layers=8),
    U1Workload("32q-k4", nqubits=32, particles=4, layers=12),
    U1Workload("40q-k3", nqubits=40, particles=3, layers=12),
    U1Workload("40q-k5", nqubits=40, particles=5, layers=4),
)
JAX_WORKLOADS = tuple(workload for workload in WORKLOADS if workload.name != "40q-k5")
PARAMETERIZED_WORKLOAD = U1Workload(
    "16q-k4-parameterized", nqubits=16, particles=4, layers=3
)


def make_native(workload: U1Workload = WORKLOADS[0]) -> tcp.U1Circuit:
    circuit = tcp.U1Circuit(
        workload.nqubits,
        particle_number=workload.particles,
        occupied=list(range(workload.particles)),
    )
    for layer in range(workload.layers):
        for wire in range(0, workload.nqubits - 1, 2):
            circuit.iswap(wire, wire + 1, theta=0.17 + 0.01 * layer)
            circuit.cphase(wire, wire + 1, theta=-0.11)
    return circuit


def native_parameters(workload: U1Workload) -> np.ndarray[Any, Any]:
    return np.asarray(
        [0.17 + 0.01 * layer for layer in range(workload.layers)],
        dtype=np.float64,
    )


def _initial_state(workload: U1Workload) -> np.ndarray[Any, Any]:
    circuit = tcp.U1Circuit(
        workload.nqubits,
        particle_number=workload.particles,
        occupied=list(range(workload.particles)),
    )
    return circuit._initial_state.copy()


def _jax_runner(
    workload: U1Workload,
) -> tuple[Any, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build a functional TensorCircuit U1 program and JIT compile its wrapper."""
    if (
        workload.name == "40q-k5"
        and os.environ.get("TENCIRPAULI_ALLOW_HEAVY_JAX") != "1"
    ):
        pytest.skip(
            "40q-k5 JAX compilation is a multi-gigabyte workload; "
            "set TENCIRPAULI_ALLOW_HEAVY_JAX=1 to opt in"
        )
    tc = pytest.importorskip("tensorcircuit")
    pytest.importorskip("jax")
    tc.set_dtype("complex128")
    tc.set_backend("jax")
    initial = _initial_state(workload)
    angles = native_parameters(workload)

    def run(state: Any, parameters: Any) -> Any:
        circuit = tc.U1Circuit(
            workload.nqubits,
            k=workload.particles,
            filled=list(range(workload.particles)),
            inputs=state,
        )
        for layer in range(workload.layers):
            for wire in range(0, workload.nqubits - 1, 2):
                circuit.iswap(wire, wire + 1, theta=parameters[layer])
                # Keep this algebraically constant parameter dependency: it is
                # part of the historical workload and makes the JAX graph
                # comparable across records. With a literal -0.11, JAX/XLA
                # may constant-fold more aggressively and fully expand/fuse
                # the circuit, causing much longer compilation (roughly
                # 30-40 s for 40q-k3 versus roughly 2 s here). This is a
                # compiler-behavior diagnostic, not a physics distinction;
                # do not simplify it without recording a separate workload.
                circuit.cphase(
                    wire,
                    wire + 1,
                    theta=-0.11 + 0.0 * parameters[layer],
                )
        return circuit.state()

    import jax

    return jax.jit(run), initial, angles


def _metadata(
    circuit: tcp.U1Circuit,
    backend: str,
    workload: U1Workload,
) -> dict[str, int | str]:
    return {
        "nqubits": workload.nqubits,
        "particle_number": workload.particles,
        "dimension": circuit.dimension,
        "gate_count": len(circuit.to_qir()),
        "state_bytes": circuit.dimension * np.dtype(np.complex128).itemsize,
        "process_maxrss_bytes": _maxrss_bytes(),
        "thread_count": int(os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1)),
        "layers": workload.layers,
        "backend": backend,
    }


def _jax_metadata(workload: U1Workload, backend: str) -> dict[str, int | str]:
    return {
        "nqubits": workload.nqubits,
        "particle_number": workload.particles,
        "dimension": math.comb(workload.nqubits, workload.particles),
        "gate_count": workload.layers * (workload.nqubits // 2) * 2,
        "state_bytes": math.comb(workload.nqubits, workload.particles) * 16,
        "process_maxrss_bytes": _maxrss_bytes(),
        "layers": workload.layers,
        "backend": backend,
    }


def _maxrss_bytes() -> int:
    """Return process peak RSS in bytes on both macOS and Linux."""
    try:
        import resource
    except ImportError:
        return 0
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _block_until_ready(value: Any) -> Any:
    return value.block_until_ready() if hasattr(value, "block_until_ready") else value


def _sync_tree(jax: Any, value: Any) -> Any:
    """Synchronize every leaf of a JAX scalar or value-and-gradient result."""
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _parameterized_observable(workload: U1Workload) -> tcp.PauliOperator:
    return tcp.PauliOperator.from_terms(
        workload.nqubits, (("Z" + "I" * (workload.nqubits - 1), 1.0),)
    )


def _parameterized_native_reference(
    workload: U1Workload,
) -> tuple[float, np.ndarray[Any, Any]]:
    """Return the native value and layer-shared gradient oracle."""
    circuit = make_native(workload)
    result = circuit.value_and_grad(_parameterized_observable(workload))
    pairs = workload.nqubits // 2
    gradient = np.asarray(result.gradient).reshape(workload.layers, pairs, 2)
    return result.value, gradient[:, :, 0].sum(axis=1)


def _parameterized_jax_runner(
    workload: U1Workload,
) -> tuple[Any, Any, Any]:
    """Build the JAX expectation and value-and-gradient objectives."""
    jax = pytest.importorskip("jax")
    tc = pytest.importorskip("tensorcircuit")
    jax.config.update("jax_enable_x64", True)
    tc.set_dtype("complex128")
    tc.set_backend("jax")
    observable = _parameterized_observable(workload)
    parameters = jax.numpy.asarray(native_parameters(workload))

    def objective(values: Any) -> Any:
        circuit = tcp.U1Circuit(
            workload.nqubits,
            particle_number=workload.particles,
            occupied=list(range(workload.particles)),
        )
        for layer in range(workload.layers):
            for wire in range(0, workload.nqubits - 1, 2):
                circuit.iswap(wire, wire + 1, theta=values[layer])
                circuit.cphase(wire, wire + 1, theta=-0.11)
        return circuit.expectation_jax(observable)

    return jax, objective, parameters


def _parameterized_metadata(backend: str, workload: U1Workload) -> dict[str, int | str]:
    return {
        "nqubits": workload.nqubits,
        "particle_number": workload.particles,
        "dimension": math.comb(workload.nqubits, workload.particles),
        "layers": workload.layers,
        "parameter_count": workload.layers,
        "angle_count": workload.layers * (workload.nqubits // 2) * 2,
        "observable_terms": 1,
        "backend": backend,
    }


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
def test_native_u1_setup(benchmark: BenchmarkFixture, workload: U1Workload) -> None:
    def setup_and_run() -> np.ndarray[Any, Any]:
        return make_native(workload).state()

    result = benchmark(setup_and_run)
    assert result.shape == (math.comb(workload.nqubits, workload.particles),)
    benchmark.extra_info.update(
        _metadata(make_native(workload), "tencirpauli-rust-setup", workload)
    )


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
def test_native_u1_steady_state(
    benchmark: BenchmarkFixture, workload: U1Workload
) -> None:
    circuit = make_native(workload)
    expected = circuit.state()
    result = benchmark.pedantic(circuit.state, rounds=5)
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
    benchmark.extra_info.update(_metadata(circuit, "tencirpauli-rust", workload))


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
def test_native_u1_end_to_end(
    benchmark: BenchmarkFixture, workload: U1Workload
) -> None:
    expected = make_native(workload).state()

    def run() -> np.ndarray[Any, Any]:
        return make_native(workload).state()

    result = benchmark.pedantic(run, rounds=5)
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
    benchmark.extra_info.update(
        _metadata(make_native(workload), "tencirpauli-rust-e2e", workload)
    )


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", JAX_WORKLOADS, ids=lambda item: item.name)
def test_tensorcircuit_jax_jit_first_call(
    benchmark: BenchmarkFixture, workload: U1Workload
) -> None:
    runner, initial, angles = _jax_runner(workload)
    expected = make_native(workload).state()

    def run() -> Any:
        return _block_until_ready(runner(initial, angles))

    result = benchmark.pedantic(run, rounds=1, iterations=1)
    np.testing.assert_allclose(np.asarray(result), expected, atol=1e-10, rtol=1e-9)
    benchmark.extra_info.update(_jax_metadata(workload, "tensorcircuit-jax-jit-first"))


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", JAX_WORKLOADS, ids=lambda item: item.name)
def test_tensorcircuit_jax_jit_steady_state(
    benchmark: BenchmarkFixture, workload: U1Workload
) -> None:
    runner, initial, angles = _jax_runner(workload)
    expected = _block_until_ready(runner(initial, angles))

    def run() -> Any:
        return _block_until_ready(runner(initial, angles))

    result = benchmark.pedantic(run, rounds=5)
    np.testing.assert_allclose(
        np.asarray(result), np.asarray(expected), atol=1e-10, rtol=1e-9
    )
    benchmark.extra_info.update(_jax_metadata(workload, "tensorcircuit-jax-jit"))


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", JAX_WORKLOADS, ids=lambda item: item.name)
def test_tensorcircuit_jax_jit_end_to_end(
    benchmark: BenchmarkFixture, workload: U1Workload
) -> None:
    expected = make_native(workload).state()

    def run() -> np.ndarray[Any, Any]:
        runner, initial, angles = _jax_runner(workload)
        return np.asarray(_block_until_ready(runner(initial, angles)))

    result = benchmark.pedantic(run, rounds=3, iterations=1)
    np.testing.assert_allclose(result, expected, atol=1e-10, rtol=1e-9)
    benchmark.extra_info.update(_jax_metadata(workload, "tensorcircuit-jax-jit-e2e"))


@pytest.mark.performance_large
def test_u1_parameterized_native_expectation(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure the public native forward expectation for a VQE-sized case."""
    workload = PARAMETERIZED_WORKLOAD
    observable = _parameterized_observable(workload)
    expected, _ = _parameterized_native_reference(workload)

    def run() -> complex:
        return make_native(workload).expectation(observable)

    result = benchmark(run)
    assert result == pytest.approx(expected, abs=1e-12)
    benchmark.extra_info.update(
        _parameterized_metadata("tencirpauli-rust-expectation", workload)
    )


@pytest.mark.performance_large
def test_u1_parameterized_native_value_and_grad(
    benchmark: BenchmarkFixture,
) -> None:
    """Measure the public native forward-plus-gradient VQE path."""
    workload = PARAMETERIZED_WORKLOAD
    circuit = make_native(workload)
    observable = _parameterized_observable(workload)
    expected_value, expected_gradient = _parameterized_native_reference(workload)
    result = benchmark(circuit.value_and_grad, observable)
    pairs = workload.nqubits // 2
    gradient = np.asarray(result.gradient).reshape(workload.layers, pairs, 2)
    np.testing.assert_allclose(result.value, expected_value, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        gradient[:, :, 0].sum(axis=1), expected_gradient, atol=1e-12, rtol=1e-12
    )
    benchmark.extra_info.update(
        _parameterized_metadata("tencirpauli-rust-value-and-grad", workload)
    )


@pytest.mark.performance_large
def test_u1_parameterized_jax_expectation_first_call(
    benchmark: BenchmarkFixture,
) -> None:
    """Record first JAX callback staging plus execution for forward expectation."""
    workload = PARAMETERIZED_WORKLOAD
    jax, objective, parameters = _parameterized_jax_runner(workload)
    expected, _ = _parameterized_native_reference(workload)
    runner = jax.jit(objective)
    result = benchmark.pedantic(
        lambda: _sync_tree(jax, runner(parameters)),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
    benchmark.extra_info.update(
        _parameterized_metadata("tencirpauli-jax-expectation-first", workload)
    )


@pytest.mark.performance_large
def test_u1_parameterized_jax_expectation_steady_state(
    benchmark: BenchmarkFixture,
) -> None:
    """Record synchronized warm JAX callback execution for forward expectation."""
    workload = PARAMETERIZED_WORKLOAD
    jax, objective, parameters = _parameterized_jax_runner(workload)
    expected, _ = _parameterized_native_reference(workload)
    runner = jax.jit(objective)
    _sync_tree(jax, runner(parameters))
    result = benchmark.pedantic(
        lambda: _sync_tree(jax, runner(parameters)), rounds=5, iterations=1
    )
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
    benchmark.extra_info.update(
        _parameterized_metadata("tencirpauli-jax-expectation-steady", workload)
    )


@pytest.mark.performance_large
def test_u1_parameterized_jax_value_and_grad_first_call(
    benchmark: BenchmarkFixture,
) -> None:
    """Record first JAX staging plus execution for expectation and gradient."""
    workload = PARAMETERIZED_WORKLOAD
    jax, objective, parameters = _parameterized_jax_runner(workload)
    expected_value, expected_gradient = _parameterized_native_reference(workload)
    runner = jax.jit(jax.value_and_grad(objective))
    result = benchmark.pedantic(
        lambda: _sync_tree(jax, runner(parameters)),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    value, gradient = result
    np.testing.assert_allclose(value, expected_value, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12, rtol=1e-12)
    benchmark.extra_info.update(
        _parameterized_metadata("tencirpauli-jax-value-and-grad-first", workload)
    )


@pytest.mark.performance_large
def test_u1_parameterized_jax_value_and_grad_steady_state(
    benchmark: BenchmarkFixture,
) -> None:
    """Record synchronized warm JAX expectation-plus-gradient execution."""
    workload = PARAMETERIZED_WORKLOAD
    jax, objective, parameters = _parameterized_jax_runner(workload)
    expected_value, expected_gradient = _parameterized_native_reference(workload)
    runner = jax.jit(jax.value_and_grad(objective))
    _sync_tree(jax, runner(parameters))
    result = benchmark.pedantic(
        lambda: _sync_tree(jax, runner(parameters)), rounds=5, iterations=1
    )
    value, gradient = result
    np.testing.assert_allclose(value, expected_value, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12, rtol=1e-12)
    benchmark.extra_info.update(
        _parameterized_metadata("tencirpauli-jax-value-and-grad-steady", workload)
    )
