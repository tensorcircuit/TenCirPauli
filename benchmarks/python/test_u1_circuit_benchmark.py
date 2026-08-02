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


def make_native(workload: U1Workload = WORKLOADS[0]) -> tcp.U1Circuit:
    circuit = tcp.U1Circuit(
        workload.nqubits,
        k=workload.particles,
        filled=list(range(workload.particles)),
    )
    for layer in range(workload.layers):
        for wire in range(0, workload.nqubits - 1, 2):
            circuit.iswap(wire, wire + 1, theta=0.17 + 0.01 * layer)
            circuit.cphase(wire, wire + 1, theta=-0.11)
    return circuit


def _initial_state(workload: U1Workload) -> np.ndarray[Any, Any]:
    circuit = tcp.U1Circuit(
        workload.nqubits,
        k=workload.particles,
        filled=list(range(workload.particles)),
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
    angles = np.asarray(
        [0.17 + 0.01 * layer for layer in range(workload.layers)] + [-0.11],
        dtype=np.float64,
    )

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
                circuit.cphase(wire, wire + 1, theta=parameters[-1])
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


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
def test_native_u1_compile(benchmark: BenchmarkFixture, workload: U1Workload) -> None:
    circuit = make_native(workload)

    def compile_cold() -> tcp.U1CircuitPlan:
        circuit._native_plan = None
        return circuit.compile()

    plan = benchmark(compile_cold)
    assert plan.dimension == circuit.dimension
    benchmark.extra_info.update(
        _metadata(circuit, "tencirpauli-rust-compile", workload)
    )


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
def test_native_u1_steady_state(
    benchmark: BenchmarkFixture, workload: U1Workload
) -> None:
    circuit = make_native(workload)
    plan = circuit.compile()
    expected = circuit.state()
    result = benchmark.pedantic(plan.run, args=(circuit._initial_state, ()), rounds=5)
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
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
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
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
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
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
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
