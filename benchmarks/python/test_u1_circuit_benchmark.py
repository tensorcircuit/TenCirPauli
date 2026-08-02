"""Release-mode U1Circuit comparisons against TensorCircuit."""

from __future__ import annotations

import os
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


def make_tensorcircuit(backend: str, workload: U1Workload = WORKLOADS[0]) -> object:
    tc = pytest.importorskip("tensorcircuit")
    tc.set_dtype("complex128")
    tc.set_backend(backend)
    circuit = tc.U1Circuit(
        workload.nqubits,
        k=workload.particles,
        filled=list(range(workload.particles)),
    )
    for layer in range(workload.layers):
        for wire in range(0, workload.nqubits - 1, 2):
            circuit.iswap(wire, wire + 1, theta=0.17 + 0.01 * layer)
            circuit.cphase(wire, wire + 1, theta=-0.11)
    return circuit


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
        "thread_count": int(os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1)),
        "layers": workload.layers,
        "backend": backend,
    }


def _tensorcircuit_metadata(
    expected: np.ndarray[Any, Any], workload: U1Workload, backend: str
) -> dict[str, int | str]:
    return {
        "nqubits": workload.nqubits,
        "particle_number": workload.particles,
        "dimension": int(expected.shape[0]),
        "gate_count": workload.layers * (workload.nqubits // 2) * 2,
        "state_bytes": expected.nbytes,
        "layers": workload.layers,
        "backend": backend,
    }


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
def test_tensorcircuit_numpy_state_retrieval(
    benchmark: BenchmarkFixture, workload: U1Workload
) -> None:
    circuit = make_tensorcircuit("numpy", workload)
    expected = np.asarray(circuit.state())
    result = benchmark.pedantic(circuit.state, rounds=5)
    np.testing.assert_allclose(result, expected, atol=1e-11, rtol=1e-10)
    benchmark.extra_info.update(
        _tensorcircuit_metadata(expected, workload, "tensorcircuit-numpy-state")
    )


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
def test_tensorcircuit_numpy_end_to_end(
    benchmark: BenchmarkFixture, workload: U1Workload
) -> None:
    expected = np.asarray(make_tensorcircuit("numpy", workload).state())

    def run() -> np.ndarray[Any, Any]:
        return np.asarray(make_tensorcircuit("numpy", workload).state())

    result = benchmark.pedantic(run, rounds=5)
    np.testing.assert_allclose(result, expected, atol=1e-11, rtol=1e-10)
    benchmark.extra_info.update(
        _tensorcircuit_metadata(expected, workload, "tensorcircuit-numpy-e2e")
    )


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
def test_tensorcircuit_jax_state_retrieval(
    benchmark: BenchmarkFixture, workload: U1Workload
) -> None:
    pytest.importorskip("jax")
    circuit = make_tensorcircuit("jax", workload)
    expected = np.asarray(circuit.state())

    def run() -> np.ndarray[Any, Any]:
        return np.asarray(circuit.state())

    result = benchmark.pedantic(run, rounds=5)
    np.testing.assert_allclose(result, expected, atol=1e-10, rtol=1e-9)
    benchmark.extra_info.update(
        _tensorcircuit_metadata(expected, workload, "tensorcircuit-jax-state")
    )


@pytest.mark.performance_large
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
def test_tensorcircuit_jax_end_to_end(
    benchmark: BenchmarkFixture, workload: U1Workload
) -> None:
    pytest.importorskip("jax")
    expected = np.asarray(make_tensorcircuit("jax", workload).state())

    def run() -> np.ndarray[Any, Any]:
        return np.asarray(make_tensorcircuit("jax", workload).state())

    result = benchmark.pedantic(run, rounds=5)
    np.testing.assert_allclose(result, expected, atol=1e-10, rtol=1e-9)
    benchmark.extra_info.update(
        _tensorcircuit_metadata(expected, workload, "tensorcircuit-jax-e2e")
    )
