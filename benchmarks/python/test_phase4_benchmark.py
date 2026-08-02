"""Release-boundary benchmarks for Phase 4 value-and-gradient engines."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


def deterministic_workload() -> tuple[tcp.GateTape, tcp.PauliOperator, np.ndarray]:
    tape = tcp.GateTape(12)
    for layer in range(3):
        for wire in range(12):
            tape.ry(wire, parameter=wire % 2)
        for wire in range(layer % 2, 11, 2):
            tape.cnot(wire, wire + 1)
    terms = []
    for wire in range(12):
        codes = [0] * 12
        codes[wire] = 1
        terms.append((codes, 0.1))
    return (
        tape,
        tcp.PauliOperator.from_terms(12, terms),
        np.array([0.13, -0.21], dtype=np.float64),
    )


def spps_workload() -> tuple[tcp.GateTape, tcp.PauliOperator, np.ndarray]:
    tape = tcp.GateTape(12)
    for wire in range(12):
        tape.ry(wire, parameter=wire % 2)
    terms = []
    for wire in range(12):
        codes = [0] * 12
        codes[wire] = 3
        terms.append((codes, 0.1))
    return (
        tape,
        tcp.PauliOperator.from_terms(12, terms),
        np.array([0.13, -0.21], dtype=np.float64),
    )


def test_deterministic_gradient_setup(benchmark: BenchmarkFixture) -> None:
    tape, observable, _ = deterministic_workload()
    engine = benchmark(tcp.PropagationEngine, tape, observable, max_weight=3)
    assert engine.nparameters == 2


def test_deterministic_gradient_first_and_steady(benchmark: BenchmarkFixture) -> None:
    tape, observable, parameters = deterministic_workload()
    engine = tcp.PropagationEngine(tape, observable, max_weight=3)
    expected = engine.value_and_grad(parameters, checkpoint_interval=4)
    result = benchmark.pedantic(
        engine.value_and_grad,
        args=(parameters,),
        kwargs={"checkpoint_interval": 4},
        rounds=10,
        iterations=1,
        warmup_rounds=1,
    )
    assert result.value == expected.value
    np.testing.assert_array_equal(result.gradient, expected.gradient)
    benchmark.extra_info["gradient_length"] = len(result.gradient)


def test_deterministic_checkpoint_scaling(benchmark: BenchmarkFixture) -> None:
    tape, observable, parameters = deterministic_workload()
    engine = tcp.PropagationEngine(tape, observable, max_weight=3)
    result = benchmark(engine.value_and_grad, parameters, checkpoint_interval=1)
    assert result.gradient.shape == (2,)
    benchmark.extra_info["checkpoint_interval"] = 1


def test_spps_fixed_budget_setup(benchmark: BenchmarkFixture) -> None:
    tape, observable, _ = spps_workload()
    engine = benchmark(tcp.SPPSEngine, tape, observable)
    assert engine.observable_terms == 12


@pytest.mark.parametrize("samples_per_term", (128, 1024))
def test_spps_fixed_budget_steady(
    benchmark: BenchmarkFixture, samples_per_term: int
) -> None:
    tape, observable, parameters = spps_workload()
    engine = tcp.SPPSEngine(tape, observable)
    result = benchmark.pedantic(
        engine.value_and_grad,
        args=(parameters,),
        kwargs={"samples_per_term": samples_per_term, "seed": 20260802},
        rounds=5,
        iterations=1,
        warmup_rounds=1,
    )
    assert np.isfinite(result.value)
    assert np.isfinite(result.gradient).all()
    benchmark.extra_info["total_paths"] = result.total_paths


def test_deterministic_100q_near_clifford_gradient(benchmark: BenchmarkFixture) -> None:
    tape = tcp.GateTape(100)
    for wire in range(0, 100, 10):
        tape.h(wire)
        tape.rz(wire, parameter=wire // 10)
    observable_terms = []
    for wire in range(0, 100, 10):
        codes = [0] * 100
        codes[wire] = 3
        observable_terms.append((codes, 0.1))
    engine = tcp.PropagationEngine(
        tape,
        tcp.PauliOperator.from_terms(100, observable_terms),
        max_weight=2,
    )
    parameters = np.linspace(0.13, 0.22, 10, dtype=np.float64)
    result = benchmark(engine.value_and_grad, parameters, checkpoint_interval=1)
    assert result.gradient.shape == (10,)
    assert np.isfinite(result.gradient).all()
    benchmark.extra_info["gradient_length"] = len(result.gradient)


def test_spps_100q_near_clifford_throughput(benchmark: BenchmarkFixture) -> None:
    tape = tcp.GateTape(100)
    for wire in range(0, 100, 10):
        tape.h(wire)
        tape.rz(wire, parameter=wire // 10)
    observable_terms = []
    for wire in range(0, 100, 10):
        codes = [0] * 100
        codes[wire] = 3
        observable_terms.append((codes, 0.1))
    engine = tcp.SPPSEngine(tape, tcp.PauliOperator.from_terms(100, observable_terms))
    parameters = np.linspace(0.13, 0.22, 10, dtype=np.float64)
    result = benchmark(
        engine.value_and_grad,
        parameters,
        samples_per_term=64,
        seed=20260802,
    )
    assert result.total_paths == 640
    assert np.isfinite(result.gradient).all()
    benchmark.extra_info["paths_per_call"] = result.total_paths


def test_spps_adaptive_budget(benchmark: BenchmarkFixture) -> None:
    tape, observable, parameters = spps_workload()
    engine = tcp.SPPSEngine(tape, observable)
    result = benchmark(
        engine.value_and_grad_adaptive,
        parameters,
        initial_samples_per_term=8,
        max_samples_per_term=32,
        gradient_tolerance=0.1,
        seed=20260802,
    )
    assert result.replicates == 2
    assert result.total_paths == 2 * sum(result.samples_per_replicate)
    benchmark.extra_info["gradient_error_proxy"] = result.gradient_error_proxy
