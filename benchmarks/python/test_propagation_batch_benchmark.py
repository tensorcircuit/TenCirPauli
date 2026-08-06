"""batch propagation independent-observable propagation benchmarks."""

from __future__ import annotations

import os

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp
from tencirpauli import advanced


def make_case(
    observable_count: int,
    *,
    nqubits: int = 12,
    layers: int = 4,
    terms_per_observable: int = 1,
    rotation_heavy: bool = True,
) -> tuple[advanced.GateTape, list[tcp.PauliOperator]]:
    tape = advanced.GateTape(nqubits)
    for layer in range(layers):
        if rotation_heavy:
            for wire in range(nqubits):
                tape.ry(wire, parameter=layer * nqubits + wire)
        else:
            for wire in range(0, nqubits - 1, 2):
                tape.cz(wire, wire + 1)
        for wire in range(0, nqubits - 1, 2):
            tape.cnot(wire, wire + 1)
    observables = []
    for index in range(observable_count):
        terms = []
        for term_index in range(terms_per_observable):
            codes = [0] * nqubits
            codes[(index + term_index) % nqubits] = 3
            codes[(index * 5 + term_index + 1) % nqubits] = 1
            terms.append((codes, 1.0 / (term_index + 1)))
        observables.append(tcp.PauliOperator(nqubits, terms))
    return tape, observables


def profile_metadata(
    tape: advanced.GateTape,
    observables: list[tcp.PauliOperator],
    parameters: np.ndarray,
) -> dict[str, int]:
    profiles = [
        advanced.PropagationEngine(tape, observable).profile(parameters).profile
        for observable in observables
    ]
    return {
        "initial_term_count_max": max(
            (item.initial_term_count for item in profiles), default=0
        ),
        "peak_term_count_max": max(
            (item.peak_term_count for item in profiles), default=0
        ),
        "final_term_count_max": max(
            (item.final_term_count for item in profiles), default=0
        ),
        "estimated_peak_bytes_max": max(
            (item.estimated_peak_bytes for item in profiles), default=0
        ),
    }


@pytest.mark.parametrize("observable_count", (1, 4, 16, 64))
def test_batch_construction(benchmark: BenchmarkFixture, observable_count: int) -> None:
    tape, observables = make_case(observable_count)
    expected = tcp.PropagationBatch(tape, observables)
    result = benchmark(tcp.PropagationBatch, tape, observables)
    assert result.observable_count == expected.observable_count
    benchmark.extra_info.update(
        {
            "observable_count": observable_count,
            "nparameters": expected.nparameters,
            "output_bytes": observable_count
            * (1 + expected.nparameters)
            * np.dtype(np.float64).itemsize,
            "thread_count": int(
                os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1)
            ),
        }
    )


@pytest.mark.parametrize("observable_count", (1, 4, 16, 64))
def test_batch_expectations(benchmark: BenchmarkFixture, observable_count: int) -> None:
    tape, observables = make_case(observable_count)
    batch = tcp.PropagationBatch(tape, observables)
    parameters = np.linspace(-0.4, 0.4, batch.nparameters)
    expected = batch.expectations(parameters)
    result = benchmark.pedantic(batch.expectations, args=(parameters,), rounds=5)
    np.testing.assert_array_equal(result, expected)
    benchmark.extra_info.update(
        {
            "observable_count": observable_count,
            "nparameters": batch.nparameters,
            "output_bytes": result.nbytes,
            "thread_count": int(
                os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1)
            ),
            "numerical_error": 0.0,
        }
    )


@pytest.mark.parametrize("observable_count", (1, 4, 16, 64))
def test_batch_values_and_gradients(
    benchmark: BenchmarkFixture, observable_count: int
) -> None:
    tape, observables = make_case(observable_count)
    batch = tcp.PropagationBatch(tape, observables)
    parameters = np.linspace(-0.4, 0.4, batch.nparameters)
    expected = batch.values_and_gradients(parameters, checkpoint_interval=4)
    result = benchmark.pedantic(
        batch.values_and_gradients,
        args=(parameters,),
        kwargs={"checkpoint_interval": 4},
        rounds=3,
    )
    np.testing.assert_array_equal(result.values, expected.values)
    np.testing.assert_array_equal(result.gradients, expected.gradients)
    benchmark.extra_info.update(
        {
            "observable_count": observable_count,
            "nparameters": batch.nparameters,
            "checkpoint_interval": 4,
            "output_bytes": result.values.nbytes + result.gradients.nbytes,
            "thread_count": int(
                os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1)
            ),
            "numerical_error": 0.0,
        }
    )


@pytest.mark.parametrize("observable_count", (4, 16, 64))
def test_scalar_serial_expectations(
    benchmark: BenchmarkFixture, observable_count: int
) -> None:
    tape, observables = make_case(observable_count)
    engines = [
        advanced.PropagationEngine(tape, observable) for observable in observables
    ]
    parameters = np.linspace(-0.4, 0.4, engines[0].nparameters)
    expected = np.array([engine.expectation(parameters) for engine in engines])
    result = benchmark.pedantic(
        lambda: np.array([engine.expectation(parameters) for engine in engines]),
        rounds=5,
    )
    np.testing.assert_array_equal(result, expected)
    benchmark.extra_info.update(
        {
            "observable_count": observable_count,
            "nparameters": engines[0].nparameters,
            "output_bytes": result.nbytes,
            "thread_count": 1,
            "numerical_error": 0.0,
        }
    )


@pytest.mark.parametrize("observable_count", (4, 16, 64))
def test_scalar_serial_values_and_gradients(
    benchmark: BenchmarkFixture, observable_count: int
) -> None:
    tape, observables = make_case(observable_count)
    engines = [
        advanced.PropagationEngine(tape, observable) for observable in observables
    ]
    parameters = np.linspace(-0.4, 0.4, engines[0].nparameters)
    expected = [
        engine.value_and_grad(parameters, checkpoint_interval=4) for engine in engines
    ]

    def run_serial() -> tuple[np.ndarray, np.ndarray]:
        results = [
            engine.value_and_grad(parameters, checkpoint_interval=4)
            for engine in engines
        ]
        return (
            np.array([result.value for result in results]),
            np.array([result.gradient for result in results]),
        )

    result = benchmark.pedantic(run_serial, rounds=3)
    np.testing.assert_array_equal(result[0], [item.value for item in expected])
    np.testing.assert_array_equal(result[1], [item.gradient for item in expected])
    benchmark.extra_info.update(
        {
            "observable_count": observable_count,
            "nparameters": engines[0].nparameters,
            "checkpoint_interval": 4,
            "output_bytes": result[0].nbytes + result[1].nbytes,
            "thread_count": 1,
            "numerical_error": 0.0,
        }
    )


def test_batch_light_clifford_crossover(benchmark: BenchmarkFixture) -> None:
    tape, observables = make_case(
        4,
        nqubits=12,
        layers=2,
        rotation_heavy=False,
    )
    batch = tcp.PropagationBatch(tape, observables)
    engines = [
        advanced.PropagationEngine(tape, observable) for observable in observables
    ]
    parameters = np.empty(0, dtype=np.float64)
    expected = np.array([engine.expectation(parameters) for engine in engines])
    result = benchmark.pedantic(batch.expectations, args=(parameters,), rounds=5)
    np.testing.assert_array_equal(result, expected)
    benchmark.extra_info.update(
        {
            "observable_count": 4,
            "nqubits": 12,
            "gate_count": len(tape),
            "nparameters": batch.nparameters,
            "output_bytes": result.nbytes,
            "thread_count": int(
                os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1)
            ),
            "numerical_error": 0.0,
        }
    )
    benchmark.extra_info.update(profile_metadata(tape, observables, parameters))


@pytest.mark.performance_large
def test_batch_100q_near_clifford(benchmark: BenchmarkFixture) -> None:
    tape, observables = make_case(
        16,
        nqubits=100,
        layers=4,
        rotation_heavy=False,
    )
    batch = tcp.PropagationBatch(tape, observables)
    parameters = np.empty(0, dtype=np.float64)
    expected = batch.expectations(parameters)
    result = benchmark.pedantic(batch.expectations, args=(parameters,), rounds=5)
    np.testing.assert_array_equal(result, expected)
    benchmark.extra_info.update(
        {
            "observable_count": 16,
            "nqubits": 100,
            "gate_count": len(tape),
            "nparameters": batch.nparameters,
            "output_bytes": result.nbytes,
            "thread_count": int(
                os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1)
            ),
            "numerical_error": 0.0,
        }
    )
    benchmark.extra_info.update(profile_metadata(tape, observables, parameters))


@pytest.mark.performance_large
def test_batch_heavy_multi_term_rows(benchmark: BenchmarkFixture) -> None:
    tape, observables = make_case(16, terms_per_observable=8)
    batch = tcp.PropagationBatch(tape, observables)
    parameters = np.linspace(-0.4, 0.4, batch.nparameters)
    expected = batch.expectations(parameters)
    result = benchmark.pedantic(batch.expectations, args=(parameters,), rounds=5)
    np.testing.assert_array_equal(result, expected)
    benchmark.extra_info.update(
        {
            "observable_count": 16,
            "terms_per_observable": 8,
            "nparameters": batch.nparameters,
            "output_bytes": result.nbytes,
            "thread_count": int(
                os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1)
            ),
            "numerical_error": 0.0,
        }
    )
    benchmark.extra_info.update(profile_metadata(tape, observables, parameters))
