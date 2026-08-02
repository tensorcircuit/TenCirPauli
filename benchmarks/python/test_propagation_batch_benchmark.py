"""Phase 5.5 independent-observable propagation benchmarks."""

from __future__ import annotations

import os

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


def make_case(observable_count: int) -> tuple[tcp.GateTape, list[tcp.PauliOperator]]:
    tape = tcp.GateTape(12)
    for layer in range(4):
        for wire in range(12):
            tape.ry(wire, parameter=layer * 12 + wire)
        for wire in range(0, 11, 2):
            tape.cnot(wire, wire + 1)
    observables = []
    for index in range(observable_count):
        codes = [0] * 12
        codes[index % 12] = 3
        codes[(index * 5 + 1) % 12] = 1
        observables.append(tcp.PauliOperator(12, [(codes, 1.0)]))
    return tape, observables


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
    engines = [tcp.PropagationEngine(tape, observable) for observable in observables]
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
    engines = [tcp.PropagationEngine(tape, observable) for observable in observables]
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
