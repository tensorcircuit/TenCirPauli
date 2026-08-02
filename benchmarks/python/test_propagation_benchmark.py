"""Release-boundary benchmarks for the public propagation handle."""

from __future__ import annotations

import numpy as np
import pytest
from matched_jax_propagation import MatchedJaxPropagation
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


def workload_12q() -> (
    tuple[tcp.GateTape, tcp.PauliOperator, np.ndarray, list[tuple[object, ...]]]
):
    tape = tcp.GateTape(12)
    reference_operations: list[tuple[object, ...]] = []
    for layer in range(3):
        for wire in range(12):
            angle = 0.07 * (wire + 1) * (layer + 1)
            tape.rx(wire, angle=angle)
            reference_operations.append(("rx", wire, None, -1, angle))
            tape.ry(wire, parameter=wire % 2)
            reference_operations.append(("ry", wire, None, wire % 2, 0.0))
        for wire in range(layer % 2, 11, 2):
            tape.cnot(wire, wire + 1)
            reference_operations.append(("cnot", wire, wire + 1, -1, 0.0))
    terms = []
    for wire in range(12):
        x = [0] * 12
        x[wire] = 1
        terms.append((x, 0.1))
        if wire < 11:
            zz = [0] * 12
            zz[wire] = zz[wire + 1] = 3
            terms.append((zz, -0.05))
    return (
        tape,
        tcp.PauliOperator.from_terms(12, terms),
        np.array([0.13, -0.21]),
        reference_operations,
    )


def test_propagation_engine_setup(benchmark: BenchmarkFixture) -> None:
    tape, observable, params, _ = workload_12q()
    result = benchmark(tcp.PropagationEngine, tape, observable, max_weight=3)
    assert result.nparameters == len(params)


def test_propagation_expectation_first_and_steady(benchmark: BenchmarkFixture) -> None:
    tape, observable, params, _ = workload_12q()
    engine = tcp.PropagationEngine(tape, observable, max_weight=3)
    expected = engine.expectation(params)
    result = benchmark.pedantic(
        engine.expectation, args=(params,), rounds=10, iterations=1, warmup_rounds=1
    )
    assert result == pytest.approx(expected)


def test_propagation_operator_materialization(benchmark: BenchmarkFixture) -> None:
    tape, observable, params, _reference_operations = workload_12q()
    engine = tcp.PropagationEngine(tape, observable, max_weight=2)
    expected = engine.propagate_operator(params)
    result = benchmark(engine.propagate_operator, params)
    assert result == expected
    benchmark.extra_info["final_terms"] = len(result.terms)


def test_matched_jax_reference_warm(benchmark: BenchmarkFixture) -> None:
    tape, observable, params, reference_operations = workload_12q()
    native = tcp.PropagationEngine(tape, observable, max_weight=3)
    reference = MatchedJaxPropagation.build(
        12,
        reference_operations,
        (term.word.to_codes() for term in observable.terms),
        (term.coefficient for term in observable.terms),
        3,
    )
    expected = reference.expectation(params)
    assert expected == pytest.approx(native.expectation(params), abs=1e-6)
    result = benchmark(reference.expectation, params)
    assert result == pytest.approx(expected)


@pytest.mark.performance_large
def test_clifford_heavy_100q_scalar(benchmark: BenchmarkFixture) -> None:
    tape = tcp.GateTape(100)
    for _layer in range(4):
        for wire in range(0, 100, 2):
            tape.h(wire)
            tape.sdg(wire + 1)
            tape.cnot(wire, wire + 1)
        for wire in range(1, 99, 2):
            tape.cz(wire, wire + 1)
    observable_terms = []
    for wire in (0, 63, 64, 99):
        codes = [0] * 100
        codes[wire] = 3
        observable_terms.append((codes, 0.25))
    engine = tcp.PropagationEngine(
        tape, tcp.PauliOperator.from_terms(100, observable_terms)
    )
    expected = engine.expectation([])
    result = benchmark(engine.expectation, [])
    assert result == pytest.approx(expected)
    benchmark.extra_info["nqubits"] = 100
    benchmark.extra_info["final_terms"] = len(engine.propagate_operator([]).terms)
