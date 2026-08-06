"""Release benchmarks for cold native-tape compilation and cached reuse."""

from __future__ import annotations

from typing import Any, Callable

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp
from tencirpauli import advanced


KINDS = (
    "propagation_engine",
    "propagation_batch",
    "spps_engine",
    "propagation_circuit",
    "spps_circuit",
)


def _tape_workload() -> tuple[advanced.GateTape, tcp.PauliOperator]:
    tape = advanced.GateTape(8)
    for _layer in range(3):
        for wire in range(8):
            tape.ry(wire, parameter=wire % 2)
        for wire in range(0, 7, 2):
            tape.cnot(wire, wire + 1)
    observable = tcp.PauliOperator.from_terms(
        8, [("Z" + "I" * 7, 1.0), ("I" * 4 + "Z" + "I" * 3, 0.5)]
    )
    return tape, observable


def _circuit_workload(circuit_type: type[Any]) -> tuple[Any, tcp.PauliOperator]:
    circuit = circuit_type(8)
    for _layer in range(3):
        for wire in range(8):
            circuit.ry(wire, theta=tcp.Parameter(wire % 2))
        for wire in range(0, 7, 2):
            circuit.cnot(wire, wire + 1)
    _, observable = _tape_workload()
    return circuit, observable


def _cold_factory(kind: str) -> Callable[[], Any]:
    def run() -> Any:
        if kind == "propagation_circuit":
            circuit, observable = _circuit_workload(tcp.PropagationCircuit)
            return circuit.compile(observable)
        if kind == "spps_circuit":
            circuit, observable = _circuit_workload(tcp.SPPSCircuit)
            return circuit.compile(observable)
        tape, observable = _tape_workload()
        if kind == "propagation_engine":
            return advanced.PropagationEngine(tape, observable)
        if kind == "propagation_batch":
            return tcp.PropagationBatch(tape, [observable, observable])
        if kind == "spps_engine":
            return advanced.SPPSEngine(tape, observable)
        raise AssertionError(kind)

    return run


def _cached_factory(kind: str) -> tuple[Callable[[], Any], dict[str, Any]]:
    if kind == "propagation_circuit":
        circuit, observable = _circuit_workload(tcp.PropagationCircuit)
        circuit.compile(observable)
        return lambda: circuit.compile(observable), {
            "gate_count": len(circuit),
            "parameter_count": circuit.nparameters,
            "observable_term_count": observable.term_count,
            "structural_conversion_included": False,
        }
    if kind == "spps_circuit":
        circuit, observable = _circuit_workload(tcp.SPPSCircuit)
        circuit.compile(observable)
        return lambda: circuit.compile(observable), {
            "gate_count": len(circuit),
            "parameter_count": circuit.nparameters,
            "observable_term_count": observable.term_count,
            "structural_conversion_included": False,
        }
    tape, observable = _tape_workload()
    if kind == "propagation_engine":

        def factory() -> Any:
            return advanced.PropagationEngine(tape, observable)

    elif kind == "propagation_batch":

        def factory() -> Any:
            return tcp.PropagationBatch(tape, [observable, observable])

    elif kind == "spps_engine":

        def factory() -> Any:
            return advanced.SPPSEngine(tape, observable)

    else:
        raise AssertionError(kind)
    return factory, {
        "gate_count": len(tape),
        "parameter_count": tape.nparameters,
        "observable_term_count": observable.term_count,
        "structural_conversion_included": False,
    }


@pytest.mark.parametrize("kind", KINDS)
def test_gate_tape_cold_compile(benchmark: BenchmarkFixture, kind: str) -> None:
    result = benchmark(_cold_factory(kind))
    benchmark.extra_info.update(
        {
            "cache_mode": "cold",
            "structural_conversion_included": True,
            "kind": kind,
            "gate_count": 36,
            "parameter_count": 2,
            "observable_term_count": 2,
        }
    )
    assert result is not None


@pytest.mark.parametrize("kind", KINDS)
def test_gate_tape_cached_reuse(benchmark: BenchmarkFixture, kind: str) -> None:
    factory, metadata = _cached_factory(kind)
    result = benchmark(factory)
    benchmark.extra_info.update({"cache_mode": "cached", "kind": kind, **metadata})
    assert result is not None
