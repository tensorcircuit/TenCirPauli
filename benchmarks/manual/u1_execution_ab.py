"""Small release-mode A/B workloads for U1 execution changes.

The script intentionally measures public Python entry points after a release
extension build. It reports setup metadata and steady medians so each
optimization can be compared with the previous recorded result on the same
machine.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

import tencirpauli as tcp
from tencirpauli import advanced


def _measure(function: Callable[[], object], repeats: int) -> float:
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return statistics.median(samples)


def _repeated_pair() -> dict[str, object]:
    circuit = tcp.U1Circuit(40, particle_number=5, occupied=list(range(5)))
    for index in range(32):
        circuit.iswap(0, 1, theta=0.013 + index * 0.001)
    plan = circuit.compile()
    value = _measure(lambda: plan.run(circuit._initial_state, ()), 5)
    return {
        "median_ms": value,
        "logical_gates": 32,
        "compiled_gates": int(plan._native.gate_count),
        "dimension": circuit.dimension,
    }


def _diagonal_heavy() -> dict[str, object]:
    circuit = tcp.U1Circuit(20, particle_number=10, occupied=list(range(10)))
    for _ in range(32):
        circuit.rz(0, theta=0.01)
    plan = circuit.compile()
    value = _measure(lambda: plan.run(circuit._initial_state, ()), 7)
    return {
        "median_ms": value,
        "logical_gates": 32,
        "compiled_gates": int(plan._native.gate_count),
        "dimension": circuit.dimension,
    }


def _gradient() -> dict[str, object]:
    circuit = tcp.U1Circuit(20, particle_number=5, occupied=list(range(5)))
    parameters = []
    for index in range(12):
        parameter = tcp.Parameter(index)
        parameters.append(0.07 + index * 0.01)
        circuit.iswap(0, 1, theta=parameter)
        circuit.rz(index % 4, theta=parameter)
    observable = tcp.PauliOperator(20, [([3] + [0] * 19, 1.0)])
    value = _measure(
        lambda: circuit.value_and_grad(observable, parameters=np.asarray(parameters)),
        5,
    )
    return {
        "median_ms": value,
        "logical_gates": 24,
        "compiled_gates": int(circuit.compile()._native.gate_count),
        "dimension": circuit.dimension,
    }


def _pair_map_compile() -> dict[str, object]:
    circuit = tcp.U1Circuit(40, particle_number=5, occupied=list(range(5)))
    circuit.iswap(0, 1, theta=0.13)

    def compile() -> advanced.U1CircuitPlan:
        circuit._native_plan = None
        return circuit.compile()

    value = _measure(compile, 5)
    return {
        "median_ms": value,
        "dimension": circuit.dimension,
        "compiled_gates": int(circuit.compile()._native.gate_count),
    }


def _facade_cache() -> dict[str, object]:
    circuit = tcp.U1Circuit(40, particle_number=5, occupied=list(range(5)))
    codes = [0] * 40
    codes[0] = 3
    observable = tcp.PauliOperator(40, [(codes, 1.0)])
    plan = circuit.compile()
    first = _measure(lambda: circuit.expectation(observable), 1)
    repeated = _measure(lambda: circuit.expectation(observable), 7)
    stateless = _measure(
        lambda: plan.expectation(circuit._initial_state, observable), 7
    )
    return {
        "first_ms": first,
        "repeated_expectation_ms": repeated,
        "stateless_expectation_ms": stateless,
        "dimension": circuit.dimension,
    }


def _projected_observable() -> dict[str, object]:
    circuit = tcp.U1Circuit(20, particle_number=5, occupied=list(range(5)))
    for layer in range(4):
        circuit.iswap(0, 1, theta=0.07 + 0.01 * layer)
    structures = []
    coefficients = []
    for wire in range(2, 18):
        codes = [0] * 20
        codes[0] = 1
        codes[1] = 1
        codes[wire] = 3
        structures.append(codes)
        coefficients.append(0.01 * (wire - 1))
    observable = tcp.PauliOperator(20, list(zip(structures, coefficients)))
    value = _measure(
        lambda: circuit.compile().expectation(circuit._initial_state, observable), 5
    )
    return {
        "median_ms": value,
        "terms": len(observable.terms),
        "dimension": circuit.dimension,
    }


CASES = {
    "repeated_pair": _repeated_pair,
    "diagonal_heavy": _diagonal_heavy,
    "gradient": _gradient,
    "pair_map_compile": _pair_map_compile,
    "facade_cache": _facade_cache,
    "projected_observable": _projected_observable,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected = CASES if args.case == "all" else {args.case: CASES[args.case]}
    result = {name: function() for name, function in selected.items()}
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
