"""Release-mode end-to-end circuit expectation/gradient measurements.

Run after ``maturin develop --release --skip-install``.  The timings include
Python construction where stated, the public facade, native execution, and
result conversion.  JAX timings include callback dispatch and host/device
transfers after one warm-up call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import tencirpauli as tcp


def _measure(function: Callable[[], Any], repeats: int = 7) -> float:
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return statistics.median(samples)


def _propagation() -> tuple[tcp.PropagationCircuit, tcp.PauliOperator]:
    circuit = tcp.PropagationCircuit(12)
    for layer in range(4):
        for wire in range(12):
            circuit.ry(wire, theta=0.01 * (layer + 1) * (wire + 1))
        for wire in range(0, 11, 2):
            circuit.cnot(wire, wire + 1)
    observable = tcp.PauliOperator.from_terms(12, [("Z" + "I" * 11, 1.0)])
    return circuit, observable


def _native() -> dict[str, Any]:
    circuit, observable = _propagation()
    forward = _measure(lambda: circuit.expectation(observable))
    gradient = _measure(lambda: circuit.value_and_grad(observable))
    return {
        "forward_expectation_ms": forward,
        "value_and_grad_ms": gradient,
        "angle_count": circuit.angle_count,
        "forward_native_parameters": circuit._native_tape(False).nparameters,
        "gradient_native_parameters": circuit._native_tape(True).nparameters,
    }


def _jax() -> dict[str, Any]:
    try:
        import jax
    except ImportError:
        return {"available": False}
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    observable = tcp.PauliOperator.from_terms(12, [("Z" + "I" * 11, 1.0)])

    def objective(weights: Any) -> Any:
        circuit = tcp.PropagationCircuit(12)
        for layer in range(4):
            for wire in range(12):
                circuit.ry(wire, theta=weights[wire] * (layer + 1))
            for wire in range(0, 11, 2):
                circuit.cnot(wire, wire + 1)
        return circuit.expectation_jax(observable)

    weights = jnp.full((12,), 0.01, dtype=jnp.float64)
    runner = jax.jit(jax.value_and_grad(objective))
    first_start = time.perf_counter_ns()
    first = runner(weights)
    first[0].block_until_ready()
    first_ms = (time.perf_counter_ns() - first_start) / 1.0e6
    steady = _measure(lambda: runner(weights)[0].block_until_ready())
    return {
        "available": True,
        "first_jit_value_and_grad_ms": first_ms,
        "warm_jit_value_and_grad_ms": steady,
        "angle_count": 48,
        "callback_output_bytes": (1 + 48) * 8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"native": _native(), "jax": _jax()}
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
