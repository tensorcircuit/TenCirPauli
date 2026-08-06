"""Release-mode end-to-end circuit differentiation measurements.

Run after ``maturin develop --release --skip-install``.  The native timings
include the public facade, the native boundary, execution, and result
materialization.  JAX timings synchronize every value-and-gradient leaf, so a
reported warm time cannot finish before the gradient transfer.  This driver is
deliberately small and workload-driven; it is an evidence source, not a wall
time test.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

import tencirpauli as tcp


def _measure(function: Callable[[], Any], repeats: int = 7) -> float:
    """Return the median warm-call time in milliseconds."""
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return statistics.median(samples)


def _synchronize(jax: Any, result: Any) -> Any:
    """Synchronize all leaves of a scalar value-and-gradient result."""
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result


def _maxrss_bytes() -> int:
    """Return process peak RSS in bytes on macOS and Linux."""
    try:
        import resource
    except ImportError:
        return 0
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _propagation_from_weights(
    weights: Sequence[float],
) -> tuple[tcp.PropagationCircuit, tcp.PauliOperator]:
    circuit = tcp.PropagationCircuit(12)
    for layer in range(4):
        for wire in range(12):
            circuit.ry(wire, theta=weights[wire] * (layer + 1))
        for wire in range(0, 11, 2):
            circuit.cnot(wire, wire + 1)
    observable = tcp.PauliOperator.from_terms(12, [("Z" + "I" * 11, 1.0)])
    return circuit, observable


def _propagation() -> tuple[tcp.PropagationCircuit, tcp.PauliOperator]:
    return _propagation_from_weights(np.full(12, 0.01, dtype=np.float64))


def _u1() -> tuple[tcp.U1Circuit, tcp.PauliOperator]:
    circuit = tcp.U1Circuit(16, particle_number=4, occupied=list(range(4)))
    for layer in range(3):
        for wire in range(0, 15, 2):
            circuit.iswap(wire, wire + 1, theta=0.17 + 0.01 * layer)
            circuit.cphase(wire, wire + 1, theta=-0.11)
    observable = tcp.PauliOperator.from_terms(16, [("Z" + "I" * 15, 1.0)])
    return circuit, observable


def _spps() -> tuple[tcp.SPPSCircuit, tcp.PauliOperator]:
    circuit = tcp.SPPSCircuit(8)
    for layer in range(2):
        for wire in range(8):
            circuit.ry(wire, theta=0.08 * (layer + 1) * (wire + 1))
        for wire in range(0, 7, 2):
            circuit.cnot(wire, wire + 1)
    terms = []
    for wire in range(8):
        word = [0] * 8
        word[wire] = 3
        terms.append((word, 0.1))
    return circuit, tcp.PauliOperator.from_terms(8, terms)


def _record_native_family(
    name: str,
    build: Callable[[], tuple[Any, tcp.PauliOperator]],
    compile_plan: Callable[[Any, tcp.PauliOperator], None],
    forward: Callable[[Any, tcp.PauliOperator], Any],
    value_and_grad: Callable[[Any, tcp.PauliOperator], Any],
    native_endpoint: Callable[[Any, tcp.PauliOperator], Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Measure construction, private plan creation, first, and steady calls."""
    construction_samples = []
    plan_samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        case = build()
        construction_samples.append((time.perf_counter_ns() - start) / 1.0e6)
        start = time.perf_counter_ns()
        compile_plan(*case)
        plan_samples.append((time.perf_counter_ns() - start) / 1.0e6)

    case = build()
    start = time.perf_counter_ns()
    first_forward = forward(*case)
    first_forward_ms = (time.perf_counter_ns() - start) / 1.0e6
    start = time.perf_counter_ns()
    first_gradient = value_and_grad(*case)
    first_gradient_ms = (time.perf_counter_ns() - start) / 1.0e6

    warm_forward = _measure(lambda: forward(*case))
    warm_gradient = _measure(lambda: value_and_grad(*case))
    native_endpoint_ms = _measure(lambda: native_endpoint(*case))
    del first_forward, first_gradient
    result = {
        "construction_ms": statistics.median(construction_samples),
        "private_plan_build_ms": statistics.median(plan_samples),
        "first_forward_ms": first_forward_ms,
        "warm_forward_ms": warm_forward,
        "first_value_and_grad_ms": first_gradient_ms,
        "warm_value_and_grad_ms": warm_gradient,
        "private_native_endpoint_warm_ms": native_endpoint_ms,
        "process_peak_rss_bytes": _maxrss_bytes(),
        "family": name,
    }
    result.update(metadata)
    return result


def _native() -> dict[str, Any]:
    propagation = _record_native_family(
        "propagation",
        _propagation,
        lambda circuit, observable: circuit._objective(
            observable,
            initial_state=circuit.initial_state,
            max_weight=None,
            max_bytes=circuit.max_bytes,
            gradient=True,
        ),
        lambda circuit, observable: circuit.expectation(observable),
        lambda circuit, observable: circuit.value_and_grad(observable),
        lambda circuit, observable: circuit._objective(
            observable,
            initial_state=circuit.initial_state,
            max_weight=None,
            max_bytes=circuit.max_bytes,
            gradient=True,
        ).engine.value_and_grad(circuit._angle_values()),
        {"angle_count": 48, "observable_terms": 1},
    )
    u1 = _record_native_family(
        "u1",
        _u1,
        lambda circuit, _observable: circuit._plan(True),
        lambda circuit, observable: circuit.expectation(observable),
        lambda circuit, observable: circuit.value_and_grad(observable),
        lambda circuit, observable: circuit._plan(True)._native.value_and_grad_handle(
            circuit._initial_state,
            observable._native_handle,
            circuit._angles(),
        ),
        {"angle_count": 24, "observable_terms": 1, "dimension": 1820},
    )
    spps = _record_native_family(
        "spps",
        _spps,
        lambda circuit, observable: circuit._spps_engine(
            observable,
            initial_state=circuit.initial_state,
            smoothing=0.01,
            max_bytes=circuit.max_bytes,
            gradient=True,
        ),
        lambda circuit, observable: circuit.expectation(
            observable, samples_per_term=128, seed=20260806
        ),
        lambda circuit, observable: circuit.value_and_grad(
            observable, samples_per_term=128, seed=20260806
        ),
        lambda circuit, observable: circuit._spps_engine(
            observable,
            initial_state=circuit.initial_state,
            smoothing=0.01,
            max_bytes=circuit.max_bytes,
            gradient=True,
        ).value_and_grad(circuit._angle_values(), samples_per_term=128, seed=20260806),
        {
            "angle_count": 16,
            "observable_terms": 8,
            "samples_per_term": 128,
            "callback_output_bytes": (1 + 16) * 8,
            "gradient_workspace_bytes": 8 * 1 * 16 * 8,
        },
    )
    return {"propagation": propagation, "u1": u1, "spps": spps}


def _jax_family(
    jax: Any,
    name: str,
    objective: Callable[[Any], Any],
    weights: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    runner = jax.jit(jax.value_and_grad(objective))
    start = time.perf_counter_ns()
    first = _synchronize(jax, runner(weights))
    first_ms = (time.perf_counter_ns() - start) / 1.0e6
    warm = _measure(lambda: _synchronize(jax, runner(weights)))
    del first
    result = {
        "family": name,
        "process_peak_rss_bytes": _maxrss_bytes(),
        "first_jit_value_and_grad_ms": first_ms,
        "warm_jit_value_and_grad_ms": warm,
    }
    result.update(metadata)
    return result


def _jax() -> dict[str, Any]:
    try:
        import jax
    except ImportError:
        return {"available": False}
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    propagation_observable = tcp.PauliOperator.from_terms(12, [("Z" + "I" * 11, 1.0)])

    def propagation_objective(weights: Any) -> Any:
        circuit, _ = _propagation_from_weights(weights)
        return circuit.expectation_jax(propagation_observable)

    propagation = _jax_family(
        jax,
        "propagation",
        propagation_objective,
        jnp.full(12, 0.01, dtype=jnp.float64),
        {"angle_count": 48, "callback_output_bytes": (1 + 48) * 8},
    )

    u1_observable = tcp.PauliOperator.from_terms(16, [("Z" + "I" * 15, 1.0)])

    def u1_objective(weights: Any) -> Any:
        circuit = tcp.U1Circuit(16, particle_number=4, occupied=list(range(4)))
        for layer in range(3):
            for wire in range(0, 15, 2):
                circuit.iswap(wire, wire + 1, theta=weights[layer])
                circuit.cphase(wire, wire + 1, theta=-0.11)
        return circuit.expectation_jax(u1_observable)

    u1 = _jax_family(
        jax,
        "u1",
        u1_objective,
        jnp.asarray([0.17, 0.18, 0.19], dtype=jnp.float64),
        {"angle_count": 24, "callback_output_bytes": (1 + 24) * 8},
    )

    spps_observable = _spps()[1]

    def spps_objective(weights: Any) -> Any:
        circuit = tcp.SPPSCircuit(8)
        for layer in range(2):
            for wire in range(8):
                circuit.ry(wire, theta=weights[layer])
            for wire in range(0, 7, 2):
                circuit.cnot(wire, wire + 1)
        return circuit.expectation_jax(
            spps_observable, samples_per_term=128, seed=20260806
        )

    spps = _jax_family(
        jax,
        "spps",
        spps_objective,
        jnp.asarray([0.08, 0.16], dtype=jnp.float64),
        {
            "angle_count": 16,
            "observable_terms": 8,
            "samples_per_term": 128,
            "gradient_workspace_bytes": 8 * 1 * 16 * 8,
            "callback_output_bytes": (1 + 16) * 8,
        },
    )
    return {"available": True, "propagation": propagation, "u1": u1, "spps": spps}


def _tensorcircuit_baseline() -> dict[str, Any]:
    try:
        import jax
        import tensorcircuit as tc
    except ImportError:
        return {"available": False}
    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    import jax.numpy as jnp

    def objective(weights: Any) -> Any:
        circuit = tc.Circuit(12)
        for layer in range(4):
            for wire in range(12):
                circuit.ry(wire, theta=weights[wire] * (layer + 1))
            for wire in range(0, 11, 2):
                circuit.cnot(wire, wire + 1)
        return tc.backend.real(circuit.expectation_ps(z=[0]))

    return _jax_family(
        jax,
        "tensorcircuit_jax",
        objective,
        jnp.full(12, 0.01, dtype=jnp.float64),
        {"angle_count": 48, "callback_output_bytes": 0},
    )


def _native_caller_chain() -> dict[str, Any]:
    weights = np.full(12, 0.01, dtype=np.float64)
    observable = tcp.PauliOperator.from_terms(12, [("Z" + "I" * 11, 1.0)])

    def run() -> tuple[float, np.ndarray[Any, Any]]:
        circuit, _ = _propagation_from_weights(weights)
        result = circuit.value_and_grad(observable)
        return result.value, result.gradient.reshape(4, 12).sum(axis=0)

    first_start = time.perf_counter_ns()
    first = run()
    first_ms = (time.perf_counter_ns() - first_start) / 1.0e6
    warm = _measure(run)
    return {
        "first_ms": first_ms,
        "warm_ms": warm,
        "outer_parameter_count": 12,
        "angle_count": 48,
        "gradient_expansion": "sum four occurrence rows per outer wire",
        "finite": bool(np.isfinite(first[0]) and np.isfinite(first[1]).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "native": _native(),
        "jax": _jax(),
        "tensorcircuit_jax": _tensorcircuit_baseline(),
        "native_caller_chain": _native_caller_chain(),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
