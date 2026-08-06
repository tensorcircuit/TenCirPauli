"""Release-mode Python-visible repeated MVP and scratch-decision measurements.

The current implementation has no retained execution scratch.  This driver
records the relevant A/B boundary (owned-output ``apply`` versus caller-owned
``apply_into``), first-versus-steady behavior, and concurrent independent calls
for generic charge, U1-lazy, and structured plans.  It is intentionally local
and informational; it does not introduce a scratch pool just to manufacture a
comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

import tencirpauli as tcp


def _median(function: Callable[[], Any], repeats: int = 7) -> float:
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return statistics.median(samples)


def _generic_charge(conserved: bool) -> Any:
    space = tcp.OperatorSpace(bosons=1, qubits=6)
    charge = tcp.AdditiveCharge(
        space,
        bosons={0: 0},
        qubits={index: (0, 1) for index in range(6)},
    )
    if conserved:
        operator = space.qubit.z(0) + space.qubit.z(1)
    else:
        # Each XX/YY term changes particle number, while their canonical sum
        # is the representative cancellation workload accepted by the charge
        # restriction boundary.
        operator = space.qubit.x(0) * space.qubit.x(1)
        operator = operator + space.qubit.y(0) * space.qubit.y(1)
    return operator.restrict_charge(charge.sector(3, boson_cutoffs={0: 0})).mvp_plan()


def _u1_lazy(particles: int, nqubits: int) -> Any:
    operator = tcp.PauliOperator.from_terms(
        nqubits,
        [("XX" + "I" * (nqubits - 2), 1.0), ("YY" + "I" * (nqubits - 2), 1.0)],
    )
    return operator.restrict_charge(tcp.U1Sector(nqubits, particles)).mvp_plan()


def _structured(kind: str) -> Any:
    if kind == "boson":
        space = tcp.OperatorSpace(bosons=4)
        operator = space.boson.create(0) * space.boson.annihilate(1)
        return operator.compile(
            "native_mvp", boson_cutoffs={index: 6 for index in range(4)}
        )
    if kind == "fermion":
        space = tcp.OperatorSpace(fermions=6)
        operator = space.fermion.create(0) * space.fermion.annihilate(1)
        operator = operator + space.fermion.create(2) * space.fermion.annihilate(3)
        return operator.compile("native_mvp")
    if kind == "hybrid":
        space = tcp.OperatorSpace(fermions=2, bosons=2)
        operator = space.fermion.create(0) * space.fermion.annihilate(1)
        operator = operator + space.boson.create(0) * space.boson.annihilate(1)
        return operator.compile("native_mvp", boson_cutoffs={0: 5, 1: 5})
    raise ValueError(f"unknown structured workload {kind!r}")


def _record(name: str, build: Callable[[], Any]) -> dict[str, Any]:
    """Record one plan's construction, repeated calls, and concurrency."""
    plan = build()
    state = np.linspace(-0.5, 0.75, plan.dimension).astype(np.complex128)
    output = np.empty_like(state)

    construction_ms = _median(build, repeats=5)
    first_start = time.perf_counter_ns()
    plan.apply_into(state, output)
    first_apply_into_ms = (time.perf_counter_ns() - first_start) / 1.0e6
    steady_apply_into_ms = _median(lambda: plan.apply_into(state, output))
    allocating_apply_ms = _median(lambda: plan.apply(state))

    def concurrent_call() -> list[np.ndarray[Any, Any]]:
        def run_once() -> np.ndarray[Any, Any]:
            result = np.empty_like(state)
            plan.apply_into(state, result)
            return result

        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(lambda _: run_once(), range(4)))

    concurrent_start = time.perf_counter_ns()
    concurrent = concurrent_call()
    concurrent_ms = (time.perf_counter_ns() - concurrent_start) / 1.0e6
    expected = plan.apply(state)
    max_abs_error = max(
        float(np.max(np.abs(output - expected))),
        *(float(np.max(np.abs(item - expected))) for item in concurrent),
    )
    return {
        "name": name,
        "construction_ms": construction_ms,
        "first_apply_into_ms": first_apply_into_ms,
        "steady_apply_into_ms": steady_apply_into_ms,
        "first_vs_steady_ratio": first_apply_into_ms / steady_apply_into_ms,
        "allocating_apply_ms": allocating_apply_ms,
        "caller_owned_speedup": allocating_apply_ms / steady_apply_into_ms,
        "concurrent_four_apply_into_ms": concurrent_ms,
        "dimension": int(plan.dimension),
        "term_count": int(getattr(plan, "term_count", -1)),
        "estimated_plan_bytes": int(plan.estimated_bytes),
        "output_bytes": int(state.nbytes),
        "retained_scratch_bytes": 0,
        "max_abs_error": max_abs_error,
        "strategy": str(plan.strategy),
        "storage": str(plan.storage),
        "decision": "defer_scratch_reuse",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [
        ("generic_charge_conserved", lambda: _generic_charge(True)),
        ("generic_charge_cancelled_nonconserving", lambda: _generic_charge(False)),
        ("u1_lazy_low", lambda: _u1_lazy(1, 12)),
        ("u1_lazy_medium", lambda: _u1_lazy(4, 16)),
        ("u1_lazy_wide", lambda: _u1_lazy(10, 20)),
        ("structured_boson", lambda: _structured("boson")),
        ("structured_fermion", lambda: _structured("fermion")),
        ("structured_hybrid", lambda: _structured("hybrid")),
    ]
    result = {
        "policy": {
            "mode": "current_no_retained_scratch",
            "decision": "defer_scratch_reuse_until_a_measured_hotspot_clears_about_10_percent",
            "comparison": "owned_output_apply_vs_caller_owned_apply_into",
            "concurrency": "four independent calls on one documented immutable plan",
        },
        "cases": [_record(name, build) for name, build in cases],
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
