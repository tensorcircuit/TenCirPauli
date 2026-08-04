"""Large matched comparisons against TensorCircuit/JAX.

Run from the repository root with the local TensorCircuit checkout available::

    PYTHONPATH=../tensorcircuit .conda/bin/python benchmarks/manual/large_tensorcircuit_compare.py

The deterministic comparison uses TensorCircuit's ``PauliPropagationEngine``
with the same global locality cutoff as TenCirPauli. The SPPS comparison uses
the repository example's JAX ``vmap`` kernel and the same circuit, observable,
smoothing, and samples-per-term. Timed calls include Python/native or
Python/JAX boundaries and synchronize JAX outputs.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import tencirpauli as tcp
from tencirpauli import advanced


@dataclass(frozen=True)
class Case:
    nqubits: int
    layers: int
    samples_per_term: int
    locality: int = 3


def build_ops(nqubits: int, layers: int) -> tuple[list[tuple[Any, ...]], int]:
    operations: list[tuple[Any, ...]] = [("H", wire) for wire in range(nqubits)]
    slot = 0
    for _ in range(layers):
        for wire in range(nqubits):
            operations.append(("R", "Z", wire, slot))
            slot += 1
        for wire in range(nqubits):
            operations.append(("R", "Y", wire, slot))
            slot += 1
        for wire in range(nqubits - 1):
            operations.append(("CNOT", wire, wire + 1))
    return operations, slot


def tfim_terms(nqubits: int) -> list[tuple[float, int, int]]:
    terms: list[tuple[float, int, int]] = []
    for wire in range(nqubits - 1):
        terms.append((-1.0, 0, (1 << wire) | (1 << (wire + 1))))
    for wire in range(nqubits):
        terms.append((-1.0, 1 << wire, 0))
    return terms


def make_native(
    case: Case,
    operations: list[tuple[Any, ...]],
    terms: list[tuple[float, int, int]],
) -> tuple[advanced.PropagationEngine, advanced.SPPSEngine, np.ndarray]:
    tape = advanced.GateTape(case.nqubits)
    for operation in operations:
        if operation[0] == "H":
            tape.h(int(operation[1]))
        elif operation[0] == "CNOT":
            tape.cnot(int(operation[1]), int(operation[2]))
        else:
            _, axis, wire, slot = operation
            getattr(tape, f"r{str(axis).lower()}")(int(wire), parameter=int(slot))

    structures: list[list[int]] = []
    coefficients: list[float] = []
    code_map = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}
    for coefficient, x_mask, z_mask in terms:
        structures.append(
            [
                code_map[((x_mask >> wire) & 1, (z_mask >> wire) & 1)]
                for wire in range(case.nqubits)
            ]
        )
        coefficients.append(coefficient)
    observable = tcp.PauliOperator.from_terms(
        case.nqubits, list(zip(structures, coefficients))
    )
    parameters = np.linspace(
        -0.19, 0.23, 2 * case.layers * case.nqubits, dtype=np.float64
    )
    deterministic = advanced.PropagationEngine(
        tape, observable, max_weight=case.locality
    )
    spps = advanced.SPPSEngine(tape, observable, smoothing=0.25 / case.layers)
    return deterministic, spps, parameters


def make_tensorcircuit(
    case: Case,
    operations: list[tuple[Any, ...]],
    terms: list[tuple[float, int, int]],
    parameters: np.ndarray,
) -> tuple[
    Callable[[], tuple[float, np.ndarray]], Callable[[], tuple[float, np.ndarray]]
]:
    import tensorcircuit as tc
    from tensorcircuit.pauliprop import PauliPropagationEngine

    tc.set_backend("jax")
    tc.set_dtype("complex128")
    structures = []
    coefficients = []
    for coefficient, x_mask, z_mask in terms:
        structures.append(
            [
                {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}[
                    ((x_mask >> wire) & 1, (z_mask >> wire) & 1)
                ]
                for wire in range(case.nqubits)
            ]
        )
        coefficients.append(coefficient)

    ppe = PauliPropagationEngine(case.nqubits, case.locality)
    weights = tc.backend.convert_to_tensor(
        np.asarray(coefficients, dtype=np.complex128)
    )
    initial = ppe.get_initial_state(np.asarray(structures), weights)

    def ppe_value(params: Any) -> Any:
        state = initial
        for operation in reversed(operations):
            if operation[0] == "H":
                state = ppe.apply_gate(state, "h", [int(operation[1])])
            elif operation[0] == "CNOT":
                state = ppe.apply_gate(
                    state, "cnot", [int(operation[1]), int(operation[2])]
                )
            else:
                _, axis, wire, slot = operation
                state = ppe.apply_gate(
                    state,
                    f"r{str(axis).lower()}",
                    [int(wire)],
                    params[int(slot)],
                )
        return tc.backend.real(ppe.expectation(state))

    import jax

    ppe_value_and_grad = jax.jit(jax.value_and_grad(ppe_value))
    term_xi = tc.backend.convert_to_tensor(np.asarray([x for _, x, _ in terms]))
    term_zi = tc.backend.convert_to_tensor(np.asarray([z for _, _, z in terms]))
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    uniforms = np.random.default_rng(20260802).random(
        (len(terms), case.samples_per_term, 2 * case.layers * case.nqubits)
    )

    example_path = (
        Path(__file__).resolve().parents[3]
        / "tensorcircuit"
        / "examples"
        / "spps_pauli_path_vqe.py"
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("tc_spps_large_example", example_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TensorCircuit SPPS example: {example_path}")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    kernel, _ = example.make_spps_kernel(
        list(reversed(operations)),
        len(parameters),
        smoothing=0.25 / case.layers,
    )

    def ppe_call() -> tuple[float, np.ndarray]:
        value, gradient = ppe_value_and_grad(parameters)
        return float(value.block_until_ready()), np.asarray(
            gradient.block_until_ready()
        )

    def spps_call() -> tuple[float, np.ndarray]:
        return example.spps_energy_grad(
            kernel,
            term_xi,
            term_zi,
            coefficient_array,
            parameters,
            uniforms,
        )

    return ppe_call, spps_call


def timed(
    call: Callable[[], tuple[float, np.ndarray]], repeats: int
) -> tuple[float, float, tuple[float, np.ndarray]]:
    samples = []
    start = time.perf_counter()
    result = call()
    first_ms = (time.perf_counter() - start) * 1e3
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - start) * 1e3)
    return first_ms, float(np.median(samples)), result


def run_case(case: Case, repeats: int, only: str, native_only: bool) -> None:
    operations, _ = build_ops(case.nqubits, case.layers)
    terms = tfim_terms(case.nqubits)
    native_deterministic, native_spps, parameters = make_native(case, operations, terms)
    native_det_first = native_det_time = native_det_result = None
    native_spps_first = native_spps_time = native_spps_result = None
    tc_det_first = tc_det_time = tc_det_result = None
    tc_spps_first = tc_spps_time = tc_spps_result = None
    if only in ("all", "deterministic"):
        native_det_first, native_det_time, native_det_result = timed(
            lambda: (lambda result: (result.value, result.gradient))(
                native_deterministic.value_and_grad(parameters, checkpoint_interval=1)
            ),
            repeats,
        )
    if only in ("all", "spps"):
        native_spps_first, native_spps_time, native_spps_result = timed(
            lambda: (lambda result: (result.value, result.gradient))(
                native_spps.value_and_grad(
                    parameters,
                    samples_per_term=case.samples_per_term,
                    seed=20260802,
                )
            ),
            repeats,
        )
    if not native_only:
        if only == "all":
            tc_ppe, tc_spps = make_tensorcircuit(case, operations, terms, parameters)
            tc_det_first, tc_det_time, tc_det_result = timed(tc_ppe, repeats)
            tc_spps_first, tc_spps_time, tc_spps_result = timed(tc_spps, repeats)
        elif only == "deterministic":
            tc_ppe, _ = make_tensorcircuit(case, operations, terms, parameters)
            tc_det_first, tc_det_time, tc_det_result = timed(tc_ppe, repeats)
        else:
            _, tc_spps = make_tensorcircuit(case, operations, terms, parameters)
            tc_spps_first, tc_spps_time, tc_spps_result = timed(tc_spps, repeats)

    print(
        f"{case.nqubits:>2}q L={case.layers:<2} k={case.locality} "
        f"terms={len(terms):>3} gates={len(operations):>4} "
        f"paths/term={case.samples_per_term:>4}"
    )
    if native_det_time is not None:
        assert native_det_first is not None
        assert native_det_result is not None
        if tc_det_time is None:
            print(
                f"  deterministic: Rust {native_det_time:9.3f} ms "
                f"(first {native_det_first:9.3f} ms)"
            )
        else:
            assert tc_det_first is not None
            assert tc_det_result is not None
            print(
                f"  deterministic: Rust {native_det_time:9.3f} ms "
                f"(first {native_det_first:9.3f}) | "
                f"TC/JAX {tc_det_time:9.3f} ms (first {tc_det_first:9.3f}) | "
                f"speedup {tc_det_time / native_det_time:6.2f}x | "
                f"Δvalue {abs(native_det_result[0] - tc_det_result[0]):.3e}"
            )
    if native_spps_time is not None:
        assert native_spps_first is not None
        assert native_spps_result is not None
        if tc_spps_time is None:
            print(
                f"  SPPS:          Rust {native_spps_time:9.3f} ms "
                f"(first {native_spps_first:9.3f} ms)"
            )
        else:
            assert tc_spps_first is not None
            assert tc_spps_result is not None
            print(
                f"  SPPS:          Rust {native_spps_time:9.3f} ms "
                f"(first {native_spps_first:9.3f}) | "
                f"TC/JAX {tc_spps_time:9.3f} ms (first {tc_spps_first:9.3f}) | "
                f"speedup {tc_spps_time / native_spps_time:6.2f}x | "
                f"Δvalue {abs(native_spps_result[0] - tc_spps_result[0]):.3e}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--only", choices=("all", "deterministic", "spps"), default="all"
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="skip TensorCircuit/JAX setup and measure only the native endpoint",
    )
    parser.add_argument(
        "--case",
        action="append",
        metavar="N,L,PATHS",
        help="repeatable case definition; defaults to 12,2,256;16,2,256",
    )
    args = parser.parse_args()
    raw_cases = args.case or ["12,2,256", "16,2,256"]
    cases = []
    for raw in raw_cases:
        nqubits, layers, paths = (int(value) for value in raw.split(","))
        cases.append(Case(nqubits, layers, paths))
    for case in cases:
        run_case(case, args.repeats, args.only, args.native_only)


if __name__ == "__main__":
    main()
