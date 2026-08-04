"""Matched end-to-end comparisons against TensorCircuit/JAX.

The TensorCircuit SPPS side intentionally imports the repository's
``examples/spps_pauli_path_vqe.py`` implementation.  Both reference calls are
timed from Python through output synchronization; no isolated Rust kernel is
used as the comparison boundary.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp
from tencirpauli import advanced


TENSORCIRCUIT_ROOT = Path(__file__).resolve().parents[3] / "tensorcircuit"
SPPS_EXAMPLE = TENSORCIRCUIT_ROOT / "examples" / "spps_pauli_path_vqe.py"


def _tensorcircuit() -> Any:
    return pytest.importorskip("tensorcircuit")


def _load_spps_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tc_spps_example", SPPS_EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TensorCircuit example {SPPS_EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workload() -> dict[str, Any]:
    tc = _tensorcircuit()
    pytest.importorskip("jax")
    example = _load_spps_example()
    nqubits, layers = 12, 2
    operations, nparameters = example.build_ops(nqubits, layers)
    terms = example.tfim_terms(nqubits, 1.0, 1.0)
    parameters = np.linspace(-0.19, 0.23, nparameters, dtype=np.float64)

    tape = advanced.GateTape(nqubits)
    for operation in operations:
        if operation[0] == "H":
            tape.h(int(operation[1]))
        elif operation[0] == "CNOT":
            tape.cnot(int(operation[1]), int(operation[2]))
        else:
            _, axis, wire, slot = operation
            getattr(tape, f"r{str(axis).lower()}")(int(wire), parameter=int(slot))
    structures = []
    coefficients = []
    for coefficient, x_mask, z_mask in terms:
        codes = []
        for qubit in range(nqubits):
            x = (int(x_mask) >> qubit) & 1
            z = (int(z_mask) >> qubit) & 1
            codes.append({(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}[(x, z)])
        structures.append(codes)
        coefficients.append(float(coefficient))
    observable = tcp.PauliOperator.from_terms(
        nqubits, list(zip(structures, coefficients))
    )

    native_deterministic = advanced.PropagationEngine(tape, observable, max_weight=3)
    native_spps = advanced.SPPSEngine(tape, observable, smoothing=0.25 / layers)

    tc.set_backend("jax")
    tc.set_dtype("complex128")
    term_xi = tc.backend.convert_to_tensor(np.asarray([x for _, x, _ in terms]))
    term_zi = tc.backend.convert_to_tensor(np.asarray([z for _, _, z in terms]))
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    uniforms = np.random.default_rng(20260802).random(
        (len(terms), 256, 2 * nqubits * layers)
    )
    spps_kernel, _ = example.make_spps_kernel(
        list(reversed(operations)), nparameters, smoothing=0.25 / layers
    )

    def tensorcircuit_spps() -> tuple[float, np.ndarray]:
        return example.spps_energy_grad(
            spps_kernel,
            term_xi,
            term_zi,
            coefficients_array,
            parameters,
            uniforms,
        )

    from tensorcircuit.pauliprop import PauliPropagationEngine

    ppe = PauliPropagationEngine(nqubits, 3)
    weights = tc.backend.convert_to_tensor(
        np.asarray(coefficients, dtype=np.complex128)
    )
    initial = ppe.get_initial_state(np.asarray(structures), weights)

    def ppe_value(parameters_array: Any) -> Any:
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
                    params=parameters_array[int(slot)],
                )
        return tc.backend.real(ppe.expectation(state))

    jax = pytest.importorskip("jax")
    ppe_value_and_grad = jax.jit(jax.value_and_grad(ppe_value))
    return {
        "native_deterministic": native_deterministic,
        "native_spps": native_spps,
        "parameters": parameters,
        "tensorcircuit_spps": tensorcircuit_spps,
        "ppe_value_and_grad": ppe_value_and_grad,
        "operations": operations,
    }


def test_tensorcircuit_value_and_gradient_matches_native_value() -> None:
    workload = _workload()
    parameters = workload["parameters"]
    native = workload["native_deterministic"].value_and_grad(
        parameters, checkpoint_interval=1
    )
    tc_value, tc_gradient = workload["ppe_value_and_grad"](parameters)
    tc_value = tc_value.block_until_ready()
    tc_gradient = tc_gradient.block_until_ready()
    assert native.value == pytest.approx(float(tc_value), abs=2e-5)
    np.testing.assert_allclose(native.gradient, np.asarray(tc_gradient), atol=2e-5)


def test_tensorcircuit_ppe_jax_value_and_gradient_steady(
    benchmark: BenchmarkFixture,
) -> None:
    workload = _workload()
    parameters = workload["parameters"]
    first_value, first_gradient = workload["ppe_value_and_grad"](parameters)
    first_value.block_until_ready()
    first_gradient.block_until_ready()

    def call() -> tuple[float, np.ndarray[Any, Any]]:
        value, gradient = workload["ppe_value_and_grad"](parameters)
        value = value.block_until_ready()
        gradient = gradient.block_until_ready()
        return float(value), np.asarray(gradient)

    value, gradient = benchmark(call)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    benchmark.extra_info["endpoint"] = (
        "TensorCircuit PauliPropagationEngine + JAX value_and_grad"
    )


def test_native_deterministic_value_and_gradient_steady(
    benchmark: BenchmarkFixture,
) -> None:
    workload = _workload()
    result = benchmark(
        workload["native_deterministic"].value_and_grad,
        workload["parameters"],
        checkpoint_interval=1,
    )
    assert np.isfinite(result.value)
    assert np.isfinite(result.gradient).all()
    benchmark.extra_info["endpoint"] = "TenCirPauli PropagationEngine.value_and_grad"


def test_tensorcircuit_spps_and_native_spps_steady(benchmark: BenchmarkFixture) -> None:
    workload = _workload()
    tc_value, tc_gradient = workload["tensorcircuit_spps"]()
    native = workload["native_spps"].value_and_grad(
        workload["parameters"], samples_per_term=256, seed=20260802
    )
    assert np.isfinite(tc_value)
    assert np.isfinite(tc_gradient).all()
    assert np.isfinite(native.value)
    assert np.isfinite(native.gradient).all()

    result = benchmark(
        workload["native_spps"].value_and_grad,
        workload["parameters"],
        samples_per_term=256,
        seed=20260802,
    )
    assert np.isfinite(result.value)
    benchmark.extra_info["native_endpoint"] = "TenCirPauli SPPSEngine.value_and_grad"
    benchmark.extra_info["comparison_endpoint"] = (
        "TensorCircuit examples/spps_pauli_path_vqe.py"
    )


def test_tensorcircuit_spps_steady(benchmark: BenchmarkFixture) -> None:
    workload = _workload()
    expected_value, expected_gradient = workload["tensorcircuit_spps"]()
    assert np.isfinite(expected_value)
    assert np.isfinite(expected_gradient).all()
    value, gradient = benchmark(workload["tensorcircuit_spps"])
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    benchmark.extra_info["endpoint"] = "TensorCircuit examples/spps_pauli_path_vqe.py"
