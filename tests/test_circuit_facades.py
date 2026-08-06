"""Acceptance tests for the unified circuit facades."""

from __future__ import annotations

import numpy as np
import pytest
import tensorcircuit as tc

import tencirpauli as tcp
import tencirpauli.propagation as propagation_module
from tencirpauli import advanced


def _z_observable(nqubits: int) -> tcp.PauliOperator:
    word = [0] * nqubits
    word[0] = 3
    return tcp.PauliOperator.from_terms(nqubits, [(word, 1.0)])


def test_propagation_facade_unifies_expectation_and_gradient() -> None:
    parameter = tcp.Parameter(0)
    circuit = tcp.PropagationCircuit(nqubits=1, initial_state=tcp.ZeroState())
    circuit.ry(0, theta=parameter)
    circuit.ry(0, theta=parameter)
    observable = _z_observable(1)

    point = np.asarray([0.19], dtype=np.float64)
    value = circuit.expectation(observable, parameters=point)
    result = circuit.value_and_grad(observable, parameters=point)
    assert value == pytest.approx(result.value)
    assert result.value == pytest.approx(np.cos(2.0 * point[0]))
    assert result.gradient.shape == (1,)
    assert not result.gradient.flags.writeable
    assert result.gradient[0] == pytest.approx(-2.0 * np.sin(2.0 * point[0]))
    assert circuit.compile(observable) is circuit.compile(observable)


def test_propagation_rejects_raw_initial_state_with_typed_error() -> None:
    circuit = tcp.PropagationCircuit(nqubits=2, initial_state=np.zeros(4))
    with pytest.raises(TypeError, match="typed state descriptor"):
        circuit.compile(_z_observable(2))


def test_propagation_expression_and_concrete_jax_array() -> None:
    jax = pytest.importorskip("jax")
    parameter = tcp.Parameter(0)
    circuit = tcp.PropagationCircuit(1)
    circuit.ry(0, theta=2.0 * parameter + 0.1)
    observable = _z_observable(1)

    point = jax.numpy.asarray([0.23])
    result = circuit.value_and_grad(observable, parameters=point)
    assert result.value == pytest.approx(np.cos(2.0 * point[0] + 0.1))
    assert result.gradient[0] == pytest.approx(-2.0 * np.sin(2.0 * point[0] + 0.1))


def test_propagation_qir_restore_and_tensorcircuit_conversion() -> None:
    qir_circuit = tcp.PropagationCircuit(2)
    qir_circuit.h(0)
    qir_circuit.cnot(0, 1)
    qir_circuit.rz(1, theta=0.17)
    restored = tcp.PropagationCircuit.from_qir(qir_circuit.to_qir(), {"nqubits": 2})
    observable = tcp.PauliOperator.from_terms(2, [("ZZ", 1.0)])
    assert restored.expectation(observable) == pytest.approx(
        qir_circuit.expectation(observable)
    )

    parameterized = tcp.PropagationCircuit(1)
    parameterized.ry(0, theta=tcp.Parameter(0))
    parameterized.ptm((0,), np.eye(4, dtype=np.float64))
    parameterized_restored = tcp.PropagationCircuit.from_qir(
        parameterized.to_qir(), {"nqubits": 1}
    )
    z_observable = _z_observable(1)
    assert parameterized_restored.expectation(z_observable, [0.23]) == pytest.approx(
        parameterized.expectation(z_observable, [0.23])
    )

    tc_circuit = tc.Circuit(2)
    tc_circuit.h(0)
    tc_circuit.cnot(0, 1)
    tc_circuit.rz(1, theta=0.17)
    converted = tcp.PropagationCircuit.from_circuit(tc_circuit)
    assert converted.expectation(observable) == pytest.approx(
        qir_circuit.expectation(observable)
    )


def test_qir_auto_discovery_reuses_duplicate_symbol_slots() -> None:
    symbol = object()
    qir = [
        {"name": "rz", "index": [0], "parameters": {"theta": symbol}},
        {"name": "rz", "index": [1], "parameters": {"theta": symbol}},
    ]
    restored = tcp.PropagationCircuit.from_qir(qir, {"nqubits": 2})
    assert restored.nparameters == 1
    assert restored.expectation(
        tcp.PauliOperator.from_terms(2, [("ZI", 1.0)]), [0.23]
    ) == pytest.approx(1.0)


def test_propagation_compile_caches_retain_key_objects() -> None:
    import gc
    import weakref

    observable = _z_observable(1)
    circuit = tcp.PropagationCircuit(1)
    circuit.compile(observable)
    reference = weakref.ref(observable)
    del observable
    gc.collect()
    assert reference() is not None

    spps = tcp.SPPSCircuit(1)
    spps_observable = _z_observable(1)
    spps.compile(spps_observable)
    spps_reference = weakref.ref(spps_observable)
    del spps_observable
    gc.collect()
    assert spps_reference() is not None


def test_gate_tape_cache_lifecycle_and_native_compile_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = propagation_module._native.pauli_gate_tape

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(propagation_module._native, "pauli_gate_tape", counted)
    tape = advanced.GateTape(2)
    tape.ry(0, parameter=0)
    tape.cnot(0, 1)
    observable_z = _z_observable(2)
    observable_x = tcp.PauliOperator.from_terms(2, [("XI", 1.0)])

    advanced.PropagationEngine(tape, observable_z)
    advanced.PropagationEngine(tape, observable_x)
    tcp.PropagationBatch(tape, [observable_z, observable_x])
    advanced.SPPSEngine(tape, observable_z)
    assert calls == 1

    tape.x(1)
    advanced.PropagationEngine(tape, observable_z)
    assert calls == 2

    circuit = tcp.PropagationCircuit(2)
    circuit.ry(0, theta=tcp.Parameter(0))
    circuit.cnot(0, 1)
    circuit.compile(observable_z)
    circuit.compile(observable_x)
    assert calls == 3
    circuit.x(1)
    circuit.compile(observable_z)
    assert calls == 4

    independent = tcp.PropagationCircuit.from_qir(circuit.to_qir(), {"nqubits": 2})
    independent.compile(observable_z)
    assert calls == 5


def test_u1_facade_expectation_and_canonical_diagonal_qir() -> None:
    parameter = tcp.Parameter(0)
    circuit = tcp.U1Circuit(2, particle_number=1, occupied=[0])
    circuit.iswap(0, 1, theta=parameter)
    circuit.diagonal(0, diagonal=np.exp(1j * np.arange(2)))
    observable = tcp.PauliOperator.from_terms(2, [("ZI", 1.0), ("IZ", -0.25)])

    result = circuit.value_and_grad(observable, parameters=[0.21])
    assert circuit.expectation(observable, parameters=[0.21]).real == pytest.approx(
        result.value
    )
    diagonal_items = [
        item for item in circuit.to_qir() if item.get("name") == "diagonal"
    ]
    assert len(diagonal_items) == 1
    assert "diagonal" in diagonal_items[0] and "diag" not in diagonal_items[0]

    restored = tcp.U1Circuit.from_qir(
        circuit.to_qir(), {"nqubits": 2, "particle_number": 1, "occupied": [0]}
    )
    np.testing.assert_allclose(restored.state([0.21]), circuit.state([0.21]))


def test_u1_tensorcircuit_classmethod_conversion() -> None:
    tc.set_backend("numpy")
    tc.set_dtype("complex128")
    reference = tc.U1Circuit(2, k=1, filled=[0])
    reference.iswap(0, 1, theta=0.31)
    converted = tcp.U1Circuit.from_circuit(reference)
    np.testing.assert_allclose(converted.state(), np.asarray(reference.state()))


def test_spps_facade_has_value_only_and_shared_parameter_paths() -> None:
    parameter = tcp.Parameter(0)
    circuit = tcp.SPPSCircuit(nqubits=1, initial_state=tcp.ZeroState())
    circuit.ry(0, theta=parameter)
    observable = _z_observable(1)

    value = circuit.expectation(
        observable, parameters=[0.19], samples_per_term=256, seed=37
    )
    result = circuit.value_and_grad(
        observable, parameters=[0.19], samples_per_term=256, seed=37
    )
    assert value.value == result.value
    assert value.value_standard_error == result.value_standard_error
    assert value.replicates == 1
    assert value.samples_per_replicate == (256,)
    assert value.total_paths == 256
    assert value.seed == 37
    assert result.gradient.shape == (1,)
    assert not result.gradient.flags.writeable

    adaptive = circuit.value_and_grad_adaptive(
        observable,
        parameters=[0.19],
        initial_samples_per_term=8,
        max_samples_per_term=16,
        gradient_tolerance=0.5,
        seed=37,
    )
    assert adaptive.gradient.shape == (1,)
    assert adaptive.replicates == 2
