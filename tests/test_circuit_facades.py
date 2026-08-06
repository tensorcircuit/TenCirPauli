"""Acceptance tests for the actual-angle circuit facades."""

from __future__ import annotations

import numpy as np
import pytest

import tencirpauli as tcp


def _z_observable(nqubits: int) -> tcp.PauliOperator:
    word = [0] * nqubits
    word[0] = 3
    return tcp.PauliOperator.from_terms(nqubits, [(word, 1.0)])


def test_propagation_occurrence_gradient_and_forward_value() -> None:
    circuit = tcp.PropagationCircuit(1)
    circuit.ry(0, theta=0.19)
    circuit.ry(0, theta=0.19)
    observable = _z_observable(1)
    value = circuit.expectation(observable)
    result = circuit.value_and_grad(observable)
    assert value == pytest.approx(result.value)
    assert result.value == pytest.approx(np.cos(0.38))
    assert result.gradient.shape == (2,)
    assert not result.gradient.flags.writeable
    np.testing.assert_allclose(result.gradient, [-np.sin(0.38), -np.sin(0.38)])
    assert circuit.angle_count == 2
    assert not hasattr(circuit, "compile")


def test_forward_expectation_does_not_call_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = tcp.PropagationCircuit(1)
    circuit.ry(0, theta=0.19)
    observable = _z_observable(1)
    native = circuit._native_tape(False)
    engine = circuit._objective(
        observable,
        initial_state=circuit.initial_state,
        max_weight=None,
        max_bytes=circuit.max_bytes,
        gradient=False,
    ).engine
    called = False
    original = engine.value_and_grad

    def fail(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "value_and_grad", fail)
    assert circuit.expectation(observable) == pytest.approx(np.cos(0.19))
    assert native.nparameters == 0
    assert not called


def test_spps_value_only_and_gradient_share_seeded_value() -> None:
    circuit = tcp.SPPSCircuit(1)
    circuit.ry(0, theta=0.19)
    observable = _z_observable(1)
    value = circuit.expectation(observable, samples_per_term=256, seed=37)
    result = circuit.value_and_grad(observable, samples_per_term=256, seed=37)
    assert value.value == result.value
    assert result.gradient.shape == (1,)
    assert not result.gradient.flags.writeable


def test_u1_actual_angles_and_state_cache() -> None:
    circuit = tcp.U1Circuit(2, particle_number=1, occupied=[0])
    circuit.iswap(0, 1, theta=0.21)
    observable = tcp.PauliOperator.from_terms(2, [("ZI", 1.0), ("IZ", -0.25)])
    result = circuit.value_and_grad(observable)
    assert result.gradient.shape == (1,)
    assert circuit.expectation(observable).real == pytest.approx(result.value)
    assert circuit.state().shape == (2,)
    assert circuit.angle_count == 1
    assert not hasattr(circuit, "compile")


def test_qir_round_trip_accepts_concrete_angles_only() -> None:
    circuit = tcp.PropagationCircuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.rz(1, theta=0.17)
    restored = tcp.PropagationCircuit.from_qir(circuit.to_qir(), {"nqubits": 2})
    observable = tcp.PauliOperator.from_terms(2, [("ZZ", 1.0)])
    assert restored.expectation(observable) == pytest.approx(
        circuit.expectation(observable)
    )


def test_nonhermitian_scalar_terminals_fail() -> None:
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0j)])
    propagation = tcp.PropagationCircuit(1)
    with pytest.raises(ValueError, match="Hermitian"):
        propagation.expectation(observable)
    spps = tcp.SPPSCircuit(1)
    with pytest.raises(ValueError, match="Hermitian"):
        spps.expectation(observable, samples_per_term=4, seed=1)
