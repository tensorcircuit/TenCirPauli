"""Numerical and API tests for the actual-angle U(1) circuit facade."""

from __future__ import annotations

import numpy as np
import pytest

import tencirpauli as tcp


def _observable(nqubits: int, word: str = "ZI") -> tcp.PauliOperator:
    return tcp.PauliOperator.from_terms(nqubits, [(word, 1.0)])


def test_required_gates_match_state_and_expectation_contract() -> None:
    circuit = tcp.U1Circuit(3, particle_number=1, occupied=[0])
    circuit.rz(0, 0.31)
    circuit.rzz(0, 2, -0.27)
    circuit.cz(0, 1)
    circuit.cphase(1, 2, 0.19)
    circuit.swap(0, 2)
    circuit.iswap(0, 1, 0.63)
    circuit.diagonal(0, 1, diagonal=np.ones(4, dtype=np.complex128))
    assert circuit.angle_count == 4
    assert circuit.state().shape == (3,)
    assert circuit.probability().shape == (3,)
    assert circuit.state_full().shape == (8,)
    assert circuit.expectation(_observable(3, "ZII")).imag == pytest.approx(0.0)


def test_every_angle_occurrence_has_an_independent_gradient() -> None:
    circuit = tcp.U1Circuit(2, particle_number=1, occupied=[0])
    circuit.rz(0, 0.13)
    circuit.rzz(0, 1, -0.21)
    circuit.cphase(0, 1, 0.37)
    circuit.iswap(0, 1, 0.29)
    observable = tcp.PauliOperator.from_terms(2, [("XX", 0.4), ("ZI", -0.7)])
    result = circuit.value_and_grad(observable)
    assert result.gradient.shape == (4,)
    assert result.gradient.flags.c_contiguous
    assert not result.gradient.flags.writeable

    epsilon = 1e-6
    angles = [0.13, -0.21, 0.37, 0.29]
    finite_difference = []
    for index, _angle in enumerate(angles):
        plus_angles = list(angles)
        minus_angles = list(angles)
        plus_angles[index] += epsilon
        minus_angles[index] -= epsilon
        plus = tcp.U1Circuit(2, particle_number=1, occupied=[0])
        minus = tcp.U1Circuit(2, particle_number=1, occupied=[0])
        for target, values in ((plus, plus_angles), (minus, minus_angles)):
            target.rz(0, values[0])
            target.rzz(0, 1, values[1])
            target.cphase(0, 1, values[2])
            target.iswap(0, 1, values[3])
        finite_difference.append(
            (plus.expectation(observable).real - minus.expectation(observable).real)
            / (2.0 * epsilon)
        )
    np.testing.assert_allclose(result.gradient, finite_difference, atol=2e-6)


def test_forward_and_gradient_plans_are_private_and_cache_state() -> None:
    circuit = tcp.U1Circuit(4, particle_number=1, occupied=[0])
    circuit.iswap(0, 1, 0.1)
    first = circuit.state()
    second = circuit.state()
    assert first is second
    assert not hasattr(circuit, "compile")
    assert not hasattr(circuit, "nparameters")


def test_u1_qir_round_trip_contains_only_concrete_angles() -> None:
    circuit = tcp.U1Circuit(2, particle_number=1, occupied=[0])
    circuit.iswap(0, 1, 0.37)
    circuit.rz(0, -0.11)
    restored = tcp.U1Circuit.from_qir(
        circuit.to_qir(), {"nqubits": 2, "particle_number": 1, "occupied": [0]}
    )
    np.testing.assert_allclose(restored.state(), circuit.state())


def test_u1_rejects_tracer_serialization_and_invalid_angles() -> None:
    with pytest.raises(ValueError):
        tcp.U1Circuit(1, particle_number=0).rz(0, float("nan"))
    with pytest.raises(TypeError):
        tcp.U1Circuit(1, particle_number=0).rz(0, 1.0 + 2.0j)
