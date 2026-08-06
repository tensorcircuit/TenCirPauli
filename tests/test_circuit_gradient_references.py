"""Independent numerical references for the public circuit gradient terminals."""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

import numpy as np
import pytest
from reference import codes_to_dense
from spps_reference import exact_value_and_gradient_slots

import tencirpauli as tcp


def _u1_basis(nqubits: int, particles: int) -> list[int]:
    return sorted(
        sum(1 << (nqubits - 1 - index) for index in occupied)
        for occupied in combinations(range(nqubits), particles)
    )


def _u1_dense_state(initial: np.ndarray, values: Sequence[float]) -> np.ndarray:
    """Apply the four U1 gates in an independent full-space reference."""
    nqubits = 3
    state = np.zeros(1 << nqubits, dtype=np.complex128)
    basis = _u1_basis(nqubits, 2)
    state[basis] = initial

    def bit(index: int, wire: int) -> int:
        return (index >> (nqubits - wire - 1)) & 1

    theta = float(values[0])
    for index in range(state.size):
        state[index] *= np.exp(-0.5j * theta * (1.0 if bit(index, 0) == 0 else -1.0))
    theta = float(values[1])
    for index in range(state.size):
        z0 = 1.0 if bit(index, 0) == 0 else -1.0
        z1 = 1.0 if bit(index, 1) == 0 else -1.0
        state[index] *= np.exp(-0.5j * theta * z0 * z1)
    theta = float(values[2])
    for index in range(state.size):
        if bit(index, 1) and bit(index, 2):
            state[index] *= np.exp(1j * theta)
    theta = float(values[3]) * np.pi / 2.0
    transformed = state.copy()
    for index in range(state.size):
        if bit(index, 0) != bit(index, 1):
            swapped = index ^ ((1 << 2) | (1 << 1))
            transformed[index] = (
                np.cos(theta) * state[index] + 1j * np.sin(theta) * state[swapped]
            )
    return transformed


def _finite_difference(
    build: Callable[[Sequence[float]], object],
    values: Sequence[float],
    observable: tcp.PauliOperator,
) -> np.ndarray:
    """Differentiate a fresh concrete circuit, independently of native AD."""
    epsilon = 1.0e-6
    point = np.asarray(values, dtype=np.float64)
    result = np.empty(point.shape, dtype=np.float64)
    for index in range(point.size):
        plus = point.copy()
        minus = point.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (
            float(np.real(build(plus).expectation(observable)))
            - float(np.real(build(minus).expectation(observable)))
        ) / (2.0 * epsilon)
    return result


def test_propagation_all_rotation_occurrences_match_finite_difference() -> None:
    values = np.asarray([0.13, -0.21, 0.37, 0.29, -0.17, 0.23])
    terms = [
        ("XX", 0.346),
        ("YY", 0.822),
        ("ZZ", 0.330),
        ("XZ", -1.303),
        ("ZX", 0.905),
        ("XY", 0.446),
        ("YX", -0.537),
        ("ZI", 0.581),
        ("IZ", 0.365),
        ("XI", 0.294),
        ("IX", 0.028),
        ("YI", 0.547),
        ("IY", -0.736),
    ]
    observable = tcp.PauliOperator.from_terms(2, terms)

    def build(point: Sequence[float]) -> tcp.PropagationCircuit:
        circuit = tcp.PropagationCircuit(2)
        circuit.rx(0, theta=point[0])
        circuit.ry(1, theta=point[1])
        circuit.rz(0, theta=point[2])
        circuit.rxx(0, 1, theta=point[3])
        circuit.ryy(0, 1, theta=point[4])
        circuit.rzz(0, 1, theta=point[5])
        return circuit

    circuit = build(values)
    result = circuit.value_and_grad(observable)
    assert result.value == pytest.approx(0.95288198844541, abs=1.0e-12)
    expected = _finite_difference(build, values, observable)
    np.testing.assert_allclose(result.gradient, expected, atol=3.0e-6, rtol=2.0e-6)
    assert np.all(np.abs(expected) > 0.02)


def test_u1_all_angle_occurrences_match_finite_difference() -> None:
    values = np.asarray([0.13, -0.21, 0.37, 0.29])
    initial = np.asarray([1.0, 0.3 + 0.2j, -0.2 + 0.1j], dtype=np.complex128)
    initial /= np.linalg.norm(initial)
    observable = tcp.PauliOperator.from_terms(3, [("XIX", 1.0)])

    def build(point: Sequence[float]) -> tcp.U1Circuit:
        circuit = tcp.U1Circuit(3, particle_number=2, initial_state=initial)
        circuit.rz(0, theta=point[0])
        circuit.rzz(0, 1, theta=point[1])
        circuit.cphase(1, 2, theta=point[2])
        circuit.iswap(0, 1, theta=point[3])
        return circuit

    circuit = build(values)
    result = circuit.value_and_grad(observable)
    assert result.value == pytest.approx(-0.2548817279661216, abs=1.0e-12)
    expected = _finite_difference(build, values, observable)
    np.testing.assert_allclose(result.gradient, expected, atol=3.0e-6, rtol=2.0e-6)
    assert np.all(np.abs(expected) > 0.1)


def test_u1_actual_angle_state_and_expectation_match_dense_reference() -> None:
    values = np.asarray([0.13, -0.21, 0.37, 0.29])
    initial = np.asarray([1.0, 0.3 + 0.2j, -0.2 + 0.1j], dtype=np.complex128)
    initial /= np.linalg.norm(initial)
    circuit = tcp.U1Circuit(3, particle_number=2, initial_state=initial)
    circuit.rz(0, theta=values[0])
    circuit.rzz(0, 1, theta=values[1])
    circuit.cphase(1, 2, theta=values[2])
    circuit.iswap(0, 1, theta=values[3])
    observable = tcp.PauliOperator.from_terms(3, [("XIX", 1.0)])

    expected_full = _u1_dense_state(initial, values)
    expected_value = np.vdot(expected_full, codes_to_dense((1, 0, 1)) @ expected_full)
    np.testing.assert_allclose(circuit.state_full(), expected_full, atol=1.0e-12)
    assert circuit.expectation(observable) == pytest.approx(expected_value)


def test_spps_all_rotation_occurrences_match_independent_path_reference() -> None:
    values = np.asarray([0.13, -0.21, 0.37, 0.29, -0.17, 0.23])
    operations = (
        ("rx", 0),
        ("ry", 1),
        ("rz", 0),
        ("rxx", 0, 1),
        ("ryy", 0, 1),
        ("rzz", 0, 1),
    )
    terms = [
        ((1, 0), 1.0),
        ((0, 1), -0.7),
        ((1, 3), 0.4),
        ((3, 1), 0.2),
        ((2, 0), -0.3),
        ((0, 2), 0.25),
        ((1, 1), 0.11),
        ((2, 2), -0.13),
    ]
    expected_value = 0.0
    expected_gradient = np.zeros(6, dtype=np.float64)
    for word, coefficient in terms:
        value, gradient = exact_value_and_gradient_slots(
            2,
            word,
            operations,
            values,
            "zero",
            coefficient=coefficient,
            parameter_slots=tuple(range(6)),
        )
        expected_value += value
        expected_gradient += gradient

    circuit = tcp.SPPSCircuit(2)
    circuit.rx(0, theta=values[0])
    circuit.ry(1, theta=values[1])
    circuit.rz(0, theta=values[2])
    circuit.rxx(0, 1, theta=values[3])
    circuit.ryy(0, 1, theta=values[4])
    circuit.rzz(0, 1, theta=values[5])
    observable = tcp.PauliOperator.from_terms(2, terms)
    result = circuit.value_and_grad(observable, samples_per_term=20_000, seed=91)

    assert expected_value == pytest.approx(0.18086563654300442, abs=1.0e-12)
    np.testing.assert_allclose(result.value, expected_value, atol=0.02)
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=0.04)
    assert np.all(np.abs(expected_gradient) > 0.05)
