"""Optional head-to-head U1Circuit differential tests against TensorCircuit."""

from __future__ import annotations

import numpy as np
import pytest
import tensorcircuit as tc

import tencirpauli as tcp


def _circuits() -> tuple[tcp.U1Circuit, object]:
    tc.set_dtype("complex128")
    tc.set_backend("numpy")
    initial = np.array([1.0, 0.3j, -0.2], dtype=np.complex128)
    initial /= np.linalg.norm(initial)
    native = tcp.U1Circuit(3, particle_number=1, initial_state=initial)
    reference = tc.U1Circuit(3, k=1, inputs=initial)
    native.rz(0, theta=0.31)
    reference.rz(0, theta=0.31)
    native.rzz(0, 2, theta=-0.27)
    reference.rzz(0, 2, theta=-0.27)
    native.cz(0, 1)
    reference.cz(0, 1)
    native.cphase(1, 2, theta=0.19)
    reference.cphase(1, 2, theta=0.19)
    native.swap(0, 2)
    reference.swap(0, 2)
    native.iswap(0, 1, theta=0.63)
    reference.iswap(0, 1, theta=0.63)
    return native, reference


def test_u1_state_and_probability_match_tensorcircuit() -> None:
    native, reference = _circuits()
    np.testing.assert_allclose(
        native.state(), np.asarray(reference.state()), atol=1e-11, rtol=1e-10
    )
    np.testing.assert_allclose(
        native.probability(),
        np.asarray(reference.probability()),
        atol=1e-11,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        native.state_full(), np.asarray(reference.to_dense()), atol=1e-11, rtol=1e-10
    )


def test_u1_observable_matches_tensorcircuit() -> None:
    native, reference = _circuits()
    observable = tcp.PauliOperator.from_terms(3, [((1, 2, 0), 1.0)])
    native_value = native.expectation(observable)
    reference_value = reference.expectation_ps(ps=[1, 2, 0])
    assert native_value == pytest.approx(np.asarray(reference_value))


def test_u1_conversion_rejects_boolean_angles() -> None:
    circuit = tc.U1Circuit(1, k=0, filled=[])
    circuit.rz(0, theta=True)
    with pytest.raises(TypeError, match="finite real"):
        tcp.U1Circuit.from_circuit(circuit)


@pytest.mark.parametrize("filled", [None, [1]])
def test_u1_conversion_preserves_custom_initial_state(filled: list[int] | None) -> None:
    with tc.runtime_backend("numpy"), tc.runtime_dtype("complex128"):
        initial = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
        reference = tc.U1Circuit(2, k=1, filled=filled, inputs=initial)
        reference.iswap(0, 1, theta=0.37)
        reference.rz(0, theta=-0.23)
        converted = tcp.U1Circuit.from_circuit(reference)
        np.testing.assert_allclose(converted.state(), reference.state(), atol=1e-12)
        np.testing.assert_allclose(
            converted.state_full(), reference.to_dense(), atol=1e-12
        )
        observable = tcp.PauliOperator.from_terms(2, [("ZI", 1.0)])
        np.testing.assert_allclose(
            converted.expectation(observable),
            reference.expectation_ps(z=[0]),
            atol=1e-12,
        )
        assert reference.circuit_param["inputs"] is initial
        assert reference.circuit_param["filled"] == ([0] if filled is None else filled)


@pytest.mark.parametrize(
    "nqubits, particles, layers",
    ((20, 2, 8), (24, 3, 6)),
    ids=("20q-k2", "24q-k3"),
)
def test_u1_benchmark_workloads_match_tensorcircuit(
    nqubits: int, particles: int, layers: int
) -> None:
    tc.set_dtype("complex128")
    tc.set_backend("numpy")
    native = tcp.U1Circuit(
        nqubits, particle_number=particles, occupied=list(range(particles))
    )
    reference = tc.U1Circuit(nqubits, k=particles, filled=list(range(particles)))
    for layer in range(layers):
        for wire in range(0, nqubits - 1, 2):
            theta = 0.17 + 0.01 * layer
            native.iswap(wire, wire + 1, theta=theta)
            reference.iswap(wire, wire + 1, theta=theta)
            native.cphase(wire, wire + 1, theta=-0.11)
            reference.cphase(wire, wire + 1, theta=-0.11)
    np.testing.assert_allclose(
        native.state(), np.asarray(reference.state()), atol=1e-11, rtol=1e-10
    )
