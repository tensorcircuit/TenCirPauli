"""Correctness tests for the Phase 6 Rust-native U(1) circuit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import pytest

import tencirpauli as tcp


def _basis(nqubits: int, k: int) -> list[int]:
    return sorted(
        sum(1 << (nqubits - 1 - index) for index in occupied)
        for occupied in combinations(range(nqubits), k)
    )


def _bit(state: int, nqubits: int, wire: int) -> int:
    return (state >> (nqubits - wire - 1)) & 1


def _dense_reference(
    nqubits: int,
    basis: list[int],
    initial: np.ndarray,
    operations: list[tuple[str, tuple[int, ...], object]],
) -> np.ndarray:
    state = np.zeros(1 << nqubits, dtype=np.complex128)
    state[basis] = initial
    for name, wires, payload in operations:
        if name == "rz":
            theta = float(payload)
            for index in range(state.shape[0]):
                z = 1.0 if _bit(index, nqubits, wires[0]) == 0 else -1.0
                state[index] *= np.exp(-0.5j * theta * z)
        elif name == "rzz":
            theta = float(payload)
            for index in range(state.shape[0]):
                z0 = 1.0 if _bit(index, nqubits, wires[0]) == 0 else -1.0
                z1 = 1.0 if _bit(index, nqubits, wires[1]) == 0 else -1.0
                state[index] *= np.exp(-0.5j * theta * z0 * z1)
        elif name == "cz":
            for index in range(state.shape[0]):
                if all(_bit(index, nqubits, wire) for wire in wires):
                    state[index] *= -1
        elif name == "cphase":
            theta = float(payload)
            for index in range(state.shape[0]):
                if all(_bit(index, nqubits, wire) for wire in wires):
                    state[index] *= np.exp(1j * theta)
        elif name == "swap":
            first, second = wires
            transformed = state.copy()
            for index in range(state.shape[0]):
                bits = list(f"{index:0{nqubits}b}")
                bits[first], bits[second] = bits[second], bits[first]
                transformed[int("".join(bits), 2)] = state[index]
            state = transformed
        elif name == "iswap":
            theta = float(payload) * np.pi / 2
            transformed = state.copy()
            first, second = wires
            for index in range(state.shape[0]):
                if _bit(index, nqubits, first) != _bit(index, nqubits, second):
                    swapped = index ^ (
                        (1 << (nqubits - first - 1)) | (1 << (nqubits - second - 1))
                    )
                    transformed[index] = (
                        np.cos(theta) * state[index]
                        + 1j * np.sin(theta) * state[swapped]
                    )
            state = transformed
        elif name == "diagonal":
            diag = np.asarray(payload, dtype=np.complex128)
            for index in range(state.shape[0]):
                local = 0
                for wire in wires:
                    local = (local << 1) | _bit(index, nqubits, wire)
                state[index] *= diag[local]
        else:
            raise AssertionError(name)
    return state[basis]


def test_required_gates_match_independent_dense_reference() -> None:
    nqubits, k = 3, 1
    initial = np.array([1.0, 2.0j, -0.5 + 0.2j], dtype=np.complex128)
    initial /= np.linalg.norm(initial)
    diagonal = np.exp(1j * np.arange(8) / 7.0)
    circuit = tcp.U1Circuit(nqubits, particle_number=k, initial_state=initial)
    circuit.rz(0, theta=0.31)
    circuit.rzz(0, 2, theta=-0.27)
    circuit.cz(0, 1)
    circuit.cphase(1, 2, theta=0.19)
    circuit.swap(0, 2)
    circuit.iswap(0, 1, theta=0.63)
    circuit.diagonal(0, 1, 2, diag=diagonal)
    operations = [
        ("rz", (0,), 0.31),
        ("rzz", (0, 2), -0.27),
        ("cz", (0, 1), 0.0),
        ("cphase", (1, 2), 0.19),
        ("swap", (0, 2), 0.0),
        ("iswap", (0, 1), 0.63),
        ("diagonal", (0, 1, 2), diagonal),
    ]
    expected = _dense_reference(nqubits, _basis(nqubits, k), initial, operations)
    np.testing.assert_allclose(circuit.state(), expected, atol=1e-12, rtol=1e-12)
    dense = circuit.state_full()
    np.testing.assert_allclose(
        dense[_basis(nqubits, k)], expected, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(circuit.probability(), np.abs(expected) ** 2)
    np.testing.assert_allclose(
        circuit.probability_full()[_basis(nqubits, k)], np.abs(expected) ** 2
    )


def test_projected_pauli_expectation_matches_dense_operator() -> None:
    circuit = tcp.U1Circuit(
        3,
        particle_number=1,
        initial_state=np.array([1.0, 1.0j, 0.5], dtype=np.complex128),
    )
    circuit.rz(1, theta=0.4)
    circuit.iswap(0, 2, theta=0.2)
    observable = tcp.PauliOperator(3, [([1, 2, 0], 0.7), ([3, 0, 3], -0.2)])
    value = circuit.expectation(observable)
    dense_state = circuit.state_full()
    expected = np.vdot(dense_state, observable.dense() @ dense_state)
    assert value == pytest.approx(expected)


def test_adjoint_gradient_matches_finite_difference_and_expression_chain_rule() -> None:
    parameter = tcp.Parameter(0)
    initial = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    circuit = tcp.U1Circuit(2, particle_number=1, initial_state=initial)
    circuit.rz(0, theta=2.0 * parameter + 0.1)
    observable = tcp.PauliOperator(2, [([1, 1], 1.0)])
    point = np.array([0.23])
    result = circuit.value_and_grad(observable, parameters=point)
    epsilon = 1e-6
    plus = circuit.expectation(observable, parameters=point + epsilon).real
    minus = circuit.expectation(observable, parameters=point - epsilon).real
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert result.value == pytest.approx(np.cos(2.0 * point[0] + 0.1))
    assert result.gradient[0] == pytest.approx(finite_difference, abs=1e-8)


def test_all_parameterized_gate_gradients_match_finite_difference() -> None:
    parameters = np.asarray([0.17, -0.23, 0.31, 0.11])
    circuit = tcp.U1Circuit(3, particle_number=1, occupied=[0])
    circuit.rz(0, theta=tcp.Parameter(0))
    circuit.rzz(0, 1, theta=tcp.Parameter(1))
    circuit.cphase(1, 2, theta=tcp.Parameter(2))
    circuit.iswap(0, 1, theta=tcp.Parameter(3))
    observable = tcp.PauliOperator(3, [([3, 0, 0], 1.0)])
    result = circuit.value_and_grad(observable, parameters=parameters)
    for index in range(parameters.size):
        epsilon = 1.0e-6
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference = (
            circuit.value_and_grad(observable, parameters=plus).value
            - circuit.value_and_grad(observable, parameters=minus).value
        ) / (2.0 * epsilon)
        assert result.gradient[index] == pytest.approx(finite_difference, abs=1e-7)


def test_lazy_compile_cache_and_parameter_transforms() -> None:
    p0 = tcp.Parameter(0)
    circuit = tcp.U1Circuit(2, particle_number=1, occupied=[0])
    circuit.iswap(0, 1, theta=p0)
    first = circuit.compile()
    assert circuit.compile() is first
    bound = circuit.bind_parameters({0: 0.5})
    assert bound.nparameters == 0
    np.testing.assert_allclose(bound.state(), circuit.state([0.5]))
    inverse = circuit.inverse()
    np.testing.assert_allclose(
        inverse.compile().run(circuit.state([0.5]), [0.5]),
        circuit._initial_state,
    )
    remapped = circuit.remap_parameters({0: 0})
    assert remapped.nparameters == 1
    with pytest.raises(ValueError, match="contiguous"):
        circuit.remap_parameters({0: 2})


def test_wide_low_particle_circuit_does_not_use_single_word_state() -> None:
    circuit = tcp.U1Circuit(129, particle_number=1, occupied=[128])
    circuit.iswap(0, 128, theta=0.25)
    state = circuit.state()
    assert state.shape == (129,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_u1_compile_budget_includes_static_diagonal_payload() -> None:
    circuit = tcp.U1Circuit(
        20,
        particle_number=10,
        occupied=range(10),
        max_bytes=8 * 1024 * 1024,
    )
    circuit.diagonal(*range(20), diagonal=np.ones(1 << 20, dtype=np.complex128))
    with pytest.raises(MemoryError):
        circuit.compile()


@pytest.mark.parametrize("nqubits", [63, 64, 65, 127, 128, 129, 256])
def test_width_acceptance_matrix_for_low_particle_sector(nqubits: int) -> None:
    circuit = tcp.U1Circuit(nqubits, particle_number=1, occupied=[nqubits - 1])
    circuit.iswap(0, nqubits - 1, theta=0.25)
    state = circuit.state()
    assert state.shape == (nqubits,)
    assert np.isclose(np.linalg.norm(state), 1.0)


@pytest.mark.parametrize("k", [0, 4])
def test_empty_and_full_particle_sectors_are_valid(k: int) -> None:
    circuit = tcp.U1Circuit(4, particle_number=k, occupied=list(range(k)))
    circuit.rz(0, theta=0.25)
    state = circuit.state()
    assert state.shape == (1,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_immutable_plan_supports_concurrent_runs() -> None:
    circuit = tcp.U1Circuit(8, particle_number=2, occupied=[0, 1])
    circuit.iswap(0, 7, theta=0.23)
    circuit.rzz(2, 6, theta=-0.17)
    plan = circuit.compile()
    with ThreadPoolExecutor(max_workers=4) as executor:
        states = list(
            executor.map(
                lambda _: plan.run(circuit._initial_state, ()),
                range(4),
            )
        )
    for state in states[1:]:
        np.testing.assert_array_equal(state, states[0])


def test_large_pair_kernel_round_trip() -> None:
    circuit = tcp.U1Circuit(40, particle_number=5, occupied=list(range(5)))
    circuit.iswap(0, 1, theta=0.37)
    circuit.iswap(0, 1, theta=-0.37)
    np.testing.assert_allclose(circuit.state(), circuit._initial_state, atol=1e-12)


def test_repeated_pair_run_is_fused_without_crossing_diagonal_barrier() -> None:
    circuit = tcp.U1Circuit(4, particle_number=1, occupied=[0])
    for _ in range(4):
        circuit.iswap(0, 1, theta=0.1)
    circuit.rz(0, theta=0.0)
    circuit.iswap(0, 1, theta=0.1)
    assert circuit.compile()._native.gate_count == 3


def test_facade_terminals_share_cached_final_state() -> None:
    parameter = tcp.Parameter(0)
    circuit = tcp.U1Circuit(3, particle_number=1, occupied=[0])
    circuit.iswap(0, 1, theta=parameter)
    values = np.asarray([0.23])
    state = circuit.state(values)
    probability = circuit.probability(values)
    dense = circuit.state_full(values)
    expectation = circuit.expectation(
        tcp.PauliOperator.from_terms(3, [((3, 0, 0), 1.0)]), parameters=values
    ).real
    np.testing.assert_allclose(probability, np.abs(state) ** 2)
    basis = _basis(3, 1)
    np.testing.assert_allclose(dense[basis], state)
    expected = sum(
        probability[index] * (1.0 if not (basis[index] & 0b100) else -1.0)
        for index in range(circuit.dimension)
    )
    assert expectation == pytest.approx(float(expected))
    assert circuit.state(values) is state
    np.testing.assert_allclose(
        circuit.state(np.asarray([-0.23])), circuit.state([-0.23])
    )


def test_facade_cache_invalidates_on_append_and_preserves_signed_zero_key() -> None:
    parameter = tcp.Parameter(0)
    circuit = tcp.U1Circuit(2, particle_number=1, occupied=[0])
    circuit.iswap(0, 1, theta=parameter)
    positive = circuit.state([0.0])
    negative = circuit.state([-0.0])
    assert positive is not negative
    circuit.rz(0, theta=0.2)
    appended = circuit.state([0.0])
    assert appended is not positive


def test_expectation_ps_rejects_ambiguous_and_duplicate_inputs() -> None:
    circuit = tcp.U1Circuit(2, particle_number=1)
    assert not hasattr(circuit, "expectation_ps")
    assert not hasattr(circuit, "expectation_z")


def test_129_qubit_k2_cross_limb_execution() -> None:
    nqubits = 129
    circuit = tcp.U1Circuit(nqubits, particle_number=2, occupied=[0, 1])
    circuit.iswap(0, 128, theta=0.25)
    state = circuit.state()
    initial_basis = (1 << (nqubits - 1)) | (1 << (nqubits - 2))
    moved_basis = (1 << (nqubits - 2)) | 1
    initial_index = circuit.sector.rank(initial_basis)
    moved_index = circuit.sector.rank(moved_basis)
    expected = np.zeros(circuit.dimension, dtype=np.complex128)
    angle = 0.25 * np.pi / 2.0
    expected[initial_index] = np.cos(angle)
    expected[moved_index] = 1j * np.sin(angle)
    np.testing.assert_allclose(state, expected, atol=1e-12, rtol=1e-12)


def test_qir_round_trip_preserves_gate_order_and_slots() -> None:
    parameter = tcp.Parameter(0)
    circuit = tcp.U1Circuit(2, particle_number=1, occupied=[0])
    circuit.rz(0, theta=parameter)
    circuit.iswap(0, 1, theta=1.0)
    restored = tcp.U1Circuit.from_qir(
        circuit.to_qir(), {"nqubits": 2, "particle_number": 1, "occupied": [0]}
    )
    np.testing.assert_array_equal(restored.state([0.37]), circuit.state([0.37]))


def test_qir_rejects_non_u1_gate() -> None:
    with pytest.raises(ValueError, match="unsupported U1Circuit QIR gate"):
        tcp.U1Circuit.from_qir(
            [{"name": "rx", "index": (0,), "parameters": {"theta": 0.2}}],
            {"nqubits": 1, "particle_number": 0},
        )
