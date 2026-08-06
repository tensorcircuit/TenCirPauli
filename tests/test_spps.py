"""Functional and deterministic-replay tests for the independent SPPS path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import islice, product

import numpy as np
import pytest
from spps_reference import (
    enumerate_paths,
    exact_value_and_gradient,
    exact_value_and_gradient_slots,
    path_value_and_gradient,
)

import tencirpauli as tcp
from tencirpauli import advanced


def _single_rotation(observable_code: int) -> advanced.SPPSEngine:
    tape = advanced.GateTape(1)
    tape.ry(0, parameter=0)
    return advanced.SPPSEngine(
        tape,
        tcp.PauliOperator(1, [((observable_code,), 1.0)]),
    )


def test_fixed_budget_replay_and_metadata() -> None:
    engine = _single_rotation(3)
    first = engine.value_and_grad([0.37], samples_per_term=128, seed=19)
    second = engine.value_and_grad([0.37], samples_per_term=128, seed=19)
    assert first.value == second.value
    np.testing.assert_array_equal(first.gradient, second.gradient)
    assert first.replicates == 1
    assert first.samples_per_replicate == (128,)
    assert first.total_paths == 128
    assert first.gradient_error_proxy is None
    assert first.converged is None
    assert not first.gradient.flags.writeable


def test_fixed_estimator_matches_exact_single_term_expectation() -> None:
    engine = _single_rotation(3)
    exact_value, exact_gradient = exact_value_and_gradient(
        1, (3,), (("ry", 0, 0.37),), (0.37,)
    )
    assert len(enumerate_paths(1, (3,), (("ry", 0, 0.37),), (0.37,))) == 2
    estimate = engine.value_and_grad([0.37], samples_per_term=20_000, seed=20260802)
    assert exact_value == pytest.approx(np.cos(0.37), abs=1e-12)
    assert exact_gradient == pytest.approx(-np.sin(0.37), abs=1e-12)
    assert estimate.value == pytest.approx(exact_value, abs=0.025)
    assert estimate.gradient[0] == pytest.approx(exact_gradient, abs=0.025)
    assert estimate.value_standard_error < 0.03


def test_adaptive_estimator_matches_exact_single_term_reference() -> None:
    engine = _single_rotation(3)
    exact_value, exact_gradient = exact_value_and_gradient(
        1, (3,), (("ry", 0, 0.37),), (0.37,)
    )
    estimate = engine.value_and_grad_adaptive(
        [0.37],
        initial_samples_per_term=128,
        max_samples_per_term=4096,
        gradient_tolerance=0.005,
        seed=11,
    )
    term_proxies = estimate.term_gradient_error_proxies
    assert term_proxies is not None
    assert estimate.value == pytest.approx(
        exact_value, abs=3.0 * estimate.value_standard_error
    )
    assert estimate.gradient[0] == pytest.approx(
        exact_gradient, abs=3.0 * term_proxies[0]
    )


def test_adaptive_estimator_uses_divergent_term_budgets() -> None:
    tape = advanced.GateTape(1)
    tape.ry(0, parameter=0)
    terms = [((3,), 1.0), ((1,), 1.0e-6)]
    engine = advanced.SPPSEngine(tape, tcp.PauliOperator(1, terms))
    estimate = engine.value_and_grad_adaptive(
        [0.37],
        initial_samples_per_term=8,
        max_samples_per_term=128,
        gradient_tolerance=0.01,
        seed=91,
    )
    expected = [
        exact_value_and_gradient(1, word, (("ry", 0, 0.37),), (0.37,))
        for word, _ in terms
    ]
    expected_value = sum(
        coefficient * value for (_, coefficient), (value, _) in zip(terms, expected)
    )
    expected_gradient = sum(
        coefficient * gradient
        for (_, coefficient), (_, gradient) in zip(terms, expected)
    )
    assert len(estimate.samples_per_replicate) == 2
    assert len(set(estimate.samples_per_replicate)) > 1
    assert estimate.value == pytest.approx(
        expected_value, abs=3.0 * estimate.value_standard_error
    )
    assert estimate.gradient_error_proxy is not None
    assert estimate.gradient[0] == pytest.approx(
        expected_gradient, abs=3.0 * estimate.gradient_error_proxy
    )


def test_zero_factor_branch_preserves_pad_derivative() -> None:
    # For RY(0), the sine factor is exactly zero but its proposal probability
    # is positive.  The sine branch has final Z expectation one and therefore
    # supplies the nonzero PAD derivative of <0|RY^† X RY|0> at zero.
    tape = advanced.GateTape(1)
    tape.ry(0, parameter=0)
    engine = advanced.SPPSEngine(tape, tcp.PauliOperator(1, [((1,), 1.0)]))
    estimate = engine.value_and_grad([0.0], samples_per_term=20_000, seed=5)
    assert estimate.value == 0.0
    assert estimate.gradient[0] == pytest.approx(1.0, abs=0.15)
    assert np.isfinite(estimate.gradient).all()


def test_near_zero_factor_uses_stable_pad_products() -> None:
    tape = advanced.GateTape(1)
    tape.ry(0, parameter=0)
    engine = advanced.SPPSEngine(tape, tcp.PauliOperator(1, [((1,), 1.0)]))
    estimate = engine.value_and_grad([1.0e-14], samples_per_term=4096, seed=17)
    assert np.isfinite(estimate.value)
    assert np.isfinite(estimate.gradient).all()
    assert estimate.gradient[0] == pytest.approx(1.0, abs=0.4)


def test_adaptive_budget_and_empty_observable_contract() -> None:
    engine = _single_rotation(3)
    estimate = engine.value_and_grad_adaptive(
        [0.37],
        initial_samples_per_term=8,
        max_samples_per_term=32,
        gradient_tolerance=1.0e-8,
        seed=11,
    )
    assert estimate.replicates == 2
    assert estimate.samples_per_replicate in ((8,), (16,), (32,))
    assert estimate.total_paths == 2 * estimate.samples_per_replicate[0]
    assert estimate.gradient_error_proxy is not None
    assert estimate.term_gradient_error_proxies is not None
    assert estimate.converged in (False, True)

    empty = advanced.SPPSEngine(advanced.GateTape(1), tcp.PauliOperator.empty(1))
    empty_estimate = empty.value_and_grad_adaptive(
        [],
        initial_samples_per_term=2,
        max_samples_per_term=4,
        gradient_tolerance=0.1,
        seed=0,
    )
    assert empty_estimate.value == 0.0
    assert empty_estimate.gradient.shape == (0,)
    assert empty_estimate.total_paths == 0
    assert empty_estimate.converged is True


def test_fixed_budget_single_term_chunk_replay_is_bitwise_stable() -> None:
    tape = advanced.GateTape(2)
    tape.h(0)
    tape.rxx(0, 1, parameter=0)
    tape.cnot(0, 1)
    tape.ry(1, parameter=1)
    engine = advanced.SPPSEngine(
        tape,
        tcp.PauliOperator(2, [((3, 1), 0.7)]),
        initial_state=tcp.ProductBlochState([[0.3, 0.4, 0.5], [0.2, -0.1, 0.6]]),
    )
    first = engine.value_and_grad([0.37, -0.29], samples_per_term=1024, seed=41)
    second = engine.value_and_grad([0.37, -0.29], samples_per_term=1024, seed=41)
    assert first.value == second.value
    assert first.value_standard_error == second.value_standard_error
    np.testing.assert_array_equal(first.gradient, second.gradient)


def test_composite_exact_paths_cover_static_two_qubit_and_shared_slots() -> None:
    theta = 0.37
    operations = (
        ("h", 0),
        ("rz", 1),
        ("cnot", 0, 1),
        ("rxx", 0, 1),
        ("ry", 0),
    )
    angles = (0.0, -0.19, 0.0, theta, theta)
    parameter_slots = (None, None, None, 0, 0)
    state = np.array([[0.31, -0.27, 0.44], [-0.22, 0.35, 0.51]])
    terms = (((1, 3), 0.7), ((2, 1), -0.4))
    expected_value = 0.0
    expected_gradient = np.zeros(1)
    for word, coefficient in terms:
        paths = enumerate_paths(
            2,
            word,
            operations,
            angles,
            parameter_slots=parameter_slots,
        )
        assert sum(path.probability for path in paths) == pytest.approx(1.0)
        path_contributions = [
            path_value_and_gradient(path, state, coefficient=coefficient, nparameters=1)
            for path in paths
        ]
        expected_value += sum(value for value, _ in path_contributions)
        expected_gradient += sum(
            (gradient for _, gradient in path_contributions), start=np.zeros(1)
        )
        exact_value, exact_gradient = exact_value_and_gradient_slots(
            2,
            word,
            operations,
            angles,
            state,
            coefficient=coefficient,
            parameter_slots=parameter_slots,
        )
        assert exact_value == pytest.approx(
            sum(value for value, _ in path_contributions)
        )
        np.testing.assert_allclose(
            exact_gradient,
            sum((gradient for _, gradient in path_contributions), start=np.zeros(1)),
        )

    tape = advanced.GateTape(2)
    tape.h(0)
    tape.rz(1, angle=-0.19)
    tape.cnot(0, 1)
    tape.rxx(0, 1, parameter=0)
    tape.ry(0, parameter=0)
    engine = advanced.SPPSEngine(
        tape,
        tcp.PauliOperator(2, list(terms)),
        initial_state=tcp.ProductBlochState(state),
    )
    estimate = engine.value_and_grad([theta], samples_per_term=20_000, seed=73)
    assert estimate.value == pytest.approx(expected_value, abs=0.04)
    assert estimate.gradient[0] == pytest.approx(expected_gradient[0], abs=0.06)


def test_composite_bloch_terminal_reduction_and_adaptive_replay() -> None:
    tape = advanced.GateTape(2)
    tape.h(0)
    tape.rz(1, angle=-0.19)
    tape.cnot(0, 1)
    tape.rxx(0, 1, parameter=0)
    tape.ry(0, parameter=0)
    engine = advanced.SPPSEngine(
        tape,
        tcp.PauliOperator(2, [((1, 3), 0.7), ((2, 1), -0.4)]),
        initial_state=tcp.ProductBlochState([[0.31, -0.27, 0.44], [-0.22, 0.35, 0.51]]),
    )
    first = engine.value_and_grad_adaptive(
        [0.37],
        initial_samples_per_term=8,
        max_samples_per_term=16,
        gradient_tolerance=1.0e-8,
        seed=91,
    )
    second = engine.value_and_grad_adaptive(
        [0.37],
        initial_samples_per_term=8,
        max_samples_per_term=16,
        gradient_tolerance=1.0e-8,
        seed=91,
    )
    assert first.value == second.value
    assert first.value_standard_error == second.value_standard_error
    assert first.samples_per_replicate in ((8, 8), (16, 16))
    np.testing.assert_array_equal(first.gradient, second.gradient)
    assert np.isfinite(first.value_standard_error)


def test_spps_concurrent_calls_are_isolated_and_replayable() -> None:
    tape = advanced.GateTape(2)
    tape.cnot(0, 1)
    tape.rzz(0, 1, parameter=0)
    tape.rx(0, parameter=1)
    engine = advanced.SPPSEngine(tape, tcp.PauliOperator(2, [((1, 2), 0.8)]))
    parameters = ([0.21, -0.17], [0.33, 0.14])
    expected = [
        engine.value_and_grad(values, samples_per_term=768, seed=seed)
        for values, seed in zip(parameters, (7, 9))
    ]

    def run(index: int):
        return engine.value_and_grad(
            parameters[index], samples_per_term=768, seed=(7, 9)[index]
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(run, (0, 1, 0, 1)))
    for result, reference in zip(
        actual, (expected[0], expected[1], expected[0], expected[1])
    ):
        assert result.value == reference.value
        assert result.value_standard_error == reference.value_standard_error
        np.testing.assert_array_equal(result.gradient, reference.gradient)


def test_spps_execution_workspace_guard_covers_term_parameter_product() -> None:
    nqubits = 10
    tape = advanced.GateTape(nqubits)
    for parameter in range(1000):
        tape.rx(parameter % nqubits, parameter=parameter)
    structures = list(islice(product(range(4), repeat=nqubits), 1000))
    observable = tcp.PauliOperator(nqubits, [(word, 1.0) for word in structures])
    engine = advanced.SPPSEngine(tape, observable, max_bytes=150_000)
    with pytest.raises(MemoryError, match="memory limit"):
        engine.value_and_grad([0.1] * 1000, samples_per_term=2, seed=3)
    with pytest.raises(MemoryError, match="memory limit"):
        engine.value_and_grad_adaptive(
            [0.1] * 1000,
            initial_samples_per_term=2,
            max_samples_per_term=2,
            gradient_tolerance=0.1,
            seed=3,
        )


def test_spps_validation_and_unsupported_ptm() -> None:
    tape = advanced.GateTape(1)
    tape.ptm((0,), np.eye(4, dtype=np.float64))
    with pytest.raises(ValueError, match="SPPS"):
        advanced.SPPSEngine(tape, tcp.PauliOperator(1, [((3,), 1.0)]))
    with pytest.raises(ValueError, match="positive"):
        advanced.SPPSEngine(
            advanced.GateTape(1),
            tcp.PauliOperator(1, [((3,), 1.0)]),
            smoothing=0.0,
        )
    engine = _single_rotation(3)
    with pytest.raises(ValueError, match="samples_per_term"):
        engine.value_and_grad([0.1], samples_per_term=1, seed=0)
    with pytest.raises(ValueError, match="seed"):
        engine.value_and_grad([0.1], samples_per_term=2, seed=-1)
