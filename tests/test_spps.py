"""Functional and deterministic-replay tests for the independent SPPS path."""

from __future__ import annotations

import numpy as np
import pytest
from spps_reference import enumerate_paths, exact_value_and_gradient

import tencirpauli as tcp


def _single_rotation(observable_code: int) -> tcp.SPPSEngine:
    tape = tcp.GateTape(1)
    tape.ry(0, parameter=0)
    return tcp.SPPSEngine(
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


def test_zero_factor_branch_preserves_pad_derivative() -> None:
    # For RY(0), the sine factor is exactly zero but its proposal probability
    # is positive.  The sine branch has final Z expectation one and therefore
    # supplies the nonzero PAD derivative of <0|RY^† X RY|0> at zero.
    tape = tcp.GateTape(1)
    tape.ry(0, parameter=0)
    engine = tcp.SPPSEngine(tape, tcp.PauliOperator(1, [((1,), 1.0)]))
    estimate = engine.value_and_grad([0.0], samples_per_term=20_000, seed=5)
    assert estimate.value == 0.0
    assert estimate.gradient[0] == pytest.approx(1.0, abs=0.15)
    assert np.isfinite(estimate.gradient).all()


def test_near_zero_factor_uses_stable_pad_products() -> None:
    tape = tcp.GateTape(1)
    tape.ry(0, parameter=0)
    engine = tcp.SPPSEngine(tape, tcp.PauliOperator(1, [((1,), 1.0)]))
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

    empty = tcp.SPPSEngine(tcp.GateTape(1), tcp.PauliOperator.empty(1))
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


def test_spps_validation_and_unsupported_ptm() -> None:
    tape = tcp.GateTape(1)
    tape.ptm((0,), np.eye(4, dtype=np.float64))
    with pytest.raises(ValueError, match="SPPS"):
        tcp.SPPSEngine(tape, tcp.PauliOperator(1, [((3,), 1.0)]))
    with pytest.raises(ValueError, match="positive"):
        tcp.SPPSEngine(
            tcp.GateTape(1),
            tcp.PauliOperator(1, [((3,), 1.0)]),
            smoothing=0.0,
        )
    engine = _single_rotation(3)
    with pytest.raises(ValueError, match="samples_per_term"):
        engine.value_and_grad([0.1], samples_per_term=1, seed=0)
    with pytest.raises(ValueError, match="seed"):
        engine.value_and_grad([0.1], samples_per_term=2, seed=-1)
