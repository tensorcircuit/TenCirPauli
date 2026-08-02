"""Independent functional checks for the deterministic Phase 4 reverse path."""

from __future__ import annotations

import numpy as np
import pytest
from propagation_reference import product_expectation, propagate_dense
from reference import codes_to_dense

import tencirpauli as tcp


def test_local_rotation_gradients_match_independent_dense_difference() -> None:
    for name, code in (("rx", 3), ("ry", 1), ("rz", 1)):
        theta = 0.371
        tape = tcp.GateTape(1)
        getattr(tape, name)(0, parameter=0)
        observable = tcp.PauliOperator(1, [((code,), 1.0)])
        engine = tcp.PropagationEngine(tape, observable)
        result = engine.value_and_grad([theta])
        operations = ((name, 0, theta),)
        h = 1.0e-6
        plus = propagate_dense(1, ((code,),), (1.0,), ((name, 0, theta + h),))
        minus = propagate_dense(1, ((code,),), (1.0,), ((name, 0, theta - h),))
        plus_matrix = sum(
            (coefficient * codes_to_dense(word) for word, coefficient in plus.items()),
            np.zeros((2, 2), dtype=np.complex128),
        )
        minus_matrix = sum(
            (coefficient * codes_to_dense(word) for word, coefficient in minus.items()),
            np.zeros((2, 2), dtype=np.complex128),
        )
        expected = (
            product_expectation(plus_matrix, "zero")
            - product_expectation(minus_matrix, "zero")
        ) / (2.0 * h)
        assert operations  # keep the gate convention explicit in this oracle
        assert result.value == pytest.approx(engine.expectation([theta]), abs=1e-12)
        assert result.gradient[0] == pytest.approx(expected, abs=2e-6)


def test_shared_slot_and_checkpoint_replay_are_identical() -> None:
    tape = tcp.GateTape(2)
    tape.ry(0, parameter=0)
    tape.cnot(0, 1)
    tape.rz(1, parameter=0)
    tape.h(0)
    observable = tcp.PauliOperator(2, [((1, 0), 0.7), ((0, 3), -0.2)])
    engine = tcp.PropagationEngine(tape, observable, max_weight=2)
    parameter = [0.37]
    results = [
        engine.value_and_grad(parameter, checkpoint_interval=interval)
        for interval in (1, 2, 3, None)
    ]
    for result in results[1:]:
        assert result.value == results[0].value
        np.testing.assert_array_equal(result.gradient, results[0].gradient)
    assert results[0].gradient.flags.c_contiguous
    assert not results[0].gradient.flags.writeable


def test_frozen_support_drops_zero_local_branches() -> None:
    tape = tcp.GateTape(1)
    tape.ry(0, parameter=0)
    x_engine = tcp.PropagationEngine(tape, tcp.PauliOperator(1, [((1,), 1.0)]))
    z_engine = tcp.PropagationEngine(tape, tcp.PauliOperator(1, [((3,), 1.0)]))
    # At zero the sine edge is not part of the deterministic support.  The
    # full dense derivative is nonzero for X, but the frozen contract returns
    # only the executed cosine edge.
    assert x_engine.value_and_grad([0.0]).value == 0.0
    assert x_engine.value_and_grad([0.0]).gradient[0] == 0.0
    assert z_engine.value_and_grad([0.0]).gradient[0] == 0.0


def test_static_ptm_transpose_path_and_projection_match_forward() -> None:
    matrix = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float64)
    tape = tcp.GateTape(1)
    tape.ptm((0,), matrix)
    tape.rx(0, parameter=0)
    engine = tcp.PropagationEngine(tape, tcp.PauliOperator(1, [((3,), 1.0)]))
    result = engine.value_and_grad([0.23])
    assert result.value == pytest.approx(engine.expectation([0.23]), abs=1e-12)
    h = 1.0e-6
    expected = (engine.expectation([0.23 + h]) - engine.expectation([0.23 - h])) / (
        2 * h
    )
    assert result.gradient[0] == pytest.approx(expected, abs=2e-6)


def test_gradient_boundaries_fail_explicitly() -> None:
    tape = tcp.GateTape(1)
    tape.rx(0, parameter=0)
    with pytest.raises(ValueError, match="checkpoint_interval"):
        tcp.PropagationEngine(tape, tcp.PauliOperator(1, [((3,), 1.0)])).value_and_grad(
            [0.2], checkpoint_interval=0
        )
    with pytest.raises(ValueError, match="Hermitian"):
        tcp.PropagationEngine(
            tcp.GateTape(1), tcp.PauliOperator(1, [((1,), 1.0j)])
        ).value_and_grad([])
