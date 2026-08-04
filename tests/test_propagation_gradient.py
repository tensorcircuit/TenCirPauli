"""Independent functional checks for the deterministic reverse path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pytest
from propagation_reference import product_expectation, propagate_dense
from reference import codes_to_dense

import tencirpauli as tcp


def _dense_expectation(
    nqubits: int,
    word: tuple[int, ...],
    coefficient: float,
    operation: tuple[object, ...],
    state: np.ndarray,
) -> float:
    propagated = propagate_dense(nqubits, (word,), (coefficient,), (operation,))
    matrix = sum(
        (value * codes_to_dense(codes) for codes, value in propagated.items()),
        np.zeros((1 << nqubits, 1 << nqubits), dtype=np.complex128),
    )
    return product_expectation(matrix, state)


def test_local_vjp_table_covers_all_rotation_axes_and_pauli_words() -> None:
    theta = 0.37
    state = np.array([[0.31, -0.27, 0.44], [-0.22, 0.35, 0.51]])
    cases = [
        (name, 1, word, (name, 0, theta))
        for name in ("rx", "ry", "rz")
        for word in product(range(4), repeat=1)
    ] + [
        (name, 2, word, (name, 0, 1, theta))
        for name in ("rxx", "ryy", "rzz")
        for word in product(range(4), repeat=2)
    ]
    for name, nqubits, word, operation in cases:
        tape = tcp.GateTape(nqubits)
        if nqubits == 1:
            getattr(tape, name)(0, parameter=0)
        else:
            getattr(tape, name)(0, 1, parameter=0)
        observable = tcp.PauliOperator(nqubits, [(word, 1.0)])
        initial_state = tcp.ProductBlochState(state[:nqubits])
        engine = tcp.PropagationEngine(tape, observable, initial_state=initial_state)
        result = engine.value_and_grad([theta])
        h = 1.0e-6
        plus_operation = (*operation[:-1], theta + h)
        minus_operation = (*operation[:-1], theta - h)
        plus = _dense_expectation(
            nqubits, word, 1.0, plus_operation, initial_state.bloch
        )
        minus = _dense_expectation(
            nqubits, word, 1.0, minus_operation, initial_state.bloch
        )
        expected = (plus - minus) / (2.0 * h)
        assert result.gradient[0] == pytest.approx(expected, abs=4.0e-6)


def test_deleted_support_and_aggregate_cancellation_do_not_enter_reverse() -> None:
    collision = np.zeros((4, 4), dtype=np.float64)
    collision[3, 1] = 1.0
    collision[3, 2] = -1.0
    collision_tape = tcp.GateTape(1)
    collision_tape.rx(0, parameter=0)
    collision_tape.ptm((0,), collision)
    collision_engine = tcp.PropagationEngine(
        collision_tape,
        tcp.PauliOperator(1, [((1,), 1.0), ((2,), 1.0)]),
    )
    collision_result = collision_engine.value_and_grad([0.37])
    assert collision_result.value == 0.0
    assert collision_result.gradient[0] == 0.0

    projection_tape = tcp.GateTape(2)
    projection_tape.rxx(0, 1, parameter=0)
    projection_engine = tcp.PropagationEngine(
        projection_tape,
        tcp.PauliOperator(2, [((3, 0), 1.0)]),
        initial_state=tcp.ProductBlochState([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        max_weight=1,
    )
    projection_result = projection_engine.value_and_grad([0.37])
    assert projection_result.value == 0.0
    assert projection_result.gradient[0] == 0.0


def test_deterministic_gradient_concurrent_calls_are_isolated() -> None:
    tape = tcp.GateTape(2)
    tape.h(0)
    tape.cnot(0, 1)
    tape.rzz(0, 1, parameter=0)
    tape.ry(0, parameter=1)
    engine = tcp.PropagationEngine(
        tape,
        tcp.PauliOperator(2, [((1, 2), 0.8), ((3, 0), -0.3)]),
    )
    parameters = ([0.19, -0.23], [0.41, 0.08])
    expected = [engine.value_and_grad(values) for values in parameters]

    def run(index: int):
        return engine.value_and_grad(parameters[index])

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(run, (0, 1, 0, 1)))
    for result, reference in zip(
        actual, (expected[0], expected[1], expected[0], expected[1])
    ):
        assert result.value == reference.value
        np.testing.assert_array_equal(result.gradient, reference.gradient)


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
