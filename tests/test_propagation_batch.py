"""Phase 5.5 row-wise deterministic propagation tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import tencirpauli as tcp


def _observables(count: int = 4) -> list[tcp.PauliOperator]:
    return [
        tcp.PauliOperator(
            2,
            [
                ((1, 0), 0.2 + 0.01 * index),
                ((0, 3), -0.4),
                ((2, 1), 0.1),
            ],
        )
        for index in range(count)
    ]


def _tape() -> tcp.GateTape:
    tape = tcp.GateTape(2)
    tape.h(0)
    tape.rzz(0, 1, parameter=0)
    tape.ry(1, parameter=1)
    tape.cnot(0, 1)
    return tape


@pytest.mark.parametrize("count", (0, 1, 4, 16))
def test_batch_rows_match_independent_scalar_engines(count: int) -> None:
    tape = _tape()
    observables = _observables(count)
    batch = tcp.PropagationBatch(tape, observables, max_weight=2)
    parameters = np.array([0.31, -0.22], dtype=np.float64)
    expected_values = np.array(
        [
            tcp.PropagationEngine(tape, observable, max_weight=2).expectation(
                parameters
            )
            for observable in observables
        ],
        dtype=np.float64,
    )
    expected_gradients = np.array(
        [
            tcp.PropagationEngine(tape, observable, max_weight=2)
            .value_and_grad(parameters, checkpoint_interval=2)
            .gradient
            for observable in observables
        ],
        dtype=np.float64,
    ).reshape(count, 2)

    np.testing.assert_array_equal(batch.expectations(parameters), expected_values)
    result = batch.values_and_gradients(parameters, checkpoint_interval=2)
    np.testing.assert_array_equal(result.values, expected_values)
    np.testing.assert_array_equal(result.gradients, expected_gradients)
    assert result.values.shape == (count,)
    assert result.gradients.shape == (count, 2)
    assert result.values.flags.c_contiguous
    assert result.gradients.flags.c_contiguous
    assert not result.values.flags.writeable
    assert not result.gradients.flags.writeable


def test_batch_preserves_states_projection_and_exact_cancellation() -> None:
    tape = tcp.GateTape(1)
    tape.rx(0, parameter=0)
    collision = np.zeros((4, 4), dtype=np.float64)
    collision[3, 1] = 1.0
    collision[3, 2] = -1.0
    tape.ptm((0,), collision)
    observables = [
        tcp.PauliOperator(1, [((1,), 1.0), ((2,), 1.0)]),
        tcp.PauliOperator(1, [((3,), 1.0)]),
    ]
    state = tcp.ComputationalBasisState((1,))
    batch = tcp.PropagationBatch(tape, observables, initial_state=state, max_weight=0)
    parameters = [0.37]
    expected = [
        tcp.PropagationEngine(
            tape, observable, initial_state=state, max_weight=0
        ).expectation(parameters)
        for observable in observables
    ]
    np.testing.assert_array_equal(batch.expectations(parameters), expected)
    result = batch.values_and_gradients(parameters)
    np.testing.assert_array_equal(result.values, expected)
    assert result.gradients.shape == (2, 1)


def test_batch_supports_product_bloch_and_empty_observables() -> None:
    tape = tcp.GateTape(2)
    tape.rz(0, parameter=0)
    state = tcp.ProductBlochState([[0.2, -0.1, 0.7], [0.0, 0.3, 0.4]])
    batch = tcp.PropagationBatch(tape, _observables(1), initial_state=state)
    scalar = tcp.PropagationEngine(tape, _observables(1)[0], initial_state=state)
    result = batch.values_and_gradients([0.19])
    expected = scalar.value_and_grad([0.19])
    assert result.values[0] == expected.value
    np.testing.assert_array_equal(result.gradients[0], expected.gradient)

    empty = tcp.PropagationBatch(tape, [])
    assert empty.expectations([0.19]).shape == (0,)
    assert empty.values_and_gradients([0.19]).values.shape == (0,)
    assert empty.values_and_gradients([0.19]).gradients.shape == (0, 1)


def test_batch_rejects_invalid_rows_parameters_and_memory() -> None:
    tape = tcp.GateTape(1)
    tape.rx(0, parameter=0)
    with pytest.raises(ValueError, match="same nqubits"):
        tcp.PropagationBatch(tape, [tcp.PauliOperator(2, [((1, 0), 1.0)])])
    nonhermitian = tcp.PropagationBatch(tape, [tcp.PauliOperator(1, [((1,), 1j)])])
    with pytest.raises(ValueError, match="Hermitian"):
        nonhermitian.expectations([0.2])
    with pytest.raises(ValueError, match="parameters"):
        nonhermitian.expectations([])
    with pytest.raises(ValueError, match="checkpoint_interval"):
        tcp.PropagationBatch(
            tape, [tcp.PauliOperator(1, [((3,), 1.0)])]
        ).values_and_gradients([0.2], checkpoint_interval=0)
    with pytest.raises(MemoryError, match="memory limit"):
        tcp.PropagationBatch(
            tcp.GateTape(1), [tcp.PauliOperator(1, [((3,), 1.0)])], max_bytes=1
        )


def test_batch_parallel_repeat_is_bitwise_stable() -> None:
    tape = _tape()
    observables = _observables(16)
    batch = tcp.PropagationBatch(tape, observables)
    parameters = [0.13, -0.47]
    first = batch.values_and_gradients(parameters, checkpoint_interval=1)
    second = batch.values_and_gradients(parameters, checkpoint_interval=1)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.gradients, second.gradients)


def test_batch_concurrent_calls_are_isolated() -> None:
    tape = _tape()
    batch = tcp.PropagationBatch(tape, _observables(16))
    parameters = ([0.13, -0.47], [0.21, 0.08])
    expected = [batch.values_and_gradients(values) for values in parameters]
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(executor.map(batch.values_and_gradients, parameters))
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result.values, reference.values)
        np.testing.assert_array_equal(result.gradients, reference.gradients)
