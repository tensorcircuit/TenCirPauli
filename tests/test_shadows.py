"""Numerical and interchange contracts for classical-shadow snapshots."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import tencirpauli as tcp


def test_pauli_import_and_readback_are_numeric_and_immutable() -> None:
    bases = np.array([[3], [3]], dtype=np.uint8)
    outcomes = np.zeros((2, 2, 1), dtype=np.uint8)
    snapshots = tcp.Snapshots.from_pauli_measurements(bases, outcomes)
    arrays = snapshots.to_arrays()
    assert isinstance(arrays, tcp.PauliSnapshotArrays)
    assert arrays.bases.dtype == np.uint8
    assert arrays.outcomes.shape == (2, 2, 1)
    assert not arrays.outcomes.flags.writeable
    np.testing.assert_allclose(
        snapshots.expectation(tcp.PauliWord.from_string("Z")), 3.0
    )
    np.testing.assert_allclose(
        snapshots.rdm(), np.diag([2.0, -1.0]).astype(np.complex128)
    )


def test_clifford_frame_import_and_numeric_estimators() -> None:
    snapshots = tcp.Snapshots.from_clifford_measurements(
        np.array([[1]], dtype=np.uint64),
        np.array([[0]], dtype=np.uint64),
        np.array([[1]], dtype=np.int8),
        np.array([[[0]]], dtype=np.uint8),
        nqubits=1,
    )
    arrays = snapshots.to_arrays()
    assert isinstance(arrays, tcp.CliffordSnapshotArrays)
    assert arrays.frame_signs.shape == (1, 1)
    np.testing.assert_allclose(
        snapshots.expectation(tcp.PauliWord.from_string("X")), 3.0
    )
    np.testing.assert_allclose(
        snapshots.rdm(),
        np.array([[0.5, 1.5], [1.5, 0.5]], dtype=np.complex128),
    )
    np.testing.assert_allclose(
        snapshots.estimate_many(
            [tcp.PauliWord.from_string("X"), tcp.PauliWord.from_string("Z")]
        ),
        [3.0, 0.0],
    )


def test_seeded_sampling_roundtrips_and_stabilizer_fidelity() -> None:
    state = tcp.StabilizerState.zero_state(1)
    snapshots = tcp.Snapshots.sample(
        state, nsettings=32, repeats=1, protocol="clifford", seed=11
    )
    arrays = snapshots.to_arrays()
    replay = tcp.Snapshots.from_clifford_measurements(
        arrays.frame_x_words,
        arrays.frame_z_words,
        arrays.frame_signs,
        arrays.outcomes,
        nqubits=1,
    )
    assert replay.nsettings == snapshots.nsettings
    np.testing.assert_array_equal(replay.to_arrays().outcomes, arrays.outcomes)
    assert replay.fidelity(state) == snapshots.fidelity(state)


def test_dense_target_fidelity_handles_arbitrary_target_states() -> None:
    plus_snapshot = tcp.Snapshots.from_clifford_measurements(
        np.array([[1]], dtype=np.uint64),
        np.array([[0]], dtype=np.uint64),
        np.array([[1]], dtype=np.int8),
        np.array([[[0]]], dtype=np.uint8),
        nqubits=1,
    )
    plus_i = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    np.testing.assert_allclose(plus_snapshot.fidelity_statevector(plus_i), 0.5)

    one_snapshot = tcp.Snapshots.from_clifford_measurements(
        np.array([[0]], dtype=np.uint64),
        np.array([[1]], dtype=np.uint64),
        np.array([[-1]], dtype=np.int8),
        np.array([[[0]]], dtype=np.uint8),
        nqubits=1,
    )
    np.testing.assert_allclose(
        one_snapshot.fidelity_statevector(np.array([0.0, 1.0], dtype=np.complex128)),
        2.0,
    )
    np.testing.assert_allclose(
        one_snapshot.fidelity_statevector(np.array([1.0, 0.0], dtype=np.complex128)),
        -1.0,
    )

    graph_snapshot = tcp.Snapshots.from_clifford_measurements(
        np.array([[1], [2]], dtype=np.uint64),
        np.array([[2], [1]], dtype=np.uint64),
        np.ones((1, 2), dtype=np.int8),
        np.zeros((1, 1, 2), dtype=np.uint8),
        nqubits=2,
    )
    graph_state = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.complex128) / 2.0
    np.testing.assert_allclose(graph_snapshot.fidelity_statevector(graph_state), 4.0)

    with pytest.raises(ValueError, match="norm"):
        plus_snapshot.fidelity_statevector(np.array([1.0, 1.0], dtype=np.complex128))
    with pytest.raises(ValueError, match="width"):
        plus_snapshot.fidelity_statevector(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        )


def test_blocks_and_shadow_bound_contracts() -> None:
    bases = np.full((4, 1), 3, dtype=np.uint8)
    outcomes = np.zeros((4, 1, 1), dtype=np.uint8)
    snapshots = tcp.Snapshots.from_pauli_measurements(bases, outcomes)
    blocks = snapshots.expectation(
        tcp.PauliWord.from_string("Z"), blocks=2, return_blocks=True
    )
    np.testing.assert_allclose(blocks, [3.0, 3.0])
    nsettings, block_count = tcp.shadow_bound(
        np.array([[3], [1]], dtype=np.uint8), epsilon=0.5
    )
    assert block_count > 0 and nsettings % block_count == 0


def test_renyi_methods_and_energy_variance_have_fixed_values() -> None:
    bases = np.full((2, 1), 3, dtype=np.uint8)
    outcomes = np.zeros((2, 1, 1), dtype=np.uint8)
    snapshots = tcp.Snapshots.from_pauli_measurements(bases, outcomes)
    np.testing.assert_allclose(
        snapshots.renyi_entropy(subsystem_to_keep=[0], method="shadow"), -np.log(5.0)
    )
    hamiltonian = tcp.PauliOperator.from_terms(1, [("Z", 1.0)])
    np.testing.assert_allclose(snapshots.energy_variance(hamiltonian), -8.0)

    repeated = tcp.Snapshots.from_pauli_measurements(
        np.array([[3]], dtype=np.uint8), np.zeros((1, 2, 1), dtype=np.uint8)
    )
    np.testing.assert_allclose(
        repeated.renyi_entropy(subsystem_to_keep=[0], method="randomized_measurement"),
        -np.log(2.0),
    )


@pytest.mark.parametrize("scale", [1.0, 1.0e-7])
def test_energy_variance_removes_only_imaginary_square_roundoff(scale: float) -> None:
    snapshots = tcp.Snapshots.from_pauli_measurements(
        np.full((2, 2), 3, dtype=np.uint8), np.zeros((2, 1, 2), dtype=np.uint8)
    )
    hamiltonian = tcp.PauliOperator.from_terms(
        2,
        [
            (word, value * scale)
            for word, value in [("ZX", 0.4), ("IX", 0.3), ("ZY", 0.4), ("IY", 0.5)]
        ],
    )
    # H^2 = scale^2 * (0.66 II + 0.64 ZI); both Z settings estimate <ZI> = 3.
    np.testing.assert_allclose(
        snapshots.energy_variance(hamiltonian), 2.58 * scale**2, atol=0.0, rtol=1e-12
    )
    nonhermitian = tcp.PauliOperator.from_terms(2, [("ZI", 1.0 + 1.0e-18j)])
    with pytest.raises(ValueError, match="Hermitian"):
        snapshots.energy_variance(nonhermitian)


def test_shadow_renyi_workspace_guard_precedes_large_materialization() -> None:
    snapshots = tcp.Snapshots.from_pauli_measurements(
        np.full((2, 16), 3, dtype=np.uint8),
        np.zeros((2, 1, 16), dtype=np.uint8),
    )
    with pytest.raises(MemoryError):
        snapshots.renyi_entropy(subsystem_to_keep=list(range(16)), max_bytes=1 << 30)


def test_shadow_seeded_calls_are_concurrent_and_replayable() -> None:
    state = np.zeros(64, dtype=np.complex128)
    state[0] = 1.0

    def sample() -> tcp.Snapshots:
        return tcp.Snapshots.sample(
            state, nsettings=8, repeats=1, protocol="clifford", seed=91
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(lambda _: sample(), range(2))
    first_arrays = first.to_arrays()
    second_arrays = second.to_arrays()
    np.testing.assert_array_equal(
        first_arrays.frame_x_words, second_arrays.frame_x_words
    )
    np.testing.assert_array_equal(
        first_arrays.frame_z_words, second_arrays.frame_z_words
    )
    np.testing.assert_array_equal(first_arrays.outcomes, second_arrays.outcomes)


def test_invalid_shadow_inputs_fail_at_the_boundary() -> None:
    with pytest.raises(TypeError):
        tcp.Snapshots.from_pauli_measurements(
            np.array([[1.0]], dtype=np.float64), np.zeros((1, 1, 1), dtype=np.uint8)
        )
    with pytest.raises(ValueError):
        tcp.Snapshots.sample(
            np.array([1.0, 0.0], dtype=np.float64),
            nsettings=1,
            protocol="clifford",
            seed=-1,
        )
    with pytest.raises(ValueError):
        tcp.Snapshots.from_clifford_measurements(
            np.array([[1]], dtype=np.uint64),
            np.array([[0]], dtype=np.uint64),
            np.array([[2]], dtype=np.int8),
            np.zeros((1, 1, 1), dtype=np.uint8),
            nqubits=1,
        )
