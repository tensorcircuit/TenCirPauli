"""grouping deterministic grouping, compatibility, and reconstruction tests."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from reference import commutes

from tencirpauli import (
    GeneralCommutingGroupingResult,
    PauliOperator,
    QWCGroupingResult,
)


def qwc_compatible(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(a == 0 or b == 0 or a == b for a, b in zip(left, right))


def test_qwc_groups_have_stable_membership_basis_and_metadata() -> None:
    operator = PauliOperator.from_terms(
        2,
        (("XX", 1.0), ("ZZ", 2.0), ("XI", 3.0), ("IZ", 4.0), ("II", 5.0)),
    )
    result = operator.group_commuting()
    assert isinstance(result, QWCGroupingResult)
    assert result.mode == "qubit_wise"
    assert result.measurement_ready is True
    assert result.algorithm == "largest_first"
    assert result.groups == operator.group_commuting().groups
    assert sorted(index for group in result.groups for index in group) == list(range(5))
    structures = tuple(term.word.to_codes() for term in operator.terms)
    for group, basis in zip(result.groups, result.bases):
        for left_position, left in enumerate(group):
            for right in group[left_position + 1 :]:
                assert qwc_compatible(structures[left], structures[right])
        for qubit, code in enumerate(basis):
            assert code in range(4)
            assert all(structures[index][qubit] in (0, code) for index in group)


def test_qwc_reconstruction_matches_term_eigenvalues() -> None:
    operator = PauliOperator.from_terms(2, (("XI", 1.0), ("IX", 2.0), ("II", 3.0)))
    result = operator.group_commuting()
    assert isinstance(result, QWCGroupingResult)
    assert result.groups == ((0, 1, 2),)
    bits = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int8)
    np.testing.assert_array_equal(
        result.reconstruct(0, bits),
        np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]], dtype=np.int8),
    )
    with pytest.raises(ValueError, match="only 0 and 1"):
        result.reconstruct(0, np.asarray([[2, 0]], dtype=np.int8))
    with pytest.raises(TypeError, match="NumPy int8"):
        result.reconstruct(0, np.asarray([[0.5, 0.0]], dtype=np.float64))
    with pytest.raises(TypeError, match="NumPy int8"):
        result.reconstruct(0, [[0, 0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        result.reconstruct(0, np.zeros((1, 1), dtype=np.int8))
    with pytest.raises(ValueError, match="C-contiguous"):
        result.reconstruct(0, np.zeros((2, 4), dtype=np.int8)[:, ::2])


@pytest.mark.parametrize(
    ("nqubits", "shots", "term_count", "support_density"),
    ((3, 7, 8, 0.25), (70, 11, 16, 0.15), (70, 23, 32, 0.75)),
)
def test_qwc_reconstruction_matches_independent_numpy_oracle(
    nqubits: int, shots: int, term_count: int, support_density: float
) -> None:
    rng = np.random.default_rng(20260806 + nqubits + term_count)
    codes = np.where(
        rng.random((term_count, nqubits)) < support_density,
        rng.integers(1, 4, size=(term_count, nqubits)),
        0,
    ).astype(np.uint8)
    for index in range(term_count):
        codes[index, index % nqubits] = 3
    operator = PauliOperator.from_code_arrays(codes, np.arange(1, term_count + 1))
    result = operator.group_commuting()
    bits = rng.integers(0, 2, size=(shots, nqubits), dtype=np.int8)
    canonical_codes = np.asarray(
        [term.word.to_codes() for term in operator.terms], dtype=np.uint8
    )

    for group_index, group in enumerate(result.groups):
        expected = np.empty((shots, len(group)), dtype=np.int8)
        for column, term_index in enumerate(group):
            support = canonical_codes[term_index] != 0
            parity = np.sum(bits[:, support], axis=1) & 1
            expected[:, column] = np.where(parity == 0, 1, -1)
        np.testing.assert_array_equal(result.reconstruct(group_index, bits), expected)


def test_qwc_reconstruction_releases_gil_for_large_workload() -> None:
    rng = np.random.default_rng(20260806)
    nqubits, term_count, shots = 128, 48, 8_000
    codes = np.where(
        rng.random((term_count, nqubits)) < 0.5,
        3,
        0,
    ).astype(np.uint8)
    for index in range(term_count):
        codes[index, index] = 3
    result = PauliOperator.from_code_arrays(codes, np.ones(term_count))
    grouping = result.group_commuting()
    bits = rng.integers(0, 2, size=(shots, nqubits), dtype=np.int8)
    progress = [0]
    ready = threading.Event()
    stop = threading.Event()

    def observe() -> None:
        ready.set()
        while not stop.is_set():
            progress[0] += 1

    observer = threading.Thread(target=observe)
    observer.start()
    assert ready.wait(2.0)
    try:
        grouping.reconstruct(0, bits)
    finally:
        stop.set()
        observer.join(2.0)
    assert not observer.is_alive()
    assert progress[0] > 0


def test_general_commuting_is_separate_and_not_measurement_ready() -> None:
    operator = PauliOperator.from_terms(2, (("XX", 1.0), ("ZZ", 2.0), ("YY", 3.0)))
    result = operator.group_commuting(mode="general", algorithm="dsatur")
    assert isinstance(result, GeneralCommutingGroupingResult)
    assert result.mode == "general"
    assert result.measurement_ready is False
    assert result.groups == ((0, 1, 2),)
    structures = tuple(term.word.to_codes() for term in operator.terms)
    for group in result.groups:
        for left_position, left in enumerate(group):
            for right in group[left_position + 1 :]:
                assert commutes(structures[left], structures[right])
    assert operator.group_commuting(mode="qubit_wise").groups != result.groups


def test_grouping_rejects_unknown_options_and_empty_is_deterministic() -> None:
    with pytest.raises(ValueError, match="mode"):
        PauliOperator.empty(2).group_commuting(mode="invalid")
    with pytest.raises(ValueError, match="algorithm"):
        PauliOperator.empty(2).group_commuting(algorithm="invalid")
    with pytest.raises(ValueError, match="non-negative integer"):
        PauliOperator.empty(2).group_commuting(max_matrix_entries=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        PauliOperator.empty(2).compatibility_matrix(max_entries=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        PauliOperator.empty(2).incompatibility_edges(max_edges=True)
    empty = PauliOperator.empty(4).group_commuting()
    assert empty.groups == ()


def test_bounded_dense_and_streaming_compatibility_kernels_agree() -> None:
    operator = PauliOperator.from_terms(
        2, (("XX", 1.0), ("ZZ", 2.0), ("XI", 3.0), ("IZ", 4.0))
    )
    matrix = operator.compatibility_matrix(mode="qubit_wise", max_entries=16)
    assert matrix.shape == (4, 4)
    assert np.all(np.diag(matrix))
    edges = operator.incompatibility_edges(mode="qubit_wise", max_edges=16)
    assert all(not matrix[left, right] for left, right in edges)
    assert len(edges) == int(np.count_nonzero(np.triu(~matrix, 1)))
    with pytest.raises(MemoryError, match="requested"):
        operator.compatibility_matrix(mode="general", max_entries=1)
    with pytest.raises(MemoryError, match="requested"):
        operator.incompatibility_edges(mode="qubit_wise", max_edges=0)
