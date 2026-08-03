"""P3 deterministic grouping, compatibility, and reconstruction tests."""

from __future__ import annotations

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
        result.reconstruct(0, [[2, 0]])
    with pytest.raises(ValueError, match="only 0 and 1"):
        result.reconstruct(0, [[0.5, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        result.reconstruct(0, [[0]])


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
