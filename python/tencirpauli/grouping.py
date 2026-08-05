"""Deterministic QWC and general-commuting Pauli grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import numpy as np

from . import _native
from ._validation import validate_nonnegative_int
from .pauli import PauliOperator


@dataclass(frozen=True, init=False)
class QWCGroupingResult:
    """A deterministic qubit-wise commuting measurement partition.

    ``groups`` contains indices into the canonical operator terms. ``bases``
    gives the required single-qubit basis code per group, where ``0`` means
    identity and ``1/2/3`` mean X/Y/Z. Use :meth:`reconstruct` to convert
    computational-basis samples after that basis rotation into term values.
    """

    nqubits: int
    groups: Tuple[Tuple[int, ...], ...]
    bases: Tuple[Tuple[int, ...], ...]
    reconstruction_masks: Tuple[Tuple[int, ...], ...]
    term_to_group: Tuple[int, ...]
    algorithm: str
    group_count: int
    term_count: int
    measurement_ready: bool
    mode: str

    def __init__(
        self,
        nqubits: int,
        groups: Tuple[Tuple[int, ...], ...],
        bases: Tuple[Tuple[int, ...], ...],
        reconstruction_masks: Tuple[Tuple[int, ...], ...],
        algorithm: str,
    ) -> None:
        normalized_groups = tuple(
            tuple(int(index) for index in group) for group in groups
        )
        if len(bases) != len(normalized_groups) or len(reconstruction_masks) != len(
            normalized_groups
        ):
            raise ValueError("group metadata must have one entry per group")
        term_count = sum(len(group) for group in normalized_groups)
        flattened = tuple(index for group in normalized_groups for index in group)
        if sorted(flattened) != list(range(term_count)):
            raise ValueError("groups must cover each canonical term exactly once")
        if any(
            len(mask) != len(group)
            for mask, group in zip(reconstruction_masks, normalized_groups)
        ):
            raise ValueError("reconstruction metadata must match group sizes")
        term_to_group = [-1] * term_count
        for group_index, group in enumerate(normalized_groups):
            for term_index in group:
                term_to_group[term_index] = group_index
        object.__setattr__(self, "nqubits", int(nqubits))
        object.__setattr__(self, "groups", normalized_groups)
        object.__setattr__(self, "bases", tuple(tuple(basis) for basis in bases))
        object.__setattr__(
            self,
            "reconstruction_masks",
            tuple(tuple(int(mask) for mask in masks) for masks in reconstruction_masks),
        )
        object.__setattr__(self, "term_to_group", tuple(term_to_group))
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "group_count", len(normalized_groups))
        object.__setattr__(self, "term_count", term_count)
        object.__setattr__(self, "measurement_ready", True)
        object.__setattr__(self, "mode", "qubit_wise")

    def reconstruct(
        self, group_index: int, bitstrings: Sequence[Sequence[int]]
    ) -> np.ndarray[Any, Any]:
        """Reconstruct eigenvalues for one QWC group from rotated samples.

        Args:
            group_index: Index of the group whose terms should be reconstructed.
            bitstrings: Binary samples with shape ``(shots, nqubits)`` after
                applying the group's basis rotations.

        Returns:
            An ``int8`` array of shape ``(shots, group_size)`` containing
            eigenvalues in ``{-1, +1}``, in canonical term order.

        Raises:
            IndexError: If ``group_index`` is not a valid group.
            ValueError: If the sample shape or binary-value contract is invalid.
        """
        if not 0 <= group_index < len(self.groups):
            raise IndexError(f"group index {group_index} is out of range")
        raw_values = np.asarray(bitstrings)
        if raw_values.ndim != 2 or raw_values.shape[1] != self.nqubits:
            raise ValueError(
                f"bitstrings must have shape (shots, {self.nqubits}), got {raw_values.shape}"
            )
        if raw_values.dtype.kind not in "biuf" or np.any(
            (raw_values != 0) & (raw_values != 1)
        ):
            raise ValueError("bitstrings must contain only 0 and 1")
        values = np.asarray(raw_values, dtype=np.int8)
        output: np.ndarray[Any, Any] = np.empty(
            (values.shape[0], len(self.groups[group_index])), dtype=np.int8
        )
        for column, mask in enumerate(self.reconstruction_masks[group_index]):
            support = tuple(index for index in range(self.nqubits) if mask >> index & 1)
            parity = np.mod(values[:, support].sum(axis=1), 2) if support else 0
            output[:, column] = 1 - 2 * parity
        return output


@dataclass(frozen=True, init=False)
class GeneralCommutingGroupingResult:
    """A deterministic algebraic commuting partition.

    Terms in each group commute as operators, but the result intentionally does
    not provide a common tensor-product measurement basis. Use this mode for
    algebraic grouping or downstream measurement schemes that handle general
    commuting sets separately.
    """

    nqubits: int
    groups: Tuple[Tuple[int, ...], ...]
    term_to_group: Tuple[int, ...]
    algorithm: str
    group_count: int
    term_count: int
    measurement_ready: bool
    mode: str

    def __init__(
        self,
        nqubits: int,
        groups: Tuple[Tuple[int, ...], ...],
        algorithm: str,
    ) -> None:
        normalized_groups = tuple(
            tuple(int(index) for index in group) for group in groups
        )
        term_count = sum(len(group) for group in normalized_groups)
        flattened = tuple(index for group in normalized_groups for index in group)
        if sorted(flattened) != list(range(term_count)):
            raise ValueError("groups must cover each canonical term exactly once")
        term_to_group = [-1] * term_count
        for group_index, group in enumerate(normalized_groups):
            for term_index in group:
                term_to_group[term_index] = group_index
        object.__setattr__(self, "nqubits", int(nqubits))
        object.__setattr__(self, "groups", normalized_groups)
        object.__setattr__(self, "term_to_group", tuple(term_to_group))
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "group_count", len(normalized_groups))
        object.__setattr__(self, "term_count", term_count)
        object.__setattr__(self, "measurement_ready", False)
        object.__setattr__(self, "mode", "general")


GroupingResult = Union[QWCGroupingResult, GeneralCommutingGroupingResult]


def group_operator(
    operator: PauliOperator,
    mode: str = "qubit_wise",
    algorithm: str = "largest_first",
    max_matrix_entries: int = 10_000_000,
) -> GroupingResult:
    """Group a Pauli operator with deterministic ordering and bounded workspace.

    Args:
        operator: Canonical Pauli operator whose terms are grouped.
        mode: ``"qubit_wise"`` for jointly measurable QWC groups or
            ``"general"`` for algebraically commuting groups.
        algorithm: ``"largest_first"`` or ``"dsatur"`` graph-coloring order.
        max_matrix_entries: Maximum compatibility-matrix entries allowed by
            the grouping preflight.

    Returns:
        A :class:`QWCGroupingResult` for ``qubit_wise`` mode or a
        :class:`GeneralCommutingGroupingResult` for ``general`` mode.

    Raises:
        ValueError: If ``mode`` or ``algorithm`` is unsupported.
        MemoryError: If the compatibility workspace exceeds the entry bound.
    """
    validate_nonnegative_int(max_matrix_entries, "max_matrix_entries")
    mode_code = {"qubit_wise": 0, "general": 1}.get(mode)
    algorithm_code = {"largest_first": 0, "dsatur": 1}.get(algorithm)
    if mode_code is None:
        raise ValueError("mode must be 'qubit_wise' or 'general'")
    if algorithm_code is None:
        raise ValueError("algorithm must be 'largest_first' or 'dsatur'")
    if operator._native_handle is not None:
        native_groups, bases_raw, supports = _native.pauli_group_handle(
            operator._native_handle, mode_code, algorithm_code, max_matrix_entries
        )
        normalized_groups = tuple(
            tuple(int(index) for index in group) for group in native_groups
        )
        if mode == "general":
            return GeneralCommutingGroupingResult(
                operator.nqubits, normalized_groups, algorithm
            )
        native_bases = tuple(tuple(int(code) for code in basis) for basis in bases_raw)
        native_masks = tuple(
            tuple(sum(1 << int(qubit) for qubit in support) for support in group)
            for group in supports
        )
        return QWCGroupingResult(
            operator.nqubits,
            normalized_groups,
            native_bases,
            native_masks,
            algorithm,
        )
    structures = operator._arrays()[0]
    size = len(structures)
    if size * size > max_matrix_entries:
        raise MemoryError(
            f"grouping requires {size * size} compatibility entries, "
            f"exceeding max_matrix_entries={max_matrix_entries}"
        )
    raw_groups = _native.pauli_group(
        operator.nqubits,
        structures,
        mode_code,
        algorithm_code,
        max_matrix_entries,
    )
    groups: Tuple[Tuple[int, ...], ...] = tuple(
        tuple(int(index) for index in group) for group in raw_groups
    )
    if mode == "general":
        return GeneralCommutingGroupingResult(operator.nqubits, groups, algorithm)
    basis_values: list[tuple[int, ...]] = []
    mask_values: list[tuple[int, ...]] = []
    for group in groups:
        basis = [0] * operator.nqubits
        group_masks = []
        for index in group:
            codes = structures[index]
            mask = 0
            for qubit, code in enumerate(codes):
                if code:
                    mask |= 1 << qubit
                    if basis[qubit] == 0:
                        basis[qubit] = code
                    elif basis[qubit] != code:
                        raise RuntimeError(
                            "native QWC grouping returned incompatible terms"
                        )
            group_masks.append(mask)
        basis_values.append(tuple(basis))
        mask_values.append(tuple(group_masks))
    return QWCGroupingResult(
        operator.nqubits,
        groups,
        tuple(basis_values),
        tuple(mask_values),
        algorithm,
    )
