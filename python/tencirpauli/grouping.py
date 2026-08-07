"""Deterministic QWC and general-commuting Pauli grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, cast

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
    term_to_group: Tuple[int, ...]
    algorithm: str
    group_count: int
    term_count: int
    measurement_ready: bool
    mode: str
    _native_handle: Optional[_native.NativeQwcGroupingHandle]

    def __init__(
        self,
        nqubits: int,
        groups: Tuple[Tuple[int, ...], ...],
        bases: Tuple[Tuple[int, ...], ...],
        algorithm: str,
        native_handle: Optional[_native.NativeQwcGroupingHandle] = None,
    ) -> None:
        if native_handle is None:
            raise RuntimeError("QWC grouping results require a native handle")
        normalized_groups = tuple(
            tuple(int(index) for index in group) for group in groups
        )
        if len(bases) != len(normalized_groups):
            raise ValueError("group metadata must have one entry per group")
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
        object.__setattr__(self, "bases", tuple(tuple(basis) for basis in bases))
        object.__setattr__(self, "term_to_group", tuple(term_to_group))
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "group_count", len(normalized_groups))
        object.__setattr__(self, "term_count", term_count)
        object.__setattr__(self, "measurement_ready", True)
        object.__setattr__(self, "mode", "qubit_wise")
        object.__setattr__(self, "_native_handle", native_handle)

    def reconstruct(
        self, group_index: int, bitstrings: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Reconstruct eigenvalues for one QWC group from rotated samples.

        Args:
            group_index: Index of the group whose terms should be reconstructed.
            bitstrings: A C-contiguous ``numpy.int8`` array with shape
                ``(shots, nqubits)`` after applying the group's basis
                rotations. Values must be binary; validation runs in Rust.

        Returns:
            An ``int8`` array of shape ``(shots, group_size)`` containing
            eigenvalues in ``{-1, +1}``, in canonical term order.

        Raises:
            IndexError: If ``group_index`` is not a valid group.
            TypeError: If ``bitstrings`` is not a C-contiguous ``numpy.int8``
                array.
            ValueError: If the sample shape or binary-value contract is invalid.
        """
        if not 0 <= group_index < len(self.groups):
            raise IndexError(f"group index {group_index} is out of range")
        if not isinstance(bitstrings, np.ndarray):
            raise TypeError("bitstrings must be a C-contiguous NumPy int8 array")
        if bitstrings.ndim != 2 or bitstrings.shape[1] != self.nqubits:
            raise ValueError(
                f"bitstrings must have shape (shots, {self.nqubits}), got {bitstrings.shape}"
            )
        if bitstrings.dtype != np.dtype(np.int8):
            raise TypeError("bitstrings must be a C-contiguous NumPy int8 array")
        if not bitstrings.flags.c_contiguous:
            raise ValueError("bitstrings must be C-contiguous")
        native_handle = self._native_handle
        if native_handle is None:
            raise RuntimeError("QWC grouping results must retain a native handle")
        shots, group_size, flat = native_handle.reconstruct(group_index, bitstrings)
        return cast(
            np.ndarray[Any, Any],
            np.asarray(flat, dtype=np.int8).reshape((shots, group_size)),
        )


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
    handle = operator._native_handle
    if handle is None:
        raise RuntimeError("Pauli operators must retain native handles")
    if mode == "qubit_wise":
        native_groups, bases_raw, grouping_handle = _native.pauli_qwc_group_handle(
            handle, algorithm_code, max_matrix_entries
        )
    else:
        native_groups, bases_raw, _ = _native.pauli_group_handle(
            handle, mode_code, algorithm_code, max_matrix_entries
        )
    groups: Tuple[Tuple[int, ...], ...] = tuple(
        tuple(int(index) for index in group) for group in native_groups
    )
    if mode == "general":
        return GeneralCommutingGroupingResult(operator.nqubits, groups, algorithm)
    native_bases = tuple(tuple(int(code) for code in basis) for basis in bases_raw)
    return QWCGroupingResult(
        operator.nqubits,
        groups,
        native_bases,
        algorithm,
        grouping_handle,
    )
