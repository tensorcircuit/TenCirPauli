"""Deterministic QWC and general-commuting Pauli grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import numpy as np

from . import _native
from ._validation import validate_nonnegative_int
from .pauli import PauliOperator


@dataclass(frozen=True)
class QWCGroupingResult:
    """A locally measurable qubit-wise commuting partition."""

    nqubits: int
    groups: Tuple[Tuple[int, ...], ...]
    bases: Tuple[Tuple[int, ...], ...]
    reconstruction_masks: Tuple[Tuple[int, ...], ...]
    coefficient_mapping: Tuple[Tuple[int, ...], ...]
    algorithm: str
    measurement_ready: bool = True
    mode: str = "qubit_wise"

    @property
    def nterms(self) -> int:
        """Return the number of canonical terms represented by this result."""
        return sum(len(group) for group in self.groups)

    def reconstruct(
        self, group_index: int, bitstrings: Sequence[Sequence[int]]
    ) -> np.ndarray:
        """Reconstruct eigenvalues for one group from rotated measurement bits."""
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


@dataclass(frozen=True)
class GeneralCommutingGroupingResult:
    """An algebraic commuting partition without a joint-measurement plan."""

    nqubits: int
    groups: Tuple[Tuple[int, ...], ...]
    coefficient_mapping: Tuple[Tuple[int, ...], ...]
    algorithm: str
    measurement_ready: bool = False
    mode: str = "general"

    @property
    def nterms(self) -> int:
        """Return the number of canonical terms represented by this result."""
        return sum(len(group) for group in self.groups)


GroupingResult = Union[QWCGroupingResult, GeneralCommutingGroupingResult]


def group_operator(
    operator: PauliOperator,
    mode: str = "qubit_wise",
    algorithm: str = "largest_first",
    max_matrix_entries: int = 10_000_000,
) -> GroupingResult:
    """Group terms with a deterministic native call and entry bound."""
    validate_nonnegative_int(max_matrix_entries, "max_matrix_entries")
    mode_code = {"qubit_wise": 0, "general": 1}.get(mode)
    algorithm_code = {"largest_first": 0, "dsatur": 1}.get(algorithm)
    if mode_code is None:
        raise ValueError("mode must be 'qubit_wise' or 'general'")
    if algorithm_code is None:
        raise ValueError("algorithm must be 'largest_first' or 'dsatur'")
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
    groups = tuple(tuple(group) for group in raw_groups)
    mapping = groups
    if mode == "general":
        return GeneralCommutingGroupingResult(
            operator.nqubits, groups, mapping, algorithm
        )
    bases = []
    masks = []
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
        bases.append(tuple(basis))
        masks.append(tuple(group_masks))
    return QWCGroupingResult(
        operator.nqubits,
        groups,
        tuple(bases),
        tuple(masks),
        mapping,
        algorithm,
    )
