"""Deterministic fermion-to-qubit occupation mappings.

The public plans in this module keep the occupation convention explicit.  A
Jordan-Wigner Pauli operator is transformed by the binary linear change of
computational-basis coordinates for parity and Fenwick Bravyi-Kitaev plans;
the same path is therefore easy to compare with an independently constructed
encoded Fock-space matrix.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from .hamiltonian import DEFAULT_MAX_BYTES
from .pauli import PauliOperator
from .structured import _PAULI_PRODUCT, HybridOperator, _Term


_CONVENTION = "tencirpauli.gf2_occupation.v1"
_BASIS_ORDERING = "qubit0_msb_matrix"
_SCHEMA_VERSION = 1


def _exact_nonnegative(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_matrix(
    rows: Sequence[Sequence[int]], n_modes: int
) -> Tuple[Tuple[int, ...], ...]:
    normalized = tuple(tuple(int(value) for value in row) for row in rows)
    if len(normalized) != n_modes or any(len(row) != n_modes for row in normalized):
        raise ValueError("encoding matrix must be square with shape (n_modes, n_modes)")
    if any(value not in (0, 1) for row in normalized for value in row):
        raise ValueError("encoding matrix entries must be binary")
    return normalized


def _gf2_inverse(matrix: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
    n_modes = len(matrix)
    augmented = [
        list(row) + [1 if row_index == column else 0 for column in range(n_modes)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(n_modes):
        pivot = next(
            (row for row in range(column, n_modes) if augmented[row][column]), None
        )
        if pivot is None:
            raise ValueError("encoding matrix must be invertible over GF(2)")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(n_modes):
            if row != column and augmented[row][column]:
                augmented[row] = [
                    left ^ right
                    for left, right in zip(augmented[row], augmented[column])
                ]
    return tuple(tuple(row[n_modes:]) for row in augmented)


def _canonical_cnot_operations(
    matrix: Tuple[Tuple[int, ...], ...],
) -> Tuple[Tuple[int, int], ...]:
    """Return the frozen row-reduction provenance for a triangular matrix."""
    n_modes = len(matrix)
    rows = [list(row) for row in matrix]
    reductions: List[Tuple[int, int]] = []
    for pivot in range(n_modes):
        if rows[pivot][pivot] != 1:
            raise ValueError("supported occupation matrices must be unit triangular")
        for target in range(pivot + 1, n_modes):
            if rows[target][pivot]:
                rows[target] = [
                    left ^ right for left, right in zip(rows[target], rows[pivot])
                ]
                reductions.append((pivot, target))
    identity = tuple(
        tuple(1 if row == column else 0 for column in range(n_modes))
        for row in range(n_modes)
    )
    if tuple(tuple(row) for row in rows) != identity:
        raise ValueError("occupation matrix reduction did not reach identity")
    return tuple(reversed(reductions))


def _matrix_to_array(matrix: Tuple[Tuple[int, ...], ...]) -> np.ndarray[Any, Any]:
    result = np.asarray(matrix, dtype=np.uint8)
    if result.size == 0:
        result = np.empty((0, 0), dtype=np.uint8)
    result = np.ascontiguousarray(result)
    result.setflags(write=False)
    return cast(np.ndarray[Any, Any], result)


def _mapping_matrix(name: str, n_modes: int) -> Tuple[Tuple[int, ...], ...]:
    if name == "jordan_wigner":
        return tuple(
            tuple(1 if row == column else 0 for column in range(n_modes))
            for row in range(n_modes)
        )
    if name == "parity":
        return tuple(
            tuple(1 if column <= row else 0 for column in range(n_modes))
            for row in range(n_modes)
        )
    if name == "bravyi_kitaev":
        rows: List[Tuple[int, ...]] = []
        for row in range(n_modes):
            endpoint = row + 1
            lowbit = endpoint & -endpoint
            start = endpoint - lowbit
            rows.append(
                tuple(1 if start <= column <= row else 0 for column in range(n_modes))
            )
        return tuple(rows)
    raise ValueError(
        "mapping must be one of 'jordan_wigner', 'parity', or 'bravyi_kitaev'"
    )


class FermionQubitMapping:
    """Immutable occupation-encoding plan for JW, parity, or BK mapping."""

    __slots__ = (
        "_clifford_operations",
        "_cnot_operations",
        "_encoding",
        "_inverse_encoding",
        "_locked",
        "basis_ordering",
        "convention",
        "estimated_bytes",
        "mapping_name",
        "mode_ordering",
        "n_modes",
        "nqubits",
        "schema_version",
    )
    schema_version: int
    mapping_name: str
    n_modes: int
    nqubits: int
    mode_ordering: str
    basis_ordering: str
    convention: str
    _encoding: np.ndarray[Any, Any]
    _inverse_encoding: np.ndarray[Any, Any]
    _cnot_operations: Tuple[Tuple[int, int], ...]
    _clifford_operations: np.ndarray[Any, Any]
    estimated_bytes: int
    _locked: bool

    def __init__(
        self,
        mapping_name: str,
        n_modes: int,
        encoding: Tuple[Tuple[int, ...], ...],
    ) -> None:
        n_modes = _exact_nonnegative(n_modes, "n_modes")
        if mapping_name not in {"jordan_wigner", "parity", "bravyi_kitaev"}:
            raise ValueError("unsupported fermion-to-qubit mapping")
        matrix = _validate_matrix(encoding, n_modes)
        inverse = _gf2_inverse(matrix)
        cnot_operations = _canonical_cnot_operations(matrix)
        clifford = np.asarray(cnot_operations, dtype=np.int64).reshape((-1, 2))
        clifford = np.ascontiguousarray(clifford)
        clifford.setflags(write=False)
        object.__setattr__(self, "schema_version", _SCHEMA_VERSION)
        object.__setattr__(self, "mapping_name", mapping_name)
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "nqubits", n_modes)
        object.__setattr__(self, "mode_ordering", "mode0_increasing")
        object.__setattr__(self, "basis_ordering", _BASIS_ORDERING)
        object.__setattr__(self, "convention", _CONVENTION)
        object.__setattr__(self, "_encoding", _matrix_to_array(matrix))
        object.__setattr__(self, "_inverse_encoding", _matrix_to_array(inverse))
        object.__setattr__(self, "_cnot_operations", cnot_operations)
        object.__setattr__(self, "_clifford_operations", clifford)
        object.__setattr__(
            self,
            "estimated_bytes",
            n_modes * n_modes * 2 + len(cnot_operations) * 16 + 256,
        )
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("FermionQubitMapping is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def _build(cls, mapping_name: str, n_modes: int) -> "FermionQubitMapping":
        return cls(mapping_name, n_modes, _mapping_matrix(mapping_name, n_modes))

    @classmethod
    def jordan_wigner(cls, n_modes: int) -> "FermionQubitMapping":
        """Build the identity occupation encoding used by Jordan-Wigner."""
        return cls._build("jordan_wigner", n_modes)

    @classmethod
    def parity(cls, n_modes: int) -> "FermionQubitMapping":
        """Build the prefix-parity occupation encoding."""
        return cls._build("parity", n_modes)

    @classmethod
    def bravyi_kitaev(cls, n_modes: int) -> "FermionQubitMapping":
        """Build the frozen Fenwick-interval Bravyi-Kitaev encoding."""
        return cls._build("bravyi_kitaev", n_modes)

    @classmethod
    def from_name(cls, name: str, n_modes: int) -> "FermionQubitMapping":
        """Build a plan from a stable mapping name."""
        if not isinstance(name, str):
            raise TypeError("mapping name must be a string")
        return cls._build(name, n_modes)

    @property
    def name(self) -> str:
        """Alias for the stable mapping name."""
        return self.mapping_name

    @property
    def mapping(self) -> str:
        """Alias retained for metadata-oriented callers."""
        return self.mapping_name

    @property
    def encoding_matrix(self) -> np.ndarray[Any, Any]:
        """Read-only binary matrix ``q = B n (mod 2)``."""
        return self._encoding

    @property
    def inverse_encoding_matrix(self) -> np.ndarray[Any, Any]:
        """Read-only inverse binary occupation transform."""
        return self._inverse_encoding

    @property
    def cnot_operations(self) -> Tuple[Tuple[int, int], ...]:
        """Canonical ``(control, target)`` provenance for the linear transform."""
        return self._cnot_operations

    @property
    def clifford_operations(self) -> np.ndarray[Any, Any]:
        """Read-only CNOT array in deterministic synthesis order."""
        return self._clifford_operations

    def encode_occupation(self, occupation: Sequence[int]) -> Tuple[int, ...]:
        """Encode one occupation vector with the frozen GF(2) convention."""
        values = tuple(occupation)
        if len(values) != self.n_modes or any(value not in (0, 1) for value in values):
            raise ValueError("occupation must be a binary vector of length n_modes")
        return tuple(
            sum(row[column] * values[column] for column in range(self.n_modes)) & 1
            for row in self._encoding
        )

    def _transform_codes_with_phase(
        self, codes: Sequence[int]
    ) -> Tuple[Tuple[int, ...], complex]:
        values = tuple(codes)
        if len(values) != self.n_modes or any(
            not isinstance(code, int) or isinstance(code, bool) or code not in range(4)
            for code in values
        ):
            raise ValueError("Pauli codes must have length n_modes and lie in 0..3")
        result = list(values)
        phase = 1.0 + 0j
        control_images = ((0, 0), (1, 1), (2, 1), (3, 0))
        target_images = ((0, 0), (0, 1), (3, 2), (3, 3))
        for control, target in self._cnot_operations:
            control_code, target_code = control_images[result[control]]
            image_control, image_target = target_images[result[target]]
            result[control], local_phase = _PAULI_PRODUCT[control_code][image_control]
            phase *= local_phase
            result[target], local_phase = _PAULI_PRODUCT[target_code][image_target]
            phase *= local_phase
        if phase not in (1.0 + 0j, -1.0 + 0j):
            raise RuntimeError("internal CNOT conjugation produced a non-real phase")
        return tuple(result), phase

    def _transform_codes(self, codes: Sequence[int]) -> Tuple[int, ...]:
        """Transform a word and discard the exact sign for diagnostics."""
        result, _ = self._transform_codes_with_phase(codes)
        return result

    def _transform_prefix(
        self, codes: Sequence[int], prefix_length: int
    ) -> Tuple[Tuple[int, ...], complex]:
        if prefix_length != self.n_modes or len(codes) < prefix_length:
            raise ValueError("mapping plan and fermion axis count are incompatible")
        transformed, phase = self._transform_codes_with_phase(
            tuple(codes[:prefix_length])
        )
        return transformed + tuple(codes[prefix_length:]), phase

    def map_pauli(self, operator: PauliOperator) -> PauliOperator:
        """Conjugate a pure fermion-axis Pauli operator by the encoding CNOTs."""
        if not isinstance(operator, PauliOperator):
            raise TypeError("map_pauli expects a PauliOperator")
        if operator.nqubits != self.n_modes:
            raise ValueError("mapping plan and Pauli qubit count are incompatible")
        return PauliOperator.from_terms(
            self.n_modes,
            (
                (transformed, term.coefficient * phase)
                for term in operator.terms
                for transformed, phase in (
                    self._transform_codes_with_phase(term.word.to_codes()),
                )
            ),
        )

    def map_fermion_operator(
        self, operator: Any, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> PauliOperator:
        """Map a pure :class:`FermionOperator` through one batched JW result."""
        from .structured import FermionOperator

        if not isinstance(operator, FermionOperator):
            raise TypeError("mapping expects a FermionOperator")
        if operator.space.fermions != self.n_modes:
            raise ValueError("mapping plan and fermion mode counts are incompatible")
        jordan_wigner = operator.map_fermions("jordan_wigner", max_bytes=max_bytes)
        return self.map_pauli(jordan_wigner)

    def map_hybrid(
        self, operator: HybridOperator, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> Union[HybridOperator, PauliOperator]:
        """Map only fermion axes of a compatible structured hybrid operator."""
        if not isinstance(operator, HybridOperator):
            raise TypeError("mapping expects a HybridOperator")
        if operator.space.fermions != self.n_modes:
            raise ValueError("mapping plan and fermion axis counts are incompatible")
        jordan_wigner = operator.map_fermions("jordan_wigner", max_bytes=max_bytes)
        if isinstance(jordan_wigner, PauliOperator):
            return PauliOperator.from_terms(
                jordan_wigner.nqubits,
                (
                    (
                        transformed,
                        term.coefficient * phase,
                    )
                    for term in jordan_wigner.terms
                    for transformed, phase in (
                        self._transform_prefix(term.word.to_codes(), self.n_modes),
                    )
                ),
            )
        terms = []
        for term in jordan_wigner._terms:
            mapped = term.mapped_fermion
            if mapped is None:
                transformed = None
                phase = 1.0 + 0j
            else:
                transformed, phase = self._transform_codes_with_phase(mapped)
            terms.append(
                _Term(
                    term.fermion,
                    term.boson,
                    term.qubit,
                    term.qudit,
                    transformed,
                    term.coefficient * phase,
                )
            )
        return HybridOperator._from_terms(jordan_wigner.space, terms, max_bytes)

    def __repr__(self) -> str:
        return f"FermionQubitMapping({self.mapping_name!r}, n_modes={self.n_modes})"
