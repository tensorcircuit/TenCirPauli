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

from . import _native
from ._validation import validate_nonnegative_int
from .hamiltonian import (
    DEFAULT_MAX_BYTES,
    _check_allocation,
    _effective_max_bytes,
    _validate_max_bytes,
)
from .pauli import PauliOperator
from .structured import (
    _PAULI_PRODUCT,
    HybridOperator,
    _hybrid_arrays,
    _hybrid_from_native,
    _Term,
)


_CONVENTION = "tencirpauli.gf2_occupation.v1"
_BASIS_ORDERING = "qubit0_msb_matrix"
_SCHEMA_VERSION = 1


_exact_nonnegative = validate_nonnegative_int


def _mapping_plan_upper_bound(n_modes: int) -> int:
    """Return a checked-before-allocation upper bound for one mapping plan."""
    max_cnot_count = n_modes * (n_modes - 1) // 2
    native_bytes = _mapping_plan_native_bytes(n_modes, max_cnot_count)
    # The public object eagerly retains two uint8 diagnostic matrices, the
    # tuple-of-pairs provenance, and its int64 CNOT mirror.  Count their
    # logical payloads once; Python headers and allocator slack remain outside
    # the best-effort contract.
    public_bytes = 2 * n_modes * n_modes + 2 * 16 * max_cnot_count
    return native_bytes + public_bytes


def _mapping_plan_native_bytes(n_modes: int, cnot_count: int) -> int:
    """Return the cheap logical payload estimate for the retained native plan."""
    packed_words = (n_modes + 63) // 64
    packed_bytes = 2 * n_modes * packed_words * 8
    return 2 * n_modes * n_modes + packed_bytes + 16 * cnot_count + 256


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
        "_native_plan",
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
    _native_plan: Optional[Any]
    _locked: bool

    def __init__(
        self,
        mapping_name: str,
        n_modes: int,
        encoding: Tuple[Tuple[int, ...], ...],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
        *,
        _native_plan: Optional[Any] = None,
    ) -> None:
        n_modes = _exact_nonnegative(n_modes, "n_modes")
        if mapping_name not in {"jordan_wigner", "parity", "bravyi_kitaev"}:
            raise ValueError("unsupported fermion-to-qubit mapping")
        _validate_max_bytes(max_bytes)
        _check_allocation(
            _mapping_plan_upper_bound(n_modes),
            max_bytes,
            "fermion mapping plan",
        )
        matrix = _validate_matrix(encoding, n_modes)
        if _native_plan is None:
            inverse = _gf2_inverse(matrix)
            cnot_operations = _canonical_cnot_operations(matrix)
            native_bytes = _mapping_plan_native_bytes(n_modes, len(cnot_operations))
        else:
            inverse = _validate_matrix(_native_plan.inverse_encoding, n_modes)
            cnot_operations = tuple(
                (int(control), int(target))
                for control, target in _native_plan.cnot_operations
            )
            native_bytes = int(_native_plan.estimated_bytes)
        clifford = np.asarray(cnot_operations, dtype=np.int64).reshape((-1, 2))
        clifford = np.ascontiguousarray(clifford)
        clifford.setflags(write=False)
        public_matrix_bytes = 2 * n_modes * n_modes
        public_cnot_bytes = len(cnot_operations) * 16
        actual_bytes = native_bytes + public_matrix_bytes + 2 * public_cnot_bytes
        # Keep the public estimate equal to the same cheap preflight upper
        # bound used before the native plan is built. It is intentionally a
        # little loose for JW/BK, which avoids an extra mapping-specific scan
        # and guarantees that a budget at the documented estimate succeeds.
        estimated_bytes = max(actual_bytes, _mapping_plan_upper_bound(n_modes))
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
        object.__setattr__(self, "estimated_bytes", estimated_bytes)
        object.__setattr__(self, "_native_plan", _native_plan)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("FermionQubitMapping is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def _build(
        cls,
        mapping_name: str,
        n_modes: int,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionQubitMapping":
        normalized_modes = _exact_nonnegative(n_modes, "n_modes")
        if mapping_name not in {"jordan_wigner", "parity", "bravyi_kitaev"}:
            raise ValueError("unsupported fermion-to-qubit mapping")
        _validate_max_bytes(max_bytes)
        _check_allocation(
            _mapping_plan_upper_bound(normalized_modes),
            max_bytes,
            "fermion mapping plan",
        )
        native_plan = _native.mapping_plan(
            mapping_name,
            normalized_modes,
            _effective_max_bytes(max_bytes),
        )
        encoding = tuple(
            tuple(int(value) for value in row) for row in native_plan.encoding
        )
        return cls(
            mapping_name,
            normalized_modes,
            encoding,
            max_bytes=max_bytes,
            _native_plan=native_plan,
        )

    @classmethod
    def jordan_wigner(
        cls,
        n_modes: int,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionQubitMapping":
        """Build the identity occupation encoding used by Jordan-Wigner."""
        return cls._build("jordan_wigner", n_modes, max_bytes=max_bytes)

    @classmethod
    def parity(
        cls,
        n_modes: int,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionQubitMapping":
        """Build the prefix-parity occupation encoding."""
        return cls._build("parity", n_modes, max_bytes=max_bytes)

    @classmethod
    def bravyi_kitaev(
        cls,
        n_modes: int,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionQubitMapping":
        """Build the frozen Fenwick-interval Bravyi-Kitaev encoding."""
        return cls._build("bravyi_kitaev", n_modes, max_bytes=max_bytes)

    @classmethod
    def from_name(
        cls,
        name: str,
        n_modes: int,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionQubitMapping":
        """Build a plan from a stable mapping name."""
        if not isinstance(name, str):
            raise TypeError("mapping name must be a string")
        return cls._build(name, n_modes, max_bytes=max_bytes)

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
            raise ValueError(
                "Pauli codes must have length n_modes and lie in 0..3 (inclusive)"
            )
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

    def map_pauli(
        self,
        operator: PauliOperator,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> PauliOperator:
        """Conjugate a pure fermion-axis Pauli operator by the encoding CNOTs."""
        if not isinstance(operator, PauliOperator):
            raise TypeError("map_pauli expects a PauliOperator")
        if operator.nqubits != self.n_modes:
            raise ValueError("mapping plan and Pauli qubit count are incompatible")
        _validate_max_bytes(max_bytes)
        _check_allocation(
            max(1, len(operator.terms)) * (self.n_modes + 4) * 16,
            max_bytes,
            "mapped Pauli operator",
        )
        if self._native_plan is not None:
            structures, coefficients_re, coefficients_im = operator._arrays()
            result = self._native_plan.transform(
                structures,
                coefficients_re,
                coefficients_im,
                _effective_max_bytes(max_bytes),
            )
            return PauliOperator._from_native(self.n_modes, result)
        structures, coefficients_re, coefficients_im = operator._arrays()
        return PauliOperator.from_terms(
            self.n_modes,
            (
                (transformed, complex(real, imaginary) * phase)
                for structure, real, imaginary in zip(
                    structures, coefficients_re, coefficients_im
                )
                for transformed, phase in (self._transform_codes_with_phase(structure),)
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
        return self.map_pauli(jordan_wigner, max_bytes=max_bytes)

    def map_majorana_operator(
        self, operator: Any, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> PauliOperator:
        """Map a Majorana operator without materializing a fermion expansion."""
        from .majorana import MajoranaOperator

        if not isinstance(operator, MajoranaOperator):
            raise TypeError("mapping expects a MajoranaOperator")
        if operator.n_modes != self.n_modes:
            raise ValueError("mapping plan and Majorana mode counts are incompatible")
        _validate_max_bytes(max_bytes)
        if self._native_plan is not None:
            result = self._native_plan.transform_majorana(
                [list(term.word.indices) for term in operator.terms],
                [term.coefficient.real for term in operator.terms],
                [term.coefficient.imag for term in operator.terms],
                _effective_max_bytes(max_bytes),
            )
            return PauliOperator._from_native(self.n_modes, result)

        # The non-native constructor is a private/reference path.  Keep it
        # semantically complete while normal public plans use the Rust batch
        # kernel above.
        from .pauli import PauliWord

        def local_codes(index: int) -> Tuple[int, ...]:
            mode, odd = divmod(index, 2)
            return tuple(
                (
                    3
                    if qubit < mode
                    else 2 if qubit == mode and odd else 1 if qubit == mode else 0
                )
                for qubit in range(self.n_modes)
            )

        terms = []
        for term in operator.terms:
            word = PauliWord.from_codes((0,) * self.n_modes)
            phase = 1.0 + 0j
            for majorana_index in term.word.indices:
                local = PauliWord.from_codes(local_codes(majorana_index))
                product = word.multiply(local)
                word = product.word
                phase *= product.phase.value_complex
            transformed, mapping_phase = self._transform_codes_with_phase(
                word.to_codes()
            )
            terms.append((transformed, term.coefficient * phase * mapping_phase))
        return PauliOperator.from_terms(self.n_modes, terms)

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
            _check_allocation(
                max(1, len(jordan_wigner.terms))
                * (operator.space.fermions + len(operator.space._axes) + 4)
                * 16,
                max_bytes,
                "mapped hybrid Pauli operator",
            )
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
        if self._native_plan is not None:
            if not isinstance(jordan_wigner, HybridOperator):
                raise TypeError(
                    "native hybrid mapping received an incompatible operator"
                )
            result = self._native_plan.transform_hybrid(
                operator.space.bosons,
                operator.space.qubits,
                len(operator.space.qudits),
                operator.space.qudits[0] if operator.space.qudits else 0,
                _hybrid_arrays(jordan_wigner),
                _effective_max_bytes(max_bytes),
            )
            return _hybrid_from_native(jordan_wigner.space, result)
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
