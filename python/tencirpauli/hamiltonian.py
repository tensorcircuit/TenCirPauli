"""NumPy-compatible Hamiltonian targets and backend MVP plans."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, cast

import numpy as np


# A practical default for explicit statevector Hamiltonian targets. Users can
# lower or raise this per call through the public ``max_bytes`` parameter.
DEFAULT_MAX_BYTES = 16 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class COOMatrix:
    """Deterministic NumPy/SciPy-compatible COO arrays."""

    row: np.ndarray[Any, Any]
    column: np.ndarray[Any, Any]
    data: np.ndarray[Any, Any]
    shape: Tuple[int, int]

    @property
    def value(self) -> np.ndarray[Any, Any]:
        """Alias for the conventional COO value field."""
        return self.data

    def to_scipy(self) -> Any:
        """Convert to SciPy COO only when the optional dependency is installed."""
        try:
            from scipy.sparse import coo_matrix  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError("COO conversion requires scipy") from error
        return coo_matrix((self.data, (self.row, self.column)), shape=self.shape)


@dataclass(frozen=True)
class CSRMatrix:
    """Deterministic NumPy/SciPy-compatible CSR arrays."""

    indptr: np.ndarray[Any, Any]
    indices: np.ndarray[Any, Any]
    data: np.ndarray[Any, Any]
    shape: Tuple[int, int]

    @property
    def value(self) -> np.ndarray[Any, Any]:
        """Alias for the conventional CSR value field."""
        return self.data

    def to_scipy(self) -> Any:
        """Convert to SciPy CSR only when the optional dependency is installed."""
        try:
            from scipy.sparse import csr_matrix
        except ImportError as error:
            raise ImportError("CSR conversion requires scipy") from error
        return csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)


@dataclass(frozen=True, init=False)
class NativeMVPPlan:
    """Reusable Rust-native matrix-free MVP plan.

    ``strategy`` is ``"x_mask_diagonal"`` for the precomputed diagonal
    kernel or ``"term_direct"`` when the explicit plan memory limit selects
    the direct term kernel.
    """

    nqubits: int
    term_count: int
    strategy: str
    _native_plan: Any
    local_dimensions: Tuple[int, ...]
    basis_ordering: str
    estimated_bytes: int
    _generic_entries: Any

    def __init__(
        self,
        nqubits: int,
        term_count: int,
        strategy: str,
        native_plan: Any,
        *,
        local_dimensions: Tuple[int, ...] = (),
        basis_ordering: str = "qubit0_msb_matrix",
        estimated_bytes: int = 0,
        generic_entries: Any = None,
    ) -> None:
        object.__setattr__(self, "nqubits", nqubits)
        object.__setattr__(self, "term_count", term_count)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "_native_plan", native_plan)
        object.__setattr__(self, "local_dimensions", tuple(local_dimensions))
        object.__setattr__(self, "basis_ordering", basis_ordering)
        object.__setattr__(self, "estimated_bytes", int(estimated_bytes))
        object.__setattr__(self, "_generic_entries", generic_entries)

    def apply(
        self,
        state: Sequence[complex],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the precompiled Rust plan without rebuilding its structure."""
        if self._generic_entries is not None:
            return _apply_generic_entries(self._generic_entries, state, max_bytes)
        dimension = self.dimension
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.shape[0] != dimension:
            raise ValueError(
                f"state must have shape ({dimension},), got {values.shape}"
            )
        contiguous = np.ascontiguousarray(values)
        return cast(
            np.ndarray[Any, Any],
            np.asarray(
                self._native_plan.apply(contiguous, _effective_max_bytes(max_bytes)),
                dtype=np.complex128,
            ),
        )

    @property
    def dimension(self) -> int:
        """Finite basis dimension represented by this plan."""
        if self.local_dimensions:
            return int(np.prod(self.local_dimensions, dtype=np.intp))
        return _dimension(self.nqubits)

    def __call__(self, state: Sequence[complex]) -> np.ndarray[Any, Any]:
        """Apply the plan using its default memory limit."""
        return self.apply(state)


@dataclass(frozen=True)
class BackendMVPPlan:
    """Versioned pure-array MVP plan with an independent NumPy executor."""

    schema_version: int
    nqubits: int
    word_count: int
    x_words: np.ndarray[Any, Any]
    z_words: np.ndarray[Any, Any]
    coefficients: np.ndarray[Any, Any]
    ordering: str = "qubit0_msb_matrix"
    integer_width: int = 64
    required_operations: Tuple[str, ...] = ("xor", "phase", "scatter_add")
    local_dimensions: Tuple[int, ...] = ()
    basis_ordering: str = "qubit0_msb_matrix"
    estimated_bytes: int = 0
    _generic_entries: Any = None

    def __post_init__(self) -> None:
        """Normalize plan buffers and freeze them after construction."""
        for name, dtype in (
            ("x_words", np.uint64),
            ("z_words", np.uint64),
            ("coefficients", np.complex128),
        ):
            values = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    def apply(
        self,
        state: Sequence[complex],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the plan using only NumPy arrays and deterministic indexing."""
        if self._generic_entries is not None:
            return _apply_generic_entries(self._generic_entries, state, max_bytes)
        dimension = _dimension(self.nqubits)
        _check_allocation(dimension * 80, max_bytes, "backend MVP working memory")
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.shape[0] != dimension:
            raise ValueError(
                f"state must have shape ({dimension},), got {values.shape}"
            )
        output: np.ndarray[Any, Any] = np.zeros(dimension, dtype=np.complex128)
        columns = np.arange(dimension, dtype=np.uint64)
        for term_index, coefficient in enumerate(self.coefficients):
            x_mask = _matrix_mask(
                self.x_words[term_index], self.z_words[term_index], self.nqubits
            )
            rows = np.bitwise_xor(columns, np.uint64(x_mask)).astype(np.intp)
            phase: np.ndarray[Any, Any] = np.ones(dimension, dtype=np.complex128)
            for qubit in range(self.nqubits):
                code = _packed_code(
                    self.x_words[term_index], self.z_words[term_index], qubit
                )
                bit = (columns >> np.uint64(self.nqubits - 1 - qubit)) & 1
                if code == 2:
                    phase *= np.where(bit == 0, 1j, -1j)
                elif code == 3:
                    phase *= np.where(bit == 1, -1.0, 1.0)
            output[rows] += coefficient * phase * values
        return output


def _dimension(nqubits: int) -> int:
    if nqubits < 0 or nqubits >= np.intp().itemsize * 8:
        raise OverflowError(
            "matrix dimension cannot be represented by platform indices"
        )
    return 1 << nqubits


def _validate_max_bytes(max_bytes: Optional[int]) -> None:
    if max_bytes is not None and (
        not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0
    ):
        raise ValueError("max_bytes must be a non-negative integer or None")


def _effective_max_bytes(max_bytes: Optional[int]) -> int:
    _validate_max_bytes(max_bytes)
    return sys.maxsize if max_bytes is None else max_bytes


def _check_allocation(requested: int, limit: Optional[int], context: str) -> None:
    _validate_max_bytes(limit)
    if limit is not None and requested > limit:
        raise MemoryError(
            f"{context} requires approximately {requested} bytes, "
            f"exceeding max_bytes={limit}"
        )


def _apply_generic_entries(
    generic_entries: Any, state: Sequence[complex], max_bytes: Optional[int]
) -> np.ndarray[Any, Any]:
    dimensions, rows, columns, coefficients = generic_entries
    dimension = int(np.prod(dimensions, dtype=np.intp))
    _check_allocation(dimension * 16, max_bytes, "structured MVP output")
    values = np.asarray(state, dtype=np.complex128)
    if values.ndim != 1 or values.shape[0] != dimension:
        raise ValueError(f"state must have shape ({dimension},), got {values.shape}")
    output: np.ndarray[Any, Any] = np.zeros(dimension, dtype=np.complex128)
    np.add.at(
        output, rows.astype(np.intp), coefficients * values[columns.astype(np.intp)]
    )
    return output


def _matrix_mask(
    x_words: np.ndarray[Any, Any], z_words: np.ndarray[Any, Any], nqubits: int
) -> int:
    mask = 0
    for qubit in range(nqubits):
        if _packed_code(x_words, z_words, qubit) in (1, 2):
            mask |= 1 << (nqubits - 1 - qubit)
    return mask


def _packed_code(
    x_words: np.ndarray[Any, Any], z_words: np.ndarray[Any, Any], qubit: int
) -> int:
    x = (int(x_words[qubit // 64]) >> (qubit % 64)) & 1
    z = (int(z_words[qubit // 64]) >> (qubit % 64)) & 1
    return {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}[(x, z)]
