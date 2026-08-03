"""NumPy-compatible Hamiltonian targets and backend MVP plans."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, Union, cast

import numpy as np


# A practical default for explicit statevector Hamiltonian targets. Users can
# lower or raise this per call through the public ``max_bytes`` parameter.
DEFAULT_MAX_BYTES = 16 * 1024 * 1024 * 1024

# These labels are part of the reusable-plan metadata contract.  Pauli plans
# keep the historical qubit spelling; structured plans use the ordered axes of
# OperatorSpace and therefore must not advertise a binary qubit basis.
MIXED_RADIX_BASIS_ORDERING = "operator_space_axis0_msb_mixed_radix"
DIRECT_WEYL_BASIS_ORDERING = "qudit0_msb_matrix"


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

    The immutable metadata fields describe the target schema, canonical term
    counts, mapping, finite cutoffs, and local Weyl convention when relevant.
    """

    nqubits: int
    term_count: int
    strategy: str
    _native_plan: Any
    local_dimensions: Tuple[int, ...]
    basis_ordering: str
    estimated_bytes: int
    _generic_entries: Any
    schema_version: int
    target: str
    source_term_count: int
    plan_term_count: int
    mapping: Optional[str]
    boson_cutoffs: Tuple[Tuple[int, int], ...]
    boson_boundary: Optional[str]
    qudit_dimension: Optional[int]
    weyl_convention: Optional[str]
    _dimension_value: int = field(init=False, repr=False, compare=False)

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
        schema_version: int = 1,
        target: str = "native_mvp",
        source_term_count: Optional[int] = None,
        plan_term_count: Optional[int] = None,
        mapping: Optional[str] = None,
        boson_cutoffs: Tuple[Tuple[int, int], ...] = (),
        boson_boundary: Optional[str] = None,
        qudit_dimension: Optional[int] = None,
        weyl_convention: Optional[str] = None,
    ) -> None:
        dimensions = tuple(local_dimensions)
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("nqubits must be a non-negative integer")
        for name, value in (
            ("term_count", term_count),
            ("estimated_bytes", estimated_bytes),
            ("schema_version", schema_version),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        dimension = _checked_dimension(dimensions, nqubits)
        if schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if target != "native_mvp":
            raise ValueError("native MVP plans must have target='native_mvp'")
        if mapping not in {None, "jordan_wigner"}:
            raise ValueError("mapping must be None or 'jordan_wigner'")
        object.__setattr__(self, "nqubits", nqubits)
        object.__setattr__(self, "term_count", term_count)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "_native_plan", native_plan)
        object.__setattr__(self, "local_dimensions", dimensions)
        object.__setattr__(self, "basis_ordering", basis_ordering)
        object.__setattr__(self, "estimated_bytes", int(estimated_bytes))
        object.__setattr__(self, "_generic_entries", generic_entries)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "target", target)
        for name, candidate in (
            ("source_term_count", source_term_count),
            ("plan_term_count", plan_term_count),
        ):
            if candidate is not None and (
                not isinstance(candidate, int)
                or isinstance(candidate, bool)
                or candidate < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(
            self,
            "source_term_count",
            term_count if source_term_count is None else source_term_count,
        )
        object.__setattr__(
            self,
            "plan_term_count",
            term_count if plan_term_count is None else plan_term_count,
        )
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "boson_cutoffs", tuple(boson_cutoffs))
        object.__setattr__(self, "boson_boundary", boson_boundary)
        if qudit_dimension is not None and (
            not isinstance(qudit_dimension, int)
            or isinstance(qudit_dimension, bool)
            or qudit_dimension < 3
        ):
            raise ValueError("qudit_dimension must be at least 3 when present")
        if qudit_dimension is None and weyl_convention is not None:
            raise ValueError("weyl_convention requires qudit_dimension")
        if qudit_dimension is not None and weyl_convention != "X^a Z^b":
            raise ValueError("Weyl plans require convention 'X^a Z^b'")
        object.__setattr__(self, "qudit_dimension", qudit_dimension)
        object.__setattr__(self, "weyl_convention", weyl_convention)
        object.__setattr__(self, "_dimension_value", dimension)

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
        return self._dimension_value

    def __call__(self, state: Sequence[complex]) -> np.ndarray[Any, Any]:
        """Apply the plan using its default memory limit."""
        return self.apply(state)


@dataclass(frozen=True)
class BackendMVPPlan:
    """Versioned pure-array MVP plan with an independent NumPy executor.

    Pauli plans contain packed X/Z words; direct-Weyl plans contain modular
    exponent arrays and advertise the factorized backend operations they use.
    """

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
    plan_kind: str = "pauli"
    qudit_dimension: int = 0
    a_exponents: np.ndarray[Any, Any] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint32)
    )
    b_exponents: np.ndarray[Any, Any] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint32)
    )
    target: str = "backend_mvp"
    source_term_count: int = -1
    plan_term_count: int = -1
    mapping: Optional[str] = None
    boson_cutoffs: Tuple[Tuple[int, int], ...] = ()
    boson_boundary: Optional[str] = None
    weyl_convention: Optional[str] = None
    _dimension_value: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize plan buffers and freeze them after construction."""
        for name, dtype in (
            ("x_words", np.uint64),
            ("z_words", np.uint64),
            ("coefficients", np.complex128),
            ("a_exponents", np.uint32),
            ("b_exponents", np.uint32),
        ):
            values = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.plan_kind not in {"pauli", "direct_weyl"}:
            raise ValueError("unknown backend MVP plan kind")
        for name, value in (
            ("schema_version", self.schema_version),
            ("nqubits", self.nqubits),
            ("word_count", self.word_count),
            ("integer_width", self.integer_width),
            ("estimated_bytes", self.estimated_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.target != "backend_mvp":
            raise ValueError("backend MVP plans must have target='backend_mvp'")
        dimensions = tuple(self.local_dimensions)
        object.__setattr__(self, "local_dimensions", dimensions)
        object.__setattr__(self, "required_operations", tuple(self.required_operations))
        object.__setattr__(self, "boson_cutoffs", tuple(self.boson_cutoffs))
        dimension = _checked_dimension(dimensions, self.nqubits)
        object.__setattr__(self, "_dimension_value", dimension)
        for name, value in (
            ("source_term_count", self.source_term_count),
            ("plan_term_count", self.plan_term_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                object.__setattr__(self, name, len(self.coefficients))
        if self.mapping not in {None, "jordan_wigner"}:
            raise ValueError("mapping must be None or 'jordan_wigner'")
        if not np.isfinite(self.coefficients).all():
            raise ValueError("backend plan coefficients must be finite")
        if self.plan_kind == "pauli":
            expected_word_count = (self.nqubits + 63) // 64
            if self.word_count != expected_word_count:
                raise ValueError("Pauli word_count does not match nqubits")
            if self.x_words.ndim != 2 or self.z_words.ndim != 2:
                raise ValueError("Pauli word arrays must be two-dimensional")
            expected_shape = (len(self.coefficients), self.word_count)
            if (
                self.x_words.shape != expected_shape
                or self.z_words.shape != expected_shape
            ):
                raise ValueError(
                    "Pauli word arrays must match plan term and word counts"
                )
            if dimensions and dimensions != (2,) * self.nqubits:
                raise ValueError("Pauli local dimensions must be binary")
        if self.plan_kind == "direct_weyl":
            if (
                not isinstance(self.qudit_dimension, int)
                or isinstance(self.qudit_dimension, bool)
                or self.qudit_dimension < 3
            ):
                raise ValueError("direct Weyl plans require dimension at least 3")
            if not self.local_dimensions or any(
                value != self.qudit_dimension for value in self.local_dimensions
            ):
                raise ValueError(
                    "direct Weyl local dimensions must match qudit_dimension"
                )
            if self.a_exponents.shape != self.b_exponents.shape:
                raise ValueError("direct Weyl exponent arrays must have equal shapes")
            if self.a_exponents.ndim != 2 or self.a_exponents.shape[0] != len(
                self.coefficients
            ):
                raise ValueError("direct Weyl exponent arrays must match term count")
            if self.a_exponents.shape[1] != len(self.local_dimensions):
                raise ValueError("direct Weyl exponents must match site count")
            if np.any(self.a_exponents >= self.qudit_dimension) or np.any(
                self.b_exponents >= self.qudit_dimension
            ):
                raise ValueError("direct Weyl exponents must be reduced modulo d")
            if self.weyl_convention != "X^a Z^b":
                raise ValueError("direct Weyl plans require convention 'X^a Z^b'")
            if self.basis_ordering != DIRECT_WEYL_BASIS_ORDERING:
                raise ValueError("direct Weyl plans require qudit0_msb_matrix ordering")
            if (
                not self.required_operations
                or "cyclic_shift" not in self.required_operations
            ):
                raise ValueError("direct Weyl plans require cyclic_shift support")

    def apply(
        self,
        state: Sequence[complex],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the plan using only NumPy arrays and deterministic indexing."""
        if self._generic_entries is not None:
            return _apply_generic_entries(self._generic_entries, state, max_bytes)
        if self.plan_kind == "direct_weyl":
            return self._apply_direct_weyl(state, max_bytes)
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

    @property
    def dimension(self) -> int:
        """Finite basis dimension represented by this plan."""
        return self._dimension_value

    @property
    def term_count(self) -> int:
        """Number of canonical plan terms."""
        return len(self.coefficients)

    def __call__(self, state: Sequence[complex]) -> np.ndarray[Any, Any]:
        """Apply the plan using its default memory limit."""
        return self.apply(state)

    def _apply_direct_weyl(
        self, state: Sequence[complex], max_bytes: Optional[int]
    ) -> np.ndarray[Any, Any]:
        dimension = self.dimension
        values = np.asarray(state, dtype=np.complex128)
        rank_shape = tuple(self.local_dimensions)
        if values.ndim == 1 and values.shape == (dimension,):
            tensor = values.reshape(rank_shape)
            flat = True
        elif values.shape == rank_shape:
            tensor = values
            flat = False
        else:
            raise ValueError(
                f"state must have shape ({dimension},) or {rank_shape}, got {values.shape}"
            )
        _check_allocation(dimension * 16 * 2, max_bytes, "backend MVP working memory")
        omega = np.exp(2j * np.pi / self.qudit_dimension)
        output = np.zeros_like(tensor, dtype=np.complex128)
        for term_index, coefficient in enumerate(self.coefficients):
            term = tensor
            for axis, (a, b) in enumerate(
                zip(self.a_exponents[term_index], self.b_exponents[term_index])
            ):
                if b:
                    phase = omega ** (int(b) * np.arange(self.qudit_dimension))
                    shape = [1] * len(rank_shape)
                    shape[axis] = self.qudit_dimension
                    term = term * phase.reshape(shape)
                shift = int(a) % self.qudit_dimension
                if shift:
                    term = np.concatenate(
                        (
                            np.take(
                                term,
                                np.arange(
                                    self.qudit_dimension - shift, self.qudit_dimension
                                ),
                                axis=axis,
                            ),
                            np.take(
                                term,
                                np.arange(0, self.qudit_dimension - shift),
                                axis=axis,
                            ),
                        ),
                        axis=axis,
                    )
            output += coefficient * term
        result = output.reshape(-1) if flat else output
        return cast(np.ndarray[Any, Any], np.asarray(result, dtype=np.complex128))


CompileResult = Union[
    np.ndarray[Any, Any], COOMatrix, CSRMatrix, NativeMVPPlan, BackendMVPPlan
]


def _dimension(nqubits: int) -> int:
    if nqubits < 0 or nqubits >= np.intp().itemsize * 8:
        raise OverflowError(
            "matrix dimension cannot be represented by platform indices"
        )
    return 1 << nqubits


def _checked_dimension(local_dimensions: Tuple[int, ...], nqubits: int) -> int:
    """Compute one platform-index-safe mixed-radix dimension."""
    if local_dimensions:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in local_dimensions
        ):
            raise ValueError("local dimensions must be positive integers")
        dimension = math.prod(local_dimensions)
    else:
        dimension = _dimension(nqubits)
    if dimension > int(np.iinfo(np.intp).max):
        raise OverflowError(
            "finite basis dimension cannot be represented by platform indices"
        )
    return dimension


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
    dimension = _checked_dimension(tuple(int(value) for value in dimensions), 0)
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
