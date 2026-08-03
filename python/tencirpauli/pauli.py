"""Typed public Pauli word and static operator APIs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Tuple, Union, cast

import numpy as np

from . import _native
from .hamiltonian import DEFAULT_MAX_BYTES, _effective_max_bytes, _validate_max_bytes


if TYPE_CHECKING:
    from .grouping import GroupingResult
    from .hamiltonian import BackendMVPPlan, COOMatrix, CSRMatrix, NativeMVPPlan
    from .symmetry import U1RestrictedOperator, U1Sector, Z2SymmetryAnalysis


class PauliPhase(IntEnum):
    """The exact four-valued phase returned by Pauli multiplication."""

    PLUS_ONE = 0
    PLUS_I = 1
    MINUS_ONE = 2
    MINUS_I = 3

    @property
    def value_complex(self) -> complex:
        """Return the phase as a Python complex scalar."""
        return (1.0, 1.0j, -1.0, -1.0j)[int(self)]


def _validate_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class PauliProduct:
    """Result of multiplying two phase-free Pauli words."""

    word: "PauliWord"
    phase: PauliPhase


@dataclass(frozen=True, init=False)
class PauliWord:
    """A phase-free Pauli word using external codes ``0=I, 1=X, 2=Y, 3=Z``.

    The packed representation uses qubit zero as its least-significant bit;
    matrix-producing APIs explicitly convert that layout to TensorCircuit's
    qubit-zero-is-MSB convention.
    """

    nqubits: int
    x_words: Tuple[int, ...]
    z_words: Tuple[int, ...]

    def __init__(
        self, nqubits: int, x_words: Sequence[int], z_words: Sequence[int]
    ) -> None:
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError(f"nqubits must be a non-negative integer, got {nqubits!r}")
        x_tuple = tuple(x_words)
        z_tuple = tuple(z_words)
        word_count = (nqubits + 63) // 64
        for name, words in (("x_words", x_tuple), ("z_words", z_tuple)):
            if len(words) != word_count:
                raise ValueError(
                    f"{name}: expected {word_count} packed words for {nqubits} "
                    f"qubits, got {len(words)}"
                )
            for index, word in enumerate(words):
                if (
                    not isinstance(word, int)
                    or isinstance(word, bool)
                    or not 0 <= word < 2**64
                ):
                    raise ValueError(
                        f"{name}[{index}] must be an unsigned 64-bit integer"
                    )
        if word_count and nqubits % 64:
            tail_mask = (1 << (nqubits % 64)) - 1
            x_tuple = (*x_tuple[:-1], x_tuple[-1] & tail_mask)
            z_tuple = (*z_tuple[:-1], z_tuple[-1] & tail_mask)
        object.__setattr__(self, "nqubits", nqubits)
        object.__setattr__(self, "x_words", x_tuple)
        object.__setattr__(self, "z_words", z_tuple)

    @classmethod
    def from_codes(cls, codes: Sequence[int]) -> "PauliWord":
        """Construct one word from an ordered sequence of I/X/Y/Z codes."""
        normalized = tuple(codes)
        if any(
            not isinstance(code, int) or isinstance(code, bool) or code not in range(4)
            for code in normalized
        ):
            raise ValueError(
                f"Pauli codes must be integers in 0..3, got {normalized!r}"
            )
        x_words, z_words = _native.pauli_from_codes(len(normalized), normalized)
        return cls(len(normalized), tuple(x_words), tuple(z_words))

    @classmethod
    def from_string(cls, value: str) -> "PauliWord":
        """Construct one word from an ``IXYZ`` string in qubit order."""
        if not isinstance(value, str):
            raise TypeError("Pauli string must be a str")
        lookup = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        try:
            return cls.from_codes(
                tuple(lookup[character] for character in value.upper())
            )
        except KeyError as error:
            raise ValueError(f"invalid Pauli character {error.args[0]!r}") from error

    @classmethod
    def batch_from_codes(
        cls, nqubits: int, structures: Iterable[Sequence[int]]
    ) -> Tuple["PauliWord", ...]:
        """Convert many structures with one coarse-grained native call."""
        normalized = tuple(tuple(structure) for structure in structures)
        for structure in normalized:
            if len(structure) != nqubits:
                raise ValueError(
                    f"expected structure length {nqubits}, got {len(structure)}"
                )
            if any(code not in range(4) for code in structure):
                raise ValueError("Pauli codes must be integers in 0..3")
        word_count, x_flat, z_flat = _native.pauli_batch_from_codes(nqubits, normalized)
        return tuple(
            cls(
                nqubits,
                tuple(x_flat[index * word_count : (index + 1) * word_count]),
                tuple(z_flat[index * word_count : (index + 1) * word_count]),
            )
            for index in range(len(normalized))
        )

    @property
    def weight(self) -> int:
        """Return the number of non-identity sites."""
        return int(_native.pauli_weight(self.nqubits, self.x_words, self.z_words))

    @property
    def support(self) -> Tuple[int, ...]:
        """Return non-identity qubit indices in ascending order."""
        return tuple(_native.pauli_support(self.nqubits, self.x_words, self.z_words))

    def to_codes(self) -> Tuple[int, ...]:
        """Return the external I/X/Y/Z codes in qubit order."""
        return tuple(_native.pauli_codes(self.nqubits, self.x_words, self.z_words))

    def to_string(self) -> str:
        """Return the canonical ``IXYZ`` string representation."""
        return "".join("IXYZ"[code] for code in self.to_codes())

    def symplectic_inner_product(self, other: "PauliWord") -> int:
        """Return the binary symplectic inner product in ``{0, 1}``."""
        _ensure_compatible(self, other)
        return int(
            _native.pauli_symplectic_inner_product(
                self.nqubits,
                self.x_words,
                self.z_words,
                other.x_words,
                other.z_words,
            )
        )

    def commutes_with(self, other: "PauliWord") -> bool:
        """Return whether this Pauli word commutes with another word."""
        _ensure_compatible(self, other)
        return bool(
            _native.pauli_commutes(
                self.nqubits,
                self.x_words,
                self.z_words,
                other.x_words,
                other.z_words,
            )
        )

    def multiply(self, other: "PauliWord") -> PauliProduct:
        """Multiply words and return a canonical word plus exact phase."""
        _ensure_compatible(self, other)
        codes, phase = _native.pauli_multiply(
            self.nqubits, self.to_codes(), other.to_codes()
        )
        return PauliProduct(PauliWord.from_codes(codes), PauliPhase(phase))

    def adjoint(self) -> "PauliWord":
        """Return the adjoint; every phase-free basis word is Hermitian."""
        return self

    def __str__(self) -> str:
        return self.to_string()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PauliWord):
            return NotImplemented
        return (self.nqubits, self.to_codes()) < (other.nqubits, other.to_codes())


def _ensure_word(other: object) -> None:
    if not isinstance(other, PauliWord):
        raise TypeError(f"expected PauliWord, got {type(other).__name__}")


def _ensure_compatible(left: PauliWord, right: object) -> None:
    _ensure_word(right)
    assert isinstance(right, PauliWord)
    if left.nqubits != right.nqubits:
        raise ValueError(
            f"incompatible qubit counts: {left.nqubits} and {right.nqubits}"
        )


PauliInput = Union[PauliWord, str, Sequence[int]]
_OperatorArrays = Tuple[
    Tuple[Tuple[int, ...], ...], Tuple[float, ...], Tuple[float, ...]
]


@dataclass(frozen=True)
class PauliTerm:
    """One canonical Pauli word and its complex128-compatible coefficient."""

    word: PauliWord
    coefficient: complex


@dataclass(frozen=True)
class CanonicalizationResult:
    """Deterministic batch canonicalization with backend reduction metadata."""

    canonical_structures: Tuple[Tuple[int, ...], ...]
    coefficients: Tuple[complex, ...]
    input_to_canonical: Tuple[int, ...]
    phase_multipliers: Tuple[PauliPhase, ...]


@dataclass(frozen=True)
class CanonicalizationArrayResult:
    """Contiguous canonicalization arrays for large backend-facing batches."""

    canonical_structures: np.ndarray[Any, Any]
    coefficients: np.ndarray[Any, Any]
    input_to_canonical: np.ndarray[Any, Any]
    phase_multipliers: np.ndarray[Any, Any]


@dataclass(frozen=True, init=False)
class PauliOperator:
    """A deterministic static Pauli operator with exact-zero aggregation."""

    nqubits: int
    terms: Tuple[PauliTerm, ...]
    _canonical_structures: Tuple[Tuple[int, ...], ...] = field(
        repr=False, compare=False
    )
    _coefficient_reals: Tuple[float, ...] = field(repr=False, compare=False)
    _coefficient_imaginaries: Tuple[float, ...] = field(repr=False, compare=False)

    def __init__(
        self, nqubits: int, terms: Iterable[Tuple[PauliInput, complex]]
    ) -> None:
        normalized_terms = tuple(terms)
        array_input = _normalize_code_array_inputs(nqubits, normalized_terms)
        if array_input is None:
            structures, coefficients_re, coefficients_im = _normalize_operator_inputs(
                nqubits, normalized_terms
            )
            result = _native.pauli_canonicalize(
                nqubits,
                structures,
                coefficients_re,
                coefficients_im,
            )
        else:
            result = _native.pauli_canonicalize_array(nqubits, *array_input)
        _initialize_operator(self, nqubits, result)

    @classmethod
    def empty(cls, nqubits: int) -> "PauliOperator":
        """Construct the additive identity on ``nqubits``."""
        return cls(nqubits, ())

    @classmethod
    def from_terms(
        cls, nqubits: int, terms: Iterable[Tuple[PauliInput, complex]]
    ) -> "PauliOperator":
        """Construct and canonicalize mixed string, code, or word terms."""
        return cls(nqubits, terms)

    @classmethod
    def from_code_arrays(
        cls,
        structures: Sequence[Sequence[int]],
        coefficients: Sequence[complex],
    ) -> "PauliOperator":
        """Construct directly from contiguous-friendly code and coefficient arrays."""
        code_array, coefficient_array = _normalize_code_arrays(structures, coefficients)
        nqubits = int(code_array.shape[1])
        result = _native.pauli_canonicalize_array(
            nqubits, code_array, coefficient_array
        )
        instance = object.__new__(cls)
        _initialize_operator(instance, nqubits, result)
        return instance

    @classmethod
    def canonicalize_batch(
        cls, nqubits: int, terms: Iterable[Tuple[PauliInput, complex]]
    ) -> CanonicalizationResult:
        """Canonicalize a batch while retaining reduction mapping and phases.

        Code-array and string inputs are phase-free, so every returned phase
        multiplier is ``PauliPhase.PLUS_ONE``. Exact-zero aggregated keys are
        retained here for backend structural plans; ``from_terms`` removes
        them for static operators.
        """
        normalized_terms = tuple(terms)
        array_input = _normalize_code_array_inputs(nqubits, normalized_terms)
        if array_input is None:
            structures, coefficients_re, coefficients_im = _normalize_operator_inputs(
                nqubits, normalized_terms
            )
            result = _native.pauli_canonicalize_batch(
                nqubits,
                structures,
                coefficients_re,
                coefficients_im,
            )
        else:
            result = _native.pauli_canonicalize_batch_array(nqubits, *array_input)
        canonical_structures, real, imaginary, mapping, phases = result
        return CanonicalizationResult(
            tuple(
                tuple(int(code) for code in structure)
                for structure in canonical_structures
            ),
            tuple(
                complex(real_value, imaginary_value)
                for real_value, imaginary_value in zip(real, imaginary)
            ),
            tuple(int(index) for index in mapping),
            tuple(PauliPhase(int(phase)) for phase in phases),
        )

    @classmethod
    def canonicalize_code_arrays(
        cls,
        structures: Sequence[Sequence[int]],
        coefficients: Sequence[complex],
    ) -> CanonicalizationResult:
        """Canonicalize code arrays without per-term Python object conversion."""
        code_array, coefficient_array = _normalize_code_arrays(structures, coefficients)
        nqubits = int(code_array.shape[1])
        canonical_structures, real, imaginary, mapping, phases = (
            _native.pauli_canonicalize_batch_array(
                nqubits, code_array, coefficient_array
            )
        )
        return CanonicalizationResult(
            tuple(tuple(int(code) for code in row) for row in canonical_structures),
            tuple(
                complex(real_value, imaginary_value)
                for real_value, imaginary_value in zip(real, imaginary)
            ),
            tuple(int(index) for index in mapping),
            tuple(PauliPhase(int(phase)) for phase in phases),
        )

    @classmethod
    def canonicalize_code_arrays_numpy(
        cls,
        structures: Sequence[Sequence[int]],
        coefficients: Sequence[complex],
    ) -> CanonicalizationArrayResult:
        """Return contiguous canonicalization arrays without Python term objects."""
        code_array, coefficient_array = _normalize_code_arrays(structures, coefficients)
        nqubits = int(code_array.shape[1])
        canonical_count, codes, values, mapping, phases = (
            _native.pauli_canonicalize_batch_numpy(
                nqubits, code_array, coefficient_array
            )
        )
        canonical_structures = np.asarray(codes, dtype=np.uint8).reshape(
            (canonical_count, nqubits)
        )
        canonical_coefficients = np.asarray(values, dtype=np.complex128)
        input_to_canonical = np.asarray(mapping, dtype=np.uintp)
        phase_multipliers = np.asarray(phases, dtype=np.uint8)
        for array in (
            canonical_structures,
            canonical_coefficients,
            input_to_canonical,
            phase_multipliers,
        ):
            array.flags.writeable = False
        return CanonicalizationArrayResult(
            canonical_structures,
            canonical_coefficients,
            input_to_canonical,
            phase_multipliers,
        )

    @classmethod
    def from_strings(cls, terms: Iterable[Tuple[str, complex]]) -> "PauliOperator":
        """Construct an operator from strings, inferring the common qubit count."""
        normalized = tuple(terms)
        if not normalized:
            raise ValueError("cannot infer nqubits from an empty term sequence")
        nqubits = len(normalized[0][0])
        return cls(nqubits, normalized)

    @classmethod
    def _from_native(
        cls,
        nqubits: int,
        result: Tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]],
    ) -> "PauliOperator":
        instance = object.__new__(cls)
        _initialize_operator(instance, nqubits, result)
        return instance

    def _arrays(self) -> _OperatorArrays:
        return (
            self._canonical_structures,
            self._coefficient_reals,
            self._coefficient_imaginaries,
        )

    def add(
        self,
        other: "PauliOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Add two operators and aggregate exact duplicate keys."""
        _validate_max_bytes(max_bytes)
        _ensure_operator_compatible(self, other)
        left = self._arrays()
        right = other._arrays()
        result = _native.pauli_operator_binary(self.nqubits, left, right, 0)
        return self._from_native(self.nqubits, result)

    def scale(
        self,
        scalar: complex,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Multiply all coefficients by a finite complex scalar."""
        _validate_max_bytes(max_bytes)
        normalized = complex(scalar)
        if not math.isfinite(normalized.real) or not math.isfinite(normalized.imag):
            raise ValueError("scale must be a finite complex128 value")
        structures, coefficients_re, coefficients_im = self._arrays()
        result = _native.pauli_operator_scale(
            self.nqubits,
            structures,
            coefficients_re,
            coefficients_im,
            normalized.real,
            normalized.imag,
        )
        return self._from_native(self.nqubits, result)

    def multiply(
        self,
        other: "PauliOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Multiply operators, absorbing exact Pauli phases into coefficients."""
        _validate_max_bytes(max_bytes)
        return self._binary(other, 1)

    def commutator(
        self,
        other: "PauliOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Return ``self * other - other * self``."""
        _validate_max_bytes(max_bytes)
        return self._binary(other, 2)

    def anticommutator(
        self,
        other: "PauliOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Return ``self * other + other * self``."""
        _validate_max_bytes(max_bytes)
        return self._binary(other, 3)

    def adjoint(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "PauliOperator":
        """Return the coefficient-conjugated adjoint operator."""
        _validate_max_bytes(max_bytes)
        structures, coefficients_re, coefficients_im = self._arrays()
        result = _native.pauli_operator_adjoint(
            self.nqubits, structures, coefficients_re, coefficients_im
        )
        return self._from_native(self.nqubits, result)

    def is_hermitian(self, tolerance: float = 0.0) -> bool:
        """Validate Hermiticity using an explicit non-negative tolerance."""
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Hermiticity tolerance must be finite and non-negative")
        structures, coefficients_re, coefficients_im = self._arrays()
        return bool(
            _native.pauli_operator_is_hermitian(
                self.nqubits,
                structures,
                coefficients_re,
                coefficients_im,
                tolerance,
            )
        )

    def group_commuting(
        self,
        mode: str = "qubit_wise",
        algorithm: str = "largest_first",
        max_matrix_entries: int = 10_000_000,
    ) -> "GroupingResult":
        """Return a deterministic QWC or general-commuting grouping result."""
        from .grouping import group_operator

        return group_operator(
            self,
            mode=mode,
            algorithm=algorithm,
            max_matrix_entries=max_matrix_entries,
        )

    def find_z2_symmetries(
        self, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "Z2SymmetryAnalysis":
        """Discover deterministic, term-wise commuting Pauli Z2 symmetries."""
        from .symmetry import Z2SymmetryAnalysis

        _validate_max_bytes(max_bytes)
        generators, constraint_rank = _native.pauli_find_z2_symmetries(
            self.nqubits, *self._arrays(), _effective_max_bytes(max_bytes)
        )
        return Z2SymmetryAnalysis(
            self.nqubits,
            tuple(PauliWord.from_codes(codes) for codes in generators),
            int(constraint_rank),
        )

    def taper_z2(
        self,
        sector: Sequence[int],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Find Z2 symmetries, select ``sector``, and taper this operator."""
        analysis = self.find_z2_symmetries(max_bytes=max_bytes)
        return analysis.tapering_plan(sector).transform_operator(self)

    def restrict_u1(
        self,
        sector: "U1Sector",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "U1RestrictedOperator":
        """Validate and restrict this operator to an explicit U1 sector."""
        from .symmetry import U1Sector, _restrict_u1

        _validate_max_bytes(max_bytes)
        if not isinstance(sector, U1Sector):
            raise TypeError(f"expected U1Sector, got {type(sector).__name__}")
        if sector.nqubits != self.nqubits:
            raise ValueError(
                f"expected sector for {self.nqubits} qubits, got {sector.nqubits}"
            )
        return _restrict_u1(self, sector, max_bytes)

    def compatibility_matrix(
        self, mode: str = "general", max_entries: int = 10_000_000
    ) -> np.ndarray[Any, Any]:
        """Return a bounded dense matrix, limited by compatibility entries."""
        _validate_nonnegative_int(max_entries, "max_entries")
        mode_code = {"qubit_wise": 0, "general": 1}.get(mode)
        if mode_code is None:
            raise ValueError("mode must be 'qubit_wise' or 'general'")
        structures = self._canonical_structures
        values = _native.pauli_compatibility_matrix(
            self.nqubits, structures, mode_code, max_entries
        )
        size = len(structures)
        matrix: np.ndarray[Any, Any] = np.asarray(values, dtype=np.bool_).reshape(
            (size, size)
        )
        return matrix

    def incompatibility_edges(
        self, mode: str = "general", max_edges: int = 10_000_000
    ) -> Tuple[Tuple[int, int], ...]:
        """Return streaming edges, limited by the number of output edges."""
        _validate_nonnegative_int(max_edges, "max_edges")
        mode_code = {"qubit_wise": 0, "general": 1}.get(mode)
        if mode_code is None:
            raise ValueError("mode must be 'qubit_wise' or 'general'")
        structures = self._canonical_structures
        return tuple(
            (left, right)
            for left, right in _native.pauli_incompatibility_edges(
                self.nqubits, structures, mode_code, max_edges
            )
        )

    def dense(
        self, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> np.ndarray[Any, Any]:
        """Materialize a bounded complex128 dense Hamiltonian matrix."""
        from . import _native

        structures, coefficients_re, coefficients_im = self._arrays()
        _validate_max_bytes(max_bytes)
        dimension, values = _native.pauli_dense_array(
            self.nqubits,
            structures,
            coefficients_re,
            coefficients_im,
            _effective_max_bytes(max_bytes),
        )
        result: np.ndarray[Any, Any] = np.asarray(values, dtype=np.complex128).reshape(
            (dimension, dimension)
        )
        return result

    def coo(self, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> "COOMatrix":
        """Compile deterministic, duplicate-aggregated COO arrays."""
        from . import _native
        from .hamiltonian import COOMatrix

        structures, coefficients_re, coefficients_im = self._arrays()
        _validate_max_bytes(max_bytes)
        dimension, rows, columns, values = _native.pauli_coo_array(
            self.nqubits,
            structures,
            coefficients_re,
            coefficients_im,
            _effective_max_bytes(max_bytes),
        )
        return COOMatrix(
            np.asarray(rows, dtype=np.uint64),
            np.asarray(columns, dtype=np.uint64),
            np.asarray(values, dtype=np.complex128),
            (dimension, dimension),
        )

    def csr(self, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> "CSRMatrix":
        """Compile deterministic CSR arrays from the canonical COO stream."""
        from . import _native
        from .hamiltonian import CSRMatrix

        structures, coefficients_re, coefficients_im = self._arrays()
        _validate_max_bytes(max_bytes)
        dimension, indptr, indices, values = _native.pauli_csr_array(
            self.nqubits,
            structures,
            coefficients_re,
            coefficients_im,
            _effective_max_bytes(max_bytes),
        )
        return CSRMatrix(
            np.asarray(indptr, dtype=np.uint64),
            np.asarray(indices, dtype=np.uint64),
            np.asarray(values, dtype=np.complex128),
            (dimension, dimension),
        )

    def mvp(
        self,
        state: Sequence[complex],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the Hamiltonian to a one-dimensional complex128 state."""
        from . import _native

        _validate_max_bytes(max_bytes)
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1:
            raise ValueError(f"state must be one-dimensional, got shape {values.shape}")
        structures, coefficients_re, coefficients_im = self._arrays()
        return cast(
            np.ndarray[Any, Any],
            np.asarray(
                _native.pauli_mvp_array(
                    self.nqubits,
                    structures,
                    coefficients_re,
                    coefficients_im,
                    np.ascontiguousarray(values),
                    _effective_max_bytes(max_bytes),
                ),
                dtype=np.complex128,
            ),
        )

    def backend_mvp_plan(
        self, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "BackendMVPPlan":
        """Compile a versioned pure-array plan for backend execution."""
        from . import _native
        from .hamiltonian import BackendMVPPlan

        _validate_max_bytes(max_bytes)
        structures, coefficients_re, coefficients_im = self._arrays()
        schema, nqubits, word_count, x_words, z_words, real, imaginary = (
            _native.pauli_backend_plan(
                self.nqubits,
                structures,
                coefficients_re,
                coefficients_im,
                _effective_max_bytes(max_bytes),
            )
        )
        return BackendMVPPlan(
            schema,
            nqubits,
            word_count,
            np.asarray(x_words, dtype=np.uint64).reshape((len(real), word_count)),
            np.asarray(z_words, dtype=np.uint64).reshape((len(real), word_count)),
            np.asarray(real, dtype=np.float64)
            + 1j * np.asarray(imaginary, dtype=np.float64),
            local_dimensions=(2,) * self.nqubits,
            basis_ordering="qubit0_msb_matrix",
        )

    def native_mvp_plan(
        self, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "NativeMVPPlan":
        """Compile a reusable Rust-native matrix-free MVP plan."""
        from . import _native
        from .hamiltonian import NativeMVPPlan

        _validate_max_bytes(max_bytes)
        structures, coefficients_re, coefficients_im = self._arrays()
        native_plan = _native.pauli_mvp_plan(
            self.nqubits,
            structures,
            coefficients_re,
            coefficients_im,
            _effective_max_bytes(max_bytes),
        )
        if native_plan.nqubits != self.nqubits:
            raise RuntimeError("native MVP plan has incompatible qubit count")
        if native_plan.term_count != len(self.terms):
            raise RuntimeError("native MVP plan has incompatible term count")
        return NativeMVPPlan(
            self.nqubits,
            len(self.terms),
            native_plan.strategy,
            native_plan,
            local_dimensions=(2,) * self.nqubits,
            basis_ordering="qubit0_msb_matrix",
            estimated_bytes=0,
        )

    def compile(self, target: str, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> Any:
        """Compile one named Hamiltonian target through the public API."""
        if target == "dense":
            return self.dense(max_bytes=max_bytes)
        if target == "coo":
            return self.coo(max_bytes=max_bytes)
        if target == "csr":
            return self.csr(max_bytes=max_bytes)
        if target == "backend_mvp":
            return self.backend_mvp_plan(max_bytes=max_bytes)
        if target == "native_mvp":
            return self.native_mvp_plan(max_bytes=max_bytes)
        raise ValueError(
            "target must be one of 'dense', 'coo', 'csr', 'native_mvp', or 'backend_mvp'"
        )

    def _binary(self, other: "PauliOperator", operation: int) -> "PauliOperator":
        _ensure_operator_compatible(self, other)
        result = _native.pauli_operator_binary(
            self.nqubits, self._arrays(), other._arrays(), operation
        )
        return self._from_native(self.nqubits, result)

    def __add__(self, other: object) -> "PauliOperator":
        if not isinstance(other, PauliOperator):
            return NotImplemented
        return self.add(other)

    def __sub__(self, other: object) -> "PauliOperator":
        if not isinstance(other, PauliOperator):
            return NotImplemented
        return self.add(other.scale(-1.0))

    def __neg__(self) -> "PauliOperator":
        return self.scale(-1.0)

    def __mul__(self, scalar: object) -> "PauliOperator":
        if isinstance(scalar, PauliOperator):
            return self.multiply(scalar)
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        return self.scale(complex(scalar))

    def __rmul__(self, scalar: object) -> "PauliOperator":
        return self * scalar

    def tensor_product(
        self,
        other: "PauliOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Return the ordinary tensor product with left axes first."""
        _validate_max_bytes(max_bytes)
        if not isinstance(other, PauliOperator):
            raise TypeError("tensor_product expects a PauliOperator")
        terms = []
        for left, right in (
            (left, right) for left in self.terms for right in other.terms
        ):
            terms.append(
                (
                    left.word.to_codes() + right.word.to_codes(),
                    left.coefficient * right.coefficient,
                )
            )
        return PauliOperator.from_terms(self.nqubits + other.nqubits, terms)


def _normalize_operator_inputs(
    nqubits: int, terms: Iterable[Tuple[PauliInput, complex]]
) -> _OperatorArrays:
    if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
        raise ValueError(f"nqubits must be a non-negative integer, got {nqubits!r}")
    structures = []
    coefficients_re = []
    coefficients_im = []
    for value, coefficient in terms:
        structures.append(_coerce_structure(nqubits, value))
        normalized = complex(coefficient)
        if not math.isfinite(normalized.real) or not math.isfinite(normalized.imag):
            raise ValueError("coefficients must be finite complex128 values")
        coefficients_re.append(normalized.real)
        coefficients_im.append(normalized.imag)
    return tuple(structures), tuple(coefficients_re), tuple(coefficients_im)


def _normalize_code_array_inputs(
    nqubits: int,
    terms: Sequence[Tuple[PauliInput, complex]],
) -> Optional[Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
        raise ValueError(f"nqubits must be a non-negative integer, got {nqubits!r}")
    if any(isinstance(value, (PauliWord, str)) for value, _ in terms):
        return None
    if not terms:
        return (
            np.empty((0, nqubits), dtype=np.uint8),
            np.empty(0, dtype=np.complex128),
        )
    try:
        structures, coefficients = _normalize_code_arrays(
            [cast(Sequence[int], value) for value, _ in terms],
            [coefficient for _, coefficient in terms],
        )
    except (TypeError, ValueError, OverflowError) as error:
        if "coefficients" in str(error) or "Pauli codes" in str(error):
            raise
        return None
    if structures.shape[1] != nqubits:
        return None
    return structures, coefficients


def _normalize_code_arrays(
    structures: Sequence[Sequence[int]], coefficients: Sequence[complex]
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    try:
        code_array = np.asarray(structures)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "structures must be a rectangular two-dimensional array"
        ) from error
    if code_array.ndim != 2:
        raise ValueError("structures must be a rectangular two-dimensional array")
    if code_array.size and code_array.dtype.kind not in ("i", "u"):
        raise ValueError("Pauli codes must be integers in 0..3")
    if code_array.size and (np.any(code_array < 0) or np.any(code_array > 3)):
        raise ValueError("Pauli codes must be integers in 0..3")
    try:
        coefficient_array = np.asarray(coefficients, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("coefficients must be finite complex128 values") from error
    if coefficient_array.ndim != 1 or coefficient_array.shape[0] != code_array.shape[0]:
        raise ValueError(
            "coefficients must be one-dimensional with one value per structure"
        )
    if not np.all(np.isfinite(coefficient_array)):
        raise ValueError("coefficients must be finite complex128 values")
    return (
        np.ascontiguousarray(code_array, dtype=np.uint8),
        np.ascontiguousarray(coefficient_array, dtype=np.complex128),
    )


def _coerce_structure(nqubits: int, value: PauliInput) -> Tuple[int, ...]:
    if isinstance(value, PauliWord):
        if value.nqubits != nqubits:
            raise ValueError(f"expected {nqubits} qubits, got {value.nqubits}")
        return _codes_from_word(value)
    if isinstance(value, str):
        lookup = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        try:
            structure = tuple(lookup[character] for character in value.upper())
        except KeyError as error:
            raise ValueError(f"invalid Pauli character {error.args[0]!r}") from error
    else:
        structure = tuple(value)
        if any(
            not isinstance(code, int) or isinstance(code, bool) or code not in range(4)
            for code in structure
        ):
            raise ValueError(f"Pauli codes must be integers in 0..3, got {structure!r}")
    if len(structure) != nqubits:
        raise ValueError(f"expected {nqubits} qubits, got {len(structure)}")
    return structure


def _codes_from_word(word: PauliWord) -> Tuple[int, ...]:
    code_lookup = (0, 1, 3, 2)
    return tuple(
        code_lookup[
            ((word.x_words[qubit // 64] >> (qubit % 64)) & 1)
            | (((word.z_words[qubit // 64] >> (qubit % 64)) & 1) << 1)
        ]
        for qubit in range(word.nqubits)
    )


def _word_from_codes(nqubits: int, structure: Sequence[int]) -> PauliWord:
    word_count = (nqubits + 63) // 64
    x_words = [0] * word_count
    z_words = [0] * word_count
    for qubit, code in enumerate(structure):
        word_index, bit_index = divmod(qubit, 64)
        if code in (1, 2):
            x_words[word_index] |= 1 << bit_index
        if code in (2, 3):
            z_words[word_index] |= 1 << bit_index
    return PauliWord(nqubits, tuple(x_words), tuple(z_words))


def _initialize_operator(
    instance: PauliOperator,
    nqubits: int,
    result: Tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]],
) -> None:
    structures, coefficients_re, coefficients_im = result
    canonical_structures = tuple(
        tuple(int(code) for code in structure) for structure in structures
    )
    real_values = tuple(float(value) for value in coefficients_re)
    imaginary_values = tuple(float(value) for value in coefficients_im)
    terms = tuple(
        PauliTerm(_word_from_codes(nqubits, structure), complex(real, imaginary))
        for structure, real, imaginary in zip(
            canonical_structures, real_values, imaginary_values
        )
    )
    object.__setattr__(instance, "nqubits", nqubits)
    object.__setattr__(instance, "terms", terms)
    object.__setattr__(instance, "_canonical_structures", canonical_structures)
    object.__setattr__(instance, "_coefficient_reals", real_values)
    object.__setattr__(instance, "_coefficient_imaginaries", imaginary_values)


def _ensure_operator_compatible(left: PauliOperator, right: object) -> None:
    if not isinstance(right, PauliOperator):
        raise TypeError(f"expected PauliOperator, got {type(right).__name__}")
    if left.nqubits != right.nqubits:
        raise ValueError(
            f"incompatible qubit counts: {left.nqubits} and {right.nqubits}"
        )
