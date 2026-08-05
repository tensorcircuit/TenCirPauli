"""Typed public Pauli word and static operator APIs."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

from . import _native
from ._validation import normalize_pauli_code, validate_nonnegative_int
from .hamiltonian import (
    _PLAN_FACTORY_TOKEN,
    DEFAULT_MAX_BYTES,
    _check_allocation,
    _effective_max_bytes,
    _validate_max_bytes,
)


if TYPE_CHECKING:
    from .charge import (
        AdditiveCharge,
        AdditiveSymmetryAnalysis,
        ChargeRestrictedOperator,
        ChargeSector,
        ChargeStorage,
    )
    from .grouping import GroupingResult
    from .hamiltonian import (
        BackendMVPPlan,
        CompileResult,
        COOMatrix,
        CSRMatrix,
        NativeMVPPlan,
    )
    from .symmetry import U1RestrictedOperator, U1Sector, Z2SymmetryAnalysis


class PauliPhase(IntEnum):
    """Exact phase labels returned by phase-free Pauli multiplication.

    The integer values encode ``+1``, ``+i``, ``-1``, and ``-i`` in that
    order. Use :attr:`value_complex` when a numerical coefficient is needed.
    """

    PLUS_ONE = 0
    PLUS_I = 1
    MINUS_ONE = 2
    MINUS_I = 3

    @property
    def value_complex(self) -> complex:
        """Return the enumerated phase as a Python complex scalar."""
        return (1.0, 1.0j, -1.0, -1.0j)[int(self)]


_PAULI_CHAR_TO_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_PAULI_CODE_TO_CHAR = "IXYZ"


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
    qubit-zero-is-MSB convention. Words are immutable and canonical, so their
    packed arrays can be safely reused as dictionary keys and native inputs.
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
        """Construct one word from ordered ``0..3`` I/X/Y/Z codes.

        The input order is public qubit order, with qubit zero first. The
        returned word stores the equivalent packed symplectic representation.
        """
        normalized = tuple(normalize_pauli_code(code) for code in codes)
        x_words, z_words = _native.pauli_from_codes(len(normalized), normalized)
        return cls(len(normalized), tuple(x_words), tuple(z_words))

    @classmethod
    def from_string(cls, value: str) -> "PauliWord":
        """Construct one word from an ``IXYZ`` string in qubit order."""
        if not isinstance(value, str):
            raise TypeError("Pauli string must be a str")
        try:
            return cls.from_codes(
                tuple(_PAULI_CHAR_TO_CODE[character] for character in value.upper())
            )
        except KeyError as error:
            raise ValueError(f"invalid Pauli character {error.args[0]!r}") from error

    @classmethod
    def batch_from_codes(
        cls, nqubits: int, structures: Iterable[Sequence[int]]
    ) -> Tuple["PauliWord", ...]:
        """Convert many same-width code rows with one native batch call.

        Each structure must have length ``nqubits`` and contain only integer
        codes ``0..3``. Results preserve input order.
        """
        normalized = tuple(tuple(structure) for structure in structures)
        checked = []
        for structure in normalized:
            if len(structure) != nqubits:
                raise ValueError(
                    f"expected structure length {nqubits}, got {len(structure)}"
                )
            checked.append(tuple(normalize_pauli_code(code) for code in structure))
        normalized = tuple(checked)
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
        """Return immutable external codes in public qubit order."""
        return tuple(_native.pauli_codes(self.nqubits, self.x_words, self.z_words))

    def to_string(self) -> str:
        """Return the canonical ``IXYZ`` string in public qubit order."""
        return "".join(_PAULI_CODE_TO_CHAR[code] for code in self.to_codes())

    def symplectic_inner_product(self, other: "PauliWord") -> int:
        """Return the binary symplectic inner product in ``{0, 1}``.

        A result of ``1`` means the words anticommute; ``0`` means they
        commute. Both words must have the same qubit count.
        """
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
        """Return whether two equal-width Pauli words commute."""
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
        """Multiply words and return the phase-free result plus exact phase.

        The returned :class:`PauliProduct` keeps the phase separate, so a
        caller can apply it to a numerical coefficient without losing the
        canonical word representation.
        """
        _ensure_compatible(self, other)
        x_words, z_words, phase = _native.pauli_multiply(
            self.nqubits,
            self.x_words,
            self.z_words,
            other.x_words,
            other.z_words,
        )
        return PauliProduct(
            PauliWord(self.nqubits, tuple(x_words), tuple(z_words)), PauliPhase(phase)
        )

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
_OperatorArrays = Tuple[Any, Any, Any]


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


@dataclass(frozen=True, init=False, eq=False)
class PauliOperator:
    """Deterministic sparse Pauli operator with exact-zero aggregation.

    Terms are canonicalized on construction, duplicate words are aggregated,
    exact zeros are removed, and surviving terms are sorted deterministically.
    All matrix and MVP targets use the package's documented qubit ordering and
    honor the best-effort ``max_bytes`` guard.
    """

    nqubits: int
    _terms: Optional[Tuple[PauliTerm, ...]] = field(
        default=None, repr=False, compare=False
    )
    _canonical_structures: Optional[Any] = field(
        default=None, repr=False, compare=False
    )
    _coefficient_reals: Optional[Any] = field(default=None, repr=False, compare=False)
    _coefficient_imaginaries: Optional[Any] = field(
        default=None, repr=False, compare=False
    )
    _hermitian_exact: Optional[bool] = field(default=None, repr=False, compare=False)
    _native_handle: Optional[_native.NativePauliOperatorHandle] = field(
        default=None, repr=False, compare=False
    )

    def __init__(
        self, nqubits: int, terms: Iterable[Tuple[PauliInput, complex]]
    ) -> None:
        normalized_terms = tuple(terms)
        array_input = _normalize_code_array_inputs(nqubits, normalized_terms)
        if array_input is None:
            structures, coefficients_re, coefficients_im = _normalize_operator_inputs(
                nqubits, normalized_terms
            )
            handle = _native.pauli_operator_native(
                nqubits,
                structures,
                coefficients_re,
                coefficients_im,
                _effective_max_bytes(DEFAULT_MAX_BYTES),
            )
            _initialize_native_handle(self, handle)
        else:
            handle = _native.pauli_operator_native_array(
                nqubits,
                array_input[0],
                array_input[1],
                _effective_max_bytes(DEFAULT_MAX_BYTES),
            )
            _initialize_native_handle(self, handle)

    @classmethod
    def empty(cls, nqubits: int) -> "PauliOperator":
        """Construct the additive identity on ``nqubits``."""
        return cls(nqubits, ())

    @property
    def terms(self) -> Tuple[PauliTerm, ...]:
        """Return canonical terms, materializing a native result on demand."""
        cached = self._terms
        if cached is None:
            handle = self._native_handle
            if handle is None:
                raise RuntimeError("PauliOperator has no term storage")
            cached = _terms_from_packed_native(self.nqubits, handle.materialize())
            object.__setattr__(self, "_terms", cached)
        return cached

    @classmethod
    def from_terms(
        cls, nqubits: int, terms: Iterable[Tuple[PauliInput, complex]]
    ) -> "PauliOperator":
        """Construct and canonicalize mixed string, code, or word terms.

        Each term is ``(word, coefficient)`` where ``word`` may be an ``IXYZ``
        string, a code sequence, or a :class:`PauliWord`. All words must have
        width ``nqubits``.

        Examples:
            >>> import tencirpauli as tcp
            >>> operator = tcp.PauliOperator.from_terms(
            ...     2, [("XX", 0.5), ("YY", 0.5)]
            ... )
            >>> operator.compile("dense").shape
            (4, 4)
        """
        return cls(nqubits, terms)

    @classmethod
    def from_code_arrays(
        cls,
        structures: Sequence[Sequence[int]],
        coefficients: Sequence[complex],
    ) -> "PauliOperator":
        """Construct from a batch of code rows and complex coefficients.

        This is the preferred constructor for large array-backed inputs because
        it makes one coarse-grained native canonicalization call.
        """
        code_array, coefficient_array = _normalize_code_arrays(structures, coefficients)
        nqubits = int(code_array.shape[1])
        handle = _native.pauli_operator_native_array(
            nqubits,
            code_array,
            coefficient_array,
            _effective_max_bytes(DEFAULT_MAX_BYTES),
        )
        instance = object.__new__(cls)
        _initialize_native_handle(instance, handle)
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
        structures, coefficients_re, coefficients_im = result
        handle = _native.pauli_operator_canonical(
            nqubits,
            structures,
            coefficients_re,
            coefficients_im,
            _effective_max_bytes(DEFAULT_MAX_BYTES),
        )
        _initialize_native_handle(instance, handle)
        return instance

    @classmethod
    def _from_native_handle(
        cls, handle: _native.NativePauliOperatorHandle
    ) -> "PauliOperator":
        instance = object.__new__(cls)
        object.__setattr__(instance, "nqubits", int(handle.nqubits))
        object.__setattr__(instance, "_terms", None)
        object.__setattr__(instance, "_canonical_structures", None)
        object.__setattr__(instance, "_coefficient_reals", None)
        object.__setattr__(instance, "_coefficient_imaginaries", None)
        object.__setattr__(instance, "_hermitian_exact", None)
        object.__setattr__(instance, "_native_handle", handle)
        return instance

    def _arrays(self) -> _OperatorArrays:
        structures = self._canonical_structures
        coefficients_re = self._coefficient_reals
        coefficients_im = self._coefficient_imaginaries
        if structures is None or coefficients_re is None or coefficients_im is None:
            handle = self._native_handle
            if handle is None:
                raise RuntimeError("PauliOperator has no array storage")
            term_count, width, raw_codes, raw_coefficients = handle.materialize_arrays()
            structures = np.asarray(raw_codes, dtype=np.uint8).reshape(
                (int(term_count), int(width))
            )
            coefficients = np.asarray(raw_coefficients, dtype=np.complex128)
            coefficients_re = np.asarray(coefficients.real, dtype=np.float64)
            coefficients_im = np.asarray(coefficients.imag, dtype=np.float64)
            object.__setattr__(self, "_canonical_structures", structures)
            object.__setattr__(self, "_coefficient_reals", coefficients_re)
            object.__setattr__(self, "_coefficient_imaginaries", coefficients_im)
        return (
            structures,
            coefficients_re,
            coefficients_im,
        )

    def _as_native_handle(
        self, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> _native.NativePauliOperatorHandle:
        handle = self._native_handle
        if handle is not None:
            return handle
        _validate_max_bytes(max_bytes)
        structures, coefficients_re, coefficients_im = self._arrays()
        return _native.pauli_operator_native(
            self.nqubits,
            structures,
            coefficients_re,
            coefficients_im,
            _effective_max_bytes(max_bytes),
        )

    @property
    def term_count(self) -> int:
        """Return the number of nonzero canonical algebraic terms."""
        handle = self._native_handle
        if handle is not None:
            return int(handle.term_count)
        terms = self._terms
        if terms is None:
            raise RuntimeError("PauliOperator has no term storage")
        return len(terms)

    def __len__(self) -> int:
        return self.term_count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliOperator):
            return NotImplemented
        return self.nqubits == other.nqubits and self.terms == other.terms

    def __hash__(self) -> int:
        return hash((self.nqubits, self.terms))

    def __repr__(self) -> str:
        storage = "native" if self._native_handle is not None else "python"
        return (
            f"PauliOperator(nqubits={self.nqubits}, "
            f"term_count={self.term_count}, storage={storage!r})"
        )

    def to_dict(self) -> Dict[str, complex]:
        """Return canonical Pauli strings and coefficients without term objects."""
        handle = self._native_handle
        if handle is not None:
            strings, coefficients = handle.materialize_strings()
            values = np.asarray(coefficients, dtype=np.complex128)
            return {string: complex(value) for string, value in zip(strings, values)}
        structures, coefficients_re, coefficients_im = self._arrays()
        return {
            "".join(_PAULI_CODE_TO_CHAR[int(code)] for code in structure): complex(
                real, imaginary
            )
            for structure, real, imaginary in zip(
                structures, coefficients_re, coefficients_im
            )
        }

    def add(
        self,
        other: "PauliOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Add two operators and aggregate exact duplicate keys."""
        _validate_max_bytes(max_bytes)
        _ensure_operator_compatible(self, other)
        if self._native_handle is not None or other._native_handle is not None:
            handle_result = self._as_native_handle(max_bytes).add(
                other._as_native_handle(max_bytes),
                _effective_max_bytes(max_bytes),
            )
            return self._from_native_handle(handle_result)
        left = self._arrays()
        right = other._arrays()
        array_result = _native.pauli_operator_binary(
            self.nqubits,
            left,
            right,
            0,
            _effective_max_bytes(max_bytes),
        )
        return self._from_native(self.nqubits, array_result)

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
        if self._native_handle is not None:
            return self._from_native_handle(
                self._native_handle.scale(normalized.real, normalized.imag)
            )
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
        return self._binary(other, 1, max_bytes)

    def commutator(
        self,
        other: "PauliOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Return ``self * other - other * self``."""
        _validate_max_bytes(max_bytes)
        return self._binary(other, 2, max_bytes)

    def anticommutator(
        self,
        other: "PauliOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Return ``self * other + other * self``."""
        _validate_max_bytes(max_bytes)
        return self._binary(other, 3, max_bytes)

    def adjoint(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "PauliOperator":
        """Return the coefficient-conjugated adjoint operator."""
        _validate_max_bytes(max_bytes)
        if self._native_handle is not None:
            return self._from_native_handle(self._native_handle.adjoint())
        structures, coefficients_re, coefficients_im = self._arrays()
        result = _native.pauli_operator_adjoint(
            self.nqubits, structures, coefficients_re, coefficients_im
        )
        return self._from_native(self.nqubits, result)

    def is_hermitian(self, tolerance: float = 0.0) -> bool:
        """Validate Hermiticity using an explicit non-negative tolerance."""
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Hermiticity tolerance must be finite and non-negative")
        if self._native_handle is not None:
            return bool(self._native_handle.is_hermitian(tolerance))
        if tolerance == 0.0:
            return self._exact_hermitian_value()
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

    def _exact_hermitian_value(self) -> bool:
        """Return exact Hermiticity, caching the immutable operator query."""
        cached = self._hermitian_exact
        if cached is None:
            structures, coefficients_re, coefficients_im = self._arrays()
            cached = bool(
                _native.pauli_operator_is_hermitian(
                    self.nqubits,
                    structures,
                    coefficients_re,
                    coefficients_im,
                    0.0,
                )
            )
            object.__setattr__(self, "_hermitian_exact", cached)
        return cached

    def analyze_charge(
        self,
        charge: "AdditiveCharge",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "AdditiveSymmetryAnalysis":
        """Analyze an exact additive charge using the complete commutator."""
        from .charge import analyze_charge

        return analyze_charge(self, charge, max_bytes=max_bytes)

    def conserves(
        self,
        charge: "AdditiveCharge",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> bool:
        """Return whether this Pauli operator exactly conserves ``charge``."""
        return self.analyze_charge(charge, max_bytes=max_bytes).is_conserved

    def restrict_charge(
        self,
        sector: Union["ChargeSector", "U1Sector"],
        *,
        storage: "ChargeStorage" = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> Union["ChargeRestrictedOperator", "U1RestrictedOperator"]:
        """Restrict an exactly conserved Pauli operator to a charge-sector MVP.

        ``U1Sector`` and equivalent canonical qubit-number charge sectors use
        the packed U(1) backend. The default CPU-native storage is lazy.
        """
        from .symmetry import U1Sector, _canonical_u1_sector, _restrict_u1

        if isinstance(sector, U1Sector):
            return _restrict_u1(
                self, sector, max_bytes, term_count=self.term_count, storage=storage
            )
        canonical_u1 = _canonical_u1_sector(sector)
        if canonical_u1 is not None:
            return _restrict_u1(
                self,
                canonical_u1,
                max_bytes,
                term_count=self.term_count,
                storage=storage,
            )
        from .charge import restrict_charge

        return restrict_charge(self, sector, storage=storage, max_bytes=max_bytes)

    def group_commuting(
        self,
        mode: str = "qubit_wise",
        algorithm: str = "largest_first",
        max_matrix_entries: int = 10_000_000,
    ) -> "GroupingResult":
        """Return a deterministic QWC or general-commuting grouping result.

        Examples:
            >>> import tencirpauli as tcp
            >>> operator = tcp.PauliOperator.from_terms(2, [("XX", 1.0), ("ZZ", 1.0)])
            >>> result = operator.group_commuting(mode="qubit_wise")
            >>> result.term_count
            2
        """
        from .grouping import group_operator

        return group_operator(
            self,
            mode=mode,
            algorithm=algorithm,
            max_matrix_entries=max_matrix_entries,
        )

    def find_z2_symmetries(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "Z2SymmetryAnalysis":
        """Discover deterministic, term-wise commuting Pauli Z2 symmetries."""
        from .symmetry import Z2SymmetryAnalysis

        _validate_max_bytes(max_bytes)
        if self._native_handle is None:
            raise RuntimeError("PauliOperator must retain a native handle")
        generators, constraint_rank = _native.pauli_find_z2_symmetries_handle(
            self._native_handle, _effective_max_bytes(max_bytes)
        )
        return Z2SymmetryAnalysis(
            self.nqubits,
            tuple(PauliWord.from_codes(codes) for codes in generators),
            int(constraint_rank),
        )

    def taper_z2(
        self,
        sector: Sequence[int],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        """Find Z2 symmetries, select ``sector``, and taper this operator."""
        analysis = self.find_z2_symmetries(max_bytes=max_bytes)
        return analysis.tapering_plan(sector).transform_operator(self)

    def restrict_u1(
        self,
        sector: "U1Sector",
        *,
        storage: "ChargeStorage" = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "U1RestrictedOperator":
        """Deprecated alias for :meth:`restrict_charge` with a ``U1Sector``."""
        from .symmetry import U1Sector, _restrict_u1

        warnings.warn(
            "PauliOperator.restrict_u1() is deprecated; use "
            "restrict_charge(U1Sector(...)) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        _validate_max_bytes(max_bytes)
        if not isinstance(sector, U1Sector):
            raise TypeError(f"expected U1Sector, got {type(sector).__name__}")
        if sector.nqubits != self.nqubits:
            raise ValueError(
                f"expected sector for {self.nqubits} qubits, got {sector.nqubits}"
            )
        return _restrict_u1(
            self, sector, max_bytes, term_count=self.term_count, storage=storage
        )

    def compatibility_matrix(
        self, mode: str = "qubit_wise", max_entries: int = 10_000_000
    ) -> np.ndarray[Any, Any]:
        """Return a bounded dense matrix, limited by compatibility entries."""
        validate_nonnegative_int(max_entries, "max_entries")
        mode_code = {"qubit_wise": 0, "general": 1}.get(mode)
        if mode_code is None:
            raise ValueError("mode must be 'qubit_wise' or 'general'")
        structures, _, _ = self._arrays()
        values = _native.pauli_compatibility_matrix(
            self.nqubits, structures, mode_code, max_entries
        )
        size = len(structures)
        matrix: np.ndarray[Any, Any] = np.asarray(values, dtype=np.bool_).reshape(
            (size, size)
        )
        return matrix

    def incompatibility_edges(
        self, mode: str = "qubit_wise", max_edges: int = 10_000_000
    ) -> Tuple[Tuple[int, int], ...]:
        """Return streaming edges, limited by the number of output edges."""
        validate_nonnegative_int(max_edges, "max_edges")
        mode_code = {"qubit_wise": 0, "general": 1}.get(mode)
        if mode_code is None:
            raise ValueError("mode must be 'qubit_wise' or 'general'")
        structures, _, _ = self._arrays()
        return tuple(
            (left, right)
            for left, right in _native.pauli_incompatibility_edges(
                self.nqubits, structures, mode_code, max_edges
            )
        )

    def dense(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> np.ndarray[Any, Any]:
        """Materialize a bounded complex128 dense Hamiltonian matrix."""
        from . import _native

        _validate_max_bytes(max_bytes)
        if self._native_handle is not None:
            dimension, real, imaginary = _native.pauli_dense_handle(
                self._native_handle, _effective_max_bytes(max_bytes)
            )
            values = np.asarray(real, dtype=np.float64) + 1j * np.asarray(
                imaginary, dtype=np.float64
            )
        else:
            structures, coefficients_re, coefficients_im = self._arrays()
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

    def coo(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> "COOMatrix":
        """Compile deterministic, duplicate-aggregated COO arrays."""
        from . import _native
        from .hamiltonian import COOMatrix

        _validate_max_bytes(max_bytes)
        rows: Any
        columns: Any
        values: Any
        if self._native_handle is not None:
            dimension, rows, columns, real, imaginary = _native.pauli_coo_handle(
                self._native_handle, _effective_max_bytes(max_bytes)
            )
            values = np.asarray(real, dtype=np.float64) + 1j * np.asarray(
                imaginary, dtype=np.float64
            )
        else:
            structures, coefficients_re, coefficients_im = self._arrays()
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

    def csr(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> "CSRMatrix":
        """Compile deterministic CSR arrays from the canonical COO stream."""
        from . import _native
        from .hamiltonian import CSRMatrix

        _validate_max_bytes(max_bytes)
        indptr: Any
        indices: Any
        values: Any
        if self._native_handle is not None:
            dimension, indptr, indices, real, imaginary = _native.pauli_csr_handle(
                self._native_handle, _effective_max_bytes(max_bytes)
            )
            values = np.asarray(real, dtype=np.float64) + 1j * np.asarray(
                imaginary, dtype=np.float64
            )
        else:
            structures, coefficients_re, coefficients_im = self._arrays()
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
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the Hamiltonian to a one-dimensional complex128 state."""
        from . import _native

        _validate_max_bytes(max_bytes)
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1:
            raise ValueError(f"state must be one-dimensional, got shape {values.shape}")
        if self._native_handle is not None:
            result = _native.pauli_mvp_handle(
                self._native_handle,
                np.ascontiguousarray(values),
                _effective_max_bytes(max_bytes),
            )
        else:
            structures, coefficients_re, coefficients_im = self._arrays()
            result = _native.pauli_mvp_array(
                self.nqubits,
                structures,
                coefficients_re,
                coefficients_im,
                np.ascontiguousarray(values),
                _effective_max_bytes(max_bytes),
            )
        return cast(np.ndarray[Any, Any], np.asarray(result, dtype=np.complex128))

    def backend_mvp_plan(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "BackendMVPPlan":
        """Compile a versioned pure-array plan for backend execution."""
        from . import _native
        from .hamiltonian import BackendMVPPlan

        _validate_max_bytes(max_bytes)
        if self._native_handle is not None:
            schema, nqubits, word_count, x_words, z_words, real, imaginary = (
                _native.pauli_backend_plan_handle(
                    self._native_handle, _effective_max_bytes(max_bytes)
                )
            )
        else:
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
            estimated_bytes=int(len(real) * (word_count * 16 + 16)),
            _factory_token=_PLAN_FACTORY_TOKEN,
        )

    def native_mvp_plan(
        self,
        *,
        storage: "Literal['lazy', 'eager']" = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "NativeMVPPlan":
        """Compile a reusable Rust-native matrix-free MVP plan."""
        from . import _native
        from .hamiltonian import NativeMVPPlan

        _validate_max_bytes(max_bytes)
        if self._native_handle is not None:
            native_plan = _native.pauli_mvp_plan_handle(
                self._native_handle,
                _effective_max_bytes(max_bytes),
                storage,
            )
        else:
            structures, coefficients_re, coefficients_im = self._arrays()
            native_plan = _native.pauli_mvp_plan(
                self.nqubits,
                structures,
                coefficients_re,
                coefficients_im,
                _effective_max_bytes(max_bytes),
                storage,
            )
        if native_plan.nqubits != self.nqubits:
            raise RuntimeError("native MVP plan has incompatible qubit count")
        if native_plan.term_count != self.term_count:
            raise RuntimeError("native MVP plan has incompatible term count")
        word_count = (self.nqubits + 63) // 64
        term_bytes = self.term_count * (word_count * 16 + 16)
        if native_plan.strategy == "x_mask_diagonal":
            x_mask_count = (
                self._native_handle.distinct_x_mask_count()
                if self._native_handle is not None
                else len(set(self._packed_x_words()))
            )
            estimated_bytes = term_bytes + x_mask_count * (1 << self.nqubits) * 16
        else:
            estimated_bytes = term_bytes
        return NativeMVPPlan(
            self.nqubits,
            self.term_count,
            native_plan.strategy,
            native_plan,
            storage=storage,
            local_dimensions=(2,) * self.nqubits,
            basis_ordering="qubit0_msb_matrix",
            estimated_bytes=estimated_bytes,
            _factory_token=_PLAN_FACTORY_TOKEN,
        )

    def compile(
        self,
        target: str,
        *,
        storage: "Literal['lazy', 'eager']" = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "CompileResult":
        """Compile one named Hamiltonian target.

        Supported targets are ``"dense"``, ``"coo"``, ``"csr"``,
        ``"native_mvp"``, and ``"backend_mvp"``. Dense and sparse targets
        materialize arrays; MVP targets return reusable plans.
        """
        if target == "dense":
            return self.dense(max_bytes=max_bytes)
        if target == "coo":
            return self.coo(max_bytes=max_bytes)
        if target == "csr":
            return self.csr(max_bytes=max_bytes)
        if target == "backend_mvp":
            return self.backend_mvp_plan(max_bytes=max_bytes)
        if target == "native_mvp":
            return self.native_mvp_plan(storage=storage, max_bytes=max_bytes)
        raise ValueError(
            "target must be one of 'dense', 'coo', 'csr', 'native_mvp', or 'backend_mvp'"
        )

    def _binary(
        self,
        other: "PauliOperator",
        operation: int,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "PauliOperator":
        _validate_max_bytes(max_bytes)
        _ensure_operator_compatible(self, other)
        if self._native_handle is not None or other._native_handle is not None:
            left = self._as_native_handle(max_bytes)
            right = other._as_native_handle(max_bytes)
            operation_name = {
                1: "multiply",
                2: "commutator",
                3: "anticommutator",
            }.get(operation)
            if operation_name is None:
                raise ValueError(f"unknown Pauli operator operation {operation}")
            result = getattr(left, operation_name)(
                right, _effective_max_bytes(max_bytes)
            )
            return self._from_native_handle(result)
        result = _native.pauli_operator_binary(
            self.nqubits,
            self._arrays(),
            other._arrays(),
            operation,
            _effective_max_bytes(max_bytes),
        )
        return self._from_native(self.nqubits, result)

    def _packed_x_words(self) -> Tuple[Tuple[int, ...], ...]:
        handle = self._native_handle
        if handle is not None:
            term_count, word_count, x_words, _, _ = handle.materialize()
            values = np.asarray(x_words, dtype=np.uint64).reshape(
                (int(term_count), int(word_count))
            )
            return tuple(tuple(int(word) for word in row) for row in values)
        return tuple(term.word.x_words for term in self.terms)

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
        left_structures, left_reals, left_imaginaries = self._arrays()
        right_structures, right_reals, right_imaginaries = other._arrays()
        pair_count = len(left_structures) * len(right_structures)
        total_qubits = self.nqubits + other.nqubits
        # The pair table is the dominant predictable workspace. Reserve room
        # for its code/coefficient arrays and one comparable canonical output
        # representation before materializing anything.
        _check_allocation(
            pair_count * (2 * total_qubits + 32),
            max_bytes,
            "Pauli tensor-product workspace",
        )
        structures: np.ndarray[Any, Any] = np.empty(
            (pair_count, total_qubits), dtype=np.uint8
        )
        coefficients: np.ndarray[Any, Any] = np.empty(pair_count, dtype=np.complex128)
        pair_index = 0
        for left_index, left_structure in enumerate(left_structures):
            left_coefficient = complex(
                left_reals[left_index], left_imaginaries[left_index]
            )
            for right_index, right_structure in enumerate(right_structures):
                right_coefficient = complex(
                    right_reals[right_index], right_imaginaries[right_index]
                )
                structures[pair_index, :] = np.concatenate(
                    (left_structure, right_structure)
                )
                coefficients[pair_index] = left_coefficient * right_coefficient
                pair_index += 1
        return PauliOperator.from_code_arrays(
            cast(Sequence[Sequence[int]], structures),
            cast(Sequence[complex], coefficients),
        )


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
    if all(isinstance(value, (tuple, list)) and len(value) == 0 for value, _ in terms):
        return _normalize_code_arrays(
            cast(
                Sequence[Sequence[int]],
                np.empty((len(terms), 0), dtype=np.uint8),
            ),
            [coefficient for _, coefficient in terms],
        )
    try:
        structures, coefficients = _normalize_code_arrays(
            [cast(Sequence[int], value) for value, _ in terms],
            [coefficient for _, coefficient in terms],
        )
    except (TypeError, ValueError, OverflowError) as error:
        if "coefficients" in str(error) or "Pauli code" in str(error):
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
    if code_array.dtype.kind not in ("i", "u"):
        raise TypeError("Pauli code arrays must use an integer dtype")
    if code_array.size and (np.any(code_array < 0) or np.any(code_array > 3)):
        raise ValueError("Pauli codes must be in the half-open range 0..4")
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
        try:
            structure = tuple(
                _PAULI_CHAR_TO_CODE[character] for character in value.upper()
            )
        except KeyError as error:
            raise ValueError(f"invalid Pauli character {error.args[0]!r}") from error
    else:
        structure = tuple(normalize_pauli_code(code) for code in value)
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


def _initialize_operator(
    instance: PauliOperator,
    nqubits: int,
    result: Tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]],
) -> None:
    structures, coefficients_re, coefficients_im = result
    handle = _native.pauli_operator_canonical(
        nqubits,
        structures,
        coefficients_re,
        coefficients_im,
        _effective_max_bytes(DEFAULT_MAX_BYTES),
    )
    _initialize_native_handle(instance, handle)


def _initialize_native_handle(
    instance: PauliOperator,
    handle: _native.NativePauliOperatorHandle,
) -> None:
    object.__setattr__(instance, "nqubits", int(handle.nqubits))
    object.__setattr__(instance, "_terms", None)
    object.__setattr__(instance, "_canonical_structures", None)
    object.__setattr__(instance, "_coefficient_reals", None)
    object.__setattr__(instance, "_coefficient_imaginaries", None)
    object.__setattr__(instance, "_hermitian_exact", None)
    object.__setattr__(instance, "_native_handle", handle)


def _terms_from_packed_native(
    nqubits: int,
    result: Tuple[int, int, object, object, object],
) -> Tuple[PauliTerm, ...]:
    term_count, word_count, x_words, z_words, coefficients = result
    canonical_x_words = np.asarray(x_words, dtype=np.uint64).reshape(
        (int(term_count), int(word_count))
    )
    canonical_z_words = np.asarray(z_words, dtype=np.uint64).reshape(
        (int(term_count), int(word_count))
    )
    values = np.asarray(coefficients, dtype=np.complex128)
    return tuple(
        PauliTerm(
            PauliWord(
                nqubits,
                tuple(int(word) for word in x),
                tuple(int(word) for word in z),
            ),
            complex(value),
        )
        for x, z, value in zip(canonical_x_words, canonical_z_words, values)
    )


def _ensure_operator_compatible(left: PauliOperator, right: object) -> None:
    if not isinstance(right, PauliOperator):
        raise TypeError(f"expected PauliOperator, got {type(right).__name__}")
    if left.nqubits != right.nqubits:
        raise ValueError(
            f"incompatible qubit counts: {left.nqubits} and {right.nqubits}"
        )
