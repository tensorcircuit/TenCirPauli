"""Typed public Pauli word and static operator APIs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Iterable, Sequence, Tuple, Union

import numpy as np

from . import _native


if TYPE_CHECKING:
    from .grouping import GroupingResult


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
        for name, words in (("x_words", x_tuple), ("z_words", z_tuple)):
            for index, word in enumerate(words):
                if (
                    not isinstance(word, int)
                    or isinstance(word, bool)
                    or not 0 <= word < 2**64
                ):
                    raise ValueError(
                        f"{name}[{index}] must be an unsigned 64-bit integer"
                    )
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


@dataclass(frozen=True)
class PauliTerm:
    """One canonical Pauli word and its complex128-compatible coefficient."""

    word: PauliWord
    coefficient: complex


@dataclass(frozen=True, init=False)
class PauliOperator:
    """A deterministic static Pauli operator with exact-zero aggregation."""

    nqubits: int
    terms: Tuple[PauliTerm, ...]

    def __init__(
        self, nqubits: int, terms: Iterable[Tuple[PauliInput, complex]]
    ) -> None:
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError(f"nqubits must be a non-negative integer, got {nqubits!r}")
        structures = []
        coefficients = []
        for value, coefficient in terms:
            word = _coerce_word(nqubits, value)
            normalized = complex(coefficient)
            if not math.isfinite(normalized.real) or not math.isfinite(normalized.imag):
                raise ValueError("coefficients must be finite complex128 values")
            structures.append(word.to_codes())
            coefficients.append(normalized)
        result = _native.pauli_canonicalize(
            nqubits,
            structures,
            tuple(value.real for value in coefficients),
            tuple(value.imag for value in coefficients),
        )
        object.__setattr__(self, "nqubits", nqubits)
        object.__setattr__(self, "terms", _terms_from_native(nqubits, result))

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
        object.__setattr__(instance, "nqubits", nqubits)
        object.__setattr__(instance, "terms", _terms_from_native(nqubits, result))
        return instance

    def _arrays(
        self,
    ) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[float, ...], Tuple[float, ...]]:
        structures = tuple(term.word.to_codes() for term in self.terms)
        coefficients = tuple(term.coefficient for term in self.terms)
        return (
            structures,
            tuple(value.real for value in coefficients),
            tuple(value.imag for value in coefficients),
        )

    def add(self, other: "PauliOperator") -> "PauliOperator":
        """Add two operators and aggregate exact duplicate keys."""
        _ensure_operator_compatible(self, other)
        left = self._arrays()
        right = other._arrays()
        result = _native.pauli_operator_binary(self.nqubits, left, right, 0)
        return self._from_native(self.nqubits, result)

    def scale(self, scalar: complex) -> "PauliOperator":
        """Multiply all coefficients by a finite complex scalar."""
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

    def multiply(self, other: "PauliOperator") -> "PauliOperator":
        """Multiply operators, absorbing exact Pauli phases into coefficients."""
        return self._binary(other, 1)

    def commutator(self, other: "PauliOperator") -> "PauliOperator":
        """Return ``self * other - other * self``."""
        return self._binary(other, 2)

    def anticommutator(self, other: "PauliOperator") -> "PauliOperator":
        """Return ``self * other + other * self``."""
        return self._binary(other, 3)

    def adjoint(self) -> "PauliOperator":
        """Return the coefficient-conjugated adjoint operator."""
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

    def compatibility_matrix(
        self, mode: str = "general", max_entries: int = 10_000_000
    ) -> np.ndarray[Any, Any]:
        """Return a bounded dense compatibility matrix for canonical terms."""
        mode_code = {"qubit_wise": 0, "general": 1}.get(mode)
        if mode_code is None:
            raise ValueError("mode must be 'qubit_wise' or 'general'")
        structures = tuple(term.word.to_codes() for term in self.terms)
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
        """Return a bounded streaming edge list without dense matrix allocation."""
        mode_code = {"qubit_wise": 0, "general": 1}.get(mode)
        if mode_code is None:
            raise ValueError("mode must be 'qubit_wise' or 'general'")
        structures = tuple(term.word.to_codes() for term in self.terms)
        return tuple(
            (left, right)
            for left, right in _native.pauli_incompatibility_edges(
                self.nqubits, structures, mode_code, max_edges
            )
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

    def __mul__(self, scalar: object) -> "PauliOperator":
        if isinstance(scalar, PauliOperator):
            return NotImplemented
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        return self.scale(complex(scalar))

    def __rmul__(self, scalar: object) -> "PauliOperator":
        return self * scalar


def _coerce_word(nqubits: int, value: PauliInput) -> PauliWord:
    if isinstance(value, PauliWord):
        if value.nqubits != nqubits:
            raise ValueError(f"expected {nqubits} qubits, got {value.nqubits}")
        return value
    if isinstance(value, str):
        word = PauliWord.from_string(value)
    else:
        word = PauliWord.from_codes(value)
    if word.nqubits != nqubits:
        raise ValueError(f"expected {nqubits} qubits, got {word.nqubits}")
    return word


def _terms_from_native(
    nqubits: int,
    result: Tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]],
) -> Tuple[PauliTerm, ...]:
    structures, coefficients_re, coefficients_im = result
    words = PauliWord.batch_from_codes(nqubits, structures)
    return tuple(
        PauliTerm(word, complex(real, imaginary))
        for word, real, imaginary in zip(words, coefficients_re, coefficients_im)
    )


def _ensure_operator_compatible(left: PauliOperator, right: object) -> None:
    if not isinstance(right, PauliOperator):
        raise TypeError(f"expected PauliOperator, got {type(right).__name__}")
    if left.nqubits != right.nqubits:
        raise ValueError(
            f"incompatible qubit counts: {left.nqubits} and {right.nqubits}"
        )
