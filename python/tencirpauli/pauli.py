"""Typed public Pauli word and static operator APIs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence, Tuple

from . import _native


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
