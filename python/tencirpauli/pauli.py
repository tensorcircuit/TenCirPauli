"""High-level Pauli word API backed by the private native extension."""

from dataclasses import dataclass
from typing import Sequence, Tuple

from . import _native


@dataclass(frozen=True)
class PauliWord:
    """A phase-free Pauli word in binary symplectic representation."""

    nqubits: int
    x_words: Tuple[int, ...]
    z_words: Tuple[int, ...]

    def __init__(
        self, nqubits: int, x_words: Sequence[int], z_words: Sequence[int]
    ) -> None:
        object.__setattr__(self, "nqubits", nqubits)
        object.__setattr__(self, "x_words", tuple(x_words))
        object.__setattr__(self, "z_words", tuple(z_words))

    @property
    def weight(self) -> int:
        """Return the number of non-identity sites."""
        return int(_native.pauli_weight(self.nqubits, self.x_words, self.z_words))

    def commutes_with(self, other: "PauliWord") -> bool:
        """Return whether this Pauli word commutes with another word."""
        if self.nqubits != other.nqubits:
            raise ValueError(
                f"incompatible qubit counts: {self.nqubits} and {other.nqubits}"
            )
        return bool(
            _native.pauli_commutes(
                self.nqubits,
                self.x_words,
                self.z_words,
                other.x_words,
                other.z_words,
            )
        )
