"""Z2 symmetry tapering and explicit U(1) sector APIs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, cast

import numpy as np

from . import _native
from .hamiltonian import (
    DEFAULT_MAX_BYTES,
    COOMatrix,
    CSRMatrix,
    _effective_max_bytes,
    _validate_max_bytes,
)
from .pauli import PauliOperator, PauliWord


@dataclass(frozen=True)
class Z2SymmetryAnalysis:
    """Deterministic, exactly validated Pauli Z2 symmetry analysis."""

    nqubits: int
    generators: Tuple[PauliWord, ...]
    constraint_rank: int

    @property
    def rank(self) -> int:
        """Return the number of independent commuting generators."""
        return len(self.generators)

    def tapering_plan(self, sector: Sequence[int]) -> "Z2TaperingPlan":
        """Build a reusable Clifford plan for a selected eigenvalue sector."""
        selected = tuple(_validate_sector_value(value) for value in sector)
        native_plan = _native.pauli_z2_tapering_plan(
            self.nqubits,
            tuple(word.to_codes() for word in self.generators),
            selected,
        )
        return Z2TaperingPlan._from_native(native_plan)


@dataclass(frozen=True, init=False)
class Z2TaperingPlan:
    """Reusable Clifford transform and selected Z2 sector."""

    nqubits_before: int
    nqubits_after: int
    generators: Tuple[PauliWord, ...]
    sector: Tuple[int, ...]
    removed_qubits: Tuple[int, ...]
    clifford_operations: np.ndarray[Any, Any]
    _native_plan: Any

    def __init__(self, native_plan: Any) -> None:
        generators = tuple(
            PauliWord.from_codes(codes) for codes in native_plan.generators
        )
        operations_raw = tuple(native_plan.clifford_operations)
        operations = np.asarray(operations_raw, dtype=np.int64).reshape(
            (len(operations_raw), 3)
        )
        operations.flags.writeable = False
        object.__setattr__(self, "nqubits_before", int(native_plan.nqubits_before))
        object.__setattr__(self, "nqubits_after", int(native_plan.nqubits_after))
        object.__setattr__(self, "generators", generators)
        object.__setattr__(
            self, "sector", tuple(int(value) for value in native_plan.sector)
        )
        object.__setattr__(
            self,
            "removed_qubits",
            tuple(int(value) for value in native_plan.removed_qubits),
        )
        object.__setattr__(self, "clifford_operations", operations)
        object.__setattr__(self, "_native_plan", native_plan)

    @classmethod
    def _from_native(cls, native_plan: Any) -> "Z2TaperingPlan":
        return cls(native_plan)

    def transform_operator(self, operator: PauliOperator) -> PauliOperator:
        """Transform a compatible operator and substitute the selected sector."""
        if not isinstance(operator, PauliOperator):
            raise TypeError(f"expected PauliOperator, got {type(operator).__name__}")
        result = self._native_plan.transform_operator(
            operator.nqubits, *operator._arrays()
        )
        return PauliOperator._from_native(self.nqubits_after, result)


@dataclass(frozen=True)
class U1Sector:
    """Fixed-Hamming-weight basis with TensorCircuit integer ordering."""

    nqubits: int
    particle_number: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.nqubits, int)
            or isinstance(self.nqubits, bool)
            or self.nqubits < 0
        ):
            raise ValueError("nqubits must be a non-negative integer")
        if (
            not isinstance(self.particle_number, int)
            or isinstance(self.particle_number, bool)
            or not 0 <= self.particle_number <= self.nqubits
        ):
            raise ValueError("particle_number must be between 0 and nqubits")
        dimension = math.comb(self.nqubits, self.particle_number)
        if dimension > np.iinfo(np.uint64).max:
            raise OverflowError("U1 sector dimension exceeds uint64 restricted indices")
        if dimension > np.iinfo(np.intp).max:
            raise OverflowError("U1 sector dimension exceeds platform indices")

    @property
    def dimension(self) -> int:
        """Number of basis states, ``C(nqubits, particle_number)``."""
        return math.comb(self.nqubits, self.particle_number)

    def rank(self, bitstring: int | Sequence[int]) -> int:
        """Return the ascending-basis rank without materializing the basis."""
        value = _coerce_bitstring(self.nqubits, bitstring)
        if bin(value).count("1") != self.particle_number:
            raise ValueError("bitstring has the wrong Hamming weight")
        rank = 0
        ones = 0
        for position in range(self.nqubits):
            if (value >> (self.nqubits - 1 - position)) & 1:
                remaining = self.nqubits - position - 1
                rank += math.comb(remaining, self.particle_number - ones)
                ones += 1
        return rank

    def unrank(self, index: int) -> int | Tuple[int, ...]:
        """Return the computational-basis integer at a restricted index."""
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < self.dimension
        ):
            raise IndexError("restricted basis index is out of range")
        remaining_index = index
        remaining_ones = self.particle_number
        value = 0
        for position in range(self.nqubits):
            zero_count = math.comb(self.nqubits - position - 1, remaining_ones)
            if remaining_index >= zero_count:
                remaining_index -= zero_count
                value |= 1 << (self.nqubits - 1 - position)
                remaining_ones -= 1
        if self.nqubits <= 64:
            return value
        return tuple(
            (value >> (self.nqubits - 1 - position)) & 1
            for position in range(self.nqubits)
        )

    def basis_words(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> np.ndarray[Any, Any]:
        """Return a read-only packed basis in ascending computational order."""
        _validate_max_bytes(max_bytes)
        dimension, word_count, values = _native.u1_basis_words(
            self.nqubits, self.particle_number, _effective_max_bytes(max_bytes)
        )
        packed = np.asarray(values, dtype=np.uint64)
        if self.nqubits <= 64:
            if self.nqubits == 0:
                result = np.zeros(dimension, dtype=np.uint64)
            else:
                result = packed.reshape((dimension,))
        else:
            result = packed.reshape((dimension, word_count))
        result.flags.writeable = False
        return cast(np.ndarray[Any, Any], result)


@dataclass(frozen=True, init=False)
class U1RestrictedOperator:
    """A validated Pauli operator restricted to one U(1) sector."""

    sector: U1Sector
    dimension: int
    _native_operator: Any

    def __init__(self, sector: U1Sector, native_operator: Any) -> None:
        object.__setattr__(self, "sector", sector)
        object.__setattr__(self, "dimension", int(native_operator.dimension))
        object.__setattr__(self, "_native_operator", native_operator)

    def apply(
        self,
        state: Sequence[complex],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the restricted operator without allocating a full-space state."""
        _validate_max_bytes(max_bytes)
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.shape[0] != self.dimension:
            raise ValueError(
                f"state must have shape ({self.dimension},), got {values.shape}"
            )
        return cast(
            np.ndarray[Any, Any],
            np.asarray(
                self._native_operator.apply(
                    np.ascontiguousarray(values), _effective_max_bytes(max_bytes)
                ),
                dtype=np.complex128,
            ),
        )

    def mvp_plan(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> "U1MvpPlan":
        """Build a reusable restricted matrix-free plan."""
        _validate_max_bytes(max_bytes)
        return U1MvpPlan(
            self.sector, self._native_operator.mvp_plan(_effective_max_bytes(max_bytes))
        )

    def dense(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> np.ndarray[Any, Any]:
        """Materialize the bounded dense matrix in restricted-space ordering."""
        _validate_max_bytes(max_bytes)
        dimension, values = self._native_operator.dense(_effective_max_bytes(max_bytes))
        return cast(
            np.ndarray[Any, Any],
            np.asarray(values, dtype=np.complex128).reshape((dimension, dimension)),
        )

    def coo(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> COOMatrix:
        """Materialize deterministic COO arrays in restricted-space ordering."""
        _validate_max_bytes(max_bytes)
        dimension, rows, columns, values = self._native_operator.coo(
            _effective_max_bytes(max_bytes)
        )
        return COOMatrix(
            np.asarray(rows, dtype=np.uint64),
            np.asarray(columns, dtype=np.uint64),
            np.asarray(values, dtype=np.complex128),
            (dimension, dimension),
        )

    def csr(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> CSRMatrix:
        """Materialize bounded CSR arrays in restricted-basis ordering."""
        _validate_max_bytes(max_bytes)
        dimension, indptr, indices, values = self._native_operator.csr(
            _effective_max_bytes(max_bytes)
        )
        return CSRMatrix(
            np.asarray(indptr, dtype=np.uint64),
            np.asarray(indices, dtype=np.uint64),
            np.asarray(values, dtype=np.complex128),
            (dimension, dimension),
        )


@dataclass(frozen=True, init=False)
class U1MvpPlan:
    """Reusable matrix-free plan over a fixed-particle-number basis."""

    sector: U1Sector
    dimension: int
    _native_plan: Any

    def __init__(self, sector: U1Sector, native_plan: Any) -> None:
        object.__setattr__(self, "sector", sector)
        object.__setattr__(self, "dimension", int(native_plan.dimension))
        object.__setattr__(self, "_native_plan", native_plan)

    def apply(
        self,
        state: Sequence[complex],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the reusable restricted plan."""
        _validate_max_bytes(max_bytes)
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.shape[0] != self.dimension:
            raise ValueError(
                f"state must have shape ({self.dimension},), got {values.shape}"
            )
        return cast(
            np.ndarray[Any, Any],
            np.asarray(
                self._native_plan.apply(
                    np.ascontiguousarray(values), _effective_max_bytes(max_bytes)
                ),
                dtype=np.complex128,
            ),
        )


def _validate_sector_value(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 1):
        raise ValueError("sector values must be +1 or -1")
    return value


def _coerce_bitstring(nqubits: int, bitstring: int | Sequence[int]) -> int:
    if isinstance(bitstring, bool):
        raise TypeError("bitstring must be an integer or a sequence of bits")
    if isinstance(bitstring, int):
        value = bitstring
    else:
        bits = tuple(bitstring)
        if len(bits) != nqubits or any(bit not in (0, 1) for bit in bits):
            raise ValueError("bitstring must contain exactly nqubits binary values")
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
    if value < 0 or (nqubits < value.bit_length()):
        raise ValueError("bitstring is outside the computational basis")
    return value


def _restrict_u1(
    operator: PauliOperator,
    sector: U1Sector,
    max_bytes: Optional[int],
) -> U1RestrictedOperator:
    native_operator = _native.pauli_restrict_u1(
        operator.nqubits,
        *operator._arrays(),
        sector.particle_number,
        _effective_max_bytes(max_bytes),
    )
    return U1RestrictedOperator(sector, native_operator)
