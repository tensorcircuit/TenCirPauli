"""Public Majorana algebra and exact fermion conversion helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

from . import _native
from ._validation import validate_nonnegative_int
from .hamiltonian import (
    DEFAULT_MAX_BYTES,
    BackendMVPPlan,
    NativeMVPPlan,
    _check_allocation,
    _effective_max_bytes,
    _validate_max_bytes,
)
from .structured import (
    FermionOperator,
    _fermion_arrays,
    _fermion_from_native,
    _finite_complex,
)


_exact_nonnegative = validate_nonnegative_int


def _bit_count(value: int) -> int:
    """Return an integer popcount on every supported Python version."""
    return bin(value).count("1")


def _canonicalize_indices(
    n_modes: int, indices: Sequence[object]
) -> Tuple[Tuple[int, ...], int]:
    support = 0
    sign = 1
    for raw_index in indices:
        index = _exact_nonnegative(raw_index, "Majorana index")
        if index >= 2 * n_modes:
            raise ValueError("Majorana index is outside 0..2*n_modes")
        if _bit_count(support >> (index + 1)) & 1:
            sign = -sign
        support ^= 1 << index
    values = []
    while support:
        low = support & -support
        values.append(low.bit_length() - 1)
        support ^= low
    return tuple(values), sign


def _multiply_canonical(
    left: "MajoranaWord", right: "MajoranaWord"
) -> Tuple["MajoranaWord", int]:
    left_support = sum(1 << index for index in left.indices)
    right_support = sum(1 << index for index in right.indices)
    inversions = sum(
        _bit_count(right_support & ((1 << left_index) - 1))
        for left_index in left.indices
    )
    support = left_support ^ right_support
    indices = []
    while support:
        low = support & -support
        indices.append(low.bit_length() - 1)
        support ^= low
    return MajoranaWord(left.n_modes, tuple(indices)), (-1 if inversions & 1 else 1)


def _guard_expansion(branches: int, max_bytes: Optional[int], context: str) -> None:
    _validate_max_bytes(max_bytes)
    if max_bytes is not None:
        _check_allocation(branches * 192, max_bytes, context)


@dataclass(frozen=True)
class MajoranaWord:
    """Canonical phase-free product of Majorana generators.

    Majorana indices use ``2 * mode`` for the creation-like generator and
    ``2 * mode + 1`` for the annihilation-like generator. ``indices`` is
    strictly increasing; raw products should be constructed with
    :meth:`from_indices` so the fermionic sign is retained.
    """

    n_modes: int
    indices: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        n_modes = _exact_nonnegative(self.n_modes, "n_modes")
        indices = tuple(self.indices)
        if any(
            not isinstance(index, int) or isinstance(index, bool) for index in indices
        ):
            raise ValueError("Majorana indices must be integers")
        if any(index < 0 or index >= 2 * n_modes for index in indices):
            raise ValueError("Majorana index is outside 0..2*n_modes")
        if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise ValueError("MajoranaWord indices must be sorted and strictly unique")
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "indices", indices)

    @property
    def degree(self) -> int:
        """Return the number of generators in the canonical word."""
        return len(self.indices)

    @property
    def is_identity(self) -> bool:
        """Return whether this word contains no Majorana generator."""
        return not self.indices

    @classmethod
    def from_indices(cls, n_modes: int, indices: Sequence[object]) -> "MajoranaProduct":
        """Canonicalize a raw generator sequence and retain its fermionic sign.

        Returns a :class:`MajoranaProduct` because sorting and cancelling
        repeated generators can contribute a sign of ``+1`` or ``-1``.
        """
        n_modes = _exact_nonnegative(n_modes, "n_modes")
        canonical, sign = _canonicalize_indices(n_modes, indices)
        return MajoranaProduct(cls(n_modes, canonical), sign)

    def multiply(self, other: "MajoranaWord") -> "MajoranaProduct":
        """Multiply two canonical words and return the canonical sign."""
        if not isinstance(other, MajoranaWord) or other.n_modes != self.n_modes:
            raise ValueError("Majorana words require equal n_modes")
        word, sign = _multiply_canonical(self, other)
        return MajoranaProduct(word, sign)

    def adjoint(self) -> "MajoranaProduct":
        """Return the word adjoint and its exact reversal sign."""
        sign = -1 if (self.degree * (self.degree - 1) // 2) & 1 else 1
        return MajoranaProduct(self, sign)


@dataclass(frozen=True)
class MajoranaProduct:
    """Coefficient-free result of a canonical Majorana word product."""

    word: MajoranaWord
    sign: int

    def __post_init__(self) -> None:
        if not isinstance(self.word, MajoranaWord) or self.sign not in (-1, 1):
            raise ValueError("MajoranaProduct requires a word and sign in {-1, +1}")


@dataclass(frozen=True)
class MajoranaTerm:
    """One canonical Majorana word and its complex coefficient."""

    word: MajoranaWord
    coefficient: complex

    def __post_init__(self) -> None:
        if not isinstance(self.word, MajoranaWord):
            raise TypeError("MajoranaTerm.word must be a MajoranaWord")
        object.__setattr__(self, "coefficient", _finite_complex(self.coefficient))


class MajoranaOperator:
    """Immutable deterministic sparse operator in the Majorana algebra.

    Input products are canonicalized, duplicate words are aggregated, exact
    zero coefficients are removed, and surviving terms are sorted
    lexicographically by generator indices.
    """

    __slots__ = ("_locked", "_terms", "n_modes")
    n_modes: int
    _terms: Tuple[MajoranaTerm, ...]
    _locked: bool

    def __init__(
        self,
        n_modes: int,
        terms: Iterable[Tuple[Sequence[object], complex]] = (),
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        n_modes = _exact_nonnegative(n_modes, "n_modes")
        raw_indices: List[List[int]] = []
        coefficients: List[complex] = []
        for indices, coefficient in terms:
            normalized_indices: List[int] = []
            for raw_index in indices:
                index = _exact_nonnegative(raw_index, "Majorana index")
                if index >= 2 * n_modes:
                    raise ValueError("Majorana index is outside 0..2*n_modes")
                normalized_indices.append(index)
            raw_indices.append(normalized_indices)
            coefficients.append(_finite_complex(coefficient))
        canonical_indices, real, imaginary = cast(
            Tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]],
            _native.majorana_canonicalize(
                n_modes,
                raw_indices,
                [value.real for value in coefficients],
                [value.imag for value in coefficients],
                _effective_max_bytes(max_bytes),
            ),
        )
        aggregate = {
            MajoranaWord(n_modes, tuple(indices)): complex(re, im)
            for indices, re, im in zip(canonical_indices, real, imaginary)
        }
        self._initialize(n_modes, aggregate)

    def _initialize(self, n_modes: int, aggregate: Dict[MajoranaWord, complex]) -> None:
        terms = tuple(
            MajoranaTerm(word, coefficient)
            for word, coefficient in sorted(
                aggregate.items(), key=lambda item: item[0].indices
            )
            if coefficient.real != 0.0 or coefficient.imag != 0.0
        )
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "_terms", terms)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("MajoranaOperator is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def from_terms(
        cls,
        n_modes: int,
        terms: Iterable[Tuple[Sequence[object], complex]],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "MajoranaOperator":
        """Construct from arbitrary raw Majorana factor sequences.

        Each input term is ``(indices, coefficient)``. Repeated indices are
        reduced with the exact Majorana sign before duplicate words are
        aggregated.
        """
        return cls(n_modes, terms, max_bytes=max_bytes)

    @classmethod
    def from_indices(
        cls,
        n_modes: int,
        indices: Sequence[object],
        coefficient: complex = 1.0,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "MajoranaOperator":
        """Construct one operator term from an arbitrary generator sequence.

        Examples:
            >>> import tencirpauli as tcp
            >>> operator = tcp.MajoranaOperator.from_indices(1, [0, 1])
            >>> operator.term_count
            1
        """
        return cls(n_modes, ((indices, coefficient),), max_bytes=max_bytes)

    @classmethod
    def _from_canonical(
        cls,
        n_modes: int,
        aggregate: Dict[MajoranaWord, complex],
    ) -> "MajoranaOperator":
        instance = object.__new__(cls)
        instance._initialize(n_modes, aggregate)
        return instance

    @classmethod
    def _from_native(
        cls,
        n_modes: int,
        indices: Sequence[Sequence[int]],
        real: Sequence[float],
        imaginary: Sequence[float],
    ) -> "MajoranaOperator":
        aggregate = {
            MajoranaWord(n_modes, tuple(word)): complex(re, im)
            for word, re, im in zip(indices, real, imaginary)
        }
        return cls._from_canonical(n_modes, aggregate)

    @property
    def terms(self) -> Tuple[MajoranaTerm, ...]:
        """Return immutable nonzero terms in deterministic lexicographic order."""
        return self._terms

    @property
    def term_count(self) -> int:
        """Return the number of nonzero canonical terms."""
        return len(self._terms)

    def __len__(self) -> int:
        return self.term_count

    def _check_other(self, other: object) -> "MajoranaOperator":
        if not isinstance(other, MajoranaOperator):
            raise TypeError(f"expected MajoranaOperator, got {type(other).__name__}")
        if other.n_modes != self.n_modes:
            raise ValueError("Majorana operators require equal n_modes")
        return other

    def add(
        self,
        other: "MajoranaOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "MajoranaOperator":
        """Return the exact canonical sum of two equal-mode operators."""
        other = self._check_other(other)
        aggregate: Dict[MajoranaWord, complex] = {
            term.word: term.coefficient for term in self._terms
        }
        for term in other._terms:
            aggregate[term.word] = aggregate.get(term.word, 0j) + term.coefficient
        _guard_expansion(len(aggregate), max_bytes, "Majorana addition")
        return self._from_canonical(self.n_modes, aggregate)

    def scale(
        self,
        coefficient: complex,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "MajoranaOperator":
        """Return a new operator with every coefficient multiplied by ``coefficient``."""
        scalar = _finite_complex(coefficient, "scale")
        _guard_expansion(len(self._terms), max_bytes, "Majorana scaling")
        return self._from_canonical(
            self.n_modes,
            {term.word: term.coefficient * scalar for term in self._terms},
        )

    def multiply(
        self,
        other: "MajoranaOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "MajoranaOperator":
        """Return the exact product and aggregate equal output words."""
        other = self._check_other(other)
        pair_count = len(self._terms) * len(other._terms)
        _guard_expansion(pair_count, max_bytes, "Majorana multiplication")
        left_coefficients = [term.coefficient for term in self._terms]
        right_coefficients = [term.coefficient for term in other._terms]
        indices, real, imaginary = cast(
            Tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]],
            _native.majorana_multiply(
                self.n_modes,
                [list(term.word.indices) for term in self._terms],
                [value.real for value in left_coefficients],
                [value.imag for value in left_coefficients],
                [list(term.word.indices) for term in other._terms],
                [value.real for value in right_coefficients],
                [value.imag for value in right_coefficients],
                _effective_max_bytes(max_bytes),
            ),
        )
        return self._from_native(self.n_modes, indices, real, imaginary)

    def commutator(
        self,
        other: "MajoranaOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "MajoranaOperator":
        """Return the exact commutator ``self * other - other * self``."""
        return self.multiply(other, max_bytes=max_bytes).add(
            other.multiply(self, max_bytes=max_bytes).scale(-1, max_bytes=max_bytes),
            max_bytes=max_bytes,
        )

    def anticommutator(
        self,
        other: "MajoranaOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "MajoranaOperator":
        """Return the exact anticommutator ``self * other + other * self``."""
        return self.multiply(other, max_bytes=max_bytes).add(
            other.multiply(self, max_bytes=max_bytes), max_bytes=max_bytes
        )

    def adjoint(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "MajoranaOperator":
        """Return the exact coefficient-conjugated operator adjoint."""
        _guard_expansion(len(self._terms), max_bytes, "Majorana adjoint")
        aggregate: Dict[MajoranaWord, complex] = {}
        for term in self._terms:
            sign = term.word.adjoint().sign
            aggregate[term.word] = term.coefficient.conjugate() * sign
        return self._from_canonical(self.n_modes, aggregate)

    def is_hermitian(self, tolerance: float = 0.0) -> bool:
        """Return whether the operator equals its adjoint within ``tolerance``."""
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or tolerance < 0
        ):
            raise ValueError("Hermiticity tolerance must be finite and non-negative")
        left = {term.word: term.coefficient for term in self._terms}
        right = {term.word: term.coefficient for term in self.adjoint()._terms}
        return left.keys() == right.keys() and all(
            abs(left[key] - right[key]) <= tolerance for key in left
        )

    def to_fermion(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> FermionOperator:
        """Expand exactly into the canonical ladder-operator algebra.

        Each degree-``d`` Majorana word can produce up to ``2**d`` fermion
        branches. The expansion is guarded by ``max_bytes`` before native
        allocation and returns a canonical :class:`FermionOperator`.
        """
        branches = sum(1 << term.word.degree for term in self._terms)
        _guard_expansion(branches, max_bytes, "Majorana-to-fermion expansion")
        result = _native.majorana_to_fermion(
            self.n_modes,
            [list(term.word.indices) for term in self._terms],
            [term.coefficient.real for term in self._terms],
            [term.coefficient.imag for term in self._terms],
            _effective_max_bytes(max_bytes),
        )
        creation, annihilation, real, imaginary = result
        return _fermion_from_native(
            FermionOperator,
            self.n_modes,
            (
                creation,
                annihilation,
                tuple(complex(re, im) for re, im in zip(real, imaginary)),
            ),
        )

    def map_fermions(
        self,
        mapping: Union[str, Any] = "jordan_wigner",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> Any:
        """Map directly to qubits through a named or reusable mapping plan.

        ``mapping`` may be ``"jordan_wigner"``, ``"parity"``,
        ``"bravyi_kitaev"``, or a :class:`FermionQubitMapping` instance.
        The Majorana expansion is handled in one batched path.
        """
        from .mapping import FermionQubitMapping

        plan = (
            FermionQubitMapping.from_name(mapping, self.n_modes, max_bytes=max_bytes)
            if isinstance(mapping, str)
            else mapping
        )
        if not isinstance(plan, FermionQubitMapping):
            raise TypeError("mapping must be a supported name or FermionQubitMapping")
        return plan.map_majorana_operator(self, max_bytes=max_bytes)

    def compile(
        self,
        target: str,
        *,
        mapping: Union[str, Any] = "jordan_wigner",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> Any:
        """Compile a mapped Majorana operator to a named Hamiltonian target.

        Supported targets are the same as :meth:`PauliOperator.compile` after
        mapping: ``dense``, ``coo``, ``csr``, ``native_mvp``, and
        ``backend_mvp``. The mapping name and source term count are attached to
        reusable plan metadata.
        """
        from .mapping import FermionQubitMapping
        from .structured import _with_plan_metadata

        plan = (
            FermionQubitMapping.from_name(mapping, self.n_modes, max_bytes=max_bytes)
            if isinstance(mapping, str)
            else mapping
        )
        if not isinstance(plan, FermionQubitMapping):
            raise TypeError("mapping must be a supported name or FermionQubitMapping")
        result = plan.map_majorana_operator(self, max_bytes=max_bytes).compile(
            target, max_bytes=max_bytes
        )
        if target in {"native_mvp", "backend_mvp"} and isinstance(
            result, (NativeMVPPlan, BackendMVPPlan)
        ):
            return _with_plan_metadata(
                result,
                mapping=plan.name,
                source_term_count=self.term_count,
            )
        return result

    def __add__(self, other: object) -> "MajoranaOperator":
        if not isinstance(other, MajoranaOperator):
            return NotImplemented
        return self.add(other)

    def __sub__(self, other: object) -> "MajoranaOperator":
        if not isinstance(other, MajoranaOperator):
            return NotImplemented
        return self.add(other.scale(-1))

    def __neg__(self) -> "MajoranaOperator":
        return self.scale(-1)

    def __mul__(self, other: object) -> "MajoranaOperator":
        if isinstance(other, MajoranaOperator):
            return self.multiply(other)
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.scale(complex(other))
        return NotImplemented

    def __rmul__(self, other: object) -> "MajoranaOperator":
        return self * other

    def __repr__(self) -> str:
        return f"MajoranaOperator(n_modes={self.n_modes}, terms={self._terms!r})"


def fermion_to_majorana(
    operator: FermionOperator,
    *,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> MajoranaOperator:
    """Convert a canonical fermion operator through one native batch call."""
    if not isinstance(operator, FermionOperator):
        raise TypeError("fermion_to_majorana expects a FermionOperator")
    creation, annihilation, real, imaginary = _fermion_arrays(operator)
    indices, result_real, result_imaginary = _native.fermion_to_majorana(
        operator.space.fermions,
        creation,
        annihilation,
        real,
        imaginary,
        _effective_max_bytes(max_bytes),
    )
    return MajoranaOperator._from_native(
        operator.space.fermions, indices, result_real, result_imaginary
    )
