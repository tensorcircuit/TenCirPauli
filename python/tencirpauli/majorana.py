"""Public Majorana algebra and exact fermion conversion helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .hamiltonian import (
    DEFAULT_MAX_BYTES,
    BackendMVPPlan,
    NativeMVPPlan,
    _check_allocation,
    _validate_max_bytes,
)
from .structured import FermionOperator


def _exact_nonnegative(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_complex(value: object, name: str = "coefficient") -> complex:
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise ValueError(f"{name} must be a finite real or complex scalar")
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _canonicalize_indices(
    n_modes: int, indices: Sequence[object]
) -> Tuple[Tuple[int, ...], int]:
    values: List[int] = []
    sign = 1
    for raw_index in indices:
        index = _exact_nonnegative(raw_index, "Majorana index")
        if index >= 2 * n_modes:
            raise ValueError("Majorana index is outside 0..2*n_modes")
        inversions = sum(existing > index for existing in values)
        if inversions & 1:
            sign = -sign
        if index in values:
            values.remove(index)
        else:
            insert_at = 0
            while insert_at < len(values) and values[insert_at] < index:
                insert_at += 1
            values.insert(insert_at, index)
    return tuple(values), sign


def _multiply_canonical(
    left: "MajoranaWord", right: "MajoranaWord"
) -> Tuple["MajoranaWord", int]:
    inversions = sum(
        left_index > right_index
        for left_index in left.indices
        for right_index in right.indices
    )
    support = set(left.indices)
    support.symmetric_difference_update(right.indices)
    return MajoranaWord(left.n_modes, tuple(sorted(support))), (
        -1 if inversions & 1 else 1
    )


def _guard_expansion(branches: int, max_bytes: Optional[int], context: str) -> None:
    _validate_max_bytes(max_bytes)
    if max_bytes is not None:
        _check_allocation(branches * 192, max_bytes, context)


@dataclass(frozen=True)
class MajoranaWord:
    """Canonical phase-free product of Majorana generators."""

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
        """Whether the word is the identity."""
        return not self.indices

    @classmethod
    def from_indices(cls, n_modes: int, indices: Sequence[object]) -> "MajoranaProduct":
        """Canonicalize an arbitrary raw generator sequence and retain its sign."""
        n_modes = _exact_nonnegative(n_modes, "n_modes")
        canonical, sign = _canonicalize_indices(n_modes, indices)
        return MajoranaProduct(cls(n_modes, canonical), sign)

    def multiply(self, other: "MajoranaWord") -> "MajoranaProduct":
        """Multiply two canonical words without branching."""
        if not isinstance(other, MajoranaWord) or other.n_modes != self.n_modes:
            raise ValueError("Majorana words require equal n_modes")
        word, sign = _multiply_canonical(self, other)
        return MajoranaProduct(word, sign)

    def adjoint(self) -> "MajoranaProduct":
        """Return the exact phase-free word adjoint and its sign."""
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
    """Immutable deterministic sparse operator in the Majorana algebra."""

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
        aggregate: Dict[MajoranaWord, complex] = {}
        for raw_indices, coefficient in terms:
            value = _finite_complex(coefficient)
            word, sign = _canonicalize_indices(n_modes, raw_indices)
            canonical = MajoranaWord(n_modes, word)
            aggregate[canonical] = aggregate.get(canonical, 0j) + sign * value
            _guard_expansion(len(aggregate), max_bytes, "Majorana canonicalization")
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
        """Construct from arbitrary raw Majorana factor sequences."""
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
        """Construct one operator from an arbitrary raw generator sequence."""
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

    @property
    def terms(self) -> Tuple[MajoranaTerm, ...]:
        """Return immutable canonical terms in lexicographic order."""
        return self._terms

    @property
    def term_count(self) -> int:
        """Return the number of nonzero canonical terms."""
        return len(self._terms)

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
        """Return the exact canonical sum."""
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
        """Return an exactly scaled operator."""
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
        """Return the exact product, which has one output word per pair."""
        other = self._check_other(other)
        pair_count = len(self._terms) * len(other._terms)
        _guard_expansion(pair_count, max_bytes, "Majorana multiplication")
        aggregate: Dict[MajoranaWord, complex] = {}
        for left in self._terms:
            for right in other._terms:
                word, sign = _multiply_canonical(left.word, right.word)
                aggregate[word] = (
                    aggregate.get(word, 0j)
                    + left.coefficient * right.coefficient * sign
                )
        return self._from_canonical(self.n_modes, aggregate)

    def commutator(
        self,
        other: "MajoranaOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "MajoranaOperator":
        """Return ``self * other - other * self``."""
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
        """Return ``self * other + other * self``."""
        return self.multiply(other, max_bytes=max_bytes).add(
            other.multiply(self, max_bytes=max_bytes), max_bytes=max_bytes
        )

    def adjoint(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "MajoranaOperator":
        """Return the exact coefficient-conjugated adjoint."""
        _guard_expansion(len(self._terms), max_bytes, "Majorana adjoint")
        aggregate: Dict[MajoranaWord, complex] = {}
        for term in self._terms:
            sign = term.word.adjoint().sign
            aggregate[term.word] = term.coefficient.conjugate() * sign
        return self._from_canonical(self.n_modes, aggregate)

    def is_hermitian(self, tolerance: float = 0.0) -> bool:
        """Check equality with the adjoint using an explicit tolerance."""
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
        """Expand exactly into the Phase 7 canonical fermion algebra."""
        branches = sum(1 << term.word.degree for term in self._terms)
        _guard_expansion(branches, max_bytes, "Majorana-to-fermion expansion")
        raw_terms: List[Tuple[Tuple[Tuple[int, str], ...], complex]] = []
        for term in self._terms:
            current: List[Tuple[Tuple[Tuple[int, str], ...], complex]] = [
                ((), term.coefficient)
            ]
            for index in term.word.indices:
                mode, component = divmod(index, 2)
                options = (
                    ((mode, "create"), 1.0 + 0j),
                    ((mode, "annihilate"), 1.0 + 0j),
                )
                if component:
                    options = (
                        ((mode, "create"), 1.0j),
                        ((mode, "annihilate"), -1.0j),
                    )
                current = [
                    ((*factors, factor), coefficient * local)
                    for factors, coefficient in current
                    for factor, local in options
                ]
            raw_terms.extend(current)
        return FermionOperator.from_terms(self.n_modes, raw_terms, max_bytes=max_bytes)

    def map_fermions(
        self,
        mapping: Union[str, Any] = "jordan_wigner",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> Any:
        """Map through a string-selected or reusable fermion mapping plan."""
        from .mapping import FermionQubitMapping

        plan = (
            FermionQubitMapping.from_name(mapping, self.n_modes, max_bytes=max_bytes)
            if isinstance(mapping, str)
            else mapping
        )
        if not isinstance(plan, FermionQubitMapping):
            raise TypeError("mapping must be a supported name or FermionQubitMapping")
        return plan.map_fermion_operator(
            self.to_fermion(max_bytes=max_bytes), max_bytes=max_bytes
        )

    def compile(
        self,
        target: str,
        *,
        mapping: Union[str, Any] = "jordan_wigner",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> Any:
        """Compile after exact fermion conversion and requested qubit mapping."""
        from .mapping import FermionQubitMapping
        from .structured import _with_plan_metadata

        plan = (
            FermionQubitMapping.from_name(mapping, self.n_modes, max_bytes=max_bytes)
            if isinstance(mapping, str)
            else mapping
        )
        if not isinstance(plan, FermionQubitMapping):
            raise TypeError("mapping must be a supported name or FermionQubitMapping")
        result = plan.map_fermion_operator(
            self.to_fermion(max_bytes=max_bytes), max_bytes=max_bytes
        ).compile(target, max_bytes=max_bytes)
        if target in {"native_mvp", "backend_mvp"} and isinstance(
            result, (NativeMVPPlan, BackendMVPPlan)
        ):
            return _with_plan_metadata(
                result,
                mapping=plan.mapping_name,
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
    """Convert a canonical fermion operator through the frozen inverse map."""
    if not isinstance(operator, FermionOperator):
        raise TypeError("fermion_to_majorana expects a FermionOperator")
    fermion_terms = [term for term in operator._terms if term.fermion is not None]
    branches = sum(
        1 << len(term.fermion.factors)
        for term in fermion_terms
        if term.fermion is not None
    )
    _guard_expansion(branches, max_bytes, "Fermion-to-Majorana expansion")
    aggregate: Dict[MajoranaWord, complex] = {}
    for term in fermion_terms:
        assert term.fermion is not None
        word = term.fermion
        current: Dict[Tuple[int, ...], complex] = {(): term.coefficient}
        for mode, action in word.factors:
            options = (
                ((2 * mode,), 0.5 + 0j),
                ((2 * mode + 1,), 0.5j if action == "annihilate" else -0.5j),
            )
            updated: Dict[Tuple[int, ...], complex] = {}
            for raw, coefficient in current.items():
                for factor, local in options:
                    indices, sign = _canonicalize_indices(
                        operator.space.fermions, raw + factor
                    )
                    updated[indices] = (
                        updated.get(indices, 0j) + coefficient * local * sign
                    )
            current = updated
        for indices, coefficient in current.items():
            majorana_word = MajoranaWord(operator.space.fermions, indices)
            aggregate[majorana_word] = aggregate.get(majorana_word, 0j) + coefficient
    return MajoranaOperator._from_canonical(operator.space.fermions, aggregate)
