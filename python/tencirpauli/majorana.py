"""Public Majorana algebra and exact fermion conversion helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

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

    __slots__ = ("_locked", "_native_handle", "_terms", "n_modes")
    n_modes: int
    _terms: Optional[Tuple[MajoranaTerm, ...]]
    _native_handle: Optional[_native.NativeMajoranaOperatorHandle]
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
        handle = _native.majorana_canonicalize(
            n_modes,
            raw_indices,
            [value.real for value in coefficients],
            [value.imag for value in coefficients],
            _effective_max_bytes(max_bytes),
        )
        self._initialize_native(n_modes, handle)

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
        object.__setattr__(self, "_native_handle", None)
        object.__setattr__(self, "_locked", True)

    def _initialize_native(
        self,
        n_modes: int,
        handle: _native.NativeMajoranaOperatorHandle,
    ) -> None:
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "_native_handle", handle)
        object.__setattr__(self, "_terms", None)
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
        handle: _native.NativeMajoranaOperatorHandle,
    ) -> "MajoranaOperator":
        instance = object.__new__(cls)
        instance._initialize_native(n_modes, handle)
        return instance

    @property
    def terms(self) -> Tuple[MajoranaTerm, ...]:
        """Return immutable nonzero terms in deterministic lexicographic order."""
        cached = self._terms
        if cached is None:
            handle = self._native_handle
            if handle is None:
                raise RuntimeError("MajoranaOperator has no term storage")
            _term_count, payload, offsets, coefficients = handle.materialize()
            values = np.asarray(coefficients, dtype=np.complex128)
            words = np.asarray(payload, dtype=np.uint64)
            stops = np.asarray(offsets, dtype=np.uintp)
            cached = tuple(
                MajoranaTerm(
                    MajoranaWord(
                        self.n_modes, tuple(int(index) for index in words[start:stop])
                    ),
                    complex(value),
                )
                for (start, stop), value in zip(zip(stops[:-1], stops[1:]), values)
            )
            object.__setattr__(self, "_terms", cached)
        return cached

    @property
    def term_count(self) -> int:
        """Return the number of nonzero canonical terms."""
        if self._native_handle is not None:
            return self._native_handle.term_count
        return len(self.terms)

    def to_dict(self) -> Dict[Tuple[int, ...], complex]:
        """Return canonical index tuples without constructing Majorana terms."""
        if self._native_handle is not None:
            _term_count, payload, offsets, coefficients = (
                self._native_handle.materialize()
            )
            words = np.asarray(payload, dtype=np.uint64)
            stops = np.asarray(offsets, dtype=np.uintp)
            values = np.asarray(coefficients, dtype=np.complex128)
            return {
                tuple(int(index) for index in words[start:stop]): complex(value)
                for (start, stop), value in zip(zip(stops[:-1], stops[1:]), values)
            }
        return {term.word.indices: term.coefficient for term in self.terms}

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
        if self._native_handle is not None and other._native_handle is not None:
            return self._from_native(
                self.n_modes,
                self._native_handle.add(
                    other._native_handle, _effective_max_bytes(max_bytes)
                ),
            )
        left_terms = self.terms
        right_terms = other.terms
        aggregate: Dict[MajoranaWord, complex] = {
            term.word: term.coefficient for term in left_terms
        }
        for term in right_terms:
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
        _guard_expansion(self.term_count, max_bytes, "Majorana scaling")
        if self._native_handle is not None:
            return self._from_native(
                self.n_modes,
                self._native_handle.scale(scalar.real, scalar.imag),
            )
        return self._from_canonical(
            self.n_modes,
            {term.word: term.coefficient * scalar for term in self.terms},
        )

    def multiply(
        self,
        other: "MajoranaOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "MajoranaOperator":
        """Return the exact product and aggregate equal output words."""
        other = self._check_other(other)
        pair_count = self.term_count * other.term_count
        _guard_expansion(pair_count, max_bytes, "Majorana multiplication")
        if self._native_handle is not None and other._native_handle is not None:
            return self._from_native(
                self.n_modes,
                self._native_handle.multiply(
                    other._native_handle, _effective_max_bytes(max_bytes)
                ),
            )
        else:
            left_terms = self.terms
            right_terms = other.terms
            left_indices = tuple(term.word.indices for term in left_terms)
            right_indices = tuple(term.word.indices for term in right_terms)
            left_values = tuple(term.coefficient for term in left_terms)
            right_values = tuple(term.coefficient for term in right_terms)
            left_real = tuple(value.real for value in left_values)
            left_imaginary = tuple(value.imag for value in left_values)
            right_real = tuple(value.real for value in right_values)
            right_imaginary = tuple(value.imag for value in right_values)
        handle = _native.majorana_multiply(
            self.n_modes,
            [list(word) for word in left_indices],
            list(left_real),
            list(left_imaginary),
            [list(word) for word in right_indices],
            list(right_real),
            list(right_imaginary),
            _effective_max_bytes(max_bytes),
        )
        return self._from_native(self.n_modes, handle)

    def commutator(
        self,
        other: "MajoranaOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "MajoranaOperator":
        """Return the exact commutator ``self * other - other * self``."""
        other = self._check_other(other)
        if self._native_handle is not None and other._native_handle is not None:
            return self._from_native(
                self.n_modes,
                self._native_handle.commutator(
                    other._native_handle, _effective_max_bytes(max_bytes)
                ),
            )
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
        other = self._check_other(other)
        if self._native_handle is not None and other._native_handle is not None:
            return self._from_native(
                self.n_modes,
                self._native_handle.anticommutator(
                    other._native_handle, _effective_max_bytes(max_bytes)
                ),
            )
        return self.multiply(other, max_bytes=max_bytes).add(
            other.multiply(self, max_bytes=max_bytes), max_bytes=max_bytes
        )

    def adjoint(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "MajoranaOperator":
        """Return the exact coefficient-conjugated operator adjoint."""
        _guard_expansion(self.term_count, max_bytes, "Majorana adjoint")
        if self._native_handle is not None:
            return self._from_native(
                self.n_modes,
                self._native_handle.adjoint(),
            )
        aggregate: Dict[MajoranaWord, complex] = {}
        for term in self.terms:
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
        if self._native_handle is not None:
            return bool(self._native_handle.is_hermitian(float(tolerance)))
        left = self.to_dict()
        right = self.adjoint().to_dict()
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
        handle = self._native_handle
        if handle is None:
            terms = self.terms
            handle = _native.majorana_canonicalize(
                self.n_modes,
                [list(term.word.indices) for term in terms],
                [term.coefficient.real for term in terms],
                [term.coefficient.imag for term in terms],
                _effective_max_bytes(max_bytes),
            )
        return _fermion_from_native(
            FermionOperator,
            self.n_modes,
            handle.to_fermion(_effective_max_bytes(max_bytes)),
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
        storage: Literal["lazy", "eager"] = "lazy",
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
            target, storage=storage, max_bytes=max_bytes
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
        storage = "native" if self._native_handle is not None else "python"
        return f"MajoranaOperator(n_modes={self.n_modes}, term_count={self.term_count}, storage={storage!r})"


def fermion_to_majorana(
    operator: FermionOperator,
    *,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> MajoranaOperator:
    """Convert a canonical fermion operator through one native batch call."""
    if not isinstance(operator, FermionOperator):
        raise TypeError("fermion_to_majorana expects a FermionOperator")
    if isinstance(operator._native_handle, _native.NativeFermionOperatorHandle):
        return MajoranaOperator._from_native(
            operator.space.fermions,
            operator._native_handle.to_majorana(_effective_max_bytes(max_bytes)),
        )
    creation, annihilation, real, imaginary = _fermion_arrays(operator)
    handle = _native.fermion_to_majorana(
        operator.space.fermions,
        creation,
        annihilation,
        real,
        imaginary,
        _effective_max_bytes(max_bytes),
    )
    return MajoranaOperator._from_native(operator.space.fermions, handle)
