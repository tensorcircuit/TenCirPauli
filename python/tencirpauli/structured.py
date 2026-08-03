"""Structured fermion, boson, Weyl, and hybrid operator algebra.

The symbolic kernels in this module deliberately keep their representations
small: fermion words are two mode tuples, boson words are power blocks, and
Weyl words are modular site triples.  Public operations canonicalize a whole
operator at a time and finite targets use one mixed-radix basis convention.
"""

from __future__ import annotations

import cmath
import math
import numbers
import struct
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np

from . import _native
from .hamiltonian import (
    DEFAULT_MAX_BYTES,
    BackendMVPPlan,
    COOMatrix,
    CSRMatrix,
    NativeMVPPlan,
    _check_allocation,
    _effective_max_bytes,
    _validate_max_bytes,
)
from .pauli import PauliOperator


_U32_MAX = 2**32 - 1
_STRUCTURED_NATIVE_SPARSE_WORK_THRESHOLD = 64
_IDENTITY_CODES = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_PAULI_PRODUCT: Tuple[Tuple[Tuple[int, complex], ...], ...] = (
    ((0, 1), (1, 1), (2, 1), (3, 1)),
    ((1, 1), (0, 1), (3, 1j), (2, -1j)),
    ((2, 1), (3, -1j), (0, 1), (1, 1j)),
    ((3, 1), (2, 1j), (1, -1j), (0, 1)),
)


def _finite_complex(value: object, name: str = "coefficient") -> complex:
    if isinstance(value, (bool, np.ndarray)) or not isinstance(
        value, (numbers.Real, numbers.Complex)
    ):
        raise ValueError(f"{name} must be a finite real or complex scalar")
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real or complex scalar") from error
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_mode(mode: object, count: int, name: str = "mode") -> int:
    value = _nonnegative_int(mode, name)
    if value >= count:
        raise ValueError(f"{name} {value} is outside 0..{count}")
    return value


def _action(action: object) -> str:
    if action not in ("create", "annihilate"):
        raise ValueError("action must be exactly 'create' or 'annihilate'")
    return str(action)


def _factor(value: object, count: int) -> Tuple[int, str]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("fermion and boson factors must be (mode, action) pairs")
    return _positive_mode(value[0], count), _action(value[1])


def _estimate_terms(term_count: int, bytes_per_term: int = 160) -> int:
    return max(1, int(term_count)) * bytes_per_term


def _guard_terms(term_count: int, max_bytes: Optional[int], context: str) -> None:
    _validate_max_bytes(max_bytes)
    if max_bytes is not None:
        _check_allocation(_estimate_terms(term_count), max_bytes, context)


def _complex_sort_key(value: complex) -> Tuple[bytes, bytes]:
    return struct.pack("!d", value.real), struct.pack("!d", value.imag)


@dataclass(frozen=True)
class FermionWord:
    """Canonical fermionic monomial ``creations * annihilations``."""

    n_modes: int
    creation_modes: Tuple[int, ...] = ()
    annihilation_modes: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        n_modes = _nonnegative_int(self.n_modes, "n_modes")
        creations = tuple(self.creation_modes)
        annihilations = tuple(self.annihilation_modes)
        if any(
            not isinstance(mode, int)
            or isinstance(mode, bool)
            or not 0 <= mode < n_modes
            for mode in (*creations, *annihilations)
        ):
            raise ValueError("fermion mode is outside the operator space")
        if creations != tuple(sorted(creations)) or len(set(creations)) != len(
            creations
        ):
            raise ValueError("creation_modes must be strictly increasing")
        if annihilations != tuple(sorted(annihilations, reverse=True)) or len(
            set(annihilations)
        ) != len(annihilations):
            raise ValueError("annihilation_modes must be strictly decreasing")
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "creation_modes", creations)
        object.__setattr__(self, "annihilation_modes", annihilations)

    @property
    def is_identity(self) -> bool:
        return not self.creation_modes and not self.annihilation_modes

    @property
    def parity(self) -> int:
        return (len(self.creation_modes) + len(self.annihilation_modes)) & 1

    @property
    def factors(self) -> Tuple[Tuple[int, str], ...]:
        return tuple((mode, "create") for mode in self.creation_modes) + tuple(
            (mode, "annihilate") for mode in self.annihilation_modes
        )

    @classmethod
    def from_factors(
        cls,
        n_modes: int,
        factors: Sequence[Tuple[int, str]],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionOperator":
        return FermionOperator.from_terms(
            n_modes, ((tuple(factors), 1.0),), max_bytes=max_bytes
        )

    def adjoint(self) -> "FermionOperator":
        factors = tuple(
            (mode, "create") for mode in reversed(self.annihilation_modes)
        ) + tuple((mode, "annihilate") for mode in reversed(self.creation_modes))
        return FermionOperator.from_terms(self.n_modes, ((factors, 1.0),))

    def multiply(self, other: "FermionWord") -> "FermionOperator":
        if not isinstance(other, FermionWord) or other.n_modes != self.n_modes:
            raise ValueError("fermion words require equal n_modes")
        return FermionOperator.from_terms(
            self.n_modes, ((self.factors + other.factors, 1.0),)
        )

    def __mul__(self, other: "FermionWord") -> "FermionOperator":
        return self.multiply(other)


@dataclass(frozen=True)
class BosonWord:
    """Canonical normal-ordered bosonic power blocks."""

    n_modes: int
    blocks: Tuple[Tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        n_modes = _nonnegative_int(self.n_modes, "n_modes")
        blocks = tuple(tuple(block) for block in self.blocks)
        previous = -1
        normalized: List[Tuple[int, int, int]] = []
        for block in blocks:
            if len(block) != 3:
                raise ValueError(
                    "boson blocks must be (mode, creation_power, annihilation_power)"
                )
            mode, creation, annihilation = block
            mode = _positive_mode(mode, n_modes)
            creation = _nonnegative_int(creation, "creation_power")
            annihilation = _nonnegative_int(annihilation, "annihilation_power")
            if mode <= previous:
                raise ValueError("boson blocks must be strictly increasing by mode")
            previous = mode
            if creation or annihilation:
                normalized.append((mode, creation, annihilation))
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "blocks", tuple(normalized))

    @property
    def is_identity(self) -> bool:
        return not self.blocks

    @property
    def factors(self) -> Tuple[Tuple[int, str], ...]:
        result: List[Tuple[int, str]] = []
        for mode, creation, annihilation in self.blocks:
            result.extend((mode, "create") for _ in range(creation))
            result.extend((mode, "annihilate") for _ in range(annihilation))
        return tuple(result)

    def adjoint(self) -> "BosonOperator":
        factors = tuple(
            (mode, "create")
            for mode, _, annihilation in reversed(self.blocks)
            for _ in range(annihilation)
        ) + tuple(
            (mode, "annihilate")
            for mode, creation, _ in reversed(self.blocks)
            for _ in range(creation)
        )
        return BosonOperator.from_terms(self.n_modes, ((factors, 1.0),))

    def multiply(self, other: "BosonWord") -> "BosonOperator":
        if not isinstance(other, BosonWord) or other.n_modes != self.n_modes:
            raise ValueError("boson words require equal n_modes")
        return BosonOperator.from_terms(
            self.n_modes, ((self.factors + other.factors, 1.0),)
        )

    def __mul__(self, other: "BosonWord") -> "BosonOperator":
        return self.multiply(other)


@dataclass(frozen=True)
class QuditWeylWord:
    """Direct-convention ``X**a Z**b`` word with modular exponents."""

    dimension: int
    triples: Tuple[Tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        dimension = _nonnegative_int(self.dimension, "dimension")
        if not 3 <= dimension <= _U32_MAX:
            raise ValueError("qudit dimension must satisfy 3 <= dimension <= 2**32-1")
        triples = tuple(tuple(triple) for triple in self.triples)
        previous = -1
        normalized: List[Tuple[int, int, int]] = []
        for triple in triples:
            if len(triple) != 3:
                raise ValueError("Weyl triples must be (site, a, b)")
            site, a, b = triple
            site = _nonnegative_int(site, "site")
            a = _nonnegative_int(a, "a") % dimension
            b = _nonnegative_int(b, "b") % dimension
            if site <= previous:
                raise ValueError("Weyl triples must be strictly increasing by site")
            previous = site
            if a or b:
                normalized.append((site, a, b))
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "triples", tuple(normalized))

    @property
    def is_identity(self) -> bool:
        return not self.triples

    @property
    def n_sites(self) -> int:
        return 0 if not self.triples else self.triples[-1][0] + 1

    def multiply(self, other: "QuditWeylWord") -> "QuditProduct":
        if not isinstance(other, QuditWeylWord) or other.dimension != self.dimension:
            raise ValueError("Weyl words require equal local dimensions")
        left = {site: (a, b) for site, a, b in self.triples}
        right = {site: (a, b) for site, a, b in other.triples}
        exponent = 0
        result: List[Tuple[int, int, int]] = []
        for site in sorted(set(left) | set(right)):
            a, b = left.get(site, (0, 0))
            c, e = right.get(site, (0, 0))
            exponent = (exponent + b * c) % self.dimension
            aa, bb = (a + c) % self.dimension, (b + e) % self.dimension
            if aa or bb:
                result.append((site, aa, bb))
        return QuditProduct(QuditWeylWord(self.dimension, tuple(result)), exponent)

    def commutes_with(self, other: "QuditWeylWord") -> bool:
        if not isinstance(other, QuditWeylWord) or other.dimension != self.dimension:
            raise ValueError("Weyl words require equal local dimensions")
        left = {site: (a, b) for site, a, b in self.triples}
        right = {site: (a, b) for site, a, b in other.triples}
        exponent = sum(
            b * c - e * a
            for site in set(left) | set(right)
            for a, b in (left.get(site, (0, 0)),)
            for c, e in (right.get(site, (0, 0)),)
        )
        return exponent % self.dimension == 0

    def adjoint(self) -> "QuditProduct":
        exponent = sum(a * b for _, a, b in self.triples) % self.dimension
        result = tuple(
            (site, (-a) % self.dimension, (-b) % self.dimension)
            for site, a, b in self.triples
            if a or b
        )
        return QuditProduct(QuditWeylWord(self.dimension, result), exponent)


@dataclass(frozen=True)
class QuditProduct:
    word: QuditWeylWord
    phase_exponent: int


@dataclass(frozen=True)
class FermionTerm:
    word: FermionWord
    coefficient: complex


@dataclass(frozen=True)
class BosonTerm:
    word: BosonWord
    coefficient: complex


@dataclass(frozen=True)
class QuditWeylTerm:
    word: QuditWeylWord
    coefficient: complex


def _normal_order_fermions(
    factors: Tuple[Tuple[int, str], ...], n_modes: int
) -> Dict[FermionWord, int]:
    def visit(
        sequence: Tuple[Tuple[int, str], ...],
    ) -> Dict[Tuple[Tuple[int, str], ...], int]:
        for index in range(len(sequence) - 1):
            left_mode, left_action = sequence[index]
            right_mode, right_action = sequence[index + 1]
            inversion = False
            zero = False
            sign = -1
            if left_action == "create" and right_action == "create":
                inversion = left_mode > right_mode
                zero = left_mode == right_mode
            elif left_action == "annihilate" and right_action == "annihilate":
                inversion = left_mode < right_mode
                zero = left_mode == right_mode
            elif left_action == "annihilate" and right_action == "create":
                inversion = True
            if zero:
                return {}
            if not inversion:
                continue
            swapped = (
                *sequence[:index],
                sequence[index + 1],
                sequence[index],
                *sequence[index + 2 :],
            )
            result: Dict[Tuple[Tuple[int, str], ...], int] = {}
            for key, value in visit(swapped).items():
                result[key] = result.get(key, 0) + sign * value
            if (
                left_action == "annihilate"
                and right_action == "create"
                and left_mode == right_mode
            ):
                contracted = sequence[:index] + sequence[index + 2 :]
                for key, value in visit(contracted).items():
                    result[key] = result.get(key, 0) + value
            return {key: value for key, value in result.items() if value}
        return {sequence: 1}

    words: Dict[FermionWord, int] = {}
    for sequence, coefficient in visit(factors).items():
        creations = tuple(mode for mode, action in sequence if action == "create")
        annihilations = tuple(
            mode for mode, action in sequence if action == "annihilate"
        )
        word = FermionWord(n_modes, creations, annihilations)
        words[word] = words.get(word, 0) + coefficient
    return {word: coefficient for word, coefficient in words.items() if coefficient}


def _normal_order_bosons(
    factors: Tuple[Tuple[int, str], ...], n_modes: int
) -> Dict[BosonWord, int]:
    def visit(
        sequence: Tuple[Tuple[int, str], ...],
    ) -> Dict[Tuple[Tuple[int, str], ...], int]:
        for index in range(len(sequence) - 1):
            left_mode, left_action = sequence[index]
            right_mode, right_action = sequence[index + 1]
            inversion = (left_mode > right_mode) or (
                left_mode == right_mode
                and left_action == "annihilate"
                and right_action == "create"
            )
            if not inversion:
                continue
            swapped = (
                *sequence[:index],
                sequence[index + 1],
                sequence[index],
                *sequence[index + 2 :],
            )
            result = dict(visit(swapped))
            if (
                left_mode == right_mode
                and left_action == "annihilate"
                and right_action == "create"
            ):
                contracted = sequence[:index] + sequence[index + 2 :]
                for key, value in visit(contracted).items():
                    result[key] = result.get(key, 0) + value
            return {key: value for key, value in result.items() if value}
        return {sequence: 1}

    words: Dict[BosonWord, int] = {}
    for sequence, coefficient in visit(factors).items():
        by_mode: Dict[int, List[int]] = {}
        for mode, action in sequence:
            powers = by_mode.setdefault(mode, [0, 0])
            powers[0 if action == "create" else 1] += 1
        blocks = tuple(
            (mode, values[0], values[1]) for mode, values in sorted(by_mode.items())
        )
        word = BosonWord(n_modes, blocks)
        words[word] = words.get(word, 0) + coefficient
    return {word: coefficient for word, coefficient in words.items() if coefficient}


def _aggregate_terms(
    terms: Iterable[Tuple[Tuple[Any, ...], complex]],
    max_bytes: Optional[int],
    context: str,
) -> Tuple[Tuple[Tuple[Any, ...], complex], ...]:
    buckets: Dict[Tuple[Any, ...], List[complex]] = {}
    count = 0
    for key, coefficient in terms:
        count += 1
        _guard_terms(count, max_bytes, context)
        buckets.setdefault(key, []).append(_finite_complex(coefficient))
    output: List[Tuple[Tuple[Any, ...], complex]] = []
    for key in sorted(buckets):
        values = sorted(buckets[key], key=_complex_sort_key)
        coefficient = sum(values, 0j)
        if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
            raise OverflowError("coefficient arithmetic produced a non-finite value")
        if coefficient.real != 0.0 or coefficient.imag != 0.0:
            output.append((key, coefficient))
    return tuple(output)


@dataclass(frozen=True)
class _Axis:
    domain: str
    index: int
    dimension: int


class OperatorSpace:
    """Immutable logical subsystem layout used by structured operators."""

    __slots__ = ("_axes", "_locked", "bosons", "fermions", "qubits", "qudits")
    fermions: int
    bosons: int
    qubits: int
    qudits: Tuple[int, ...]
    _axes: Tuple[_Axis, ...]
    _locked: bool

    def __init__(
        self,
        fermions: int = 0,
        bosons: int = 0,
        qubits: int = 0,
        qudits: Sequence[int] = (),
    ) -> None:
        fermions = _nonnegative_int(fermions, "fermions")
        bosons = _nonnegative_int(bosons, "bosons")
        qubits = _nonnegative_int(qubits, "qubits")
        if not isinstance(qudits, (tuple, list)):
            raise ValueError("qudits must be a tuple of local dimensions")
        dimensions = tuple(
            _nonnegative_int(value, "qudit dimension") for value in qudits
        )
        if dimensions and (
            any(value < 3 for value in dimensions) or len(set(dimensions)) != 1
        ):
            raise ValueError("qudits must have one uniform dimension d >= 3")
        axes = tuple(
            [_Axis("fermion", index, 2) for index in range(fermions)]
            + [_Axis("boson", index, 0) for index in range(bosons)]
            + [_Axis("qubit", index, 2) for index in range(qubits)]
            + [_Axis("qudit", index, dimensions[0]) for index in range(len(dimensions))]
        )
        self._initialize(fermions, bosons, qubits, dimensions, axes)

    @classmethod
    def _from_axes(cls, axes: Sequence[_Axis]) -> "OperatorSpace":
        instance = object.__new__(cls)
        counts = {domain: 0 for domain in ("fermion", "boson", "qubit", "qudit")}
        dimensions: List[int] = []
        for axis in axes:
            counts[axis.domain] += 1
            if axis.domain == "qudit":
                dimensions.append(axis.dimension)
        instance._initialize(
            counts["fermion"],
            counts["boson"],
            counts["qubit"],
            tuple(dimensions),
            tuple(axes),
        )
        return instance

    def _initialize(
        self,
        fermions: int,
        bosons: int,
        qubits: int,
        qudits: Tuple[int, ...],
        axes: Tuple[_Axis, ...],
    ) -> None:
        object.__setattr__(self, "fermions", fermions)
        object.__setattr__(self, "bosons", bosons)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "qudits", qudits)
        object.__setattr__(self, "_axes", axes)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("OperatorSpace is immutable")
        object.__setattr__(self, name, value)

    @property
    def axes(self) -> Tuple[Tuple[str, int, int], ...]:
        return tuple((axis.domain, axis.index, axis.dimension) for axis in self._axes)

    @property
    def layout_fingerprint(self) -> Tuple[Tuple[str, int, int], ...]:
        return self.axes

    @property
    def local_dimensions(self) -> Tuple[int, ...]:
        if any(axis.domain == "boson" and axis.dimension == 0 for axis in self._axes):
            raise ValueError("boson local dimensions require explicit finite cutoffs")
        return tuple(axis.dimension for axis in self._axes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OperatorSpace) and self.axes == other.axes

    def __hash__(self) -> int:
        return hash(self.axes)

    @property
    def fermion(self) -> "_FermionFactory":
        return _FermionFactory(self)

    @property
    def boson(self) -> "_BosonFactory":
        return _BosonFactory(self)

    @property
    def qubit(self) -> "_QubitFactory":
        return _QubitFactory(self)

    @property
    def qudit(self) -> "_QuditFactory":
        return _QuditFactory(self)

    def builder(self) -> "OperatorBuilder":
        return OperatorBuilder(self)

    def embed(
        self, operator: "_StructuredOperator", **maps: object
    ) -> "_StructuredOperator":
        if not isinstance(operator, _StructuredOperator):
            raise TypeError("embed expects a Phase 7 structured operator")
        if operator.space == self:
            return operator
        mappings: Dict[str, Dict[int, int]] = {}
        for domain, source_count in (
            ("fermions", operator.space.fermions),
            ("bosons", operator.space.bosons),
            ("qubits", operator.space.qubits),
            ("qudits", len(operator.space.qudits)),
        ):
            supplied = maps.get(domain)
            if source_count == 0:
                mappings[domain] = {}
                continue
            if supplied is None:
                if getattr(self, domain if domain != "qudits" else "qudits") == getattr(
                    operator.space, domain if domain != "qudits" else "qudits"
                ):
                    mappings[domain] = {index: index for index in range(source_count)}
                else:
                    raise ValueError(f"an explicit {domain} embedding map is required")
            elif isinstance(supplied, Mapping):
                mappings[domain] = {
                    int(key): int(value) for key, value in supplied.items()
                }
            else:
                if not isinstance(supplied, (tuple, list)):
                    raise ValueError(
                        f"{domain} embedding must be a mapping or sequence"
                    )
                values = tuple(supplied)
                mappings[domain] = {
                    index: int(value) for index, value in enumerate(values)
                }
            if set(mappings[domain]) != set(range(source_count)):
                raise ValueError(
                    f"{domain} embedding must map every source index exactly once"
                )
            target_count = (
                len(self.qudits) if domain == "qudits" else getattr(self, domain)
            )
            if any(
                value < 0 or value >= target_count
                for value in mappings[domain].values()
            ):
                raise ValueError(f"{domain} embedding indices must be non-negative")
        terms: List[_Term] = []
        for term in operator._terms:
            fermion = (
                None
                if term.fermion is None
                else FermionWord(
                    self.fermions,
                    tuple(
                        mappings["fermions"][mode]
                        for mode in term.fermion.creation_modes
                    ),
                    tuple(
                        mappings["fermions"][mode]
                        for mode in term.fermion.annihilation_modes
                    ),
                )
            )
            boson = (
                None
                if term.boson is None
                else BosonWord(
                    self.bosons,
                    tuple(
                        (mappings["bosons"][mode], create, annihilate)
                        for mode, create, annihilate in term.boson.blocks
                    ),
                )
            )
            qubit = [0] * self.qubits
            for index, code in enumerate(term.qubit):
                if code:
                    qubit[mappings["qubits"][index]] = code
            qudit = (
                None
                if term.qudit is None
                else QuditWeylWord(
                    self.qudits[0],
                    tuple(
                        (mappings["qudits"][site], a, b)
                        for site, a, b in term.qudit.triples
                    ),
                )
            )
            mapped = None
            if term.mapped_fermion is not None:
                expanded = [0] * self.fermions
                for index, code in enumerate(term.mapped_fermion):
                    expanded[mappings["fermions"][index]] = code
                mapped = tuple(expanded)
            terms.append(
                _Term(fermion, boson, tuple(qubit), qudit, mapped, term.coefficient)
            )
        return _make_operator(self, terms, DEFAULT_MAX_BYTES)


class _Factory:
    def __init__(self, space: OperatorSpace) -> None:
        self.space = space


class _FermionFactory(_Factory):
    @property
    def annihilate(self) -> Any:
        return lambda mode: self._make(mode, "annihilate")

    @property
    def create(self) -> Any:
        return lambda mode: self._make(mode, "create")

    def _make(self, mode: object, action: str) -> "HybridOperator":
        mode_value = _positive_mode(mode, self.space.fermions)
        word = FermionWord(
            self.space.fermions,
            (mode_value,) if action == "create" else (),
            (mode_value,) if action == "annihilate" else (),
        )
        return HybridOperator._from_terms(
            self.space, ((_Term(word, None, (), None, None, 1.0),))
        )


class _BosonFactory(_Factory):
    @property
    def annihilate(self) -> Any:
        return lambda mode: self._make(mode, "annihilate")

    @property
    def create(self) -> Any:
        return lambda mode: self._make(mode, "create")

    def _make(self, mode: object, action: str) -> "HybridOperator":
        mode_value = _positive_mode(mode, self.space.bosons)
        word = BosonWord(
            self.space.bosons,
            ((mode_value, int(action == "create"), int(action == "annihilate")),),
        )
        return HybridOperator._from_terms(
            self.space, ((_Term(None, word, (), None, None, 1.0),))
        )


class _QubitFactory(_Factory):
    @property
    def x(self) -> Any:
        return lambda index: self._make(index, 1)

    @property
    def y(self) -> Any:
        return lambda index: self._make(index, 2)

    @property
    def z(self) -> Any:
        return lambda index: self._make(index, 3)

    def _make(self, index: object, code: int) -> "HybridOperator":
        index_value = _positive_mode(index, self.space.qubits, "qubit")
        codes = [0] * self.space.qubits
        codes[index_value] = code
        return HybridOperator._from_terms(
            self.space, ((_Term(None, None, tuple(codes), None, None, 1.0),))
        )


class _QuditFactory(_Factory):
    @property
    def weyl(self) -> Any:
        return self._make

    def _make(self, site: object, a: object, b: object) -> "HybridOperator":
        site_value = _positive_mode(site, len(self.space.qudits), "site")
        word = QuditWeylWord(
            self.space.qudits[0],
            ((site_value, _nonnegative_int(a, "a"), _nonnegative_int(b, "b")),),
        )
        return HybridOperator._from_terms(
            self.space, ((_Term(None, None, (), word, None, 1.0),))
        )


@dataclass(frozen=True)
class _Term:
    fermion: Optional[FermionWord]
    boson: Optional[BosonWord]
    qubit: Tuple[int, ...]
    qudit: Optional[QuditWeylWord]
    mapped_fermion: Optional[Tuple[int, ...]]
    coefficient: complex

    def key(self) -> Tuple[Any, ...]:
        return (
            self.fermion.creation_modes if self.fermion else (),
            self.fermion.annihilation_modes if self.fermion else (),
            self.boson.blocks if self.boson else (),
            self.qubit,
            self.mapped_fermion or (),
            self.qudit.triples if self.qudit else (),
        )


class _StructuredOperator:
    _domain = "hybrid"
    space: OperatorSpace
    _terms: Tuple[_Term, ...]
    _locked: bool

    def __init__(self, space: OperatorSpace, terms: Sequence[_Term]) -> None:
        if not isinstance(space, OperatorSpace):
            raise TypeError("space must be an OperatorSpace")
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "_terms", tuple(terms))
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("operators are immutable")
        object.__setattr__(self, name, value)

    @property
    def terms(self) -> Tuple[Any, ...]:
        result: List[Any] = []
        for term in self._terms:
            if self._domain == "fermion":
                assert term.fermion is not None
                result.append(FermionTerm(term.fermion, term.coefficient))
            elif self._domain == "boson":
                assert term.boson is not None
                result.append(BosonTerm(term.boson, term.coefficient))
            elif self._domain == "qudit":
                assert term.qudit is not None
                result.append(QuditWeylTerm(term.qudit, term.coefficient))
            else:
                result.append(term)
        return tuple(result)

    @property
    def term_count(self) -> int:
        return len(self._terms)

    def _check_other(self, other: object) -> "_StructuredOperator":
        if not isinstance(other, _StructuredOperator):
            raise TypeError(f"expected structured operator, got {type(other).__name__}")
        if self.space != other.space:
            raise ValueError("operators use incompatible OperatorSpace layouts")
        return other

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        raw = tuple(terms)
        _guard_terms(len(raw), max_bytes, "operator canonicalization")
        aggregated = _aggregate_terms(
            ((term.key(), term.coefficient) for term in raw),
            max_bytes,
            "operator canonicalization",
        )
        canonical: List[_Term] = []
        by_key = {term.key(): term for term in raw}
        for key, coefficient in aggregated:
            template = by_key[key]
            canonical.append(
                _Term(
                    template.fermion,
                    template.boson,
                    template.qubit,
                    template.qudit,
                    template.mapped_fermion,
                    coefficient,
                )
            )
        return cls(space, canonical)

    def add(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        other = self._check_other(other)
        return _make_operator(self.space, self._terms + other._terms, max_bytes)

    def scale(
        self, coefficient: complex, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "_StructuredOperator":
        scalar = _finite_complex(coefficient, "scale")
        return _make_operator(
            self.space,
            (
                _replace(term, coefficient=term.coefficient * scalar)
                for term in self._terms
            ),
            max_bytes,
        )

    def multiply(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        other = self._check_other(other)
        if isinstance(self, HybridOperator) and isinstance(other, HybridOperator):
            left_arrays = _hybrid_arrays(self)
            right_arrays = _hybrid_arrays(other)
            result = _native.structured_hybrid_multiply(
                self.space.fermions,
                self.space.bosons,
                self.space.qubits,
                len(self.space.qudits),
                self.space.qudits[0] if self.space.qudits else 0,
                left_arrays,
                right_arrays,
                _effective_max_bytes(max_bytes),
            )
            return _hybrid_from_native(self.space, result)
        products: List[_Term] = []
        for left_term, right_term in product(self._terms, other._terms):
            products.extend(_multiply_terms(left_term, right_term, self.space))
            _guard_terms(len(products), max_bytes, "operator multiplication")
        return _make_operator(self.space, products, max_bytes)

    def commutator(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        return self.multiply(other, max_bytes=max_bytes).add(
            other.multiply(self, max_bytes=max_bytes).scale(-1, max_bytes=max_bytes),
            max_bytes=max_bytes,
        )

    def anticommutator(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        return self.multiply(other, max_bytes=max_bytes).add(
            other.multiply(self, max_bytes=max_bytes), max_bytes=max_bytes
        )

    def adjoint(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "_StructuredOperator":
        output: List[_Term] = []
        for term in self._terms:
            fermion_terms: Tuple[Tuple[Optional[FermionWord], complex], ...] = (
                ((None, 1.0 + 0j),)
                if term.fermion is None
                else tuple(
                    (item.word, item.coefficient)
                    for item in term.fermion.adjoint().terms
                )
            )
            boson_terms: Tuple[Tuple[Optional[BosonWord], complex], ...] = (
                ((None, 1.0 + 0j),)
                if term.boson is None
                else tuple(
                    (item.word, item.coefficient) for item in term.boson.adjoint().terms
                )
            )
            if term.qudit is None:
                qudit_terms: Tuple[Tuple[Optional[QuditWeylWord], complex], ...] = (
                    (None, 1.0 + 0j),
                )
            else:
                qudit_adjoint = term.qudit.adjoint()
                qudit_terms = (
                    (
                        qudit_adjoint.word,
                        cmath.exp(
                            2j
                            * math.pi
                            * qudit_adjoint.phase_exponent
                            / term.qudit.dimension
                        ),
                    ),
                )
            for fword, fcoefficient in fermion_terms:
                for bword, bcoefficient in boson_terms:
                    for qword, qcoefficient in qudit_terms:
                        output.append(
                            _Term(
                                fword,
                                bword,
                                _adjoint_qubit_codes(term.qubit),
                                qword,
                                term.mapped_fermion,
                                term.coefficient.conjugate()
                                * fcoefficient
                                * bcoefficient
                                * qcoefficient,
                            )
                        )
        return _make_operator(self.space, output, max_bytes)

    def is_hermitian(self, tolerance: float = 0.0) -> bool:
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or tolerance < 0
        ):
            raise ValueError("Hermiticity tolerance must be finite and non-negative")
        other = self.adjoint()
        keys = {term.key(): term.coefficient for term in self._terms}
        other_keys = {term.key(): term.coefficient for term in other._terms}
        if keys.keys() != other_keys.keys():
            return False
        return all(abs(keys[key] - other_keys[key]) <= tolerance for key in keys)

    def tensor_product(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "HybridOperator":
        if not isinstance(other, _StructuredOperator):
            raise TypeError("tensor_product expects a structured operator")
        if (
            self.space.qudits
            and other.space.qudits
            and self.space.qudits[0] != other.space.qudits[0]
        ):
            raise ValueError("mixed-dimension tensor products are not supported")
        offsets = {
            "fermion": self.space.fermions,
            "boson": self.space.bosons,
            "qubit": self.space.qubits,
            "qudit": len(self.space.qudits),
        }
        axes = self.space._axes + tuple(
            _Axis(axis.domain, axis.index + offsets[axis.domain], axis.dimension)
            for axis in other.space._axes
        )
        space = OperatorSpace._from_axes(axes)
        terms = []
        for left, right in product(self._terms, other._terms):
            left_shifted = _shift_term(
                left,
                {"fermion": 0, "boson": 0, "qubit": 0, "qudit": 0},
                space,
                self.space,
            )
            right_shifted = _shift_term(right, offsets, space, other.space)
            terms.append(
                _combine_nonexpanding_terms(left_shifted, right_shifted, space)
            )
        return HybridOperator._from_terms(space, terms, max_bytes=max_bytes)

    def map_fermions(
        self,
        mapping: str = "jordan_wigner",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> Any:
        if mapping != "jordan_wigner":
            raise NotImplementedError("Phase 7 implements only mapping='jordan_wigner'")
        if self.space.fermions == 0:
            return self
        if isinstance(self, HybridOperator):
            result = _native.structured_hybrid_jordan_wigner(
                self.space.fermions,
                self.space.bosons,
                self.space.qubits,
                len(self.space.qudits),
                self.space.qudits[0] if self.space.qudits else 0,
                _hybrid_arrays(self),
                _effective_max_bytes(max_bytes),
            )
            mapped_operator = _hybrid_from_native(self.space, result)
            if all(axis.domain in ("fermion", "qubit") for axis in self.space._axes):
                return PauliOperator.from_terms(
                    len(self.space._axes),
                    (
                        (_codes_on_space_axes(term, self.space), term.coefficient)
                        for term in mapped_operator._terms
                    ),
                )
            return mapped_operator
        mapped: List[_Term] = []
        for term in self._terms:
            if term.fermion is None:
                mapped.append(term)
                continue
            for codes, coefficient in _jordan_wigner_word(term.fermion):
                mapped.append(
                    _Term(
                        None,
                        term.boson,
                        term.qubit,
                        term.qudit,
                        codes,
                        term.coefficient * coefficient,
                    )
                )
                _guard_terms(len(mapped), max_bytes, "Jordan-Wigner expansion")
        if all(axis.domain in ("fermion", "qubit") for axis in self.space._axes):
            structures = []
            coefficients = []
            for term in mapped:
                structures.append(_codes_on_space_axes(term, self.space))
                coefficients.append(term.coefficient)
            return PauliOperator.from_terms(
                len(structures[0]) if structures else len(self.space._axes),
                zip(structures, coefficients),
            )
        return HybridOperator._from_terms(self.space, mapped, max_bytes=max_bytes)

    def compile(
        self,
        target: str,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
        **kwargs: Any,
    ) -> Any:
        if target not in {"dense", "coo", "csr", "native_mvp", "backend_mvp"}:
            raise ValueError(
                "target must be one of 'dense', 'coo', 'csr', 'native_mvp', or 'backend_mvp'"
            )
        mapped = self.map_fermions(
            kwargs.pop("fermion_mapping", "jordan_wigner"), max_bytes=max_bytes
        )
        if isinstance(mapped, PauliOperator):
            return mapped.compile(target, max_bytes=max_bytes)
        if kwargs.get("mapping") is not None:
            raise ValueError("use fermion_mapping for structured hybrid compilation")
        cutoffs = kwargs.pop("boson_cutoffs", None)
        if kwargs:
            raise TypeError(
                f"unexpected compile arguments: {', '.join(sorted(kwargs))}"
            )
        if mapped.space.bosons:
            if not isinstance(cutoffs, Mapping):
                raise ValueError(
                    "boson_cutoffs is required for every finite boson target"
                )
            normalized_cutoffs = _validate_cutoffs(mapped.space, cutoffs)
        else:
            if cutoffs is not None:
                raise ValueError(
                    "boson_cutoffs is only valid for spaces containing bosons"
                )
            normalized_cutoffs = {}
        if target == "backend_mvp" and mapped.space.bosons:
            raise NotImplementedError(
                "backend_mvp is deferred for finite boson and mixed hybrid layouts"
            )
        return _compile_finite(mapped, target, normalized_cutoffs, max_bytes)

    def __add__(self, other: object) -> Any:
        if not isinstance(other, _StructuredOperator):
            return NotImplemented
        return self.add(other)

    def __sub__(self, other: object) -> Any:
        if not isinstance(other, _StructuredOperator):
            return NotImplemented
        return self.add(other.scale(-1))

    def __neg__(self) -> Any:
        return self.scale(-1)

    def __mul__(self, other: object) -> Any:
        if isinstance(other, _StructuredOperator):
            return self.multiply(other)
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.scale(complex(other))
        return NotImplemented

    def __rmul__(self, other: object) -> Any:
        return self * other


def _replace(term: _Term, **changes: Any) -> _Term:
    values: Dict[str, Any] = {
        "fermion": term.fermion,
        "boson": term.boson,
        "qubit": term.qubit,
        "qudit": term.qudit,
        "mapped_fermion": term.mapped_fermion,
        "coefficient": term.coefficient,
    }
    values.update(changes)
    return _Term(
        cast(Optional[FermionWord], values["fermion"]),
        cast(Optional[BosonWord], values["boson"]),
        cast(Tuple[int, ...], values["qubit"]),
        cast(Optional[QuditWeylWord], values["qudit"]),
        cast(Optional[Tuple[int, ...]], values["mapped_fermion"]),
        cast(complex, values["coefficient"]),
    )


def _make_operator(
    space: OperatorSpace, terms: Iterable[_Term], max_bytes: Optional[int]
) -> "_StructuredOperator":
    # A domain-specific object is retained for low-level one-domain spaces.
    terms = tuple(terms)
    if space.fermions and space.bosons == space.qubits == len(space.qudits) == 0:
        return FermionOperator._from_terms(space, terms, max_bytes)
    if space.bosons and space.fermions == space.qubits == len(space.qudits) == 0:
        return BosonOperator._from_terms(space, terms, max_bytes)
    if len(space.qudits) and space.fermions == space.bosons == space.qubits == 0:
        return QuditWeylOperator._from_terms(space, terms, max_bytes)
    return HybridOperator._from_terms(space, terms, max_bytes)


def _combine_nonexpanding_terms(
    left: _Term, right: _Term, space: OperatorSpace
) -> _Term:
    coefficient = left.coefficient * right.coefficient
    qubit = []
    for a, b in zip(left.qubit, right.qubit):
        code, phase = _PAULI_PRODUCT[a][b]
        qubit.append(code)
        coefficient *= phase
    fword: Optional[FermionWord] = None
    if left.fermion is not None or right.fermion is not None:
        lf = left.fermion or FermionWord(space.fermions)
        rf = right.fermion or FermionWord(space.fermions)
        fermion_products = _normal_order_fermions(
            lf.factors + rf.factors, space.fermions
        )
        if len(fermion_products) != 1:
            raise ValueError(
                "tensor_product unexpectedly generated a fermion contraction"
            )
        fword, sign = next(iter(fermion_products.items()))
        coefficient *= sign
    bword: Optional[BosonWord] = None
    if left.boson is not None or right.boson is not None:
        lb = left.boson or BosonWord(space.bosons)
        rb = right.boson or BosonWord(space.bosons)
        boson_products: Dict[BosonWord, int] = _normal_order_bosons(
            lb.factors + rb.factors, space.bosons
        )
        if len(boson_products) != 1:
            raise ValueError(
                "tensor_product unexpectedly generated a boson contraction"
            )
        bword, factor = next(iter(boson_products.items()))
        coefficient *= factor
    qudit = left.qudit or right.qudit
    if left.qudit is not None and right.qudit is not None:
        assert left.qudit is not None and right.qudit is not None
        product_result = left.qudit.multiply(right.qudit)
        qudit = product_result.word
        coefficient *= cmath.exp(
            2j * math.pi * product_result.phase_exponent / product_result.word.dimension
        )
    mapped, mapped_phase = _multiply_mapped(left.mapped_fermion, right.mapped_fermion)[
        0
    ]
    coefficient *= mapped_phase
    return _Term(fword, bword, tuple(qubit), qudit, mapped, coefficient)


def _shift_term(
    term: _Term,
    offsets: Mapping[str, int],
    space: OperatorSpace,
    source_space: OperatorSpace,
) -> _Term:
    fermion = (
        None
        if term.fermion is None
        else FermionWord(
            space.fermions,
            tuple(mode + offsets["fermion"] for mode in term.fermion.creation_modes),
            tuple(
                mode + offsets["fermion"] for mode in term.fermion.annihilation_modes
            ),
        )
    )
    boson = (
        None
        if term.boson is None
        else BosonWord(
            space.bosons,
            tuple(
                (mode + offsets["boson"], create, annihilate)
                for mode, create, annihilate in term.boson.blocks
            ),
        )
    )
    qubit_values = [0] * space.qubits
    for index, code in enumerate(term.qubit):
        if code:
            qubit_values[index + offsets["qubit"]] = code
    qubit = tuple(qubit_values)
    qudit = (
        None
        if term.qudit is None
        else QuditWeylWord(
            term.qudit.dimension,
            tuple((site + offsets["qudit"], a, b) for site, a, b in term.qudit.triples),
        )
    )
    if term.mapped_fermion is None:
        mapped = None
    else:
        mapped_values = [0] * space.fermions
        for index, code in enumerate(term.mapped_fermion):
            if code:
                mapped_values[index + offsets["fermion"]] = code
        mapped = tuple(mapped_values)
    return _Term(fermion, boson, qubit, qudit, mapped, term.coefficient)


def _multiply_terms(
    left: _Term, right: _Term, space: OperatorSpace
) -> Tuple[_Term, ...]:
    f_products: Dict[Optional[FermionWord], complex] = {None: 1.0}
    if left.fermion is not None or right.fermion is not None:
        lf = left.fermion or FermionWord(space.fermions)
        rf = right.fermion or FermionWord(space.fermions)
        f_products = {
            word: complex(value)
            for word, value in _normal_order_fermions(
                lf.factors + rf.factors, space.fermions
            ).items()
        }
    b_products: Dict[Optional[BosonWord], complex] = {None: 1.0}
    if left.boson is not None or right.boson is not None:
        lb = left.boson or BosonWord(space.bosons)
        rb = right.boson or BosonWord(space.bosons)
        b_products = {
            word: complex(value)
            for word, value in _normal_order_bosons(
                lb.factors + rb.factors, space.bosons
            ).items()
        }
    q_codes = (
        tuple(_PAULI_PRODUCT[a][b][0] for a, b in zip(left.qubit, right.qubit))
        if left.qubit and right.qubit
        else left.qubit or right.qubit
    )
    q_phase = 1.0 + 0j
    if left.qubit and right.qubit:
        for a, b in zip(left.qubit, right.qubit):
            q_phase *= _PAULI_PRODUCT[a][b][1]
    qudit_products: Tuple[Tuple[Optional[QuditWeylWord], complex], ...]
    if left.qudit is not None or right.qudit is not None:
        if left.qudit is None:
            assert right.qudit is not None
            qudit_left = QuditWeylWord(right.qudit.dimension)
        else:
            qudit_left = left.qudit
        if right.qudit is None:
            qudit_right = QuditWeylWord(qudit_left.dimension)
        else:
            qudit_right = right.qudit
        dimension = qudit_left.dimension
        product_result = qudit_left.multiply(qudit_right)
        qudit_products = (
            (
                product_result.word,
                cmath.exp(2j * math.pi * product_result.phase_exponent / dimension),
            ),
        )
    else:
        qudit_products = ((None, 1.0 + 0j),)
    mapped_products = _multiply_mapped(left.mapped_fermion, right.mapped_fermion)
    output: List[_Term] = []
    for fword, fvalue in f_products.items():
        for bword, bvalue in b_products.items():
            for qword, qvalue in qudit_products:
                for mcodes, mvalue in mapped_products:
                    output.append(
                        _Term(
                            fword,
                            bword,
                            q_codes,
                            qword,
                            mcodes,
                            left.coefficient
                            * right.coefficient
                            * fvalue
                            * bvalue
                            * q_phase
                            * qvalue
                            * mvalue,
                        )
                    )
    return tuple(output)


def _multiply_mapped(
    left: Optional[Tuple[int, ...]], right: Optional[Tuple[int, ...]]
) -> Tuple[Tuple[Optional[Tuple[int, ...]], complex], ...]:
    if left is None and right is None:
        return ((None, 1.0 + 0j),)
    if left is None:
        return ((right, 1.0 + 0j),)
    if right is None:
        return ((left, 1.0 + 0j),)
    codes = []
    phase = 1.0 + 0j
    for a, b in zip(left, right):
        code, local_phase = _PAULI_PRODUCT[a][b]
        codes.append(code)
        phase *= local_phase
    return ((tuple(codes), phase),)


def _adjoint_qubit_codes(codes: Tuple[int, ...]) -> Tuple[int, ...]:
    return codes


def _jordan_wigner_word(
    word: FermionWord,
) -> Tuple[Tuple[Tuple[int, ...], complex], ...]:
    current: Dict[Tuple[int, ...], complex] = {(0,) * word.n_modes: 1.0 + 0j}
    for mode, action in word.factors:
        factor: Dict[Tuple[int, ...], complex] = {}
        for code, coefficient in (
            (1, 0.5),
            (2, 0.5j if action == "annihilate" else -0.5j),
        ):
            codes = [0] * word.n_modes
            for lower in range(mode):
                codes[lower] = 3
            codes[mode] = code
            factor[tuple(codes)] = coefficient
        updated: Dict[Tuple[int, ...], complex] = {}
        for left, left_value in current.items():
            for right, right_value in factor.items():
                result = [0] * word.n_modes
                phase = 1.0 + 0j
                for index, (a, b) in enumerate(zip(left, right)):
                    result[index], local_phase = _PAULI_PRODUCT[a][b]
                    phase *= local_phase
                key = tuple(result)
                updated[key] = updated.get(key, 0j) + left_value * right_value * phase
        current = {
            key: value
            for key, value in updated.items()
            if value.real != 0 or value.imag != 0
        }
    return tuple(sorted(current.items()))


def _codes_on_space_axes(term: _Term, space: OperatorSpace) -> Tuple[int, ...]:
    fcodes = term.mapped_fermion or (0,) * space.fermions
    result = []
    for axis in space._axes:
        result.append(
            fcodes[axis.index]
            if axis.domain == "fermion"
            else (
                term.qubit[axis.index]
                if axis.domain == "qubit" and axis.index < len(term.qubit)
                else 0
            )
        )
    return tuple(result)


def _native_fermion_raw(
    n_modes: int,
    terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
    max_bytes: Optional[int],
) -> Tuple[
    Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...], Tuple[complex, ...]
]:
    n_modes = _nonnegative_int(n_modes, "n_modes")
    factors: List[List[Tuple[int, int]]] = []
    coefficients: List[complex] = []
    for raw_factors, coefficient in terms:
        normalized = []
        for factor in raw_factors:
            mode, action = _factor(factor, n_modes)
            normalized.append((mode, int(action == "annihilate")))
        factors.append(normalized)
        coefficients.append(_finite_complex(coefficient))
    result = _native.structured_fermion_canonicalize(
        n_modes,
        factors,
        [value.real for value in coefficients],
        [value.imag for value in coefficients],
        _effective_max_bytes(max_bytes),
    )
    creation, annihilation, real, imaginary = result
    return (
        tuple(tuple(int(mode) for mode in word) for word in creation),
        tuple(tuple(int(mode) for mode in word) for word in annihilation),
        tuple(complex(re, im) for re, im in zip(real, imaginary)),
    )


def _native_boson_raw(
    n_modes: int,
    terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
    max_bytes: Optional[int],
) -> Tuple[Tuple[Tuple[Tuple[int, int, int], ...], ...], Tuple[complex, ...]]:
    n_modes = _nonnegative_int(n_modes, "n_modes")
    factors: List[List[Tuple[int, int]]] = []
    coefficients: List[complex] = []
    for raw_factors, coefficient in terms:
        normalized = []
        for factor in raw_factors:
            mode, action = _factor(factor, n_modes)
            normalized.append((mode, int(action == "annihilate")))
        factors.append(normalized)
        coefficients.append(_finite_complex(coefficient))
    blocks_raw, real, imaginary = _native.structured_boson_canonicalize(
        n_modes,
        factors,
        [value.real for value in coefficients],
        [value.imag for value in coefficients],
        _effective_max_bytes(max_bytes),
    )
    blocks = cast(Sequence[Sequence[Tuple[int, int, int]]], blocks_raw)
    return (
        tuple(
            tuple((int(block[0]), int(block[1]), int(block[2])) for block in word)
            for word in blocks
        ),
        tuple(complex(re, im) for re, im in zip(real, imaginary)),
    )


def _fermion_arrays(
    operator: "FermionOperator",
) -> Tuple[List[List[int]], List[List[int]], List[float], List[float]]:
    creation = [list(term.word.creation_modes) for term in operator.terms]
    annihilation = [list(term.word.annihilation_modes) for term in operator.terms]
    coefficients = [complex(term.coefficient) for term in operator.terms]
    return (
        creation,
        annihilation,
        [value.real for value in coefficients],
        [value.imag for value in coefficients],
    )


def _boson_arrays(
    operator: "BosonOperator",
) -> Tuple[List[List[Tuple[int, int, int]]], List[float], List[float]]:
    blocks = [list(term.word.blocks) for term in operator.terms]
    coefficients = [complex(term.coefficient) for term in operator.terms]
    return (
        blocks,
        [value.real for value in coefficients],
        [value.imag for value in coefficients],
    )


def _hybrid_arrays(
    operator: "HybridOperator",
) -> Tuple[
    List[bool],
    List[List[int]],
    List[List[int]],
    List[bool],
    List[List[Tuple[int, int, int]]],
    List[List[int]],
    List[bool],
    List[List[int]],
    List[bool],
    List[List[Tuple[int, int, int]]],
    List[float],
    List[float],
]:
    fermion_present: List[bool] = []
    fermion_creation: List[List[int]] = []
    fermion_annihilation: List[List[int]] = []
    boson_present: List[bool] = []
    boson_blocks: List[List[Tuple[int, int, int]]] = []
    qubit_codes: List[List[int]] = []
    mapped_present: List[bool] = []
    mapped_codes: List[List[int]] = []
    qudit_present: List[bool] = []
    qudit_triples: List[List[Tuple[int, int, int]]] = []
    coefficients: List[complex] = []
    for term in operator._terms:
        fermion_present.append(term.fermion is not None)
        fermion_creation.append(
            list(term.fermion.creation_modes) if term.fermion is not None else []
        )
        fermion_annihilation.append(
            list(term.fermion.annihilation_modes) if term.fermion is not None else []
        )
        boson_present.append(term.boson is not None)
        boson_blocks.append(list(term.boson.blocks) if term.boson is not None else [])
        qubit_codes.append(
            list(term.qubit)
            if len(term.qubit) == operator.space.qubits
            else [0] * operator.space.qubits
        )
        mapped_present.append(term.mapped_fermion is not None)
        mapped_codes.append(
            list(term.mapped_fermion)
            if term.mapped_fermion is not None
            else [0] * operator.space.fermions
        )
        qudit_present.append(term.qudit is not None)
        qudit_triples.append(list(term.qudit.triples) if term.qudit is not None else [])
        coefficients.append(complex(term.coefficient))
    return (
        fermion_present,
        fermion_creation,
        fermion_annihilation,
        boson_present,
        boson_blocks,
        qubit_codes,
        mapped_present,
        mapped_codes,
        qudit_present,
        qudit_triples,
        [value.real for value in coefficients],
        [value.imag for value in coefficients],
    )


def _fermion_from_native(
    cls: Any,
    n_modes: int,
    result: Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[complex]],
) -> "FermionOperator":
    creation, annihilation, coefficients = result
    terms = tuple(
        _Term(
            FermionWord(n_modes, tuple(left), tuple(right)),
            None,
            (),
            None,
            None,
            complex(coefficient),
        )
        for left, right, coefficient in zip(creation, annihilation, coefficients)
    )
    instance = object.__new__(cls)
    _StructuredOperator.__init__(instance, OperatorSpace(fermions=n_modes), terms)
    return cast("FermionOperator", instance)


def _boson_from_native(
    cls: Any,
    n_modes: int,
    result: Tuple[Sequence[Sequence[Tuple[int, int, int]]], Sequence[complex]],
) -> "BosonOperator":
    blocks, coefficients = result
    terms = tuple(
        _Term(
            None,
            BosonWord(
                n_modes,
                tuple((int(block[0]), int(block[1]), int(block[2])) for block in word),
            ),
            (),
            None,
            None,
            complex(coefficient),
        )
        for word, coefficient in zip(blocks, coefficients)
    )
    instance = object.__new__(cls)
    _StructuredOperator.__init__(instance, OperatorSpace(bosons=n_modes), terms)
    return cast("BosonOperator", instance)


def _hybrid_from_native(
    space: OperatorSpace,
    result: Tuple[Any, ...],
) -> "HybridOperator":
    (
        fermion_present,
        fermion_creation,
        fermion_annihilation,
        boson_present,
        boson_blocks,
        qubit_codes,
        mapped_present,
        mapped_codes,
        qudit_present,
        qudit_triples,
        real,
        imaginary,
    ) = result
    qudit_dimension = space.qudits[0] if space.qudits else 0
    terms = []
    for index in range(len(real)):
        fermion = (
            FermionWord(
                space.fermions,
                tuple(fermion_creation[index]),
                tuple(fermion_annihilation[index]),
            )
            if fermion_present[index]
            else None
        )
        boson = (
            BosonWord(space.bosons, tuple(boson_blocks[index]))
            if boson_present[index]
            else None
        )
        qudit = (
            QuditWeylWord(qudit_dimension, tuple(qudit_triples[index]))
            if qudit_present[index]
            else None
        )
        terms.append(
            _Term(
                fermion,
                boson,
                tuple(qubit_codes[index]),
                qudit,
                tuple(mapped_codes[index]) if mapped_present[index] else None,
                complex(real[index], imaginary[index]),
            )
        )
    instance = object.__new__(HybridOperator)
    _StructuredOperator.__init__(instance, space, tuple(terms))
    return instance


class FermionOperator(_StructuredOperator):
    _domain = "fermion"

    def __init__(
        self, n_modes: int, terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]]
    ) -> None:
        result = _native_fermion_raw(n_modes, terms, DEFAULT_MAX_BYTES)
        _StructuredOperator.__init__(
            self,
            OperatorSpace(fermions=_nonnegative_int(n_modes, "n_modes")),
            tuple(
                _Term(
                    FermionWord(n_modes, left, right),
                    None,
                    (),
                    None,
                    None,
                    coefficient,
                )
                for left, right, coefficient in zip(*result)
            ),
        )

    @classmethod
    def _from_words(
        cls, n_modes: int, products: Mapping[FermionWord, int]
    ) -> "FermionOperator":
        return cls._from_terms(
            OperatorSpace(fermions=n_modes),
            (
                _Term(word, None, (), None, None, coefficient)
                for word, coefficient in products.items()
            ),
        )

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionOperator":
        canonical = _canonical_terms(space, terms, max_bytes)
        instance = object.__new__(cls)
        _StructuredOperator.__init__(instance, space, canonical)
        return instance

    @classmethod
    def from_terms(
        cls,
        n_modes: int,
        terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionOperator":
        return cls._from_raw(n_modes, terms, max_bytes)

    @classmethod
    def _from_raw(
        cls,
        n_modes: int,
        terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
        max_bytes: Optional[int],
    ) -> "FermionOperator":
        return _fermion_from_native(
            cls,
            _nonnegative_int(n_modes, "n_modes"),
            _native_fermion_raw(n_modes, terms, max_bytes),
        )

    def multiply(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        if not isinstance(other, FermionOperator) or other.space != self.space:
            return super().multiply(other, max_bytes=max_bytes)
        left = _fermion_arrays(self)
        right = _fermion_arrays(other)
        creation, annihilation, real, imaginary = _native.structured_fermion_multiply(
            self.space.fermions,
            left,
            right,
            _effective_max_bytes(max_bytes),
        )
        return _fermion_from_native(
            FermionOperator,
            self.space.fermions,
            (
                creation,
                annihilation,
                tuple(complex(re, im) for re, im in zip(real, imaginary)),
            ),
        )

    def map_fermions(
        self,
        mapping: str = "jordan_wigner",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> Any:
        if mapping != "jordan_wigner":
            raise NotImplementedError("Phase 7 implements only mapping='jordan_wigner'")
        creation, annihilation, real, imaginary = _fermion_arrays(self)
        structures, mapped_real, mapped_imaginary = (
            _native.structured_fermion_jordan_wigner(
                self.space.fermions,
                creation,
                annihilation,
                real,
                imaginary,
                _effective_max_bytes(max_bytes),
            )
        )
        return PauliOperator.from_terms(
            self.space.fermions,
            zip(
                structures,
                (complex(re, im) for re, im in zip(mapped_real, mapped_imaginary)),
            ),
        )

    def compile(
        self,
        target: str,
        *,
        mapping: str = "jordan_wigner",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise TypeError(
                f"unexpected compile arguments: {', '.join(sorted(kwargs))}"
            )
        return self.map_fermions(mapping, max_bytes=max_bytes).compile(
            target, max_bytes=max_bytes
        )


class BosonOperator(_StructuredOperator):
    _domain = "boson"

    @classmethod
    def _from_words(
        cls, n_modes: int, products: Mapping[BosonWord, int]
    ) -> "BosonOperator":
        return cls._from_terms(
            OperatorSpace(bosons=n_modes),
            (
                _Term(None, word, (), None, None, coefficient)
                for word, coefficient in products.items()
            ),
        )

    @classmethod
    def from_terms(
        cls,
        n_modes: int,
        terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "BosonOperator":
        return _boson_from_native(
            cls,
            _nonnegative_int(n_modes, "n_modes"),
            _native_boson_raw(n_modes, terms, max_bytes),
        )

    def multiply(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        if not isinstance(other, BosonOperator) or other.space != self.space:
            return super().multiply(other, max_bytes=max_bytes)
        left = _boson_arrays(self)
        right = _boson_arrays(other)
        blocks, real, imaginary = _native.structured_boson_multiply(
            self.space.bosons,
            left,
            right,
            _effective_max_bytes(max_bytes),
        )
        typed_blocks = cast(Sequence[Sequence[Tuple[int, int, int]]], blocks)
        return _boson_from_native(
            BosonOperator,
            self.space.bosons,
            (
                typed_blocks,
                tuple(complex(re, im) for re, im in zip(real, imaginary)),
            ),
        )

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "BosonOperator":
        instance = object.__new__(cls)
        _StructuredOperator.__init__(
            instance, space, _canonical_terms(space, terms, max_bytes)
        )
        return instance


class QuditWeylOperator(_StructuredOperator):
    _domain = "qudit"

    @classmethod
    def from_terms(
        cls,
        dimension: int,
        terms: Iterable[Tuple[Sequence[Tuple[int, int, int]], complex]],
        *,
        n_sites: Optional[int] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "QuditWeylOperator":
        dimension = _nonnegative_int(dimension, "dimension")
        if not 3 <= dimension <= _U32_MAX:
            raise ValueError("qudit dimension must satisfy 3 <= dimension <= 2**32-1")
        input_terms = tuple(terms)
        raw: List[_Term] = []
        max_site = -1
        for factors, coefficient in input_terms:
            value = _finite_complex(coefficient)
            word = QuditWeylWord(dimension)
            phase = 0
            for factor in factors:
                if not isinstance(factor, (tuple, list)) or len(factor) != 3:
                    raise ValueError("Weyl factors must be (site, a, b) triples")
                site, a, b = (
                    _nonnegative_int(factor[0], "site"),
                    _nonnegative_int(factor[1], "a"),
                    _nonnegative_int(factor[2], "b"),
                )
                max_site = max(max_site, site)
                next_word = QuditWeylWord(
                    dimension, ((site, a % dimension, b % dimension),)
                )
                result = word.multiply(next_word)
                word, phase = result.word, (phase + result.phase_exponent) % dimension
            raw.append(
                _Term(
                    None,
                    None,
                    (),
                    word,
                    None,
                    value * cmath.exp(2j * math.pi * phase / dimension),
                )
            )
        if n_sites is not None:
            n_sites = _nonnegative_int(n_sites, "n_sites")
            if n_sites <= max_site:
                raise ValueError("n_sites is smaller than a supplied Weyl site")
        else:
            n_sites = max_site + 1
        space = OperatorSpace._from_axes(
            tuple(_Axis("qudit", i, dimension) for i in range(n_sites))
        )
        return cls._from_terms(space, raw, max_bytes)

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "QuditWeylOperator":
        instance = object.__new__(cls)
        _StructuredOperator.__init__(
            instance, space, _canonical_terms(space, terms, max_bytes)
        )
        return instance


class HybridOperator(_StructuredOperator):
    _domain = "hybrid"

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "HybridOperator":
        instance = object.__new__(cls)
        _StructuredOperator.__init__(
            instance, space, _canonical_terms(space, terms, max_bytes)
        )
        return instance


def _canonical_terms(
    space: OperatorSpace,
    terms: Iterable[_Term],
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> Tuple[_Term, ...]:
    raw = tuple(terms)
    for term in raw:
        _finite_complex(term.coefficient)
    aggregate = _aggregate_terms(
        ((term.key(), term.coefficient) for term in raw),
        max_bytes,
        "operator canonicalization",
    )
    by_key = {term.key(): term for term in raw}
    return tuple(
        _replace(by_key[key], coefficient=coefficient) for key, coefficient in aggregate
    )


class OperatorBuilder:
    """Single-owner batched structured-term builder."""

    def __init__(self, space: OperatorSpace) -> None:
        self.space = space
        self._products: List[Dict[str, Any]] = []

    def add_product(
        self,
        coefficient: complex = 1.0,
        *,
        fermions: Sequence[Tuple[int, str]] = (),
        bosons: Sequence[Tuple[int, str]] = (),
        qubits: Sequence[Tuple[int, object]] = (),
        qudits: Sequence[Tuple[int, int, int]] = (),
    ) -> "OperatorBuilder":
        self._products.append(
            {
                "coefficient": _finite_complex(coefficient),
                "fermions": tuple(fermions),
                "bosons": tuple(bosons),
                "qubits": tuple(qubits),
                "qudits": tuple(qudits),
            }
        )
        return self

    def finish(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> HybridOperator:
        fermion_factors: List[List[Tuple[int, int]]] = []
        boson_factors: List[List[Tuple[int, int]]] = []
        qubit_codes: List[List[int]] = []
        qudit_present: List[bool] = []
        qudit_triples: List[List[Tuple[int, int, int]]] = []
        coefficients: List[complex] = []
        for product_spec in self._products:
            normalized_factors: List[Tuple[int, int]] = []
            for factor in product_spec["fermions"]:
                mode, action = _factor(factor, self.space.fermions)
                normalized_factors.append((mode, int(action == "annihilate")))
            normalized_bosons: List[Tuple[int, int]] = []
            for factor in product_spec["bosons"]:
                mode, action = _factor(factor, self.space.bosons)
                normalized_bosons.append((mode, int(action == "annihilate")))
            qcodes = [0] * self.space.qubits
            qword: Optional[QuditWeylWord] = None
            qphase = 1.0 + 0j
            for index, code in product_spec["qubits"]:
                if isinstance(code, str):
                    if code not in _IDENTITY_CODES:
                        raise ValueError("Pauli code must be one of I, X, Y, Z")
                    code_value = _IDENTITY_CODES[code]
                else:
                    code_value = _nonnegative_int(code, "Pauli code")
                if code_value not in range(4):
                    raise ValueError("Pauli code must be an integer in 0..3")
                qcodes[_positive_mode(index, self.space.qubits, "qubit")] = code_value
            if self.space.qudits:
                qword = QuditWeylWord(self.space.qudits[0])
                for site, a, b in product_spec["qudits"]:
                    site = _positive_mode(site, len(self.space.qudits), "site")
                    factor = QuditWeylWord(self.space.qudits[0], ((site, a, b),))
                    product_result = qword.multiply(factor)
                    qword = product_result.word
                    qphase *= cmath.exp(
                        2j
                        * math.pi
                        * product_result.phase_exponent
                        / self.space.qudits[0]
                    )
            fermion_factors.append(normalized_factors)
            boson_factors.append(normalized_bosons)
            qubit_codes.append(qcodes)
            qudit_present.append(qword is not None)
            qudit_triples.append(list(qword.triples) if qword is not None else [])
            coefficients.append(product_spec["coefficient"] * qphase)
        result = _native.structured_hybrid_canonicalize(
            self.space.fermions,
            self.space.bosons,
            self.space.qubits,
            len(self.space.qudits),
            self.space.qudits[0] if self.space.qudits else 0,
            (
                fermion_factors,
                boson_factors,
                qubit_codes,
                qudit_present,
                qudit_triples,
                [value.real for value in coefficients],
                [value.imag for value in coefficients],
            ),
            _effective_max_bytes(max_bytes),
        )
        return _hybrid_from_native(self.space, result)


def _validate_cutoffs(
    space: OperatorSpace, cutoffs: Mapping[object, object]
) -> Dict[int, int]:
    if set(cutoffs) != set(range(space.bosons)):
        raise ValueError(
            "boson_cutoffs must contain exactly one inclusive cutoff for every boson mode"
        )
    result = {}
    for mode, cutoff in cutoffs.items():
        result[_positive_mode(mode, space.bosons)] = _nonnegative_int(
            cutoff, "boson cutoff"
        )
    return result


def _compile_finite(
    operator: _StructuredOperator,
    target: str,
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
) -> Any:
    dimensions = tuple(
        cutoffs[axis.index] + 1 if axis.domain == "boson" else axis.dimension
        for axis in operator.space._axes
    )
    if not dimensions:
        dimensions = (1,)
    dimension = math.prod(dimensions)
    if dimension > np.iinfo(np.intp).max:
        raise OverflowError(
            "finite basis dimension cannot be represented by platform indices"
        )
    if target == "dense":
        _check_allocation(
            dimension * dimension * 16, max_bytes, "dense structured matrix"
        )
        if hasattr(_native, "structured_dense"):
            operations, coefficients_re, coefficients_im = _native_term_arrays(
                operator, dimensions, cutoffs
            )
            native_dimension, values = _native.structured_dense(
                list(dimensions),
                operations,
                coefficients_re,
                coefficients_im,
                _effective_max_bytes(max_bytes),
            )
            return np.asarray(values, dtype=np.complex128).reshape(
                (native_dimension, native_dimension)
            )
    if _prefer_native_structured_sparse(dimension, len(operator._terms)) and hasattr(
        _native, "structured_sparse"
    ):
        if target == "native_mvp" and hasattr(_native, "structured_sparse_plan"):
            native_plan = _native_structured_mvp_plan(
                operator, dimensions, cutoffs, max_bytes
            )
            return NativeMVPPlan(
                0,
                len(operator._terms),
                "structured_sparse_native",
                native_plan,
                local_dimensions=dimensions,
                estimated_bytes=dimension * 16,
            )
        native_dimension, rows, columns, values = _native_structured_sparse(
            operator, dimensions, cutoffs, max_bytes
        )
        if target == "coo":
            _check_allocation(
                int(rows.nbytes + columns.nbytes + values.nbytes),
                max_bytes,
                "COO structured matrix",
            )
            return COOMatrix(
                rows, columns, values, (native_dimension, native_dimension)
            )
        if target == "csr":
            indptr = np.zeros(native_dimension + 1, dtype=np.uint64)
            for row in rows:
                indptr[int(row) + 1] += 1
            np.cumsum(indptr, out=indptr)
            _check_allocation(
                int(indptr.nbytes + columns.nbytes + values.nbytes),
                max_bytes,
                "CSR structured matrix",
            )
            return CSRMatrix(
                indptr, columns, values, (native_dimension, native_dimension)
            )
        generic = (dimensions, rows, columns, values)
        if target == "native_mvp":
            return NativeMVPPlan(
                0,
                len(operator._terms),
                "structured_sparse",
                None,
                local_dimensions=dimensions,
                generic_entries=generic,
            )
        if target == "backend_mvp":
            return BackendMVPPlan(
                1,
                0,
                len(operator._terms),
                np.empty((0, 0), dtype=np.uint64),
                np.empty((0, 0), dtype=np.uint64),
                np.empty(0, dtype=np.complex128),
                local_dimensions=dimensions,
                _generic_entries=generic,
            )
    entries = _finite_entries(operator, dimensions, cutoffs, max_bytes)
    if target == "dense":
        matrix: np.ndarray[Any, Any] = np.zeros(
            (dimension, dimension), dtype=np.complex128
        )
        for (row, column), value in entries.items():
            matrix[row, column] = value
        return matrix
    rows = np.fromiter((key[0] for key in sorted(entries)), dtype=np.uint64)
    columns = np.fromiter((key[1] for key in sorted(entries)), dtype=np.uint64)
    values = np.asarray([entries[key] for key in sorted(entries)], dtype=np.complex128)
    if target == "coo":
        _check_allocation(
            int(rows.nbytes + columns.nbytes + values.nbytes),
            max_bytes,
            "COO structured matrix",
        )
        return COOMatrix(rows, columns, values, (dimension, dimension))
    if target == "csr":
        indptr = np.zeros(dimension + 1, dtype=np.uint64)
        for row in rows:
            indptr[int(row) + 1] += 1
        np.cumsum(indptr, out=indptr)
        _check_allocation(
            int(indptr.nbytes + columns.nbytes + values.nbytes),
            max_bytes,
            "CSR structured matrix",
        )
        return CSRMatrix(indptr, columns, values, (dimension, dimension))
    generic = (dimensions, rows, columns, values)
    if target == "native_mvp":
        return NativeMVPPlan(
            0,
            len(operator._terms),
            "structured_sparse",
            None,
            local_dimensions=dimensions,
            generic_entries=generic,
        )
    if target == "backend_mvp":
        return BackendMVPPlan(
            1,
            0,
            len(operator._terms),
            np.empty((0, 0), dtype=np.uint64),
            np.empty((0, 0), dtype=np.uint64),
            np.empty(0, dtype=np.complex128),
            local_dimensions=dimensions,
            _generic_entries=generic,
        )
    raise AssertionError(target)


def _native_term_arrays(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
) -> Tuple[List[List[Tuple[int, int, int, int]]], List[float], List[float]]:
    operations: List[List[Tuple[int, int, int, int]]] = []
    real: List[float] = []
    imaginary: List[float] = []
    for term in operator._terms:
        term_operations: List[Tuple[int, int, int, int]] = []
        fcodes = term.mapped_fermion or (0,) * operator.space.fermions
        boson_blocks = (
            {
                mode: (creation, annihilation)
                for mode, creation, annihilation in term.boson.blocks
            }
            if term.boson is not None
            else {}
        )
        qudit_triples = (
            {site: (a, b) for site, a, b in term.qudit.triples}
            if term.qudit is not None
            else {}
        )
        for position, axis in enumerate(operator.space._axes):
            if axis.domain == "fermion" and fcodes[axis.index]:
                term_operations.append((position, 0, fcodes[axis.index], 0))
            elif (
                axis.domain == "qubit"
                and axis.index < len(term.qubit)
                and term.qubit[axis.index]
            ):
                term_operations.append((position, 0, term.qubit[axis.index], 0))
            elif axis.domain == "boson" and axis.index in boson_blocks:
                creation, annihilation = boson_blocks[axis.index]
                term_operations.append((position, 1, creation, annihilation))
            elif axis.domain == "qudit" and axis.index in qudit_triples:
                a, b = qudit_triples[axis.index]
                term_operations.append((position, 2, a, b))
        operations.append(term_operations)
        real.append(term.coefficient.real)
        imaginary.append(term.coefficient.imag)
    return operations, real, imaginary


def _prefer_native_structured_sparse(dimension: int, term_count: int) -> bool:
    """Select the Rust transition kernel only after enough work amortizes FFI."""
    return (
        term_count > 0
        and dimension * term_count >= _STRUCTURED_NATIVE_SPARSE_WORK_THRESHOLD
    )


def _native_structured_sparse(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
) -> Tuple[int, np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    operations, coefficients_re, coefficients_im = _native_term_arrays(
        operator, dimensions, cutoffs
    )
    native_dimension, rows, columns, real, imaginary = _native.structured_sparse(
        list(dimensions),
        operations,
        coefficients_re,
        coefficients_im,
        _effective_max_bytes(max_bytes),
    )
    return (
        native_dimension,
        np.asarray(rows, dtype=np.uint64),
        np.asarray(columns, dtype=np.uint64),
        np.asarray(real, dtype=np.float64)
        + 1j * np.asarray(imaginary, dtype=np.float64),
    )


def _native_structured_mvp_plan(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
) -> Any:
    operations, coefficients_re, coefficients_im = _native_term_arrays(
        operator, dimensions, cutoffs
    )
    return _native.structured_sparse_plan(
        list(dimensions),
        operations,
        coefficients_re,
        coefficients_im,
        _effective_max_bytes(max_bytes),
    )


def _finite_entries(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
) -> Dict[Tuple[int, int], complex]:
    dimension = math.prod(dimensions)
    entries: Dict[Tuple[int, int], complex] = {}
    for column in range(dimension):
        digits = _decode_index(column, dimensions)
        for term in operator._terms:
            transitions = _term_transition(term, operator.space, dimensions, digits)
            for row_digits, amplitude in transitions:
                row = _encode_index(row_digits, dimensions)
                key = (row, column)
                entries[key] = entries.get(key, 0j) + term.coefficient * amplitude
        _guard_terms(len(entries), max_bytes, "finite structured matrix construction")
    return {
        key: value
        for key, value in entries.items()
        if value.real != 0 or value.imag != 0
    }


def _decode_index(index: int, dimensions: Tuple[int, ...]) -> List[int]:
    result = [0] * len(dimensions)
    remainder = index
    for position in range(len(dimensions) - 1, -1, -1):
        result[position] = remainder % dimensions[position]
        remainder //= dimensions[position]
    return result


def _encode_index(digits: Sequence[int], dimensions: Tuple[int, ...]) -> int:
    result = 0
    for digit, dimension in zip(digits, dimensions):
        result = result * dimension + digit
    return result


def _term_transition(
    term: _Term,
    space: OperatorSpace,
    dimensions: Tuple[int, ...],
    digits: Sequence[int],
) -> Tuple[Tuple[List[int], complex], ...]:
    current: List[Tuple[List[int], complex]] = [(list(digits), 1.0 + 0j)]
    fcodes = term.mapped_fermion or (0,) * space.fermions
    for position, axis in enumerate(space._axes):
        code = (
            fcodes[axis.index]
            if axis.domain == "fermion"
            else (
                term.qubit[axis.index]
                if axis.domain == "qubit" and axis.index < len(term.qubit)
                else 0
            )
        )
        if code:
            next_values = []
            for state, value in current:
                updated = list(state)
                if code in (1, 2):
                    updated[position] = 1 - updated[position]
                    value *= 1 if code == 1 or state[position] == 0 else -1j
                    if code == 2 and state[position] == 0:
                        value *= 1j
                elif code == 3:
                    value *= 1 if state[position] == 0 else -1
                next_values.append((updated, value))
            current = next_values
    if term.boson is not None:
        blocks = {
            mode: (creation, annihilation)
            for mode, creation, annihilation in term.boson.blocks
        }
        for position, axis in enumerate(space._axes):
            if axis.domain != "boson" or axis.index not in blocks:
                continue
            creation, annihilation = blocks[axis.index]
            next_values = []
            for state, value in current:
                occupation = state[position]
                if (
                    occupation < annihilation
                    or occupation - annihilation + creation >= dimensions[position]
                ):
                    continue
                amplitude = 1.0
                for offset in range(annihilation):
                    amplitude *= math.sqrt(occupation - offset)
                remaining = occupation - annihilation
                for offset in range(creation):
                    amplitude *= math.sqrt(remaining + offset + 1)
                updated = list(state)
                updated[position] = remaining + creation
                next_values.append((updated, value * amplitude))
            current = next_values
    if term.qudit is not None:
        triples = {site: (a, b) for site, a, b in term.qudit.triples}
        for position, axis in enumerate(space._axes):
            if axis.domain != "qudit" or axis.index not in triples:
                continue
            a, b = triples[axis.index]
            next_values = []
            for state, value in current:
                updated = list(state)
                updated[position] = (state[position] + a) % dimensions[position]
                phase = cmath.exp(
                    2j * math.pi * b * state[position] / dimensions[position]
                )
                next_values.append((updated, value * phase))
            current = next_values
    return tuple(current)


__all__ = [
    "BosonOperator",
    "BosonTerm",
    "BosonWord",
    "FermionOperator",
    "FermionTerm",
    "FermionWord",
    "HybridOperator",
    "OperatorBuilder",
    "OperatorSpace",
    "QuditProduct",
    "QuditWeylOperator",
    "QuditWeylTerm",
    "QuditWeylWord",
]
