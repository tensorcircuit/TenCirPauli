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
from dataclasses import dataclass, replace
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
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
    DIRECT_WEYL_BASIS_ORDERING,
    MIXED_RADIX_BASIS_ORDERING,
    BackendMVPPlan,
    CompileResult,
    COOMatrix,
    CSRMatrix,
    NativeMVPPlan,
    _check_allocation,
    _effective_max_bytes,
    _validate_max_bytes,
)
from .pauli import _PAULI_CHAR_TO_CODE, PauliOperator


if TYPE_CHECKING:
    from .charge import (
        AdditiveCharge,
        AdditiveSymmetryAnalysis,
        ChargeRestrictedOperator,
        ChargeSector,
        ChargeStorage,
    )
    from .majorana import MajoranaOperator
    from .mapping import FermionQubitMapping


_U32_MAX = 2**32 - 1
_IDENTITY_CODES = _PAULI_CHAR_TO_CODE
_PAULI_PRODUCT: Tuple[Tuple[Tuple[int, complex], ...], ...] = (
    ((0, 1), (1, 1), (2, 1), (3, 1)),
    ((1, 1), (0, 1), (3, 1j), (2, -1j)),
    ((2, 1), (3, -1j), (0, 1), (1, 1j)),
    ((3, 1), (2, 1j), (1, -1j), (0, 1)),
)

MappedOperator = Union["_StructuredOperator", PauliOperator]


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


_nonnegative_int = validate_nonnegative_int


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
    """Canonical fermionic monomial ``creations * annihilations``.

    Creation modes are stored in increasing order and annihilation modes in
    decreasing order. Use :meth:`from_factors` for arbitrary raw products so
    CAR contractions and signs are canonicalized.
    """

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
        """Whether the word contains no fermionic ladder factors."""
        return not self.creation_modes and not self.annihilation_modes

    @property
    def parity(self) -> int:
        """Return the fermion-number parity of the word."""
        return (len(self.creation_modes) + len(self.annihilation_modes)) & 1

    @property
    def factors(self) -> Tuple[Tuple[int, str], ...]:
        """Return the canonical raw factor sequence."""
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
        """Construct and canonicalize one raw fermion word."""
        return FermionOperator.from_terms(
            n_modes, ((tuple(factors), 1.0),), max_bytes=max_bytes
        )

    def adjoint(self) -> "FermionOperator":
        """Return the coefficient-free adjoint word as an operator."""
        factors = tuple(
            (mode, "create") for mode in reversed(self.annihilation_modes)
        ) + tuple((mode, "annihilate") for mode in reversed(self.creation_modes))
        return FermionOperator.from_terms(self.n_modes, ((factors, 1.0),))

    def multiply(self, other: "FermionWord") -> "FermionOperator":
        """Multiply two words and retain all CAR contraction terms."""
        if not isinstance(other, FermionWord) or other.n_modes != self.n_modes:
            raise ValueError("fermion words require equal n_modes")
        return FermionOperator.from_terms(
            self.n_modes, ((self.factors + other.factors, 1.0),)
        )

    def __mul__(self, other: "FermionWord") -> "FermionOperator":
        return self.multiply(other)


@dataclass(frozen=True)
class BosonWord:
    """Canonical normal-ordered bosonic power blocks.

    Each block is ``(mode, creation_power, annihilation_power)``. Raw products
    should be constructed through :class:`BosonOperator` so CCR contractions
    are retained exactly.
    """

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
        """Whether the word contains no bosonic ladder factors."""
        return not self.blocks

    @property
    def factors(self) -> Tuple[Tuple[int, str], ...]:
        """Expand canonical power blocks into raw ladder factors."""
        result: List[Tuple[int, str]] = []
        for mode, creation, annihilation in self.blocks:
            result.extend((mode, "create") for _ in range(creation))
            result.extend((mode, "annihilate") for _ in range(annihilation))
        return tuple(result)

    def adjoint(self) -> "BosonOperator":
        """Return the coefficient-free adjoint word as an operator."""
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
        """Multiply two words and retain all CCR contraction terms."""
        if not isinstance(other, BosonWord) or other.n_modes != self.n_modes:
            raise ValueError("boson words require equal n_modes")
        return BosonOperator.from_terms(
            self.n_modes, ((self.factors + other.factors, 1.0),)
        )

    def __mul__(self, other: "BosonWord") -> "BosonOperator":
        return self.multiply(other)


@dataclass(frozen=True)
class QuditWeylWord:
    """Direct-convention ``X**a Z**b`` word with modular exponents.

    Exponents are reduced modulo ``dimension`` and triples are sorted by site.
    Multiplication returns a :class:`QuditProduct` with the modular phase
    exponent kept separate from the canonical word.
    """

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
        """Whether every site carries the identity Weyl factor."""
        return not self.triples

    @property
    def n_sites(self) -> int:
        """Return one past the largest explicitly stored site index."""
        return 0 if not self.triples else self.triples[-1][0] + 1

    def multiply(self, other: "QuditWeylWord") -> "QuditProduct":
        """Multiply words and return the modular phase exponent separately."""
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
        """Check the exact modular Weyl commutation condition."""
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
        """Return the adjoint word and its modular phase exponent."""
        exponent = sum(a * b for _, a, b in self.triples) % self.dimension
        result = tuple(
            (site, (-a) % self.dimension, (-b) % self.dimension)
            for site, a, b in self.triples
            if a or b
        )
        return QuditProduct(QuditWeylWord(self.dimension, result), exponent)


@dataclass(frozen=True)
class QuditProduct:
    """A Weyl word together with its modular phase exponent."""

    word: QuditWeylWord
    phase_exponent: int


@dataclass(frozen=True)
class FermionTerm:
    """One canonical fermion word and its complex coefficient."""

    word: FermionWord
    coefficient: complex


@dataclass(frozen=True)
class BosonTerm:
    """One canonical boson word and its complex coefficient."""

    word: BosonWord
    coefficient: complex


@dataclass(frozen=True)
class QuditWeylTerm:
    """One direct-convention Weyl word and its complex coefficient."""

    word: QuditWeylWord
    coefficient: complex


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
    """Immutable ordered logical subsystem layout for structured operators.

    Axes are ordered as fermions, bosons, qubits, then uniform-dimension
    qudits. This ordering controls term serialization, mixed-radix basis
    ordering, embedding, tensor products, and finite matrix targets.

    Examples:
        >>> import tencirpauli as tcp
        >>> space = tcp.OperatorSpace(fermions=1, qubits=1)
        >>> operator = space.fermion.create(0) * space.qubit.z(0)
        >>> operator.term_count
        1
    """

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
        """Create an immutable ordered layout with uniform qudit dimensions."""
        fermions = _nonnegative_int(fermions, "fermions")
        bosons = _nonnegative_int(bosons, "bosons")
        qubits = _nonnegative_int(qubits, "qubits")
        if not isinstance(qudits, (tuple, list)):
            raise ValueError("qudits must be a tuple of local dimensions")
        dimensions = tuple(
            _nonnegative_int(value, "qudit dimension") for value in qudits
        )
        if dimensions and (
            any(value < 3 or value > _U32_MAX for value in dimensions)
            or len(set(dimensions)) != 1
        ):
            raise ValueError("qudits must have one uniform dimension 3 <= d <= 2**32-1")
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
        """Return ordered ``(domain, index, local-dimension)`` descriptors."""
        return tuple((axis.domain, axis.index, axis.dimension) for axis in self._axes)

    @property
    def layout_fingerprint(self) -> Tuple[Tuple[str, int, int], ...]:
        """Return the immutable compatibility fingerprint for this layout."""
        return self.axes

    @property
    def local_dimensions(self) -> Tuple[int, ...]:
        """Return finite local dimensions, rejecting uncut boson axes."""
        if any(axis.domain == "boson" and axis.dimension == 0 for axis in self._axes):
            raise ValueError("boson local dimensions require explicit finite cutoffs")
        return tuple(axis.dimension for axis in self._axes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OperatorSpace) and self.axes == other.axes

    def __hash__(self) -> int:
        return hash(self.axes)

    @property
    def fermion(self) -> "_FermionFactory":
        """Return factories for fermion creation and annihilation."""
        return _FermionFactory(self)

    @property
    def boson(self) -> "_BosonFactory":
        """Return factories for boson creation and annihilation."""
        return _BosonFactory(self)

    @property
    def qubit(self) -> "_QubitFactory":
        """Return factories for physical Pauli X, Y, and Z factors."""
        return _QubitFactory(self)

    @property
    def qudit(self) -> "_QuditFactory":
        """Return the direct-convention Weyl factory."""
        return _QuditFactory(self)

    def builder(self) -> "OperatorBuilder":
        """Create a mutable batched builder for this immutable space."""
        return OperatorBuilder(self)

    def embed(
        self, operator: "_StructuredOperator", **maps: object
    ) -> "_StructuredOperator":
        """Embed an operator into this space with explicit domain index maps.

        Each supplied map is a source-to-target index mapping for one of
        ``fermions``, ``bosons``, ``qubits``, or ``qudits``. Unmapped domains
        must be empty; no implicit axis matching is performed.
        """
        if not isinstance(operator, _StructuredOperator):
            raise TypeError("embed expects a Phase 7 structured operator")
        valid_domains = {"fermions", "bosons", "qubits", "qudits"}
        unknown = set(maps) - valid_domains
        if unknown:
            raise TypeError("unexpected embedding maps: " + ", ".join(sorted(unknown)))
        mappings: Dict[str, Dict[int, int]] = {}
        for domain, source_count in (
            ("fermions", operator.space.fermions),
            ("bosons", operator.space.bosons),
            ("qubits", operator.space.qubits),
            ("qudits", len(operator.space.qudits)),
        ):
            has_supplied = domain in maps
            supplied = maps.get(domain)
            if source_count == 0:
                if has_supplied:
                    raise ValueError(f"{domain} embedding must be empty")
                mappings[domain] = {}
                continue
            if domain == "qudits":
                source_dimension = operator.space.qudits[0]
                if not self.qudits or self.qudits[0] != source_dimension:
                    raise ValueError(
                        "qudit embedding requires equal source and target dimensions"
                    )
            if not has_supplied:
                source_value = (
                    len(operator.space.qudits)
                    if domain == "qudits"
                    else getattr(operator.space, domain)
                )
                target_value = (
                    len(self.qudits) if domain == "qudits" else getattr(self, domain)
                )
                if source_value == target_value:
                    mappings[domain] = {index: index for index in range(source_count)}
                else:
                    raise ValueError(f"an explicit {domain} embedding map is required")
            elif supplied is None:
                raise ValueError(f"{domain} embedding must be a mapping or sequence")
            elif isinstance(supplied, Mapping):
                if any(
                    not isinstance(key, int)
                    or isinstance(key, bool)
                    or not isinstance(value, int)
                    or isinstance(value, bool)
                    for key, value in supplied.items()
                ):
                    raise ValueError(f"{domain} embedding indices must be integers")
                mappings[domain] = dict(supplied)
            else:
                if not isinstance(supplied, (tuple, list)):
                    raise ValueError(
                        f"{domain} embedding must be a mapping or sequence"
                    )
                values = tuple(supplied)
                if any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in values
                ):
                    raise ValueError(f"{domain} embedding indices must be integers")
                mappings[domain] = {index: value for index, value in enumerate(values)}
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
                raise ValueError(f"{domain} embedding indices are outside the target")
            if len(set(mappings[domain].values())) != source_count:
                raise ValueError(f"{domain} embedding targets must be injective")
        if operator.space == self and not any(maps.values()):
            return operator
        terms: List[_Term] = []
        for term in operator._materialized_terms():
            coefficient = term.coefficient
            fermion = (
                None
                if term.fermion is None
                else FermionWord(
                    self.fermions,
                    tuple(
                        sorted(
                            mappings["fermions"][mode]
                            for mode in term.fermion.creation_modes
                        )
                    ),
                    tuple(
                        sorted(
                            (
                                mappings["fermions"][mode]
                                for mode in term.fermion.annihilation_modes
                            ),
                            reverse=True,
                        )
                    ),
                )
            )
            if term.fermion is not None:
                creation_image = [
                    mappings["fermions"][mode] for mode in term.fermion.creation_modes
                ]
                annihilation_image = [
                    mappings["fermions"][mode]
                    for mode in term.fermion.annihilation_modes
                ]
                inversions = sum(
                    1
                    for values, descending in (
                        (creation_image, False),
                        (annihilation_image, True),
                    )
                    for index, left in enumerate(values)
                    for right in values[index + 1 :]
                    if (left > right if not descending else left < right)
                )
                if inversions & 1:
                    coefficient = -coefficient
            boson = (
                None
                if term.boson is None
                else BosonWord(
                    self.bosons,
                    tuple(
                        sorted(
                            (mappings["bosons"][mode], create, annihilate)
                            for mode, create, annihilate in term.boson.blocks
                        )
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
                        sorted(
                            (mappings["qudits"][site], a, b)
                            for site, a, b in term.qudit.triples
                        )
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
                _Term(fermion, boson, tuple(qubit), qudit, mapped, coefficient)
            )
        return _make_operator(self, terms, DEFAULT_MAX_BYTES)


class _Factory:
    def __init__(self, space: OperatorSpace) -> None:
        self.space = space


def _factory_native_operator(
    space: OperatorSpace,
    *,
    fermion_creation: Sequence[int] = (),
    fermion_annihilation: Sequence[int] = (),
    boson_blocks: Sequence[Tuple[int, int, int]] = (),
    qubit_codes: Optional[Sequence[int]] = None,
    qudit_triples: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> "HybridOperator":
    """Build one factory result from native canonical hybrid arrays."""
    result = _native.structured_hybrid_canonicalize(
        space.fermions,
        space.bosons,
        space.qubits,
        len(space.qudits),
        space.qudits[0] if space.qudits else 0,
        (
            [
                list((mode, 0) for mode in fermion_creation)
                + list((mode, 1) for mode in fermion_annihilation)
            ],
            [
                list(
                    (mode, 0) for mode, create, _ in boson_blocks for _ in range(create)
                )
                + list(
                    (mode, 1)
                    for mode, _, annihilate in boson_blocks
                    for _ in range(annihilate)
                )
            ],
            [list(qubit_codes or (0,) * space.qubits)],
            [qudit_triples is not None],
            [list(qudit_triples or ())],
            [1.0],
            [0.0],
        ),
        _effective_max_bytes(DEFAULT_MAX_BYTES),
    )
    return _hybrid_from_native(space, result)


class _FermionFactory(_Factory):
    @property
    def annihilate(self) -> Callable[[object], "HybridOperator"]:
        """Return the fermion annihilation factory."""
        return lambda mode: self._make(mode, "annihilate")

    @property
    def create(self) -> Callable[[object], "HybridOperator"]:
        """Return the fermion creation factory."""
        return lambda mode: self._make(mode, "create")

    def _make(self, mode: object, action: str) -> "HybridOperator":
        mode_value = _positive_mode(mode, self.space.fermions)
        return _factory_native_operator(
            self.space,
            fermion_creation=(mode_value,) if action == "create" else (),
            fermion_annihilation=(mode_value,) if action == "annihilate" else (),
        )


class _BosonFactory(_Factory):
    @property
    def annihilate(self) -> Callable[[object], "HybridOperator"]:
        """Return the boson annihilation factory."""
        return lambda mode: self._make(mode, "annihilate")

    @property
    def create(self) -> Callable[[object], "HybridOperator"]:
        """Return the boson creation factory."""
        return lambda mode: self._make(mode, "create")

    def _make(self, mode: object, action: str) -> "HybridOperator":
        mode_value = _positive_mode(mode, self.space.bosons)
        return _factory_native_operator(
            self.space,
            boson_blocks=(
                (mode_value, int(action == "create"), int(action == "annihilate")),
            ),
        )


class _QubitFactory(_Factory):
    @property
    def x(self) -> Callable[[object], "HybridOperator"]:
        """Return the Pauli-X factory."""
        return lambda index: self._make(index, 1)

    @property
    def y(self) -> Callable[[object], "HybridOperator"]:
        """Return the Pauli-Y factory."""
        return lambda index: self._make(index, 2)

    @property
    def z(self) -> Callable[[object], "HybridOperator"]:
        """Return the Pauli-Z factory."""
        return lambda index: self._make(index, 3)

    def _make(self, index: object, code: int) -> "HybridOperator":
        index_value = _positive_mode(index, self.space.qubits, "qubit")
        codes = [0] * self.space.qubits
        codes[index_value] = code
        return _factory_native_operator(self.space, qubit_codes=codes)


class _QuditFactory(_Factory):
    @property
    def weyl(self) -> Callable[[object, object, object], "HybridOperator"]:
        """Return the direct-convention Weyl factory."""
        return self._make

    def _make(self, site: object, a: object, b: object) -> "HybridOperator":
        site_value = _positive_mode(site, len(self.space.qudits), "site")
        return _factory_native_operator(
            self.space,
            qudit_triples=(
                (
                    site_value,
                    _nonnegative_int(a, "a") % self.space.qudits[0],
                    _nonnegative_int(b, "b") % self.space.qudits[0],
                ),
            ),
        )


@dataclass(frozen=True)
class _Term:
    """Canonical hybrid term; exposed as :class:`HybridTerm` in ``terms``."""

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


HybridTerm = _Term

StructuredNativeHandle = Union[
    _native.NativeFermionOperatorHandle,
    _native.NativeBosonOperatorHandle,
    _native.NativeHybridOperatorHandle,
]


class _StructuredOperator:
    _domain = "hybrid"
    space: OperatorSpace
    _terms: Optional[Tuple[_Term, ...]]
    _native_handle: Optional[StructuredNativeHandle]
    _locked: bool

    def __init__(
        self,
        space: OperatorSpace,
        terms: Optional[Sequence[_Term]] = None,
        native_handle: Optional[StructuredNativeHandle] = None,
    ) -> None:
        if not isinstance(space, OperatorSpace):
            raise TypeError("space must be an OperatorSpace")
        if sum(value is not None for value in (terms, native_handle)) != 1:
            raise ValueError("provide exactly one structured operator storage form")
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "_terms", None if terms is None else tuple(terms))
        object.__setattr__(self, "_native_handle", native_handle)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("operators are immutable")
        object.__setattr__(self, name, value)

    @property
    def terms(
        self,
    ) -> Tuple[Union[FermionTerm, BosonTerm, QuditWeylTerm, HybridTerm], ...]:
        """Return immutable typed canonical terms in deterministic order."""
        terms = self._materialized_terms()
        result: List[Union[FermionTerm, BosonTerm, QuditWeylTerm, HybridTerm]] = []
        for term in terms:
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
        """Return the number of nonzero canonical terms."""
        if self._native_handle is not None:
            return self._native_handle.term_count
        terms = self._terms
        if terms is None:
            raise RuntimeError("structured operator has no term storage")
        return len(terms)

    def __len__(self) -> int:
        return self.term_count

    def _check_other(self, other: object) -> "_StructuredOperator":
        if not isinstance(other, _StructuredOperator):
            raise TypeError(f"expected structured operator, got {type(other).__name__}")
        if self.space != other.space:
            raise ValueError("operators use incompatible OperatorSpace layouts")
        return other

    def _materialized_terms(self) -> Tuple[_Term, ...]:
        cached = self._terms
        if cached is not None:
            return cached
        if self._native_handle is not None:
            cached = _terms_from_native_handle(
                self.space, self._domain, self._native_handle
            )
            object.__setattr__(self, "_terms", cached)
            return cached
        terms = self._terms
        if terms is None:
            raise RuntimeError("structured operator has no native storage")
        return terms

    def _native_or_terms(self) -> Tuple[_Term, ...]:
        return self._materialized_terms()

    def to_dict(self) -> Dict[Any, complex]:
        """Return plain canonical word data without constructing typed terms."""
        if self._domain == "fermion":
            creation, annihilation, real, imaginary = _fermion_arrays(
                cast(FermionOperator, self)
            )
            return {
                tuple(
                    [(int(mode), "create") for mode in left]
                    + [(int(mode), "annihilate") for mode in right]
                ): complex(re, im)
                for left, right, re, im in zip(creation, annihilation, real, imaginary)
            }
        if self._domain == "boson":
            blocks, real, imaginary = _boson_arrays(cast(BosonOperator, self))
            return {
                tuple(tuple(block) for block in word): complex(re, im)
                for word, re, im in zip(blocks, real, imaginary)
            }
        data = _hybrid_arrays(self)
        real, imaginary = data[-2], data[-1]
        if self._domain == "qudit":
            return {
                tuple(tuple(triple) for triple in data[9][index]): complex(re, im)
                for index, (re, im) in enumerate(zip(real, imaginary))
            }
        result: Dict[Any, complex] = {}
        for index, (re, im) in enumerate(zip(real, imaginary)):
            key = (
                tuple(data[1][index]) if data[0][index] else (),
                tuple(data[2][index]) if data[0][index] else (),
                tuple(data[4][index]) if data[3][index] else (),
                tuple(data[5][index]),
                tuple(data[7][index]) if data[6][index] else (),
                tuple(data[9][index]) if data[8][index] else (),
            )
            result[key] = complex(re, im)
        return result

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
        """Return the exact structural sum of two compatible operators."""
        other = self._check_other(other)
        if self._native_handle is not None and other._native_handle is not None:
            _guard_terms(
                self.term_count + other.term_count,
                max_bytes,
                "operator addition",
            )
            return _add_native_handles(self, other, max_bytes)
        return _make_operator(
            self.space,
            self._materialized_terms() + other._materialized_terms(),
            max_bytes,
        )

    def scale(
        self, coefficient: complex, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "_StructuredOperator":
        """Return a copy with every coefficient multiplied by ``coefficient``."""
        scalar = _finite_complex(coefficient, "scale")
        if self._native_handle is not None:
            _guard_terms(self.term_count, max_bytes, "operator scaling")
            return _scale_native_handle(self, scalar)
        assert self._terms is not None
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
    ) -> MappedOperator:
        """Return the exact ordered product of compatible operators."""
        other = self._check_other(other)
        if isinstance(
            self._native_handle, _native.NativeHybridOperatorHandle
        ) and isinstance(other._native_handle, _native.NativeHybridOperatorHandle):
            if (
                isinstance(self, HybridOperator)
                and isinstance(other, HybridOperator)
                and _requires_eager_fermion_mapping(self, other)
            ):
                mapped_left = self.map_fermions(max_bytes=max_bytes)
                mapped_right = other.map_fermions(max_bytes=max_bytes)
                if isinstance(mapped_left, PauliOperator):
                    if not isinstance(mapped_right, PauliOperator):
                        raise TypeError(
                            "mapped fermion operands have incompatible types"
                        )
                    return mapped_left.multiply(mapped_right, max_bytes=max_bytes)
                if not isinstance(mapped_right, _StructuredOperator):
                    raise TypeError("mapped fermion operands have incompatible types")
                return mapped_left.multiply(mapped_right, max_bytes=max_bytes)
            return _hybrid_from_native(
                self.space,
                self._native_handle.multiply(
                    other._native_handle, _effective_max_bytes(max_bytes)
                ),
                _native_result_class(self, other),
            )
        if isinstance(self, HybridOperator) and isinstance(other, HybridOperator):
            if _requires_eager_fermion_mapping(self, other):
                # A canonical term cannot encode whether its raw fermion word
                # occurred before or after an already mapped Pauli word.  Map
                # both operands while their order is still explicit, then do
                # the Pauli product in the original operand order.
                mapped_left = self.map_fermions(max_bytes=max_bytes)
                mapped_right = other.map_fermions(max_bytes=max_bytes)
                if isinstance(mapped_left, PauliOperator):
                    if not isinstance(mapped_right, PauliOperator):
                        raise TypeError(
                            "mapped fermion operands have incompatible types"
                        )
                    return mapped_left.multiply(mapped_right, max_bytes=max_bytes)
                if not isinstance(mapped_right, _StructuredOperator):
                    raise TypeError("mapped fermion operands have incompatible types")
                return mapped_left.multiply(mapped_right, max_bytes=max_bytes)
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
        if (
            not isinstance(self, QuditWeylOperator)
            or not isinstance(other, QuditWeylOperator)
            or not isinstance(self._native_handle, _native.NativeHybridOperatorHandle)
            or not isinstance(other._native_handle, _native.NativeHybridOperatorHandle)
        ):
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
        for left_term, right_term in product(
            self._materialized_terms(), other._materialized_terms()
        ):
            products.extend(_multiply_terms(left_term, right_term, self.space))
            _guard_terms(len(products), max_bytes, "operator multiplication")
        return _make_operator(self.space, products, max_bytes)

    def commutator(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> MappedOperator:
        """Return ``self * other - other * self``."""
        other = self._check_other(other)
        if self._native_handle is not None and other._native_handle is not None:
            if isinstance(
                self._native_handle, _native.NativeFermionOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeFermionOperatorHandle):
                return _fermion_from_native(
                    FermionOperator,
                    self.space.fermions,
                    self._native_handle.commutator(
                        other._native_handle, _effective_max_bytes(max_bytes)
                    ),
                )
            if isinstance(
                self._native_handle, _native.NativeBosonOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeBosonOperatorHandle):
                return _boson_from_native(
                    BosonOperator,
                    self.space.bosons,
                    self._native_handle.commutator(
                        other._native_handle, _effective_max_bytes(max_bytes)
                    ),
                )
            if isinstance(
                self._native_handle, _native.NativeHybridOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeHybridOperatorHandle):
                if not (
                    isinstance(self, HybridOperator)
                    and isinstance(other, HybridOperator)
                    and _requires_eager_fermion_mapping(self, other)
                ):
                    return _hybrid_from_native(
                        self.space,
                        self._native_handle.commutator(
                            other._native_handle, _effective_max_bytes(max_bytes)
                        ),
                        _native_result_class(self, other),
                    )
        left = self.multiply(other, max_bytes=max_bytes)
        right = other.multiply(self, max_bytes=max_bytes)
        if isinstance(left, PauliOperator):
            if not isinstance(right, PauliOperator):
                raise TypeError("commutator operands have incompatible mapped types")
            return left.add(right.scale(-1, max_bytes=max_bytes), max_bytes=max_bytes)
        if not isinstance(right, _StructuredOperator):
            raise TypeError("commutator operands have incompatible mapped types")
        return left.add(
            right.scale(-1, max_bytes=max_bytes),
            max_bytes=max_bytes,
        )

    def anticommutator(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> MappedOperator:
        """Return ``self * other + other * self``."""
        other = self._check_other(other)
        if self._native_handle is not None and other._native_handle is not None:
            if isinstance(
                self._native_handle, _native.NativeFermionOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeFermionOperatorHandle):
                return _fermion_from_native(
                    FermionOperator,
                    self.space.fermions,
                    self._native_handle.anticommutator(
                        other._native_handle, _effective_max_bytes(max_bytes)
                    ),
                )
            if isinstance(
                self._native_handle, _native.NativeBosonOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeBosonOperatorHandle):
                return _boson_from_native(
                    BosonOperator,
                    self.space.bosons,
                    self._native_handle.anticommutator(
                        other._native_handle, _effective_max_bytes(max_bytes)
                    ),
                )
            if isinstance(
                self._native_handle, _native.NativeHybridOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeHybridOperatorHandle):
                if not (
                    isinstance(self, HybridOperator)
                    and isinstance(other, HybridOperator)
                    and _requires_eager_fermion_mapping(self, other)
                ):
                    return _hybrid_from_native(
                        self.space,
                        self._native_handle.anticommutator(
                            other._native_handle, _effective_max_bytes(max_bytes)
                        ),
                        _native_result_class(self, other),
                    )
        left = self.multiply(other, max_bytes=max_bytes)
        right = other.multiply(self, max_bytes=max_bytes)
        if isinstance(left, PauliOperator):
            if not isinstance(right, PauliOperator):
                raise TypeError(
                    "anticommutator operands have incompatible mapped types"
                )
            return left.add(right, max_bytes=max_bytes)
        if not isinstance(right, _StructuredOperator):
            raise TypeError("anticommutator operands have incompatible mapped types")
        return left.add(right, max_bytes=max_bytes)

    def adjoint(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "_StructuredOperator":
        """Return the coefficient-conjugated structural adjoint."""
        if self._native_handle is not None:
            return _adjoint_native_handle(self, max_bytes)
        assert self._terms is not None
        output: List[_Term] = []
        for term in self._terms:
            if _term_has_raw_and_mapped_fermions(term):
                raise ValueError(
                    "cannot take the adjoint of a term containing both raw "
                    "and mapped fermion factors"
                )
            fword = (
                None
                if term.fermion is None
                else FermionWord(
                    self.space.fermions,
                    tuple(reversed(term.fermion.annihilation_modes)),
                    tuple(reversed(term.fermion.creation_modes)),
                )
            )
            bword = (
                None
                if term.boson is None
                else BosonWord(
                    self.space.bosons,
                    tuple(
                        (mode, annihilation, creation)
                        for mode, creation, annihilation in term.boson.blocks
                    ),
                )
            )
            qword = None
            qcoefficient = 1.0 + 0j
            if term.qudit is not None:
                qudit_adjoint = term.qudit.adjoint()
                qword = qudit_adjoint.word
                qcoefficient = cmath.exp(
                    2j * math.pi * qudit_adjoint.phase_exponent / term.qudit.dimension
                )
            output.append(
                _Term(
                    fword,
                    bword,
                    _adjoint_qubit_codes(term.qubit),
                    qword,
                    term.mapped_fermion,
                    term.coefficient.conjugate() * qcoefficient,
                )
            )
            _guard_terms(len(output), max_bytes, "operator adjoint")
        return _make_operator(self.space, output, max_bytes)

    def is_hermitian(self, tolerance: float = 0.0) -> bool:
        """Check adjoint equality with an explicit coefficient tolerance."""
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or tolerance < 0
        ):
            raise ValueError("Hermiticity tolerance must be finite and non-negative")
        if self._native_handle is not None and hasattr(
            self._native_handle, "is_hermitian"
        ):
            return bool(self._native_handle.is_hermitian(float(tolerance)))
        other = self.adjoint()
        keys = self.to_dict()
        other_keys = other.to_dict()
        if keys.keys() != other_keys.keys():
            return False
        return all(abs(keys[key] - other_keys[key]) <= tolerance for key in keys)

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
        """Return whether this operator exactly conserves ``charge``."""
        return self.analyze_charge(charge, max_bytes=max_bytes).is_conserved

    def restrict_charge(
        self,
        sector: "ChargeSector",
        *,
        storage: "ChargeStorage" = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "ChargeRestrictedOperator":
        """Restrict an exactly conserved operator to a charge-sector MVP.

        CPU-native restricted plans default to ``storage="lazy"``. An explicit
        eager request or later materialization may populate a retained cache.
        """
        from .symmetry import _canonical_u1_sector, _restrict_u1

        canonical_u1 = _canonical_u1_sector(sector)
        if canonical_u1 is not None and all(
            axis.domain == "qubit" for axis in self.space._axes
        ):
            as_pauli = PauliOperator.from_terms(
                len(self.space._axes),
                ((term.qubit, term.coefficient) for term in self._materialized_terms()),
            )
            return _restrict_u1(
                as_pauli,
                canonical_u1,
                max_bytes,
                term_count=self.term_count,
                storage=storage,
            )  # type: ignore[return-value]
        from .charge import restrict_charge

        return restrict_charge(self, sector, storage=storage, max_bytes=max_bytes)

    def tensor_product(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "HybridOperator":
        """Combine independent spaces using ordinary or graded tensor rules."""
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
        terms: List[_Term] = []
        for left, right in product(
            self._materialized_terms(), other._materialized_terms()
        ):
            left_shifted = _shift_term(
                left,
                {"fermion": 0, "boson": 0, "qubit": 0, "qudit": 0},
                space,
                self.space,
            )
            right_shifted = _shift_term(right, offsets, space, other.space)
            if _term_has_raw_and_mapped_fermions(
                left_shifted
            ) or _term_has_raw_and_mapped_fermions(right_shifted):
                raise ValueError(
                    "tensor_product cannot operate on a term containing both "
                    "raw and mapped fermion factors"
                )
            if (
                left_shifted.mapped_fermion is not None
                or right_shifted.mapped_fermion is not None
            ):
                terms.extend(
                    _tensor_mapped_fermion_terms(
                        left_shifted,
                        right_shifted,
                        space,
                        self.space.fermions,
                        right_shifted.fermion is None
                        and right_shifted.mapped_fermion is not None,
                    )
                )
            else:
                terms.append(
                    _combine_nonexpanding_terms(left_shifted, right_shifted, space)
                )
        return HybridOperator._from_terms(space, terms, max_bytes=max_bytes)

    def map_fermions(
        self,
        mapping: Union[str, "FermionQubitMapping"] = "jordan_wigner",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> MappedOperator:
        """Map every raw fermion factor with the Phase 7 JW convention."""
        if not isinstance(mapping, str):
            from .mapping import FermionQubitMapping

            if not isinstance(mapping, FermionQubitMapping):
                raise TypeError(
                    "mapping must be a supported name or FermionQubitMapping"
                )
            if self.space.fermions == 0:
                if mapping.n_modes != 0:
                    raise ValueError(
                        "mapping plan and fermion axis counts are incompatible"
                    )
                return self
            if not isinstance(self, HybridOperator):
                raise TypeError(
                    "reusable mappings require a hybrid or fermion operator"
                )
            return mapping.map_hybrid(self, max_bytes=max_bytes)
        if mapping != "jordan_wigner":
            from .mapping import FermionQubitMapping

            plan = FermionQubitMapping.from_name(
                mapping, self.space.fermions, max_bytes=max_bytes
            )
            if self.space.fermions == 0:
                return self
            if not isinstance(self, HybridOperator):
                raise TypeError(
                    "reusable mappings require a hybrid or fermion operator"
                )
            return plan.map_hybrid(self, max_bytes=max_bytes)
        if self.space.fermions == 0:
            return self
        if isinstance(self._native_handle, _native.NativeHybridOperatorHandle):
            if self._native_handle.has_mixed_fermion_roles():
                raise ValueError(
                    "cannot map a term containing both raw and mapped fermion factors"
                )
            if isinstance(self, HybridOperator):
                result = self._native_handle.jordan_wigner(
                    _effective_max_bytes(max_bytes)
                )
                if all(
                    axis.domain in ("fermion", "qubit") for axis in self.space._axes
                ):
                    return PauliOperator._from_native_handle(
                        result.to_pauli(
                            _pauli_axis_descriptor(self.space),
                            _effective_max_bytes(max_bytes),
                        )
                    )
                return _hybrid_from_native(self.space, result)
        if isinstance(self._native_handle, _native.NativeFermionOperatorHandle):
            return PauliOperator._from_native_handle(
                self._native_handle.jordan_wigner(_effective_max_bytes(max_bytes))
            )
        if any(
            _term_has_raw_and_mapped_fermions(term)
            for term in self._materialized_terms()
        ):
            raise ValueError(
                "cannot map a term containing both raw and mapped fermion factors"
            )
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
            if all(axis.domain in ("fermion", "qubit") for axis in self.space._axes):
                return PauliOperator._from_native_handle(
                    result.to_pauli(
                        _pauli_axis_descriptor(self.space),
                        _effective_max_bytes(max_bytes),
                    )
                )
            return _hybrid_from_native(self.space, result)
        raise TypeError(
            "raw fermion mapping requires a FermionOperator or HybridOperator"
        )

    def compile(
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        mapping: Union[str, "FermionQubitMapping"] = "jordan_wigner",
        boson_cutoffs: Optional[Mapping[object, object]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Compile into a dense, sparse, native, or backend MVP target."""
        if target not in {"dense", "coo", "csr", "native_mvp", "backend_mvp"}:
            raise ValueError(
                "target must be one of 'dense', 'coo', 'csr', 'native_mvp', or 'backend_mvp'"
            )
        mapped = self.map_fermions(mapping, max_bytes=max_bytes)
        if isinstance(mapped, PauliOperator):
            result: CompileResult = mapped.compile(
                target, storage=storage, max_bytes=max_bytes
            )
            if isinstance(result, (NativeMVPPlan, BackendMVPPlan)):
                return _with_plan_metadata(
                    result,
                    mapping=(_mapping_name(mapping) if self.space.fermions else None),
                    source_term_count=self.term_count,
                )
            return result
        cutoffs = boson_cutoffs
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
        if target == "backend_mvp" and mapped.space.qudits:
            if not all(axis.domain == "qudit" for axis in mapped.space._axes):
                raise NotImplementedError(
                    "backend_mvp is deferred for mixed local-dimension layouts"
                )
            return _direct_weyl_backend_plan(mapped, max_bytes)
        return _compile_finite(
            mapped,
            target,
            normalized_cutoffs,
            max_bytes,
            storage=storage,
            mapping=(_mapping_name(mapping) if self.space.fermions else None),
        )

    def __add__(self, other: object) -> "_StructuredOperator":
        if not isinstance(other, _StructuredOperator):
            return NotImplemented
        return self.add(other)

    def __sub__(self, other: object) -> "_StructuredOperator":
        if not isinstance(other, _StructuredOperator):
            return NotImplemented
        return self.add(other.scale(-1))

    def __neg__(self) -> "_StructuredOperator":
        return self.scale(-1)

    def __mul__(self, other: object) -> MappedOperator:
        if isinstance(other, _StructuredOperator):
            return self.multiply(other)
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.scale(complex(other))
        return NotImplemented

    def __rmul__(self, other: object) -> MappedOperator:
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


def _with_plan_metadata(
    plan: Union[NativeMVPPlan, BackendMVPPlan],
    *,
    mapping: Optional[str],
    source_term_count: int,
) -> CompileResult:
    """Copy a reusable plan while preserving its private native handle."""
    if isinstance(plan, NativeMVPPlan):
        return NativeMVPPlan(
            plan.nqubits,
            plan.term_count,
            plan.strategy,
            plan._native_plan,
            storage=plan.storage,
            local_dimensions=plan.local_dimensions,
            basis_ordering=plan.basis_ordering,
            estimated_bytes=plan.estimated_bytes,
            generic_entries=plan._generic_entries,
            schema_version=plan.schema_version,
            target=plan.target,
            source_term_count=source_term_count,
            plan_term_count=plan.plan_term_count,
            mapping=mapping,
            boson_cutoffs=plan.boson_cutoffs,
            boson_boundary=plan.boson_boundary,
            qudit_dimension=plan.qudit_dimension,
            weyl_convention=plan.weyl_convention,
            _factory_token=_PLAN_FACTORY_TOKEN,
        )
    return replace(
        plan,
        mapping=mapping,
        source_term_count=source_term_count,
    )


def _mapping_name(mapping: Union[str, "FermionQubitMapping"]) -> str:
    if isinstance(mapping, str):
        return mapping
    name = mapping.name
    if not isinstance(name, str):
        raise TypeError("mapping plan has an invalid name")
    return name


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
        fword = FermionWord(
            space.fermions,
            lf.creation_modes + rf.creation_modes,
            rf.annihilation_modes + lf.annihilation_modes,
        )
        if (len(lf.annihilation_modes) * rf.parity) & 1:
            coefficient = -coefficient
    bword: Optional[BosonWord] = None
    if left.boson is not None or right.boson is not None:
        lb = left.boson or BosonWord(space.bosons)
        rb = right.boson or BosonWord(space.bosons)
        bword = BosonWord(space.bosons, lb.blocks + rb.blocks)
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
    b_products: Dict[Optional[BosonWord], complex] = {None: 1.0}
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


def _term_has_raw_and_mapped_fermions(term: _Term) -> bool:
    """Whether a term has an ambiguous raw/mapped fermion product."""
    return term.fermion is not None and term.mapped_fermion is not None


def _requires_eager_fermion_mapping(
    left: "HybridOperator", right: "HybridOperator"
) -> bool:
    """Detect a product whose operands do not share one fermion encoding."""
    if isinstance(
        left._native_handle, _native.NativeHybridOperatorHandle
    ) and isinstance(right._native_handle, _native.NativeHybridOperatorHandle):
        return (
            left._native_handle.has_raw_fermions()
            and right._native_handle.has_mapped_fermions()
        ) or (
            left._native_handle.has_mapped_fermions()
            and right._native_handle.has_raw_fermions()
        )
    terms = left._materialized_terms() + right._materialized_terms()
    has_raw = any(term.fermion is not None for term in terms)
    has_mapped = any(term.mapped_fermion is not None for term in terms)
    return has_raw and has_mapped


def _tensor_mapped_fermion_terms(
    left: _Term,
    right: _Term,
    space: OperatorSpace,
    left_fermion_count: int,
    right_is_mapped: bool,
) -> Tuple[_Term, ...]:
    """Map a fermionic tensor product using the combined global JW order.

    A raw right factor is mapped with its already shifted global mode indices,
    which naturally supplies the parity string over all left modes.  A mapped
    right factor has a local representation, so the same parity string is
    inserted explicitly according to its fermionic parity.
    """
    left_expansion = (
        _jordan_wigner_word(left.fermion)
        if left.fermion is not None
        else ((left.mapped_fermion or (0,) * space.fermions, 1.0 + 0j),)
    )
    right_expansion = (
        _jordan_wigner_word(right.fermion)
        if right.fermion is not None
        else ((right.mapped_fermion or (0,) * space.fermions, 1.0 + 0j),)
    )
    left_nonfermion = _replace(left, fermion=None, mapped_fermion=None)
    right_nonfermion = _replace(right, fermion=None, mapped_fermion=None)
    base = _combine_nonexpanding_terms(left_nonfermion, right_nonfermion, space)
    output: List[_Term] = []
    for left_codes, left_coefficient in left_expansion:
        for right_codes, right_coefficient in right_expansion:
            corrected_right = list(right_codes)
            if right_is_mapped and _pauli_fermion_parity(right_codes):
                for mode in range(left_fermion_count):
                    code, phase = _PAULI_PRODUCT[3][corrected_right[mode]]
                    corrected_right[mode] = code
                    right_coefficient *= phase
            mapped_codes, phase = _multiply_mapped(
                tuple(left_codes), tuple(corrected_right)
            )[0]
            assert mapped_codes is not None
            output.append(
                _replace(
                    base,
                    mapped_fermion=mapped_codes,
                    coefficient=base.coefficient
                    * left_coefficient
                    * right_coefficient
                    * phase,
                )
            )
    return tuple(output)


def _pauli_fermion_parity(codes: Sequence[int]) -> int:
    """Return the fermionic parity encoded by a mapped Pauli word."""
    return sum(code in (1, 2) for code in codes) & 1


def _adjoint_qubit_codes(codes: Tuple[int, ...]) -> Tuple[int, ...]:
    return codes


def _jordan_wigner_word(
    word: FermionWord,
) -> Tuple[Tuple[Tuple[int, ...], complex], ...]:
    # Keep this tensor-product adapter aligned with Rust's
    # `jordan_wigner_word_expansion`: X=1/2, Y=+i/2 for annihilation and
    # -i/2 for creation, with Z on modes below the active mode.
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


def _pauli_axis_descriptor(space: OperatorSpace) -> List[Tuple[int, int]]:
    """Encode ordered fermion/qubit axes for native Hybrid-to-Pauli lowering."""
    domains = {"fermion": 0, "qubit": 1}
    if any(axis.domain not in domains for axis in space._axes):
        raise ValueError("Pauli projection requires only fermion and qubit axes")
    return [(domains[axis.domain], axis.index) for axis in space._axes]


def _native_fermion_raw(
    n_modes: int,
    terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
    max_bytes: Optional[int],
) -> _native.NativeFermionOperatorHandle:
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
    return result


def _native_boson_raw(
    n_modes: int,
    terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
    max_bytes: Optional[int],
) -> _native.NativeBosonOperatorHandle:
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
    return _native.structured_boson_canonicalize(
        n_modes,
        factors,
        [value.real for value in coefficients],
        [value.imag for value in coefficients],
        _effective_max_bytes(max_bytes),
    )


def _fermion_arrays(
    operator: "FermionOperator",
) -> Tuple[List[List[int]], List[List[int]], List[float], List[float]]:
    if isinstance(operator._native_handle, _native.NativeFermionOperatorHandle):
        (
            _,
            creation_flat,
            creation_offsets,
            annihilation_flat,
            annihilation_offsets,
            coefficients,
        ) = operator._native_handle.materialize()
        creation_values = np.asarray(creation_flat, dtype=np.uint32)
        annihilation_values = np.asarray(annihilation_flat, dtype=np.uint32)
        creation_stops = np.asarray(creation_offsets, dtype=np.uintp)
        annihilation_stops = np.asarray(annihilation_offsets, dtype=np.uintp)
        creation_rows = [
            [int(mode) for mode in creation_values[start:stop]]
            for start, stop in zip(creation_stops[:-1], creation_stops[1:])
        ]
        annihilation_rows = [
            [int(mode) for mode in annihilation_values[start:stop]]
            for start, stop in zip(annihilation_stops[:-1], annihilation_stops[1:])
        ]
        values = np.asarray(coefficients, dtype=np.complex128)
        return (
            creation_rows,
            annihilation_rows,
            [float(value.real) for value in values],
            [float(value.imag) for value in values],
        )
    terms = operator._materialized_terms()
    fermion_terms = [term for term in terms if term.fermion is not None]
    creation_words: List[List[int]] = [
        list(cast(FermionWord, term.fermion).creation_modes) for term in fermion_terms
    ]
    annihilation_words: List[List[int]] = [
        list(cast(FermionWord, term.fermion).annihilation_modes)
        for term in fermion_terms
    ]
    coefficients = [complex(term.coefficient) for term in fermion_terms]
    return (
        creation_words,
        annihilation_words,
        [value.real for value in coefficients],
        [value.imag for value in coefficients],
    )


def _boson_arrays(
    operator: "BosonOperator",
) -> Tuple[List[List[Tuple[int, int, int]]], List[float], List[float]]:
    if isinstance(operator._native_handle, _native.NativeBosonOperatorHandle):
        _, blocks_flat, block_offsets, coefficients = (
            operator._native_handle.materialize()
        )
        blocks_values = np.asarray(blocks_flat, dtype=np.uint32).reshape((-1, 3))
        offsets = np.asarray(block_offsets, dtype=np.uintp)
        block_rows: List[List[Tuple[int, int, int]]] = [
            [
                cast(Tuple[int, int, int], tuple(int(value) for value in block))
                for block in blocks_values[start:stop]
            ]
            for start, stop in zip(offsets[:-1], offsets[1:])
        ]
        values = np.asarray(coefficients, dtype=np.complex128)
        return (
            block_rows,
            [float(value.real) for value in values],
            [float(value.imag) for value in values],
        )
    terms = operator._materialized_terms()
    boson_terms = [term for term in terms if term.boson is not None]
    block_words: List[List[Tuple[int, int, int]]] = [
        list(cast(BosonWord, term.boson).blocks) for term in boson_terms
    ]
    coefficients = [complex(term.coefficient) for term in boson_terms]
    return (
        block_words,
        [value.real for value in coefficients],
        [value.imag for value in coefficients],
    )


def _hybrid_arrays(
    operator: "_StructuredOperator",
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
    if isinstance(operator._native_handle, _native.NativeHybridOperatorHandle):
        (
            term_count,
            nqubits,
            structural_arrays,
            fixed_arrays,
        ) = operator._native_handle.materialize()
        (
            flags,
            fermion_creation,
            fermion_creation_offsets,
            fermion_annihilation,
            fermion_annihilation_offsets,
            boson_blocks,
            boson_offsets,
        ) = structural_arrays
        qubit_codes, mapped_codes, qudit_triples, qudit_offsets, coefficients = (
            fixed_arrays
        )
        flags_array = np.asarray(flags, dtype=np.uint8).reshape((int(term_count), 4))
        f_creation_values = np.asarray(fermion_creation, dtype=np.uint32)
        f_creation_stops = np.asarray(fermion_creation_offsets, dtype=np.uintp)
        f_annihilation_values = np.asarray(fermion_annihilation, dtype=np.uint32)
        f_annihilation_stops = np.asarray(fermion_annihilation_offsets, dtype=np.uintp)
        boson_values = np.asarray(boson_blocks, dtype=np.uint32).reshape((-1, 3))
        boson_stops = np.asarray(boson_offsets, dtype=np.uintp)
        qudit_values = np.asarray(qudit_triples, dtype=np.uint32).reshape((-1, 3))
        qudit_stops = np.asarray(qudit_offsets, dtype=np.uintp)
        f_creation_rows = [
            [int(value) for value in f_creation_values[start:stop]]
            for start, stop in zip(f_creation_stops[:-1], f_creation_stops[1:])
        ]
        f_annihilation_rows = [
            [int(value) for value in f_annihilation_values[start:stop]]
            for start, stop in zip(f_annihilation_stops[:-1], f_annihilation_stops[1:])
        ]
        boson_rows: List[List[Tuple[int, int, int]]] = [
            [
                cast(Tuple[int, int, int], tuple(int(value) for value in block))
                for block in boson_values[start:stop]
            ]
            for start, stop in zip(boson_stops[:-1], boson_stops[1:])
        ]
        qudit_rows: List[List[Tuple[int, int, int]]] = [
            [
                cast(Tuple[int, int, int], tuple(int(value) for value in triple))
                for triple in qudit_values[start:stop]
            ]
            for start, stop in zip(qudit_stops[:-1], qudit_stops[1:])
        ]
        values = np.asarray(coefficients, dtype=np.complex128)
        qubit_values = np.asarray(qubit_codes, dtype=np.uint8).reshape(
            (int(term_count), int(nqubits))
        )
        mapped_values = np.asarray(mapped_codes, dtype=np.uint8).reshape(
            (int(term_count), int(operator.space.fermions))
        )
        return (
            [bool(row[0]) for row in flags_array],
            f_creation_rows,
            f_annihilation_rows,
            [bool(row[1]) for row in flags_array],
            boson_rows,
            [list(row) for row in qubit_values],
            [bool(row[2]) for row in flags_array],
            [list(row) for row in mapped_values],
            [bool(row[3]) for row in flags_array],
            qudit_rows,
            [float(value.real) for value in values],
            [float(value.imag) for value in values],
        )
    fallback_fermion_present: List[bool] = []
    fallback_fermion_creation: List[List[int]] = []
    fallback_fermion_annihilation: List[List[int]] = []
    fallback_boson_present: List[bool] = []
    fallback_boson_blocks: List[List[Tuple[int, int, int]]] = []
    fallback_qubit_codes: List[List[int]] = []
    fallback_mapped_present: List[bool] = []
    fallback_mapped_codes: List[List[int]] = []
    fallback_qudit_present: List[bool] = []
    fallback_qudit_triples: List[List[Tuple[int, int, int]]] = []
    fallback_coefficients: List[complex] = []
    for term in operator._materialized_terms():
        fallback_fermion_present.append(term.fermion is not None)
        fallback_fermion_creation.append(
            list(term.fermion.creation_modes) if term.fermion is not None else []
        )
        fallback_fermion_annihilation.append(
            list(term.fermion.annihilation_modes) if term.fermion is not None else []
        )
        fallback_boson_present.append(term.boson is not None)
        fallback_boson_blocks.append(
            list(term.boson.blocks) if term.boson is not None else []
        )
        fallback_qubit_codes.append(
            list(term.qubit)
            if len(term.qubit) == operator.space.qubits
            else [0] * operator.space.qubits
        )
        fallback_mapped_present.append(term.mapped_fermion is not None)
        fallback_mapped_codes.append(
            list(term.mapped_fermion)
            if term.mapped_fermion is not None
            else [0] * operator.space.fermions
        )
        fallback_qudit_present.append(term.qudit is not None)
        fallback_qudit_triples.append(
            list(term.qudit.triples) if term.qudit is not None else []
        )
        fallback_coefficients.append(complex(term.coefficient))
    return (
        fallback_fermion_present,
        fallback_fermion_creation,
        fallback_fermion_annihilation,
        fallback_boson_present,
        fallback_boson_blocks,
        fallback_qubit_codes,
        fallback_mapped_present,
        fallback_mapped_codes,
        fallback_qudit_present,
        fallback_qudit_triples,
        [value.real for value in fallback_coefficients],
        [value.imag for value in fallback_coefficients],
    )


def _terms_from_native_handle(
    space: OperatorSpace, domain: str, handle: StructuredNativeHandle
) -> Tuple[_Term, ...]:
    """Materialize typed structural words at the explicit ``terms`` boundary."""
    native: Any = handle
    if domain == "fermion":
        (
            _term_count,
            creation_flat,
            creation_offsets,
            annihilation_flat,
            annihilation_offsets,
            coefficients,
        ) = native.materialize()
        creation_values = np.asarray(creation_flat, dtype=np.uint32)
        creation_stops = np.asarray(creation_offsets, dtype=np.uintp)
        annihilation_values = np.asarray(annihilation_flat, dtype=np.uint32)
        annihilation_stops = np.asarray(annihilation_offsets, dtype=np.uintp)
        values = np.asarray(coefficients, dtype=np.complex128)
        return tuple(
            _Term(
                FermionWord(
                    space.fermions,
                    tuple(int(value) for value in creation_values[start:stop]),
                    tuple(
                        int(value) for value in annihilation_values[ann_start:ann_stop]
                    ),
                ),
                None,
                (),
                None,
                None,
                complex(value),
            )
            for (start, stop), (ann_start, ann_stop), value in zip(
                zip(creation_stops[:-1], creation_stops[1:]),
                zip(annihilation_stops[:-1], annihilation_stops[1:]),
                values,
            )
        )
    if domain == "boson":
        _term_count, blocks_flat, block_offsets, coefficients = native.materialize()
        blocks_values = np.asarray(blocks_flat, dtype=np.uint32).reshape((-1, 3))
        offsets = np.asarray(block_offsets, dtype=np.uintp)
        values = np.asarray(coefficients, dtype=np.complex128)
        return tuple(
            _Term(
                None,
                BosonWord(
                    space.bosons,
                    tuple(
                        cast(Tuple[int, int, int], tuple(int(value) for value in block))
                        for block in blocks_values[start:stop]
                    ),
                ),
                (),
                None,
                None,
                complex(value),
            )
            for (start, stop), value in zip(zip(offsets[:-1], offsets[1:]), values)
        )
    (
        term_count,
        nqubits,
        structural_arrays,
        fixed_arrays,
    ) = native.materialize()
    (
        flags,
        fermion_creation,
        fermion_creation_offsets,
        fermion_annihilation,
        fermion_annihilation_offsets,
        boson_blocks,
        boson_offsets,
    ) = structural_arrays
    qubit_codes, mapped_codes, qudit_triples, qudit_offsets, coefficients = fixed_arrays
    flags_array = np.asarray(flags, dtype=np.uint8).reshape((int(term_count), 4))
    creation_values = np.asarray(fermion_creation, dtype=np.uint32)
    creation_stops = np.asarray(fermion_creation_offsets, dtype=np.uintp)
    annihilation_values = np.asarray(fermion_annihilation, dtype=np.uint32)
    annihilation_stops = np.asarray(fermion_annihilation_offsets, dtype=np.uintp)
    boson_values = np.asarray(boson_blocks, dtype=np.uint32).reshape((-1, 3))
    boson_stops = np.asarray(boson_offsets, dtype=np.uintp)
    qudit_values = np.asarray(qudit_triples, dtype=np.uint32).reshape((-1, 3))
    qudit_stops = np.asarray(qudit_offsets, dtype=np.uintp)
    qubit_values = np.asarray(qubit_codes, dtype=np.uint8).reshape(
        (int(term_count), int(nqubits))
    )
    mapped_values = np.asarray(mapped_codes, dtype=np.uint8).reshape(
        (int(term_count), int(space.fermions))
    )
    values = np.asarray(coefficients, dtype=np.complex128)
    qudit_dimension = space.qudits[0] if space.qudits else 0
    return tuple(
        _Term(
            (
                FermionWord(
                    space.fermions,
                    tuple(int(value) for value in creation_values[start:stop]),
                    tuple(
                        int(value) for value in annihilation_values[ann_start:ann_stop]
                    ),
                )
                if flags_array[index, 0]
                else None
            ),
            (
                BosonWord(
                    space.bosons,
                    tuple(
                        cast(Tuple[int, int, int], tuple(int(value) for value in block))
                        for block in boson_values[b_start:b_stop]
                    ),
                )
                if flags_array[index, 1]
                else None
            ),
            tuple(int(code) for code in qubit_values[index]),
            (
                QuditWeylWord(
                    qudit_dimension,
                    tuple(
                        cast(
                            Tuple[int, int, int], tuple(int(value) for value in triple)
                        )
                        for triple in qudit_values[q_start:q_stop]
                    ),
                )
                if flags_array[index, 3]
                else None
            ),
            (
                tuple(int(code) for code in mapped_values[index])
                if flags_array[index, 2]
                else None
            ),
            complex(value),
        )
        for index, (
            (start, stop),
            (ann_start, ann_stop),
            (b_start, b_stop),
            (q_start, q_stop),
            value,
        ) in enumerate(
            zip(
                zip(creation_stops[:-1], creation_stops[1:]),
                zip(annihilation_stops[:-1], annihilation_stops[1:]),
                zip(boson_stops[:-1], boson_stops[1:]),
                zip(qudit_stops[:-1], qudit_stops[1:]),
                values,
            )
        )
    )


def _add_native_handles(
    left: _StructuredOperator,
    right: _StructuredOperator,
    max_bytes: Optional[int],
) -> _StructuredOperator:
    left_handle = left._native_handle
    right_handle = right._native_handle
    if left_handle is None or right_handle is None:
        raise RuntimeError("native handle addition requires native handles")
    if isinstance(left_handle, _native.NativeFermionOperatorHandle) and isinstance(
        right_handle, _native.NativeFermionOperatorHandle
    ):
        return _fermion_from_native(
            type(left),
            left.space.fermions,
            left_handle.add(right_handle, _effective_max_bytes(max_bytes)),
        )
    if isinstance(left_handle, _native.NativeBosonOperatorHandle) and isinstance(
        right_handle, _native.NativeBosonOperatorHandle
    ):
        return _boson_from_native(
            type(left),
            left.space.bosons,
            left_handle.add(right_handle, _effective_max_bytes(max_bytes)),
        )
    if isinstance(left_handle, _native.NativeHybridOperatorHandle) and isinstance(
        right_handle, _native.NativeHybridOperatorHandle
    ):
        result = left_handle.add(right_handle, _effective_max_bytes(max_bytes))
        return _hybrid_from_native(
            left.space, result, _native_result_class(left, right)
        )
    if isinstance(left_handle, _native.NativeFermionOperatorHandle) and isinstance(
        right_handle, _native.NativeHybridOperatorHandle
    ):
        result = left_handle.to_hybrid().add(
            right_handle, _effective_max_bytes(max_bytes)
        )
        return _hybrid_from_native(left.space, result)
    if isinstance(left_handle, _native.NativeHybridOperatorHandle) and isinstance(
        right_handle, _native.NativeFermionOperatorHandle
    ):
        result = left_handle.add(
            right_handle.to_hybrid(), _effective_max_bytes(max_bytes)
        )
        return _hybrid_from_native(left.space, result)
    if isinstance(left_handle, _native.NativeBosonOperatorHandle) and isinstance(
        right_handle, _native.NativeHybridOperatorHandle
    ):
        result = left_handle.to_hybrid().add(
            right_handle, _effective_max_bytes(max_bytes)
        )
        return _hybrid_from_native(left.space, result)
    if isinstance(left_handle, _native.NativeHybridOperatorHandle) and isinstance(
        right_handle, _native.NativeBosonOperatorHandle
    ):
        result = left_handle.add(
            right_handle.to_hybrid(), _effective_max_bytes(max_bytes)
        )
        return _hybrid_from_native(left.space, result)
    raise TypeError("native structured handles have incompatible families")


def _scale_native_handle(
    operator: _StructuredOperator, scalar: complex
) -> _StructuredOperator:
    handle = operator._native_handle
    if handle is None:
        raise RuntimeError("native handle scaling requires native storage")
    if isinstance(handle, _native.NativeFermionOperatorHandle):
        return _fermion_from_native(
            type(operator),
            operator.space.fermions,
            handle.scale(scalar.real, scalar.imag),
        )
    if isinstance(handle, _native.NativeBosonOperatorHandle):
        return _boson_from_native(
            type(operator),
            operator.space.bosons,
            handle.scale(scalar.real, scalar.imag),
        )
    return _hybrid_from_native(
        operator.space, handle.scale(scalar.real, scalar.imag), type(operator)
    )


def _adjoint_native_handle(
    operator: _StructuredOperator, max_bytes: Optional[int]
) -> _StructuredOperator:
    handle = operator._native_handle
    if handle is None:
        raise RuntimeError("native handle adjoint requires native storage")
    if isinstance(handle, _native.NativeFermionOperatorHandle):
        return _fermion_from_native(
            type(operator),
            operator.space.fermions,
            handle.adjoint(_effective_max_bytes(max_bytes)),
        )
    if isinstance(handle, _native.NativeBosonOperatorHandle):
        return _boson_from_native(
            type(operator), operator.space.bosons, handle.adjoint()
        )
    return _hybrid_from_native(operator.space, handle.adjoint(), type(operator))


def _fermion_from_native(
    cls: Any,
    n_modes: int,
    result: _native.NativeFermionOperatorHandle,
) -> "FermionOperator":
    instance = object.__new__(cls)
    _StructuredOperator.__init__(
        instance,
        OperatorSpace(fermions=n_modes),
        native_handle=result,
    )
    return cast("FermionOperator", instance)


def _boson_from_native(
    cls: Any,
    n_modes: int,
    result: _native.NativeBosonOperatorHandle,
) -> "BosonOperator":
    instance = object.__new__(cls)
    _StructuredOperator.__init__(
        instance,
        OperatorSpace(bosons=n_modes),
        native_handle=result,
    )
    return cast("BosonOperator", instance)


def _hybrid_from_native(
    space: OperatorSpace,
    result: _native.NativeHybridOperatorHandle,
    cls: Any = None,
) -> "HybridOperator":
    if cls is None:
        cls = HybridOperator
    instance = object.__new__(cls)
    _StructuredOperator.__init__(instance, space, native_handle=result)
    return cast("HybridOperator", instance)


def _native_result_class(left: _StructuredOperator, right: _StructuredOperator) -> Any:
    """Keep pure-family facades while promoting genuinely mixed results."""
    if type(left) is type(right) and isinstance(
        left, (QuditWeylOperator, HybridOperator)
    ):
        return type(left)
    return HybridOperator


class FermionOperator(_StructuredOperator):
    """Immutable canonical fermionic operator governed by CAR.

    Raw ladder products are expanded and aggregated exactly. Use
    :meth:`map_fermions` for a Pauli representation or :meth:`compile` for a
    matrix/MVP target.
    """

    _domain = "fermion"

    def __init__(
        self, n_modes: int, terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]]
    ) -> None:
        result = _native_fermion_raw(n_modes, terms, DEFAULT_MAX_BYTES)
        _StructuredOperator.__init__(
            self,
            OperatorSpace(fermions=_nonnegative_int(n_modes, "n_modes")),
            native_handle=result,
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
        raw_terms = []
        for term in terms:
            if (
                term.fermion is None
                or any(
                    value is not None
                    for value in (term.boson, term.qudit, term.mapped_fermion)
                )
                or term.qubit
            ):
                raise ValueError("fermion terms must contain only fermion factors")
            raw_terms.append(
                (
                    tuple(
                        [(mode, "create") for mode in term.fermion.creation_modes]
                        + [
                            (mode, "annihilate")
                            for mode in term.fermion.annihilation_modes
                        ]
                    ),
                    term.coefficient,
                )
            )
        return _fermion_from_native(
            cls,
            space.fermions,
            _native_fermion_raw(space.fermions, raw_terms, max_bytes),
        )

    @classmethod
    def from_terms(
        cls,
        n_modes: int,
        terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionOperator":
        """Construct and CAR-canonicalize raw ordered fermion factors.

        Each term is ``(factors, coefficient)`` with factors such as
        ``(mode, "create")`` and ``(mode, "annihilate")``. Equal canonical
        words are aggregated and exact zero coefficients are removed.

        Examples:
            >>> import tencirpauli as tcp
            >>> number = tcp.FermionOperator.from_terms(
            ...     1, [(((0, "create"), (0, "annihilate")), 1.0)]
            ... )
            >>> number.term_count
            1
        """
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
        """Multiply two fermion operators while retaining CAR contractions."""
        if not isinstance(other, FermionOperator) or other.space != self.space:
            return cast(
                "_StructuredOperator", super().multiply(other, max_bytes=max_bytes)
            )
        if self._native_handle is not None and other._native_handle is not None:
            assert isinstance(self._native_handle, _native.NativeFermionOperatorHandle)
            assert isinstance(other._native_handle, _native.NativeFermionOperatorHandle)
            result = self._native_handle.multiply(
                other._native_handle, _effective_max_bytes(max_bytes)
            )
        else:
            left = _fermion_arrays(self)
            right = _fermion_arrays(other)
            result = _native.structured_fermion_multiply(
                self.space.fermions,
                left,
                right,
                _effective_max_bytes(max_bytes),
            )
        return _fermion_from_native(
            FermionOperator,
            self.space.fermions,
            result,
        )

    def map_fermions(
        self,
        mapping: Union[str, "FermionQubitMapping"] = "jordan_wigner",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> PauliOperator:
        """Map the fermion operator to a canonical Pauli operator.

        ``mapping`` may be a supported mapping name or a reusable
        :class:`FermionQubitMapping`. The result has one qubit per fermion
        mode and includes exact Jordan-Wigner/CNOT conjugation phases.
        """
        if not isinstance(mapping, str):
            from .mapping import FermionQubitMapping

            if not isinstance(mapping, FermionQubitMapping):
                raise TypeError(
                    "mapping must be a supported name or FermionQubitMapping"
                )
            return mapping.map_fermion_operator(self, max_bytes=max_bytes)
        if mapping != "jordan_wigner":
            from .mapping import FermionQubitMapping

            return FermionQubitMapping.from_name(
                mapping, self.space.fermions, max_bytes=max_bytes
            ).map_fermion_operator(self, max_bytes=max_bytes)
        if isinstance(self._native_handle, _native.NativeFermionOperatorHandle):
            return PauliOperator._from_native_handle(
                self._native_handle.jordan_wigner(_effective_max_bytes(max_bytes))
            )
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

    def compile(  # type: ignore[override]
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        mapping: Union[str, "FermionQubitMapping"] = "jordan_wigner",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Map to qubits and compile to a dense, sparse, or MVP target."""
        result: CompileResult = self.map_fermions(mapping, max_bytes=max_bytes).compile(
            target, storage=storage, max_bytes=max_bytes
        )
        if isinstance(result, (NativeMVPPlan, BackendMVPPlan)):
            return _with_plan_metadata(
                result,
                mapping=_mapping_name(mapping),
                source_term_count=self.term_count,
            )
        return result

    def to_majorana(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "MajoranaOperator":
        """Convert this canonical fermion operator to the Phase 7.5 algebra."""
        from .majorana import fermion_to_majorana

        return fermion_to_majorana(self, max_bytes=max_bytes)


class BosonOperator(_StructuredOperator):
    """Immutable symbolic boson operator with infinite-Fock CCR semantics.

    Finite matrix and MVP targets require explicit per-mode occupation cutoffs;
    symbolic algebra itself remains cutoff-free.
    """

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
        """Construct and CCR-canonicalize raw ordered boson factors."""
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
        """Multiply two boson operators while retaining CCR contractions."""
        if not isinstance(other, BosonOperator) or other.space != self.space:
            return cast(
                "_StructuredOperator", super().multiply(other, max_bytes=max_bytes)
            )
        if self._native_handle is not None and other._native_handle is not None:
            assert isinstance(self._native_handle, _native.NativeBosonOperatorHandle)
            assert isinstance(other._native_handle, _native.NativeBosonOperatorHandle)
            result = self._native_handle.multiply(
                other._native_handle, _effective_max_bytes(max_bytes)
            )
        else:
            left = _boson_arrays(self)
            right = _boson_arrays(other)
            result = _native.structured_boson_multiply(
                self.space.bosons,
                left,
                right,
                _effective_max_bytes(max_bytes),
            )
        return _boson_from_native(
            BosonOperator,
            self.space.bosons,
            result,
        )

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "BosonOperator":
        raw_terms = []
        for term in terms:
            if (
                term.boson is None
                or any(
                    value is not None
                    for value in (term.fermion, term.qudit, term.mapped_fermion)
                )
                or term.qubit
            ):
                raise ValueError("boson terms must contain only boson factors")
            raw_terms.append(
                (
                    tuple(
                        [
                            (mode, "create")
                            for mode, create, _ in term.boson.blocks
                            for _ in range(create)
                        ]
                        + [
                            (mode, "annihilate")
                            for mode, _, annihilate in term.boson.blocks
                            for _ in range(annihilate)
                        ]
                    ),
                    term.coefficient,
                )
            )
        return _boson_from_native(
            cls,
            space.bosons,
            _native_boson_raw(space.bosons, raw_terms, max_bytes),
        )

    def compile(  # type: ignore[override]
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        boson_cutoffs: Mapping[object, object],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Compile a boson operator with its required finite cutoffs."""
        return super().compile(
            target,
            storage=storage,
            boson_cutoffs=boson_cutoffs,
            max_bytes=max_bytes,
        )


class QuditWeylOperator(_StructuredOperator):
    """Immutable uniform-dimension direct-convention Weyl operator.

    Factors use ``X^a Z^b`` with exponents reduced modulo the common local
    dimension. Matrix targets use deterministic qudit-zero-MSB mixed-radix
    ordering.
    """

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
        """Construct a modular ``X^a Z^b`` operator and aggregate phases."""
        dimension = _nonnegative_int(dimension, "dimension")
        if not 3 <= dimension <= _U32_MAX:
            raise ValueError("qudit dimension must satisfy 3 <= dimension <= 2**32-1")
        input_terms = tuple(terms)
        raw_triples: List[List[Tuple[int, int, int]]] = []
        raw_coefficients: List[complex] = []
        max_site = -1
        for factors, coefficient in input_terms:
            value = _finite_complex(coefficient)
            current: Dict[int, Tuple[int, int]] = {}
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
                a %= dimension
                b %= dimension
                previous_a, previous_b = current.get(site, (0, 0))
                phase = (phase + previous_b * a) % dimension
                combined_a = (previous_a + a) % dimension
                combined_b = (previous_b + b) % dimension
                if combined_a or combined_b:
                    current[site] = (combined_a, combined_b)
                else:
                    current.pop(site, None)
            raw_triples.append(
                [(site, a, b) for site, (a, b) in sorted(current.items())]
            )
            raw_coefficients.append(value * cmath.exp(2j * math.pi * phase / dimension))
        if n_sites is not None:
            n_sites = _nonnegative_int(n_sites, "n_sites")
            if n_sites <= max_site:
                raise ValueError("n_sites is smaller than a supplied Weyl site")
        else:
            n_sites = max_site + 1
        space = OperatorSpace._from_axes(
            tuple(_Axis("qudit", i, dimension) for i in range(n_sites))
        )
        result = _native.structured_hybrid_canonicalize(
            0,
            0,
            0,
            n_sites,
            dimension,
            (
                [[] for _ in raw_triples],
                [[] for _ in raw_triples],
                [[] for _ in raw_triples],
                [True for _ in raw_triples],
                raw_triples,
                [value.real for value in raw_coefficients],
                [value.imag for value in raw_coefficients],
            ),
            _effective_max_bytes(max_bytes),
        )
        return cast("QuditWeylOperator", _hybrid_from_native(space, result, cls))

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "QuditWeylOperator":
        raw_terms = tuple(terms)
        triples: List[List[Tuple[int, int, int]]] = []
        coefficients: List[complex] = []
        for term in raw_terms:
            if (
                term.qudit is None
                or term.fermion is not None
                or term.boson is not None
                or term.qubit
                or term.mapped_fermion is not None
            ):
                raise ValueError("qudit terms must contain only qudit factors")
            triples.append(list(term.qudit.triples))
            coefficients.append(term.coefficient)
        result = _native.structured_hybrid_canonicalize(
            0,
            0,
            0,
            len(space.qudits),
            space.qudits[0] if space.qudits else 0,
            (
                [[] for _ in raw_terms],
                [[] for _ in raw_terms],
                [[] for _ in raw_terms],
                [bool(word) for word in triples],
                triples,
                [value.real for value in coefficients],
                [value.imag for value in coefficients],
            ),
            _effective_max_bytes(max_bytes),
        )
        return cast("QuditWeylOperator", _hybrid_from_native(space, result, cls))

    def compile(  # type: ignore[override]
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Compile a uniform-dimension Weyl operator."""
        return super().compile(target, storage=storage, max_bytes=max_bytes)


class HybridOperator(_StructuredOperator):
    """Immutable mixed-domain operator with canonical domain factors.

    Hybrid terms may combine fermion, boson, qubit, and qudit factors. Fermion
    mapping and finite boson cutoffs are explicit at compilation time.
    """

    _domain = "hybrid"

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "HybridOperator":
        raw_terms = tuple(terms)
        if any(term.mapped_fermion is not None for term in raw_terms):
            instance = object.__new__(cls)
            _StructuredOperator.__init__(
                instance, space, _canonical_terms(space, raw_terms, max_bytes)
            )
            return instance
        fermion_factors: List[List[Tuple[int, int]]] = []
        boson_factors: List[List[Tuple[int, int]]] = []
        qubit_codes: List[List[int]] = []
        qudit_present: List[bool] = []
        qudit_triples: List[List[Tuple[int, int, int]]] = []
        coefficients: List[complex] = []
        for term in raw_terms:
            fermion_factors.append(
                []
                if term.fermion is None
                else [(mode, 0) for mode in term.fermion.creation_modes]
                + [(mode, 1) for mode in term.fermion.annihilation_modes]
            )
            boson_factors.append(
                []
                if term.boson is None
                else [
                    (mode, 0)
                    for mode, create, _ in term.boson.blocks
                    for _ in range(create)
                ]
                + [
                    (mode, 1)
                    for mode, _, annihilate in term.boson.blocks
                    for _ in range(annihilate)
                ]
            )
            qubit_codes.append(list(term.qubit))
            qudit_present.append(term.qudit is not None)
            qudit_triples.append([] if term.qudit is None else list(term.qudit.triples))
            coefficients.append(term.coefficient)
        result = _native.structured_hybrid_canonicalize(
            space.fermions,
            space.bosons,
            space.qubits,
            len(space.qudits),
            space.qudits[0] if space.qudits else 0,
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
        return _hybrid_from_native(space, result, cls)

    def compile(
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        mapping: Union[str, "FermionQubitMapping"] = "jordan_wigner",
        boson_cutoffs: Optional[Mapping[object, object]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Compile a hybrid operator after optional Jordan-Wigner mapping."""
        return super().compile(
            target,
            storage=storage,
            mapping=mapping,
            boson_cutoffs=boson_cutoffs,
            max_bytes=max_bytes,
        )


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
    """Single-owner batched structured-term builder.

    Add raw products incrementally, then call :meth:`finish` once to perform
    native canonicalization and duplicate aggregation in one coarse-grained
    operation.
    """

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
        """Append one raw hybrid product to the construction buffer.

        The builder accepts sparse factors for each domain and returns itself
        so products can be chained. Validation and canonicalization occur in
        :meth:`finish`.
        """
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
        """Canonicalize all buffered products in one native batch call.

        Returns a deterministic :class:`HybridOperator`; repeated equal
        products are aggregated and exact zeros are removed.
        """
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
                    code = code.upper()
                    if code not in _IDENTITY_CODES:
                        raise ValueError("Pauli code must be one of I, X, Y, Z")
                    code_value = _IDENTITY_CODES[code]
                else:
                    code_value = normalize_pauli_code(code)
                qubit_index = _positive_mode(index, self.space.qubits, "qubit")
                current_code = qcodes[qubit_index]
                qcodes[qubit_index], local_phase = _PAULI_PRODUCT[current_code][
                    code_value
                ]
                qphase *= local_phase
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
    storage: Literal["lazy", "eager"] = "lazy",
    mapping: Optional[str] = None,
) -> CompileResult:
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
    if target == "native_mvp":
        native_plan = _native_structured_mvp_plan(
            operator, dimensions, cutoffs, max_bytes
        )
        qudit_dimension = operator.space.qudits[0] if operator.space.qudits else None
        return NativeMVPPlan(
            0,
            operator.term_count,
            "structured_mvp_native",
            native_plan,
            storage=storage,
            local_dimensions=dimensions,
            basis_ordering=MIXED_RADIX_BASIS_ORDERING,
            estimated_bytes=int(native_plan.estimated_bytes),
            mapping=mapping,
            boson_cutoffs=tuple(sorted(cutoffs.items())),
            boson_boundary="projected_fock" if cutoffs else None,
            qudit_dimension=qudit_dimension,
            weyl_convention="X^a Z^b" if qudit_dimension is not None else None,
            _factory_token=_PLAN_FACTORY_TOKEN,
        )
    if target == "dense":
        _check_allocation(
            dimension * dimension * 16, max_bytes, "dense structured matrix"
        )
        native_dimension, values = _native_structured_dense(
            operator, dimensions, cutoffs, max_bytes
        )
        return cast(
            np.ndarray[Any, Any],
            np.asarray(values, dtype=np.complex128).reshape(
                (native_dimension, native_dimension)
            ),
        )
    if target == "backend_mvp":
        raise NotImplementedError(
            "backend_mvp is available only for Pauli and uniform-dimension Weyl plans"
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
    raise AssertionError(target)


def _direct_weyl_backend_plan(
    operator: _StructuredOperator, max_bytes: Optional[int]
) -> BackendMVPPlan:
    """Build the compact, versioned plan for a uniform Weyl layout."""
    dimensions = operator.space.local_dimensions
    if (
        not dimensions
        or len(set(dimensions)) != 1
        or any(axis.domain != "qudit" for axis in operator.space._axes)
    ):
        raise NotImplementedError(
            "direct Weyl backend MVP requires a uniform pure-qudit layout"
        )
    # Check the Python-int product before allocating exponent arrays.  NumPy's
    # intp product wraps on overflow and would publish a false plan dimension.
    dimension = math.prod(dimensions)
    if dimension > int(np.iinfo(np.intp).max):
        raise OverflowError(
            "finite basis dimension cannot be represented by platform indices"
        )
    dimension = dimensions[0]
    term_count = operator.term_count
    a_exponents = np.zeros((term_count, len(dimensions)), dtype=np.uint32)
    b_exponents = np.zeros_like(a_exponents)
    coefficients: np.ndarray[Any, Any] = np.empty(term_count, dtype=np.complex128)
    native_data = _hybrid_arrays(operator)
    for term_index, triples in enumerate(native_data[9]):
        coefficients[term_index] = complex(
            native_data[-2][term_index], native_data[-1][term_index]
        )
        if native_data[8][term_index]:
            for site, a, b in triples:
                a_exponents[term_index, site] = a
                b_exponents[term_index, site] = b
    estimated_bytes = int(
        a_exponents.nbytes
        + b_exponents.nbytes
        + coefficients.nbytes
        + len(dimensions) * 8
        + term_count * 64
    )
    _check_allocation(estimated_bytes, max_bytes, "direct Weyl backend MVP plan")
    return BackendMVPPlan(
        2,
        0,
        0,
        np.empty((0, 0), dtype=np.uint64),
        np.empty((0, 0), dtype=np.uint64),
        coefficients,
        local_dimensions=dimensions,
        basis_ordering=DIRECT_WEYL_BASIS_ORDERING,
        ordering=DIRECT_WEYL_BASIS_ORDERING,
        estimated_bytes=estimated_bytes,
        plan_kind="direct_weyl",
        qudit_dimension=dimension,
        a_exponents=a_exponents,
        b_exponents=b_exponents,
        required_operations=("broadcast_phase", "cyclic_shift", "multiply", "add"),
        target="backend_mvp",
        source_term_count=term_count,
        plan_term_count=term_count,
        weyl_convention="X^a Z^b",
        _factory_token=_PLAN_FACTORY_TOKEN,
    )


def _native_term_arrays(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
) -> Tuple[List[List[Tuple[int, int, int, int]]], List[float], List[float]]:
    operations: List[List[Tuple[int, int, int, int]]] = []
    real: List[float] = []
    imaginary: List[float] = []
    native_data = _hybrid_arrays(operator)
    for term_index in range(len(native_data[-2])):
        term_operations: List[Tuple[int, int, int, int]] = []
        fcodes = (
            native_data[7][term_index]
            if native_data[6][term_index]
            else (0,) * operator.space.fermions
        )
        boson_blocks = (
            {
                mode: (creation, annihilation)
                for mode, creation, annihilation in native_data[4][term_index]
            }
            if native_data[3][term_index]
            else {}
        )
        qudit_triples = (
            {site: (a, b) for site, a, b in native_data[9][term_index]}
            if native_data[8][term_index]
            else {}
        )
        for position, axis in enumerate(operator.space._axes):
            if axis.domain == "fermion" and fcodes[axis.index]:
                term_operations.append((position, 0, fcodes[axis.index], 0))
            elif (
                axis.domain == "qubit"
                and axis.index < len(native_data[5][term_index])
                and native_data[5][term_index][axis.index]
            ):
                term_operations.append(
                    (position, 0, native_data[5][term_index][axis.index], 0)
                )
            elif axis.domain == "boson" and axis.index in boson_blocks:
                creation, annihilation = boson_blocks[axis.index]
                term_operations.append((position, 1, creation, annihilation))
            elif axis.domain == "qudit" and axis.index in qudit_triples:
                a, b = qudit_triples[axis.index]
                term_operations.append((position, 2, a, b))
        operations.append(term_operations)
        real.append(native_data[-2][term_index])
        imaginary.append(native_data[-1][term_index])
    return operations, real, imaginary


def _native_structured_sparse(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
) -> Tuple[int, np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    compile_handle = _compile_native_hybrid_handle(operator)
    if compile_handle is not None:
        native_dimension, rows, columns, real, imaginary = (
            _native.structured_sparse_handle(
                compile_handle,
                list(dimensions),
                _structured_axis_descriptor(operator.space),
                _effective_max_bytes(max_bytes),
            )
        )
        return (
            native_dimension,
            np.asarray(rows, dtype=np.uint64),
            np.asarray(columns, dtype=np.uint64),
            np.asarray(real, dtype=np.float64)
            + 1j * np.asarray(imaginary, dtype=np.float64),
        )
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
    compile_handle = _compile_native_hybrid_handle(operator)
    if compile_handle is not None:
        return _native.structured_sparse_plan_handle(
            compile_handle,
            list(dimensions),
            _structured_axis_descriptor(operator.space),
            _effective_max_bytes(max_bytes),
        )
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


def _structured_axis_descriptor(space: OperatorSpace) -> List[Tuple[int, int]]:
    domains = {"fermion": 0, "boson": 1, "qubit": 2, "qudit": 3}
    return [(domains[axis.domain], axis.index) for axis in space._axes]


def _native_structured_dense(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
) -> Tuple[int, Any]:
    compile_handle = _compile_native_hybrid_handle(operator)
    if compile_handle is not None:
        return _native.structured_dense_handle(
            compile_handle,
            list(dimensions),
            _structured_axis_descriptor(operator.space),
            _effective_max_bytes(max_bytes),
        )
    operations, coefficients_re, coefficients_im = _native_term_arrays(
        operator, dimensions, cutoffs
    )
    return _native.structured_dense(
        list(dimensions),
        operations,
        coefficients_re,
        coefficients_im,
        _effective_max_bytes(max_bytes),
    )


def _compile_native_hybrid_handle(
    operator: _StructuredOperator,
) -> Optional[_native.NativeHybridOperatorHandle]:
    """Promote specialized native storage without exporting operator terms."""
    if isinstance(operator._native_handle, _native.NativeHybridOperatorHandle):
        return operator._native_handle
    if isinstance(operator._native_handle, _native.NativeBosonOperatorHandle):
        return operator._native_handle.to_hybrid()
    return None


__all__ = [
    "BosonOperator",
    "BosonTerm",
    "BosonWord",
    "FermionOperator",
    "FermionTerm",
    "FermionWord",
    "HybridOperator",
    "HybridTerm",
    "OperatorBuilder",
    "OperatorSpace",
    "QuditProduct",
    "QuditWeylOperator",
    "QuditWeylTerm",
    "QuditWeylWord",
]
