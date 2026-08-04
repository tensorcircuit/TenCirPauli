"""Exact additive charges, finite charge sectors, and restricted execution."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

from . import _native
from ._validation import validate_nonnegative_int
from .hamiltonian import (
    DEFAULT_MAX_BYTES,
    MIXED_RADIX_BASIS_ORDERING,
    COOMatrix,
    CSRMatrix,
    _check_allocation,
    _effective_max_bytes,
    _validate_max_bytes,
)
from .pauli import PauliOperator
from .structured import (
    BosonWord,
    FermionWord,
    OperatorSpace,
    QuditWeylWord,
    _make_operator,
    _StructuredOperator,
    _Term,
)


_I128_MIN = -(1 << 127)
_I128_MAX = (1 << 127) - 1


def _checked_float(value: Union[int, Fraction], name: str) -> float:
    """Convert an exact scalar only when binary64 preserves it exactly."""
    exact = value if isinstance(value, Fraction) else Fraction(value, 1)
    try:
        result = float(exact)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"{name} is not representable exactly as a finite complex128 coefficient"
        ) from error
    if not math.isfinite(result) or Fraction.from_float(result) != exact:
        raise ValueError(
            f"{name} is not representable exactly as a finite complex128 coefficient"
        )
    return result


def _exact_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _exact_nonnegative(value: object, name: str) -> int:
    return validate_nonnegative_int(value, name)


def _sparse_vector(
    values: Optional[Mapping[object, object]], count: int, name: str
) -> Tuple[int, ...]:
    result = [0] * count
    if values is None:
        return tuple(result)
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key, value in values.items():
        index = _exact_nonnegative(key, f"{name} index")
        if index >= count:
            raise ValueError(f"{name} index {index} is outside the operator space")
        result[index] = _exact_int(value, f"{name} weight")
    return tuple(result)


def _qubit_levels(
    values: Optional[Mapping[object, object]], count: int
) -> Tuple[Tuple[int, int], ...]:
    result = [(0, 0) for _ in range(count)]
    if values is None:
        return tuple(result)
    if not isinstance(values, Mapping):
        raise TypeError("qubits must be a mapping")
    for key, value in values.items():
        index = _exact_nonnegative(key, "qubit index")
        if index >= count:
            raise ValueError(f"qubit index {index} is outside the operator space")
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("each qubit charge must be a (level0, level1) pair")
        result[index] = (
            _exact_int(value[0], "qubit level"),
            _exact_int(value[1], "qubit level"),
        )
    return tuple(result)


class AdditiveCharge:
    """Immutable exact integer-valued diagonal charge on an OperatorSpace."""

    __slots__ = (
        "_locked",
        "boson_weights",
        "fermion_weights",
        "name",
        "offset",
        "qubit_levels",
        "space",
    )
    space: OperatorSpace
    name: str
    offset: int
    fermion_weights: Tuple[int, ...]
    boson_weights: Tuple[int, ...]
    qubit_levels: Tuple[Tuple[int, int], ...]
    _locked: bool

    def __init__(
        self,
        space: OperatorSpace,
        *,
        name: str = "",
        fermions: Optional[Mapping[object, object]] = None,
        bosons: Optional[Mapping[object, object]] = None,
        qubits: Optional[Mapping[object, object]] = None,
        offset: int = 0,
    ) -> None:
        if not isinstance(space, OperatorSpace):
            raise TypeError("space must be an OperatorSpace")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "offset", _exact_int(offset, "offset"))
        object.__setattr__(
            self,
            "fermion_weights",
            _sparse_vector(fermions, space.fermions, "fermions"),
        )
        object.__setattr__(
            self, "boson_weights", _sparse_vector(bosons, space.bosons, "bosons")
        )
        object.__setattr__(self, "qubit_levels", _qubit_levels(qubits, space.qubits))
        if space.qudits:
            # Qudit axes are deliberately uncharged spectators in this phase.
            pass
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("AdditiveCharge is immutable")
        object.__setattr__(self, name, value)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AdditiveCharge)
            and self.space == other.space
            and self.offset == other.offset
            and self.fermion_weights == other.fermion_weights
            and self.boson_weights == other.boson_weights
            and self.qubit_levels == other.qubit_levels
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.space,
                self.offset,
                self.fermion_weights,
                self.boson_weights,
                self.qubit_levels,
            )
        )

    @property
    def layout_fingerprint(self) -> Tuple[Tuple[str, int, int], ...]:
        """Return the compatible OperatorSpace fingerprint."""
        return self.space.layout_fingerprint

    def as_operator(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> _StructuredOperator:
        """Materialize the exact diagonal generator in structured form."""
        identity_qubits = (0,) * self.space.qubits
        identity_qudit = (
            QuditWeylWord(self.space.qudits[0])
            if self.space.qudits
            and not (self.space.fermions or self.space.bosons or self.space.qubits)
            else None
        )
        terms: List[_Term] = []
        if self.offset:
            terms.append(
                _Term(
                    None,
                    None,
                    identity_qubits,
                    identity_qudit,
                    None,
                    complex(_checked_float(self.offset, "charge offset")),
                )
            )
        for mode, weight in enumerate(self.fermion_weights):
            if weight:
                terms.append(
                    _Term(
                        FermionWord(self.space.fermions, (mode,), (mode,)),
                        None,
                        identity_qubits,
                        identity_qudit,
                        None,
                        complex(_checked_float(weight, "fermion charge weight")),
                    )
                )
        for mode, weight in enumerate(self.boson_weights):
            if weight:
                terms.append(
                    _Term(
                        None,
                        BosonWord(self.space.bosons, ((mode, 1, 1),)),
                        identity_qubits,
                        identity_qudit,
                        None,
                        complex(_checked_float(weight, "boson charge weight")),
                    )
                )
        for index, (level_zero, level_one) in enumerate(self.qubit_levels):
            identity_coefficient = Fraction(level_zero + level_one, 2)
            z_coefficient = Fraction(level_zero - level_one, 2)
            if identity_coefficient:
                terms.append(
                    _Term(
                        None,
                        None,
                        identity_qubits,
                        identity_qudit,
                        None,
                        complex(
                            _checked_float(
                                identity_coefficient, "qubit charge identity"
                            )
                        ),
                    )
                )
            if z_coefficient:
                codes = [0] * self.space.qubits
                codes[index] = 3
                terms.append(
                    _Term(
                        None,
                        None,
                        tuple(codes),
                        identity_qudit,
                        None,
                        complex(
                            _checked_float(z_coefficient, "qubit charge Z coefficient")
                        ),
                    )
                )
        return _make_operator(self.space, terms, max_bytes)

    def as_pauli(self) -> PauliOperator:
        """Materialize a qubit-only charge as a Pauli operator."""
        if self.space.fermions or self.space.bosons or self.space.qudits:
            raise ValueError("as_pauli requires a pure qubit charge space")
        terms: List[Tuple[Tuple[int, ...], complex]] = []
        if self.offset:
            terms.append(
                (
                    (0,) * self.space.qubits,
                    complex(_checked_float(self.offset, "charge offset")),
                )
            )
        for index, (level_zero, level_one) in enumerate(self.qubit_levels):
            identity_coefficient = Fraction(level_zero + level_one, 2)
            z_coefficient = Fraction(level_zero - level_one, 2)
            if identity_coefficient:
                terms.append(
                    (
                        (0,) * self.space.qubits,
                        complex(
                            _checked_float(
                                identity_coefficient, "qubit charge identity"
                            )
                        ),
                    )
                )
            if z_coefficient:
                codes = [0] * self.space.qubits
                codes[index] = 3
                terms.append(
                    (
                        tuple(codes),
                        complex(
                            _checked_float(z_coefficient, "qubit charge Z coefficient")
                        ),
                    )
                )
        return PauliOperator.from_terms(self.space.qubits, terms)

    def sector(
        self,
        value: int,
        *,
        boson_cutoffs: Optional[Mapping[object, object]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "ChargeSector":
        """Select one exact charge value as a reusable finite sector plan."""
        return ChargeSector(
            ((self, _exact_int(value, "sector value")),),
            boson_cutoffs=boson_cutoffs,
            max_bytes=max_bytes,
        )

    def __repr__(self) -> str:
        return f"AdditiveCharge(name={self.name!r}, space={self.space.axes!r})"


class AdditiveSymmetryAnalysis:
    """Lightweight exact commutator result for one additive charge."""

    __slots__ = ("charge", "commutator_term_count", "is_conserved", "method")
    charge: AdditiveCharge
    is_conserved: bool
    commutator_term_count: int
    method: str

    def __init__(
        self,
        charge: AdditiveCharge,
        is_conserved: bool,
        commutator_term_count: int,
        method: str = "canonical_commutator",
    ) -> None:
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "is_conserved", bool(is_conserved))
        object.__setattr__(self, "commutator_term_count", int(commutator_term_count))
        object.__setattr__(self, "method", method)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AdditiveSymmetryAnalysis is immutable")

    def __repr__(self) -> str:
        return (
            "AdditiveSymmetryAnalysis("
            f"is_conserved={self.is_conserved}, "
            f"commutator_term_count={self.commutator_term_count})"
        )


def _validate_constraint_layout(
    constraints: Sequence[Tuple[AdditiveCharge, int]],
) -> Tuple[Tuple[AdditiveCharge, int], ...]:
    if not constraints:
        raise ValueError("ChargeSector requires at least one charge constraint")
    normalized: List[Tuple[AdditiveCharge, int]] = []
    for item in constraints:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("constraints must contain (AdditiveCharge, value) pairs")
        charge, value = item
        if not isinstance(charge, AdditiveCharge):
            raise TypeError("constraints must contain AdditiveCharge values")
        normalized_value = _exact_int(value, "sector value")
        if any(existing == charge for existing, _ in normalized):
            raise ValueError("ChargeSector constraints must not repeat a charge")
        normalized.append((charge, normalized_value))
    return tuple(normalized)


def _resolve_boson_cutoffs(
    constraints: Tuple[Tuple[AdditiveCharge, int], ...],
    supplied: Optional[Mapping[object, object]],
) -> Dict[int, int]:
    boson_count = constraints[0][0].space.bosons
    result: Dict[int, int] = {}
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise TypeError("boson_cutoffs must be a mapping")
        for key, value in supplied.items():
            mode = _exact_nonnegative(key, "boson cutoff index")
            if mode >= boson_count:
                raise ValueError("boson cutoff index is outside the operator space")
            result[mode] = _exact_nonnegative(value, "boson cutoff")
    for mode in range(boson_count):
        if mode in result:
            continue
        inferred: List[int] = []
        for charge, value in constraints:
            residual = value - charge.offset
            if residual < 0:
                inferred = []
                break
            if any(weight < 0 for weight in charge.fermion_weights):
                inferred = []
                break
            if any(weight < 0 for weight in charge.boson_weights):
                inferred = []
                break
            if any(level < 0 for levels in charge.qubit_levels for level in levels):
                inferred = []
                break
            weight = charge.boson_weights[mode]
            if weight > 0:
                inferred.append(residual // weight)
        if inferred:
            result[mode] = min(inferred)
        else:
            raise ValueError(
                "boson sector finiteness cannot be proved; provide an inclusive "
                "cutoff for every unresolved boson mode"
            )
    return result


def _local_dimensions(
    space: OperatorSpace, cutoffs: Mapping[int, int]
) -> Tuple[int, ...]:
    """Return finite local dimensions without multiplying the full space.

    A selected charge sector can be small even when the Cartesian product of
    all local axes is wider than a platform index (for example, one fermion
    in 65 modes).  The selected dimension is checked by the rank/unrank plan
    after its suffix counts are built; this helper must not impose a full
    Hilbert-space ceiling of its own.
    """
    return tuple(
        cutoffs[axis.index] + 1 if axis.domain == "boson" else axis.dimension
        for axis in space._axes
    )


def _charge_contribution(
    charge: AdditiveCharge, axis_domain: str, axis_index: int, value: int
) -> int:
    if axis_domain == "fermion":
        return charge.fermion_weights[axis_index] * value
    if axis_domain == "boson":
        return charge.boson_weights[axis_index] * value
    if axis_domain == "qubit":
        return charge.qubit_levels[axis_index][value]
    return 0


class ChargeSector:
    """Immutable checked rank/unrank plan for simultaneous additive charges."""

    __slots__ = (
        "_contributions",
        "_locked",
        "_native_plan",
        "_suffix_counts",
        "basis_ordering",
        "boson_cutoffs",
        "constraints",
        "dimension",
        "estimated_bytes",
        "local_dimensions",
        "space",
    )
    constraints: Tuple[Tuple[AdditiveCharge, int], ...]
    space: OperatorSpace
    boson_cutoffs: Tuple[Tuple[int, int], ...]
    local_dimensions: Tuple[int, ...]
    basis_ordering: str
    dimension: int
    estimated_bytes: int
    _suffix_counts: Tuple[Dict[Tuple[int, ...], int], ...]
    _contributions: Tuple[Tuple[Tuple[int, ...], ...], ...]
    _native_plan: Optional[Any]
    _locked: bool

    def __init__(
        self,
        constraints: Sequence[Tuple[AdditiveCharge, int]],
        *,
        boson_cutoffs: Optional[Mapping[object, object]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        _validate_max_bytes(max_bytes)
        normalized = _validate_constraint_layout(constraints)
        space = normalized[0][0].space
        if any(charge.space != space for charge, _ in normalized):
            raise ValueError("all charge constraints require the same OperatorSpace")
        cutoffs = _resolve_boson_cutoffs(normalized, boson_cutoffs)
        dimensions = _local_dimensions(space, cutoffs)
        target = tuple(value - charge.offset for charge, value in normalized)
        native_plan: Optional[Any] = None
        contribution_entries = sum(dimensions)
        contribution_upper_bound = (
            contribution_entries * len(normalized) * 16
            + len(dimensions) * 64
            + len(normalized) * 16
        )
        _check_allocation(
            contribution_upper_bound,
            max_bytes,
            "charge-sector contribution preflight",
        )
        axis_kinds = {
            "fermion": 0,
            "boson": 1,
            "qubit": 2,
            "qudit": 3,
        }
        compact_possible = all(
            _I128_MIN <= value <= _I128_MAX
            for charge, _ in normalized
            for value in (
                *charge.fermion_weights,
                *charge.boson_weights,
                *(level for levels in charge.qubit_levels for level in levels),
            )
        ) and all(_I128_MIN <= value <= _I128_MAX for value in target)
        suffix: List[Dict[Tuple[int, ...], int]] = []
        contribution_table: List[Tuple[Tuple[int, ...], ...]] = []
        if compact_possible:
            try:
                native_plan = _native.charge_sector_plan_compact(
                    list(dimensions),
                    [axis_kinds[axis.domain] for axis in space._axes],
                    [axis.index for axis in space._axes],
                    [list(charge.fermion_weights) for charge, _ in normalized],
                    [list(charge.boson_weights) for charge, _ in normalized],
                    [list(charge.qubit_levels) for charge, _ in normalized],
                    list(target),
                    _effective_max_bytes(max_bytes),
                )
            except OverflowError:
                native_plan = None
        if native_plan is None:
            for axis, dimension in zip(space._axes, dimensions):
                contribution_table.append(
                    tuple(
                        tuple(
                            _charge_contribution(charge, axis.domain, axis.index, value)
                            for charge, _ in normalized
                        )
                        for value in range(dimension)
                    )
                )
            zero = (0,) * len(normalized)
            suffix = [{} for _ in range(len(dimensions) + 1)]
            suffix[-1][zero] = 1
            for position in range(len(dimensions) - 1, -1, -1):
                table: Dict[Tuple[int, ...], int] = {}
                for contribution in contribution_table[position]:
                    for remainder, count in suffix[position + 1].items():
                        key = tuple(
                            contribution[index] + remainder[index]
                            for index in range(len(normalized))
                        )
                        table[key] = table.get(key, 0) + count
                suffix[position] = table
                _check_allocation(
                    len(table) * (64 + 24 * len(normalized)),
                    max_bytes,
                    "charge-sector dynamic-programming plan",
                )
            dimension = suffix[0].get(target, 0)
            if dimension > int(np.iinfo(np.intp).max):
                raise OverflowError(
                    "charge-sector dimension cannot be represented by platform indices"
                )
            estimated_bytes = (
                sum(len(table) * (64 + 24 * len(normalized)) for table in suffix)
                + contribution_upper_bound
                + len(dimensions) * 8
            )
            _check_allocation(estimated_bytes, max_bytes, "charge-sector plan")
        else:
            dimension = int(native_plan.dimension)
            estimated_bytes = int(native_plan.estimated_bytes)
        object.__setattr__(self, "constraints", normalized)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "boson_cutoffs", tuple(sorted(cutoffs.items())))
        object.__setattr__(self, "local_dimensions", dimensions)
        object.__setattr__(self, "basis_ordering", MIXED_RADIX_BASIS_ORDERING)
        object.__setattr__(self, "dimension", int(dimension))
        object.__setattr__(self, "estimated_bytes", int(estimated_bytes))
        object.__setattr__(self, "_suffix_counts", tuple(suffix))
        object.__setattr__(
            self,
            "_contributions",
            tuple(contribution_table) if native_plan is None else (),
        )
        object.__setattr__(self, "_native_plan", native_plan)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("ChargeSector is immutable")
        object.__setattr__(self, name, value)

    def _target(self) -> Tuple[int, ...]:
        return tuple(value - charge.offset for charge, value in self.constraints)

    def _validate_occupations(self, occupations: Sequence[object]) -> Tuple[int, ...]:
        values = tuple(_exact_nonnegative(value, "occupation") for value in occupations)
        if len(values) != len(self.local_dimensions):
            raise ValueError("occupations must match OperatorSpace.axes length")
        if any(
            value >= dimension
            for value, dimension in zip(values, self.local_dimensions)
        ):
            raise ValueError("occupation is outside the finite sector layout")
        return values

    def rank(self, occupations: Sequence[object]) -> int:
        """Return the lexicographic rank of one selected basis state."""
        values = self._validate_occupations(occupations)
        if self._native_plan is not None:
            return int(self._native_plan.rank(values))
        remaining = self._target()
        rank = 0
        for position, value in enumerate(values):
            table = self._contributions[position]
            for candidate in range(value):
                candidate_remaining = tuple(
                    remaining[index] - table[candidate][index]
                    for index in range(len(remaining))
                )
                rank += self._suffix_counts[position + 1].get(candidate_remaining, 0)
            remaining = tuple(
                remaining[index] - table[value][index]
                for index in range(len(remaining))
            )
        if remaining != (0,) * len(remaining):
            raise ValueError("occupations do not satisfy every charge constraint")
        return rank

    def unrank(self, index: int) -> Tuple[int, ...]:
        """Return one selected basis state without materializing earlier states."""
        index = _exact_nonnegative(index, "sector index")
        if index >= self.dimension:
            raise IndexError("sector index is out of range")
        if self._native_plan is not None:
            return tuple(int(value) for value in self._native_plan.unrank(index))
        remaining = self._target()
        values: List[int] = []
        for position, dimension in enumerate(self.local_dimensions):
            table = self._contributions[position]
            for candidate in range(dimension):
                candidate_remaining = tuple(
                    remaining[index] - table[candidate][index]
                    for index in range(len(remaining))
                )
                count = self._suffix_counts[position + 1].get(candidate_remaining, 0)
                if index < count:
                    values.append(candidate)
                    remaining = candidate_remaining
                    break
                index -= count
            else:
                raise RuntimeError("charge-sector rank/unrank plan is inconsistent")
        return tuple(values)

    def basis_states(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> np.ndarray[Any, Any]:
        """Materialize selected occupations only when explicitly requested."""
        _validate_max_bytes(max_bytes)
        axis_count = len(self.local_dimensions)
        _check_allocation(
            self.dimension * max(axis_count, 1) * 8,
            max_bytes,
            "charge-sector basis states",
        )
        if self._native_plan is not None:
            native_values: np.ndarray[Any, Any] = np.asarray(
                self._native_plan.basis_states(_effective_max_bytes(max_bytes)),
                dtype=np.uint64,
            ).reshape((self.dimension, axis_count))
            native_values.setflags(write=False)
            return native_values
        values: np.ndarray[Any, Any] = np.asarray(
            [self.unrank(index) for index in range(self.dimension)], dtype=np.uint64
        ).reshape((self.dimension, axis_count))
        values.setflags(write=False)
        return values

    def __repr__(self) -> str:
        return f"ChargeSector(dimension={self.dimension}, constraints={len(self.constraints)})"


class ChargeMvpPlan:
    """Reusable restricted-basis matrix-free transition plan."""

    __slots__ = (
        "_locked",
        "basis_ordering",
        "coefficients",
        "columns",
        "dimension",
        "estimated_bytes",
        "rows",
        "term_count",
        "transition_count",
    )
    dimension: int
    term_count: int
    transition_count: int
    rows: np.ndarray[Any, Any]
    columns: np.ndarray[Any, Any]
    coefficients: np.ndarray[Any, Any]
    estimated_bytes: int
    basis_ordering: str
    _locked: bool

    def __init__(
        self,
        dimension: int,
        term_count: int,
        rows: np.ndarray[Any, Any],
        columns: np.ndarray[Any, Any],
        coefficients: np.ndarray[Any, Any],
    ) -> None:
        row_values = np.ascontiguousarray(rows, dtype=np.uint64)
        column_values = np.ascontiguousarray(columns, dtype=np.uint64)
        coefficient_values = np.ascontiguousarray(coefficients, dtype=np.complex128)
        for value in (row_values, column_values, coefficient_values):
            value.setflags(write=False)
        object.__setattr__(self, "dimension", int(dimension))
        object.__setattr__(self, "term_count", int(term_count))
        object.__setattr__(self, "transition_count", len(row_values))
        object.__setattr__(self, "rows", row_values)
        object.__setattr__(self, "columns", column_values)
        object.__setattr__(self, "coefficients", coefficient_values)
        object.__setattr__(
            self,
            "estimated_bytes",
            int(row_values.nbytes + column_values.nbytes + coefficient_values.nbytes),
        )
        object.__setattr__(self, "basis_ordering", MIXED_RADIX_BASIS_ORDERING)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("ChargeMvpPlan is immutable")
        object.__setattr__(self, name, value)

    def apply(
        self,
        state: Sequence[complex],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply restricted transitions without a full-space allocation."""
        _validate_max_bytes(max_bytes)
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.shape[0] != self.dimension:
            raise ValueError(
                f"state must have shape ({self.dimension},), got {values.shape}"
            )
        _check_allocation(self.dimension * 16, max_bytes, "charge MVP output")
        return cast(
            np.ndarray[Any, Any],
            np.asarray(
                _native.charge_mvp_apply(
                    self.dimension,
                    self.rows,
                    self.columns,
                    self.coefficients,
                    np.ascontiguousarray(values),
                    _effective_max_bytes(max_bytes),
                ),
                dtype=np.complex128,
            ),
        )

    def __call__(self, state: Sequence[complex]) -> np.ndarray[Any, Any]:
        return self.apply(state)


class ChargeRestrictedOperator:
    """Exact action of a conserved structured/Pauli operator in one sector."""

    __slots__ = (
        "_locked",
        "_plan",
        "dimension",
        "operator",
        "sector",
    )
    operator: Union[_StructuredOperator, PauliOperator]
    sector: ChargeSector
    dimension: int
    _plan: ChargeMvpPlan
    _locked: bool

    def __init__(
        self,
        operator: Union[_StructuredOperator, PauliOperator],
        sector: ChargeSector,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        if not isinstance(sector, ChargeSector):
            raise TypeError("sector must be a ChargeSector")
        if isinstance(operator, PauliOperator):
            if sector.space != OperatorSpace(qubits=operator.nqubits):
                raise ValueError(
                    "Pauli operator and charge sector layouts are incompatible"
                )
        elif isinstance(operator, _StructuredOperator):
            if operator.space != sector.space:
                raise ValueError("operator and charge sector layouts are incompatible")
        else:
            raise TypeError("operator must be a structured or Pauli operator")
        for charge, _ in sector.constraints:
            if not analyze_charge(operator, charge, max_bytes=max_bytes).is_conserved:
                raise ValueError(
                    "selected charge sector requires an exactly conserved operator"
                )
        rows, columns, coefficients = _compile_restricted_transitions(
            operator, sector, max_bytes
        )
        plan = ChargeMvpPlan(
            sector.dimension,
            (
                len(operator.terms)
                if isinstance(operator, PauliOperator)
                else operator.term_count
            ),
            rows,
            columns,
            coefficients,
        )
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "sector", sector)
        object.__setattr__(self, "dimension", sector.dimension)
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("ChargeRestrictedOperator is immutable")
        object.__setattr__(self, name, value)

    def apply(
        self,
        state: Sequence[complex],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        return self._plan.apply(state, max_bytes=max_bytes)

    @property
    def estimated_bytes(self) -> int:
        """Return the immutable transition-array storage estimate."""
        return self._plan.estimated_bytes

    def mvp_plan(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> ChargeMvpPlan:
        _validate_max_bytes(max_bytes)
        _check_allocation(self._plan.estimated_bytes, max_bytes, "charge MVP plan")
        return self._plan

    def dense(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> np.ndarray[Any, Any]:
        _validate_max_bytes(max_bytes)
        _check_allocation(
            self.dimension * self.dimension * 16, max_bytes, "charge dense matrix"
        )
        result: np.ndarray[Any, Any] = np.zeros(
            (self.dimension, self.dimension), dtype=np.complex128
        )
        result[self._plan.rows, self._plan.columns] = self._plan.coefficients
        return result

    def coo(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> COOMatrix:
        _validate_max_bytes(max_bytes)
        _check_allocation(self._plan.estimated_bytes, max_bytes, "charge COO matrix")
        return COOMatrix(
            self._plan.rows,
            self._plan.columns,
            self._plan.coefficients,
            (self.dimension, self.dimension),
        )

    def csr(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> CSRMatrix:
        _validate_max_bytes(max_bytes)
        indptr = np.bincount(self._plan.rows + 1, minlength=self.dimension + 1).astype(
            np.intp, copy=False
        )
        np.cumsum(indptr, out=indptr)
        _check_allocation(
            int(
                indptr.nbytes
                + self._plan.columns.nbytes
                + self._plan.coefficients.nbytes
            ),
            max_bytes,
            "charge CSR matrix",
        )
        return CSRMatrix(
            indptr,
            self._plan.columns,
            self._plan.coefficients,
            (self.dimension, self.dimension),
        )


def _compile_restricted_transitions(
    operator: Union[_StructuredOperator, PauliOperator],
    sector: ChargeSector,
    max_bytes: Optional[int],
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    _validate_max_bytes(max_bytes)
    axis_positions = {
        (axis.domain, axis.index): position
        for position, axis in enumerate(sector.space._axes)
    }
    basis: Optional[np.ndarray[Any, Any]] = None
    if sector._native_plan is not None:
        _check_allocation(
            sector.estimated_bytes,
            max_bytes,
            "charge-sector rank/unrank plan",
        )
        if max_bytes is None:
            remaining = None
        else:
            remaining = max_bytes - sector.estimated_bytes
    else:
        basis = sector.basis_states(max_bytes=max_bytes)
        _check_allocation(
            int(basis.nbytes), max_bytes, "charge restricted basis workspace"
        )
        if max_bytes is None:
            remaining = None
        else:
            if basis.nbytes > max_bytes:
                raise MemoryError("charge restricted basis workspace exceeds max_bytes")
            remaining = max_bytes - int(basis.nbytes)

    fermion_positions = [
        axis_positions[("fermion", index)] for index in range(sector.space.fermions)
    ]
    boson_positions = [
        axis_positions[("boson", index)] for index in range(sector.space.bosons)
    ]
    qubit_positions = [
        axis_positions[("qubit", index)] for index in range(sector.space.qubits)
    ]
    qudit_positions = [
        axis_positions[("qudit", index)] for index in range(len(sector.space.qudits))
    ]
    term_count = (
        len(operator.terms)
        if isinstance(operator, PauliOperator)
        else operator.term_count
    )
    fermion_creation: List[List[int]] = []
    fermion_annihilation: List[List[int]] = []
    boson_blocks: List[List[Tuple[int, int, int]]] = []
    qubit_codes: List[List[int]] = []
    mapped_present: List[bool] = []
    mapped_codes: List[List[int]] = []
    qudit_present: List[bool] = []
    qudit_triples: List[List[Tuple[int, int, int]]] = []
    coefficients: List[complex] = []
    if isinstance(operator, PauliOperator):
        structures, coefficients_re, coefficients_im = operator._arrays()
        for structure, coefficient_real, coefficient_imaginary in zip(
            structures, coefficients_re, coefficients_im
        ):
            fermion_creation.append([])
            fermion_annihilation.append([])
            boson_blocks.append([])
            qubit_codes.append(list(structure))
            mapped_present.append(False)
            mapped_codes.append([0] * sector.space.fermions)
            qudit_present.append(False)
            qudit_triples.append([])
            coefficients.append(complex(coefficient_real, coefficient_imaginary))
    else:
        for structured_term in operator._terms:
            if (
                structured_term.fermion is not None
                and structured_term.mapped_fermion is not None
            ):
                raise ValueError(
                    "cannot restrict a term containing both raw and mapped fermion factors"
                )
            fermion_creation.append(
                list(structured_term.fermion.creation_modes)
                if structured_term.fermion is not None
                else []
            )
            fermion_annihilation.append(
                list(structured_term.fermion.annihilation_modes)
                if structured_term.fermion is not None
                else []
            )
            boson_blocks.append(
                list(structured_term.boson.blocks)
                if structured_term.boson is not None
                else []
            )
            qubit_codes.append(list(structured_term.qubit))
            mapped_present.append(structured_term.mapped_fermion is not None)
            mapped_codes.append(
                list(structured_term.mapped_fermion)
                if structured_term.mapped_fermion is not None
                else [0] * sector.space.fermions
            )
            qudit_present.append(structured_term.qudit is not None)
            qudit_triples.append(
                list(structured_term.qudit.triples)
                if structured_term.qudit is not None
                else []
            )
            coefficients.append(structured_term.coefficient)
    if len(coefficients) != term_count:
        raise RuntimeError("restricted transition term serialization is inconsistent")
    coefficient_array = np.ascontiguousarray(coefficients, dtype=np.complex128)
    _check_allocation(
        int(coefficient_array.nbytes),
        remaining,
        "charge restricted coefficient workspace",
    )
    if remaining is None:
        native_limit: Optional[int] = None
    else:
        native_limit = remaining - int(coefficient_array.nbytes)
    if sector._native_plan is not None:
        rows, columns, real, imaginary = sector._native_plan.compile_transitions(
            sector.dimension,
            list(sector.local_dimensions),
            fermion_positions,
            boson_positions,
            qubit_positions,
            qudit_positions,
            fermion_creation,
            fermion_annihilation,
            boson_blocks,
            qubit_codes,
            mapped_present,
            mapped_codes,
            qudit_present,
            qudit_triples,
            coefficient_array,
            sector.space.qudits[0] if sector.space.qudits else 0,
            _effective_max_bytes(native_limit),
        )
    else:
        assert basis is not None
        basis_array = np.ascontiguousarray(basis, dtype=np.uint64)
        rows, columns, real, imaginary = _native.charge_compile_transitions(
            sector.dimension,
            basis_array,
            list(sector.local_dimensions),
            fermion_positions,
            boson_positions,
            qubit_positions,
            qudit_positions,
            fermion_creation,
            fermion_annihilation,
            boson_blocks,
            qubit_codes,
            mapped_present,
            mapped_codes,
            qudit_present,
            qudit_triples,
            coefficient_array,
            sector.space.qudits[0] if sector.space.qudits else 0,
            _effective_max_bytes(native_limit),
        )
    return (
        np.asarray(rows, dtype=np.intp),
        np.asarray(columns, dtype=np.intp),
        np.asarray(real, dtype=np.float64)
        + 1j * np.asarray(imaginary, dtype=np.float64),
    )


def _add_exact_scaled(
    aggregate: Dict[Tuple[object, ...], List[Fraction]],
    key: Tuple[object, ...],
    coefficient: complex,
    real_scale: int,
    imaginary_scale: int,
) -> None:
    """Accumulate a complex128 coefficient times exact integer scales."""
    if real_scale == 0 and imaginary_scale == 0:
        return
    coefficient = complex(coefficient)
    real = (
        Fraction.from_float(coefficient.real) * real_scale
        - Fraction.from_float(coefficient.imag) * imaginary_scale
    )
    imaginary = (
        Fraction.from_float(coefficient.real) * imaginary_scale
        + Fraction.from_float(coefficient.imag) * real_scale
    )
    current = aggregate.setdefault(key, [Fraction(0), Fraction(0)])
    current[0] += real
    current[1] += imaginary


def _structured_charge_delta(term: _Term, charge: AdditiveCharge) -> int:
    delta = 0
    if term.fermion is not None:
        delta += sum(
            charge.fermion_weights[mode] for mode in term.fermion.creation_modes
        )
        delta -= sum(
            charge.fermion_weights[mode] for mode in term.fermion.annihilation_modes
        )
    if term.boson is not None:
        delta += sum(
            charge.boson_weights[mode] * (creation - annihilation)
            for mode, creation, annihilation in term.boson.blocks
        )
    return delta


def _structured_commutator_key(
    term: _Term, qubit_codes: Sequence[int]
) -> Tuple[object, ...]:
    key = list(term.key())
    key[3] = tuple(qubit_codes)
    return tuple(key)


def _exact_charge_commutator(
    operator: Union[_StructuredOperator, PauliOperator],
    charge: AdditiveCharge,
    max_bytes: Optional[int],
) -> Tuple[bool, int]:
    """Return conservation and term count using integer selection rules.

    Fermion and boson monomials have a constant charge delta.  A qubit X/Y
    factor is split into its exact local commutator with the diagonal Z part
    of the charge.  The resulting canonical keys are aggregated with exact
    binary-float coefficient fractions, so neither large integer weights nor
    cancellation decisions pass through a lossy charge generator.
    """
    if isinstance(operator, PauliOperator):
        expected_space = OperatorSpace(qubits=operator.nqubits)
        if charge.space != expected_space:
            raise ValueError("operator and charge layouts are incompatible")
        term_count = len(operator.terms)
        qubit_count = operator.nqubits
        structures, coefficients_re, coefficients_im = operator._arrays()
        terms: Iterable[Tuple[Sequence[int], complex, Optional[_Term]]] = zip(
            structures,
            (
                complex(real, imaginary)
                for real, imaginary in zip(coefficients_re, coefficients_im)
            ),
            (None for _ in structures),
        )
    elif isinstance(operator, _StructuredOperator):
        if operator.space != charge.space:
            raise ValueError("operator and charge layouts are incompatible")
        if any(term.mapped_fermion is not None for term in operator._terms):
            raise ValueError(
                "charge analysis is defined before fermion-to-qubit mapping; "
                "analyze the raw structured operator"
            )
        term_count = len(operator._terms)
        qubit_count = operator.space.qubits
        terms = ((term.qubit, term.coefficient, term) for term in operator._terms)
    else:
        raise TypeError("operator must be a structured or Pauli operator")

    estimated = term_count * max(1, qubit_count + 1) * 128
    _check_allocation(estimated, max_bytes, "exact additive-charge analysis")
    aggregate: Dict[Tuple[object, ...], List[Fraction]] = {}
    for codes, coefficient, structured_term in terms:
        if structured_term is None:
            base_key: Tuple[object, ...] = tuple(codes)
            raw_delta = 0
        else:
            base_key = structured_term.key()
            raw_delta = _structured_charge_delta(structured_term, charge)
        if raw_delta:
            _add_exact_scaled(aggregate, base_key, coefficient, -raw_delta, 0)
        for index, code in enumerate(codes):
            if code not in (1, 2):
                continue
            difference = charge.qubit_levels[index][0] - charge.qubit_levels[index][1]
            if not difference:
                continue
            changed = list(codes)
            changed[index] = 2 if code == 1 else 1
            key = (
                tuple(changed)
                if structured_term is None
                else _structured_commutator_key(structured_term, changed)
            )
            _add_exact_scaled(
                aggregate,
                key,
                coefficient,
                0,
                -difference if code == 1 else difference,
            )
    nonzero = sum(1 for real, imaginary in aggregate.values() if real or imaginary)
    return nonzero == 0, nonzero


def analyze_charge(
    operator: Union[_StructuredOperator, PauliOperator],
    charge: AdditiveCharge,
    *,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> AdditiveSymmetryAnalysis:
    """Analyze an exact additive charge through integer selection rules."""
    if not isinstance(charge, AdditiveCharge):
        raise TypeError("charge must be an AdditiveCharge")
    _validate_max_bytes(max_bytes)
    is_conserved, commutator_term_count = _exact_charge_commutator(
        operator, charge, max_bytes
    )
    return AdditiveSymmetryAnalysis(
        charge,
        is_conserved,
        commutator_term_count,
        method="exact_integer_selection_rules",
    )


def restrict_charge(
    operator: Union[_StructuredOperator, PauliOperator],
    sector: ChargeSector,
    *,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> ChargeRestrictedOperator:
    """Validate exact conservation and build a restricted transition plan."""
    if not isinstance(sector, ChargeSector):
        raise TypeError("sector must be a ChargeSector")
    return ChargeRestrictedOperator(operator, sector, max_bytes=max_bytes)
