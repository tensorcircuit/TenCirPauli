"""Exact additive charges, finite charge sectors, and restricted execution."""

from __future__ import annotations

import math
from typing import (
    Any,
    Dict,
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


def _exact_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _exact_nonnegative(value: object, name: str) -> int:
    result = _exact_int(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


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
                    complex(self.offset),
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
                        complex(weight),
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
                        complex(weight),
                    )
                )
        for index, (level_zero, level_one) in enumerate(self.qubit_levels):
            identity_coefficient = (level_zero + level_one) / 2
            z_coefficient = (level_zero - level_one) / 2
            if identity_coefficient:
                terms.append(
                    _Term(
                        None,
                        None,
                        identity_qubits,
                        identity_qudit,
                        None,
                        complex(identity_coefficient),
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
                        complex(z_coefficient),
                    )
                )
        return _make_operator(self.space, terms, max_bytes)

    def as_pauli(self) -> PauliOperator:
        """Materialize a qubit-only charge as a Pauli operator."""
        if self.space.fermions or self.space.bosons or self.space.qudits:
            raise ValueError("as_pauli requires a pure qubit charge space")
        terms: List[Tuple[Tuple[int, ...], complex]] = []
        if self.offset:
            terms.append(((0,) * self.space.qubits, complex(self.offset)))
        for index, (level_zero, level_one) in enumerate(self.qubit_levels):
            identity_coefficient = (level_zero + level_one) / 2
            z_coefficient = (level_zero - level_one) / 2
            if identity_coefficient:
                terms.append(((0,) * self.space.qubits, complex(identity_coefficient)))
            if z_coefficient:
                codes = [0] * self.space.qubits
                codes[index] = 3
                terms.append((tuple(codes), complex(z_coefficient)))
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
    dimensions = tuple(
        cutoffs[axis.index] + 1 if axis.domain == "boson" else axis.dimension
        for axis in space._axes
    )
    dimension = math.prod(dimensions)
    if dimension > int(np.iinfo(np.intp).max):
        raise OverflowError(
            "charge-sector dimension cannot be represented by platform indices"
        )
    return dimensions


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
        contribution_table: List[Tuple[Tuple[int, ...], ...]] = []
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
        suffix: List[Dict[Tuple[int, ...], int]] = [
            {} for _ in range(len(dimensions) + 1)
        ]
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
        target = tuple(value - charge.offset for charge, value in normalized)
        dimension = suffix[0].get(target, 0)
        if dimension > int(np.iinfo(np.intp).max):
            raise OverflowError(
                "charge-sector dimension cannot be represented by platform indices"
            )
        estimated_bytes = (
            sum(len(table) * (64 + 24 * len(normalized)) for table in suffix)
            + len(dimensions) * 8
        )
        _check_allocation(estimated_bytes, max_bytes, "charge-sector plan")
        object.__setattr__(self, "constraints", normalized)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "boson_cutoffs", tuple(sorted(cutoffs.items())))
        object.__setattr__(self, "local_dimensions", dimensions)
        object.__setattr__(self, "basis_ordering", MIXED_RADIX_BASIS_ORDERING)
        object.__setattr__(self, "dimension", int(dimension))
        object.__setattr__(self, "estimated_bytes", int(estimated_bytes))
        object.__setattr__(self, "_suffix_counts", tuple(suffix))
        object.__setattr__(self, "_contributions", tuple(contribution_table))
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
        indptr: np.ndarray[Any, Any] = np.zeros(self.dimension + 1, dtype=np.intp)
        np.add.at(indptr, self._plan.rows + 1, 1)
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
    basis = sector.basis_states(max_bytes=max_bytes)
    _check_allocation(int(basis.nbytes), max_bytes, "charge restricted basis workspace")
    if max_bytes is None:
        remaining: Optional[int] = None
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
        for pauli_term in operator.terms:
            fermion_creation.append([])
            fermion_annihilation.append([])
            boson_blocks.append([])
            qubit_codes.append(list(pauli_term.word.to_codes()))
            mapped_present.append(False)
            mapped_codes.append([0] * sector.space.fermions)
            qudit_present.append(False)
            qudit_triples.append([])
            coefficients.append(pauli_term.coefficient)
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
    basis_array = np.ascontiguousarray(basis, dtype=np.uint64)
    _check_allocation(
        int(coefficient_array.nbytes),
        remaining,
        "charge restricted coefficient workspace",
    )
    if remaining is None:
        native_limit: Optional[int] = None
    else:
        native_limit = remaining - int(coefficient_array.nbytes)
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


def analyze_charge(
    operator: Union[_StructuredOperator, PauliOperator],
    charge: AdditiveCharge,
    *,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> AdditiveSymmetryAnalysis:
    """Compute the complete canonical commutator without term-wise shortcuts."""
    if not isinstance(charge, AdditiveCharge):
        raise TypeError("charge must be an AdditiveCharge")
    _validate_max_bytes(max_bytes)
    commutator: Union[_StructuredOperator, PauliOperator]
    if isinstance(operator, PauliOperator):
        pauli_generator = charge.as_pauli()
        if operator.nqubits != pauli_generator.nqubits:
            raise ValueError("operator and charge layouts are incompatible")
        commutator = operator.commutator(pauli_generator)
    elif isinstance(operator, _StructuredOperator):
        if operator.space != charge.space:
            raise ValueError("operator and charge layouts are incompatible")
        structured_generator = charge.as_operator(max_bytes=max_bytes)
        commutator = operator.commutator(structured_generator, max_bytes=max_bytes)
    else:
        raise TypeError("operator must be a structured or Pauli operator")
    if not isinstance(commutator, (PauliOperator, _StructuredOperator)):
        raise TypeError("commutator returned an incompatible operator type")
    commutator_term_count = len(commutator.terms)
    return AdditiveSymmetryAnalysis(
        charge,
        commutator_term_count == 0,
        commutator_term_count,
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
