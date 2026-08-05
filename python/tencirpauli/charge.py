"""Exact additive charges, finite charge sectors, and restricted execution."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import (
    Any,
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
from ._validation import validate_nonnegative_int
from .hamiltonian import (
    DEFAULT_MAX_BYTES,
    MIXED_RADIX_BASIS_ORDERING,
    COOMatrix,
    CSRMatrix,
    _check_allocation,
    _effective_max_bytes,
    _validate_apply_into_buffers,
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
ChargeStorage = Literal["eager", "lazy"]


def _validate_storage(value: object) -> ChargeStorage:
    if value == "eager":
        return "eager"
    if value == "lazy":
        return "lazy"
    raise ValueError("storage must be either 'eager' or 'lazy'")


def _charge_materialization_bytes(
    dimension: int, transition_count: int, target: str
) -> int:
    if target == "dense":
        return dimension * dimension * 16
    if target == "coo":
        return transition_count * 32
    if target == "csr":
        return (dimension + 1) * 8 + transition_count * 24
    raise ValueError(f"unsupported charge materialization target: {target}")


def _checked_float(value: Union[int, float], name: str) -> float:
    """Convert a public charge scalar to the ordinary binary64 representation."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be representable as a finite float64") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
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
    """Immutable exact integer-valued diagonal charge on an ``OperatorSpace``.

    Fermion and boson weights contribute ``weight * occupation``. Qubit axes
    use the explicit ``(level_0, level_1)`` pair, and ``offset`` is added to
    every eigenvalue. Qudit axes are currently spectators. Charges are the
    input to :meth:`sector` and :func:`analyze_charge`.
    """

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
        """Return the axis layout required by operators using this charge.

        The fingerprint is deterministic and can be compared with another
        operator-space fingerprint before constructing a sector.
        """
        return self.space.layout_fingerprint

    def as_operator(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> _StructuredOperator:
        """Materialize the charge generator as a structured operator.

        Returns:
            A canonical operator whose diagonal eigenvalue on each basis state
            is the charge value. The result preserves the charge's operator
            space and deterministic term ordering.

        Raises:
            MemoryError: If the estimated operator workspace exceeds
                ``max_bytes``.
        """
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
            identity_coefficient = (level_zero + level_one) / 2.0
            z_coefficient = (level_zero - level_one) / 2.0
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
        """Materialize a pure-qubit charge as a Pauli operator.

        The qubit level pair ``(level_0, level_1)`` is decomposed into an
        identity coefficient and a ``Z`` coefficient. Fermion, boson, and
        qudit charges are rejected because they are not Pauli operators.
        """
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
            identity_coefficient = (level_zero + level_one) / 2.0
            z_coefficient = (level_zero - level_one) / 2.0
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
        """Select one exact charge value as a reusable finite sector plan.

        Args:
            value: Required charge eigenvalue.
            boson_cutoffs: Optional inclusive upper occupation bound for each
                boson mode. Every boson mode must have a finite cutoff when a
                finite sector basis is needed.
            max_bytes: Best-effort bound for the rank/unrank plan workspace.

        Returns:
            A :class:`ChargeSector` with deterministic mixed-radix ordering.

        Examples:
            >>> import tencirpauli as tcp
            >>> space = tcp.OperatorSpace(qubits=2)
            >>> charge = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
            >>> charge.sector(1).dimension
            2
        """
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
    """Immutable rank/unrank plan for simultaneous additive charge constraints.

    A sector enumerates only finite operator-space basis states satisfying all
    constraints. Its ordering is ``operator_space_axis0_msb_mixed_radix`` and
    is shared by ``rank``, ``unrank``, ``basis_states`` and restricted matrix
    targets.
    """

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
        """Return the deterministic rank of a selected basis occupation.

        Args:
            occupations: One non-negative occupation per ``OperatorSpace``
                axis, in ``local_dimensions`` order.

        Returns:
            An integer in ``[0, dimension)``. No basis array is materialized.

        Raises:
            ValueError: If the occupations have the wrong length, exceed a
                local dimension, or violate any charge constraint.
        """
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
        """Return the occupation tuple at a deterministic sector index.

        Args:
            index: Zero-based index in the restricted sector ordering.

        Returns:
            One occupation per operator-space axis. This operation does not
            materialize any preceding basis state.

        Raises:
            IndexError: If ``index`` is outside ``[0, dimension)``.
        """
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
        """Materialize all selected occupations as a read-only ``uint64`` array.

        The result has shape ``(dimension, len(local_dimensions))`` and follows
        the same ordering as :meth:`rank` and :meth:`unrank`. This is an
        explicit potentially large allocation; use ``max_bytes`` to guard it.
        """
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


@dataclass(frozen=True)
class _RestrictedTransitionInputs:
    local_dimensions: Tuple[int, ...]
    fermion_positions: List[int]
    boson_positions: List[int]
    qubit_positions: List[int]
    qudit_positions: List[int]
    fermion_creation: List[List[int]]
    fermion_annihilation: List[List[int]]
    boson_blocks: List[List[Tuple[int, int, int]]]
    qubit_codes: List[List[int]]
    mapped_present: List[bool]
    mapped_codes: List[List[int]]
    qudit_present: List[bool]
    qudit_triples: List[List[Tuple[int, int, int]]]
    coefficients: np.ndarray[Any, Any]
    qudit_dimension: int
    termwise_conserved: bool
    fast_fermion_particles: Optional[int]


class ChargeMvpPlan:
    """Reusable matrix-free transition plan in a finite charge-sector basis.

    The native handle owns validated destination-major CSR storage and can
    apply the same restricted operator to many state vectors without
    materializing a dense matrix. Public COO row indices are derived only when
    requested.
    """

    __slots__ = (
        "_locked",
        "_native_plan",
        "basis_ordering",
        "dimension",
        "estimated_bytes",
        "storage",
        "strategy",
        "target",
        "term_count",
        "transition_count",
    )
    dimension: int
    term_count: int
    transition_count: int
    estimated_bytes: int
    basis_ordering: str
    storage: ChargeStorage
    strategy: str
    target: str
    _native_plan: Any
    _locked: bool

    def __init__(
        self,
        dimension: int,
        term_count: int,
        *,
        native_plan: Any,
        storage: ChargeStorage = "eager",
    ) -> None:
        object.__setattr__(self, "dimension", int(dimension))
        object.__setattr__(self, "term_count", int(term_count))
        object.__setattr__(self, "transition_count", int(native_plan.transition_count))
        object.__setattr__(self, "_native_plan", native_plan)
        object.__setattr__(
            self,
            "estimated_bytes",
            int(native_plan.estimated_bytes),
        )
        object.__setattr__(self, "basis_ordering", MIXED_RADIX_BASIS_ORDERING)
        object.__setattr__(self, "storage", _validate_storage(storage))
        object.__setattr__(self, "strategy", "destination_major_csr")
        object.__setattr__(self, "target", "native_mvp")
        object.__setattr__(self, "_locked", True)

    def _csr_arrays(
        self,
    ) -> Tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ]:
        native_indptr, native_columns, native_coefficients = self._native_plan.csr()
        indptr = np.asarray(native_indptr, dtype=np.uint64)
        columns = np.asarray(native_columns, dtype=np.uint64)
        coefficients = np.asarray(native_coefficients, dtype=np.complex128)
        for value in (indptr, columns, coefficients):
            value.setflags(write=False)
        return indptr, columns, coefficients

    def _coo_arrays(
        self,
        *,
        max_bytes: Optional[int] = None,
    ) -> Tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ]:
        native_rows, native_columns, native_coefficients = self._native_plan.coo(
            _effective_max_bytes(max_bytes)
        )
        rows = np.asarray(native_rows, dtype=np.uint64)
        columns = np.asarray(native_columns, dtype=np.uint64)
        coefficients = np.asarray(native_coefficients, dtype=np.complex128)
        for value in (rows, columns, coefficients):
            value.setflags(write=False)
        return rows, columns, coefficients

    @property
    def indptr(self) -> np.ndarray[Any, Any]:
        """Return a read-only CSR pointer array generated from the native handle."""
        return self._csr_arrays()[0]

    @property
    def columns(self) -> np.ndarray[Any, Any]:
        """Return read-only CSR column indices generated from the native handle."""
        return self._csr_arrays()[1]

    @property
    def coefficients(self) -> np.ndarray[Any, Any]:
        """Return read-only CSR values generated from the native handle."""
        return self._csr_arrays()[2]

    @property
    def rows(self) -> np.ndarray[Any, Any]:
        """Return read-only COO row indices generated from the native handle."""
        return self._coo_arrays()[0]

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
                self._native_plan.apply(
                    np.ascontiguousarray(values), _effective_max_bytes(max_bytes)
                ),
                dtype=np.complex128,
                order="C",
            ),
        )

    def apply_into(
        self,
        input_state: np.ndarray[Any, Any],
        output_state: np.ndarray[Any, Any],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        """Apply into strict caller-owned buffers without changing the input."""
        _validate_apply_into_buffers(input_state, output_state, self.dimension)
        self._native_plan.apply_into(
            input_state,
            output_state,
            _effective_max_bytes(max_bytes),
        )

    def __call__(self, state: Sequence[complex]) -> np.ndarray[Any, Any]:
        return self.apply(state)


class ChargeLazyMvpPlan:
    """Explicitly lazy native MVP plan for a finite charge sector.

    Unlike :class:`ChargeMvpPlan`, this plan does not retain the complete
    restricted transition graph. The native implementation enumerates source
    basis states and aggregates destinations for one source column at a time.
    Dense and sparse materialization are intentionally unavailable for this
    storage strategy.
    """

    __slots__ = (
        "_locked",
        "_native_execution",
        "_native_plan",
        "basis_ordering",
        "dimension",
        "estimated_bytes",
        "storage",
        "strategy",
        "target",
        "term_count",
    )
    dimension: int
    term_count: int
    estimated_bytes: int
    basis_ordering: str
    storage: ChargeStorage
    strategy: str
    target: str
    _locked: bool
    _native_plan: Any
    _native_execution: Any

    def __init__(
        self,
        sector: ChargeSector,
        term_count: int,
        inputs: "_RestrictedTransitionInputs",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        if sector._native_plan is None:
            raise NotImplementedError(
                "storage='lazy' requires a native compact ChargeSector plan"
            )
        object.__setattr__(self, "_native_plan", sector._native_plan)
        object.__setattr__(
            self,
            "_native_execution",
            sector._native_plan.compile_mvp(
                sector.dimension,
                list(inputs.local_dimensions),
                inputs.fermion_positions,
                inputs.boson_positions,
                inputs.qubit_positions,
                inputs.qudit_positions,
                inputs.fermion_creation,
                inputs.fermion_annihilation,
                inputs.boson_blocks,
                inputs.qubit_codes,
                inputs.mapped_present,
                inputs.mapped_codes,
                inputs.qudit_present,
                inputs.qudit_triples,
                inputs.coefficients,
                inputs.qudit_dimension,
                inputs.termwise_conserved,
                _effective_max_bytes(max_bytes),
                inputs.fast_fermion_particles,
            ),
        )
        object.__setattr__(self, "dimension", sector.dimension)
        object.__setattr__(self, "term_count", int(term_count))
        object.__setattr__(
            self,
            "estimated_bytes",
            int(self._native_execution.estimated_bytes),
        )
        object.__setattr__(self, "basis_ordering", MIXED_RADIX_BASIS_ORDERING)
        object.__setattr__(self, "storage", "lazy")
        object.__setattr__(self, "strategy", "term_direct")
        object.__setattr__(self, "target", "native_mvp")
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("ChargeLazyMvpPlan is immutable")
        object.__setattr__(self, name, value)

    def apply(
        self,
        state: Sequence[complex],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the operator without storing all restricted transitions."""
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
                self._native_execution.apply(
                    np.ascontiguousarray(values), _effective_max_bytes(max_bytes)
                ),
                dtype=np.complex128,
                order="C",
            ),
        )

    def apply_into(
        self,
        input_state: np.ndarray[Any, Any],
        output_state: np.ndarray[Any, Any],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        """Apply the lazy native handle into strict caller-owned storage."""
        _validate_apply_into_buffers(input_state, output_state, self.dimension)
        self._native_execution.apply_into(
            input_state,
            output_state,
            _effective_max_bytes(max_bytes),
        )

    def __call__(self, state: Sequence[complex]) -> np.ndarray[Any, Any]:
        return self.apply(state)


class ChargeRestrictedOperator:
    """Exact action of a conserved structured or Pauli operator in one sector.

    Construction validates exact charge conservation and retains a compact
    lazy plan. Explicit sparse/dense materialization or an eager MVP request
    installs one immutable transition-graph cache on this mathematical facade.
    """

    __slots__ = (
        "_eager_plan",
        "_lazy_plan",
        "_lock",
        "_locked",
        "dimension",
        "operator",
        "sector",
        "storage",
    )
    operator: Union[_StructuredOperator, PauliOperator]
    sector: ChargeSector
    dimension: int
    _lazy_plan: ChargeLazyMvpPlan
    _eager_plan: Optional[ChargeMvpPlan]
    _lock: threading.Lock
    storage: ChargeStorage
    _locked: bool

    def __init__(
        self,
        operator: Union[_StructuredOperator, PauliOperator],
        sector: ChargeSector,
        *,
        storage: ChargeStorage = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        storage = _validate_storage(storage)
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
        term_count = (
            len(operator.terms)
            if isinstance(operator, PauliOperator)
            else operator.term_count
        )
        inputs = _restricted_transition_inputs(operator, sector)
        _check_allocation(
            int(sector.estimated_bytes + inputs.coefficients.nbytes),
            max_bytes,
            "lazy charge MVP plan",
        )
        lazy_plan = ChargeLazyMvpPlan(sector, term_count, inputs, max_bytes)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "sector", sector)
        object.__setattr__(self, "dimension", sector.dimension)
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "_lazy_plan", lazy_plan)
        object.__setattr__(self, "_eager_plan", None)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_locked", True)
        if storage == "eager":
            self._ensure_eager(max_bytes)

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
        """Apply the restricted operator to one sector-state vector.

        Args:
            state: Complex vector with shape ``(dimension,)`` in sector order.
            max_bytes: Best-effort bound for the output allocation.

        Returns:
            A new ``complex128`` vector with the same restricted dimension.

        Raises:
            ValueError: If ``state`` is not one-dimensional or has the wrong
                length.
        """
        plan = self._eager_plan or self._lazy_plan
        return plan.apply(state, max_bytes=max_bytes)

    def apply_into(
        self,
        input_state: np.ndarray[Any, Any],
        output_state: np.ndarray[Any, Any],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        """Apply into strict caller-owned buffers using the retained strategy."""
        _validate_apply_into_buffers(input_state, output_state, self.dimension)
        plan = self._eager_plan or self._lazy_plan
        plan.apply_into(input_state, output_state, max_bytes=max_bytes)

    @property
    def estimated_bytes(self) -> int:
        """Return the current best-effort retained-byte estimate."""
        value = self._lazy_plan.estimated_bytes
        if self._eager_plan is not None:
            value += self._eager_plan.estimated_bytes
        return value

    def mvp_plan(
        self,
        *,
        storage: ChargeStorage = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> Union[ChargeMvpPlan, ChargeLazyMvpPlan]:
        """Return the reusable MVP plan for this restricted operator.

        The returned plan is immutable and its storage estimate is checked
        against ``max_bytes`` before returning. Eager plans retain all
        transitions; lazy plans retain only the sector plan and term metadata.
        """
        _validate_max_bytes(max_bytes)
        storage = _validate_storage(storage)
        if storage == "eager":
            return self._ensure_eager(max_bytes)
        _check_allocation(self._lazy_plan.estimated_bytes, max_bytes, "charge MVP plan")
        return self._lazy_plan

    def _ensure_eager(self, max_bytes: Optional[int]) -> ChargeMvpPlan:
        return self._ensure_eager_for_target(max_bytes, None)

    def _ensure_eager_for_target(
        self, max_bytes: Optional[int], target: Optional[str]
    ) -> ChargeMvpPlan:
        cached = self._eager_plan
        if cached is not None:
            if target is None:
                _check_allocation(cached.estimated_bytes, max_bytes, "charge MVP plan")
            else:
                _check_allocation(
                    _charge_materialization_bytes(
                        self.dimension, cached.transition_count, target
                    ),
                    max_bytes,
                    f"charge {target} materialization",
                )
            return cached
        with self._lock:
            cached = self._eager_plan
            if cached is not None:
                if target is None:
                    _check_allocation(
                        cached.estimated_bytes, max_bytes, "charge MVP plan"
                    )
                else:
                    _check_allocation(
                        _charge_materialization_bytes(
                            self.dimension, cached.transition_count, target
                        ),
                        max_bytes,
                        f"charge {target} materialization",
                    )
                return cached
            _check_allocation(
                self._lazy_plan.estimated_bytes,
                max_bytes,
                "charge lazy MVP plan",
            )
            if max_bytes is None:
                remaining = None
            else:
                remaining = max_bytes - self._lazy_plan.estimated_bytes
            target_floor = (
                0
                if target is None
                else _charge_materialization_bytes(self.dimension, 0, target)
            )
            if remaining is not None:
                _check_allocation(
                    target_floor,
                    remaining,
                    (
                        f"charge {target} materialization preflight"
                        if target is not None
                        else "charge eager MVP plan"
                    ),
                )
                construction_budget = remaining - target_floor
            else:
                construction_budget = None
            native = self._lazy_plan._native_execution.compile_eager(
                _effective_max_bytes(construction_budget)
            )
            target_bytes = (
                0
                if target is None
                else _charge_materialization_bytes(
                    self.dimension, native.transition_count, target
                )
            )
            _check_allocation(
                native.estimated_bytes + target_bytes,
                remaining,
                "charge eager cache and materialization",
            )
            cached = ChargeMvpPlan(
                self.sector.dimension,
                self._lazy_plan.term_count,
                native_plan=native,
            )
            object.__setattr__(self, "_eager_plan", cached)
            return cached

    def dense(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> np.ndarray[Any, Any]:
        """Materialize the restricted operator as a bounded dense matrix.

        The matrix has shape ``(dimension, dimension)`` and follows sector
        ordering. Use :meth:`mvp_plan` or :meth:`apply` when dense materialization
        is not required.
        """
        _validate_max_bytes(max_bytes)
        plan = self._ensure_eager_for_target(max_bytes, "dense")
        values = plan._native_plan.dense(_effective_max_bytes(max_bytes))
        return cast(
            np.ndarray[Any, Any],
            np.asarray(values, dtype=np.complex128).reshape(
                (self.dimension, self.dimension)
            ),
        )

    def coo(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> COOMatrix:
        """Materialize deterministic duplicate-aggregated COO arrays.

        The returned :class:`COOMatrix` uses restricted-sector row and column
        indices and can be converted to SciPy with ``to_scipy()``.
        """
        _validate_max_bytes(max_bytes)
        plan = self._ensure_eager_for_target(max_bytes, "coo")
        rows, columns, coefficients = plan._coo_arrays(max_bytes=max_bytes)
        _check_allocation(
            int(rows.nbytes + columns.nbytes + coefficients.nbytes),
            max_bytes,
            "charge COO matrix",
        )
        return COOMatrix(
            rows,
            columns,
            coefficients,
            (self.dimension, self.dimension),
        )

    def csr(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> CSRMatrix:
        """Materialize deterministic CSR arrays in sector ordering.

        ``indptr``, ``indices`` and ``data`` are returned in the public
        :class:`CSRMatrix` container. The operation allocates only the CSR
        arrays and is guarded by ``max_bytes``.
        """
        _validate_max_bytes(max_bytes)
        plan = self._ensure_eager_for_target(max_bytes, "csr")
        indptr, columns, coefficients = plan._csr_arrays()
        _check_allocation(
            int(indptr.nbytes + columns.nbytes + coefficients.nbytes),
            max_bytes,
            "charge CSR matrix",
        )
        return CSRMatrix(
            indptr,
            columns,
            coefficients,
            (self.dimension, self.dimension),
        )


def _termwise_charge_conserved(
    operator: Union[_StructuredOperator, PauliOperator], sector: ChargeSector
) -> bool:
    """Return whether every serialized term preserves every charge directly."""
    if isinstance(operator, PauliOperator):
        structures, _, _ = operator._arrays()
        return all(
            all(code not in (1, 2) for code in structure) for structure in structures
        )
    for term in operator._materialized_terms():
        if any(code in (1, 2) for code in term.qubit):
            return False
        if term.mapped_fermion is not None and any(
            code in (1, 2) for code in term.mapped_fermion
        ):
            return False
        for charge, _ in sector.constraints:
            if _structured_charge_delta(term, charge) != 0:
                return False
    return True


def _fast_fermion_particles(
    operator: Union[_StructuredOperator, PauliOperator], sector: ChargeSector
) -> Optional[int]:
    """Detect the fixed-spin-sector layout used by the Hubbard MVP fast path."""
    if (
        isinstance(operator, PauliOperator)
        or getattr(operator, "_domain", None) != "fermion"
    ):
        return None
    space = sector.space
    if space.bosons or space.qubits or space.qudits or space.fermions == 0:
        return None
    if space.fermions % 2 or sector.local_dimensions != (2,) * space.fermions:
        return None
    sites = space.fermions // 2
    total_weights = (1,) * space.fermions
    balance_weights = (1,) * sites + (-1,) * sites
    total_target: Optional[int] = None
    balance_target: Optional[int] = None
    for charge, requested in sector.constraints:
        target = requested - charge.offset
        if charge.fermion_weights == total_weights:
            if total_target is not None:
                return None
            total_target = target
        elif charge.fermion_weights == balance_weights:
            if balance_target is not None:
                return None
            balance_target = target
        else:
            return None
    if total_target is None or balance_target != 0 or total_target % 2:
        return None
    particles = total_target // 2
    if particles < 1 or particles > sites:
        return None
    for term in operator._materialized_terms():
        if (
            term.boson is not None
            or term.qubit
            or term.qudit is not None
            or term.mapped_fermion is not None
        ):
            return None
    return particles


def _restricted_transition_inputs(
    operator: Union[_StructuredOperator, PauliOperator], sector: ChargeSector
) -> _RestrictedTransitionInputs:
    axis_positions = {
        (axis.domain, axis.index): position
        for position, axis in enumerate(sector.space._axes)
    }
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
        for structured_term in operator._materialized_terms():
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
    coefficient_array = np.ascontiguousarray(coefficients, dtype=np.complex128)
    termwise_conserved = _termwise_charge_conserved(operator, sector)
    return _RestrictedTransitionInputs(
        tuple(sector.local_dimensions),
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
        termwise_conserved,
        _fast_fermion_particles(operator, sector) if termwise_conserved else None,
    )


def _add_scaled(
    aggregate: Dict[Tuple[object, ...], complex],
    key: Tuple[object, ...],
    coefficient: complex,
    real_scale: int,
    imaginary_scale: int,
) -> None:
    """Accumulate a complex128 coefficient times an integer selection scale."""
    if real_scale == 0 and imaginary_scale == 0:
        return
    scaled = complex(coefficient) * complex(real_scale, imaginary_scale)
    aggregate[key] = aggregate.get(key, 0j) + scaled


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
        if any(
            term.mapped_fermion is not None for term in operator._materialized_terms()
        ):
            raise ValueError(
                "charge analysis is defined before fermion-to-qubit mapping; "
                "analyze the raw structured operator"
            )
        term_count = operator.term_count
        qubit_count = operator.space.qubits
        terms = (
            (term.qubit, term.coefficient, term)
            for term in operator._materialized_terms()
        )
    else:
        raise TypeError("operator must be a structured or Pauli operator")

    estimated = term_count * max(1, qubit_count + 1) * 128
    _check_allocation(estimated, max_bytes, "exact additive-charge analysis")
    aggregate: Dict[Tuple[object, ...], complex] = {}
    for codes, coefficient, structured_term in terms:
        if structured_term is None:
            base_key: Tuple[object, ...] = tuple(codes)
            raw_delta = 0
        else:
            base_key = structured_term.key()
            raw_delta = _structured_charge_delta(structured_term, charge)
        if raw_delta:
            _add_scaled(aggregate, base_key, coefficient, -raw_delta, 0)
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
            _add_scaled(
                aggregate,
                key,
                coefficient,
                0,
                -difference if code == 1 else difference,
            )
    nonzero = sum(
        1 for value in aggregate.values() if value.real != 0.0 or value.imag != 0.0
    )
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
        method="native_float_selection_rules",
    )


def restrict_charge(
    operator: Union[_StructuredOperator, PauliOperator],
    sector: ChargeSector,
    *,
    storage: ChargeStorage = "lazy",
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> ChargeRestrictedOperator:
    """Validate exact conservation and build a restricted MVP plan.

    ``storage="lazy"`` is the compact default. ``storage="eager"`` is an
    explicit prewarm request; dense and sparse materialization also populate
    the eager transition cache on demand.
    """
    if not isinstance(sector, ChargeSector):
        raise TypeError("sector must be a ChargeSector")
    return ChargeRestrictedOperator(
        operator, sector, storage=storage, max_bytes=max_bytes
    )
