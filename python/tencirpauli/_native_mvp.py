"""Shared immutable Python facade for Rust-native MVP plans.

This module owns the public-call contract shared by Pauli, Charge, and U(1)
plans.  Domain facades keep their own construction and materialization data,
while numerical application remains one coarse native call.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Tuple, cast

import numpy as np


# A practical default for explicit statevector targets.  It lives here so the
# shared facade and its validation helpers do not need to import hamiltonian.
DEFAULT_MAX_BYTES = 16 * 1024 * 1024 * 1024

# ``max_bytes`` is deliberately a coarse, best-effort guard rather than an
# allocation contract.  Sub-gigabyte values have repeatedly caused useful
# native caches and ordinary workspaces to be rejected on machines where they
# are immaterial, while still being far too imprecise to describe peak RSS.
# Treat them as advisory.  At and above this floor we retain cheap preflights
# for major outputs and workspaces; allocator failure remains the final guard
# for requests that cannot be estimated cheaply.
MIN_ENFORCED_MAX_BYTES = 1024 * 1024 * 1024


def _validate_max_bytes(max_bytes: Optional[int]) -> None:
    if max_bytes is not None and (
        not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0
    ):
        raise ValueError("max_bytes must be a non-negative integer or None")


def _effective_max_bytes(max_bytes: Optional[int]) -> int:
    """Return the native allocation limit after the public advisory floor."""
    enforced_limit = _enforced_max_bytes(max_bytes)
    return sys.maxsize if enforced_limit is None else enforced_limit


def _enforced_max_bytes(max_bytes: Optional[int]) -> Optional[int]:
    """Return a materially enforceable public memory limit, if any."""
    _validate_max_bytes(max_bytes)
    if max_bytes is None or max_bytes < MIN_ENFORCED_MAX_BYTES:
        return None
    return max_bytes


def _check_allocation(requested: int, limit: Optional[int], context: str) -> None:
    enforced_limit = _enforced_max_bytes(limit)
    if enforced_limit is not None and requested > enforced_limit:
        raise MemoryError(
            f"{context} requires approximately {requested} bytes, "
            f"exceeding max_bytes={enforced_limit}"
        )


def _validate_apply_into_buffers(
    input_state: object, output_state: object, dimension: int
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Validate the strict zero-copy MVP buffer protocol."""
    if not isinstance(input_state, np.ndarray) or not isinstance(
        output_state, np.ndarray
    ):
        raise TypeError("apply_into input and output must be NumPy arrays")
    expected: np.dtype[Any] = np.dtype(np.complex128)
    if input_state.dtype != expected or output_state.dtype != expected:
        raise TypeError("apply_into input and output must have dtype complex128")
    for name, value in (("input", input_state), ("output", output_state)):
        if value.ndim != 1 or value.shape != (dimension,):
            raise ValueError(f"apply_into {name} must have shape ({dimension},)")
        if not value.flags.c_contiguous:
            raise ValueError(f"apply_into {name} must be C-contiguous")
    if not output_state.flags.writeable:
        raise ValueError("apply_into output must be writable")
    if np.shares_memory(input_state, output_state):
        raise ValueError("apply_into input and output must not overlap")
    return input_state, output_state


@dataclass(frozen=True)
class NativePlanMetadata:
    """The validated metadata shared by all Rust-native MVP plan facades."""

    __slots__ = (
        "basis_ordering",
        "dimension",
        "estimated_construction_peak_bytes",
        "estimated_retained_bytes",
        "execution_family",
        "native_variant",
        "retained_format",
        "schedule_count",
        "source_term_count",
        "storage",
        "target",
        "term_count",
        "transition_count",
    )

    dimension: int
    term_count: int
    storage: Literal["lazy", "eager"]
    basis_ordering: str
    target: Literal["native_mvp"]
    execution_family: str
    native_variant: str
    retained_format: str
    source_term_count: int
    schedule_count: int
    estimated_retained_bytes: int
    estimated_construction_peak_bytes: int
    transition_count: Optional[int]

    def __post_init__(self) -> None:
        for name in (
            "dimension",
            "term_count",
            "source_term_count",
            "schedule_count",
            "estimated_retained_bytes",
            "estimated_construction_peak_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.storage not in {"lazy", "eager"}:
            raise ValueError("storage must be either 'eager' or 'lazy'")
        if self.target != "native_mvp":
            raise ValueError("native MVP plans must have target='native_mvp'")
        for name in (
            "basis_ordering",
            "execution_family",
            "native_variant",
            "retained_format",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.transition_count is not None and (
            not isinstance(self.transition_count, int)
            or isinstance(self.transition_count, bool)
            or self.transition_count < 0
        ):
            raise ValueError("transition_count must be a non-negative integer or None")


def _build_native_plan_metadata(
    *,
    dimension: int,
    term_count: int,
    storage: Literal["lazy", "eager"],
    basis_ordering: str,
    execution_family: str,
    native_variant: Optional[str],
    retained_format: Optional[str],
    source_term_count: Optional[int],
    schedule_count: Optional[int],
    estimated_retained_bytes: int,
    estimated_construction_peak_bytes: Optional[int],
    transition_count: Optional[int],
) -> NativePlanMetadata:
    """Normalize metadata fields shared by every native MVP family.

    Callers provide family-specific semantics explicitly. Only the common
    count and storage fallbacks live here, so a new family cannot accidentally
    inherit another family's basis or execution labels.
    """
    resolved_retained = int(estimated_retained_bytes)
    return NativePlanMetadata(
        dimension=int(dimension),
        term_count=int(term_count),
        storage=storage,
        basis_ordering=basis_ordering,
        target="native_mvp",
        execution_family=execution_family,
        native_variant=execution_family if native_variant is None else native_variant,
        retained_format=(
            ("compiled_actions" if storage == "lazy" else "csr")
            if retained_format is None
            else retained_format
        ),
        source_term_count=(
            int(term_count) if source_term_count is None else int(source_term_count)
        ),
        schedule_count=(
            int(term_count) if schedule_count is None else int(schedule_count)
        ),
        estimated_retained_bytes=resolved_retained,
        estimated_construction_peak_bytes=(
            resolved_retained
            if estimated_construction_peak_bytes is None
            else int(estimated_construction_peak_bytes)
        ),
        transition_count=(None if transition_count is None else int(transition_count)),
    )


class _NativeMvpFacade:
    """Mixin implementing the common native-plan public-call contract."""

    __slots__ = ()

    if TYPE_CHECKING:
        _execution: Any
        _metadata: NativePlanMetadata

    @property
    def _native_metadata(self) -> NativePlanMetadata:
        return self._metadata

    @property
    def dimension(self) -> int:
        return self._native_metadata.dimension

    @property
    def term_count(self) -> int:
        return self._native_metadata.term_count

    @property
    def storage(self) -> Literal["lazy", "eager"]:
        return self._native_metadata.storage

    @property
    def basis_ordering(self) -> str:
        return self._native_metadata.basis_ordering

    @property
    def target(self) -> Literal["native_mvp"]:
        return self._native_metadata.target

    @property
    def execution_family(self) -> str:
        return self._native_metadata.execution_family

    @property
    def native_variant(self) -> str:
        return self._native_metadata.native_variant

    @property
    def retained_format(self) -> str:
        return self._native_metadata.retained_format

    @property
    def source_term_count(self) -> int:
        return self._native_metadata.source_term_count

    @property
    def schedule_count(self) -> int:
        return self._native_metadata.schedule_count

    @property
    def estimated_retained_bytes(self) -> int:
        return self._native_metadata.estimated_retained_bytes

    @property
    def estimated_construction_peak_bytes(self) -> int:
        return self._native_metadata.estimated_construction_peak_bytes

    @property
    def transition_count(self) -> Optional[int]:
        return self._native_metadata.transition_count

    @property
    def estimated_bytes(self) -> int:
        """Compatibility spelling for the retained native-plan storage."""
        return self.estimated_retained_bytes

    def apply(
        self,
        state: Sequence[complex],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the plan through exactly one coarse native call."""
        return self._apply(state, max_bytes=max_bytes, adjoint=False)

    def apply_adjoint(
        self,
        state: Sequence[complex],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the conjugate transpose in this plan's unchanged basis.

        Reuse the retained native plan without materializing a matrix or
        constructing a second plan. ``max_bytes`` follows :meth:`apply`.
        """
        return self._apply(state, max_bytes=max_bytes, adjoint=True)

    def _apply(
        self,
        state: Sequence[complex],
        *,
        max_bytes: Optional[int],
        adjoint: bool,
    ) -> np.ndarray[Any, Any]:
        _validate_max_bytes(max_bytes)
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.shape[0] != self.dimension:
            raise ValueError(
                f"state must have shape ({self.dimension},), got {values.shape}"
            )
        _check_allocation(self.dimension * 16, max_bytes, "requested native MVP output")
        apply = self._execution.apply_adjoint if adjoint else self._execution.apply
        return cast(
            np.ndarray[Any, Any],
            np.asarray(
                apply(np.ascontiguousarray(values), _effective_max_bytes(max_bytes)),
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
        """Apply into strict caller-owned buffers without a Python state copy."""
        _validate_apply_into_buffers(input_state, output_state, self.dimension)
        self._execution.apply_into(
            input_state, output_state, _effective_max_bytes(max_bytes)
        )

    def __call__(self, state: Sequence[complex]) -> np.ndarray[Any, Any]:
        return self.apply(state)

    def to_scipy_linear_operator(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> Any:
        """Expose this native plan as a SciPy ``LinearOperator``."""
        from .integrations.scipy import to_scipy_linear_operator

        return to_scipy_linear_operator(self, max_bytes=max_bytes)


def _read_csr_arrays(
    execution: Any, max_bytes: Optional[int]
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Read one eager native plan's flat, read-only CSR arrays.

    The caller owns construction of the public ``CSRMatrix`` container.  Keeping
    this helper at the flat-array ABI prevents an eager charge plan from
    constructing a temporary sparse container merely to unpack it again.
    """
    indptr, columns, values = execution.csr(_effective_max_bytes(max_bytes))
    indptr_array = np.asarray(indptr, dtype=np.uint64)
    columns_array = np.asarray(columns, dtype=np.uint64)
    values_array = np.asarray(values, dtype=np.complex128)
    _check_allocation(
        int(indptr_array.nbytes + columns_array.nbytes + values_array.nbytes),
        max_bytes,
        "native eager CSR materialization",
    )
    for array in (indptr_array, columns_array, values_array):
        array.setflags(write=False)
    return indptr_array, columns_array, values_array


def _read_coo_arrays(
    execution: Any, max_bytes: Optional[int]
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Read one eager native plan's flat, read-only COO arrays.

    The caller owns construction of the public ``COOMatrix`` container for the
    same reason as :func:`_read_csr_arrays`.
    """
    rows, columns, values = execution.coo(_effective_max_bytes(max_bytes))
    rows_array = np.asarray(rows, dtype=np.uint64)
    columns_array = np.asarray(columns, dtype=np.uint64)
    values_array = np.asarray(values, dtype=np.complex128)
    _check_allocation(
        int(rows_array.nbytes + columns_array.nbytes + values_array.nbytes),
        max_bytes,
        "native eager COO materialization",
    )
    for array in (rows_array, columns_array, values_array):
        array.setflags(write=False)
    return rows_array, columns_array, values_array
