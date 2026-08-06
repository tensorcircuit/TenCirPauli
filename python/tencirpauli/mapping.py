"""Deterministic fermion-to-qubit occupation mappings.

The public plans in this module keep the occupation convention explicit.  A
Jordan-Wigner Pauli operator is transformed by the binary linear change of
computational-basis coordinates for parity and Fenwick Bravyi-Kitaev plans;
the same path is therefore easy to compare with an independently constructed
encoded Fock-space matrix.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from . import _native
from ._validation import validate_nonnegative_int
from .hamiltonian import (
    DEFAULT_MAX_BYTES,
    _check_allocation,
    _effective_max_bytes,
    _validate_max_bytes,
)
from .pauli import PauliOperator
from .structured import HybridOperator, _hybrid_from_native


_CONVENTION = "tencirpauli.gf2_occupation.v1"
_BASIS_ORDERING = "qubit0_msb_matrix"
_SCHEMA_VERSION = 1
_MAPPING_FACTORY_TOKEN = object()


_exact_nonnegative = validate_nonnegative_int


def _mapping_plan_upper_bound(n_modes: int) -> int:
    """Return a checked-before-allocation upper bound for one mapping plan."""
    max_cnot_count = n_modes * (n_modes - 1) // 2
    native_bytes = _mapping_plan_native_bytes(n_modes, max_cnot_count)
    # Public matrix and provenance exports are materialized only when their
    # properties are requested; this remains a conservative plan bound.
    public_bytes = 2 * n_modes * n_modes + 2 * 16 * max_cnot_count
    return native_bytes + public_bytes


def _mapping_plan_native_bytes(n_modes: int, cnot_count: int) -> int:
    """Return the cheap logical payload estimate for the retained native plan."""
    packed_words = (n_modes + 63) // 64
    packed_bytes = 2 * n_modes * packed_words * 8
    return 2 * n_modes * n_modes + packed_bytes + 16 * cnot_count + 256


def _validate_matrix(
    rows: Sequence[Sequence[int]], n_modes: int
) -> Tuple[Tuple[int, ...], ...]:
    normalized = tuple(tuple(int(value) for value in row) for row in rows)
    if len(normalized) != n_modes or any(len(row) != n_modes for row in normalized):
        raise ValueError("encoding matrix must be square with shape (n_modes, n_modes)")
    if any(value not in (0, 1) for row in normalized for value in row):
        raise ValueError("encoding matrix entries must be binary")
    return normalized


class FermionQubitMapping:
    """Immutable occupation-encoding plan for JW, parity, or BK mapping.

    The plan maps ``n_modes`` fermion occupations to the same number of qubits
    over GF(2), and stores the inverse transform plus a deterministic CNOT
    synthesis. Public mapped operators use mode-zero-increasing ordering and
    the ``qubit0_msb_matrix`` basis convention.
    """

    __slots__ = (
        "_clifford_operations",
        "_cnot_operations",
        "_encoding",
        "_inverse_encoding",
        "_locked",
        "_name",
        "_native_plan",
        "basis_ordering",
        "convention",
        "estimated_bytes",
        "mode_ordering",
        "n_modes",
        "nqubits",
        "schema_version",
    )
    schema_version: int
    _name: str
    n_modes: int
    nqubits: int
    mode_ordering: str
    basis_ordering: str
    convention: str
    _encoding: Optional[np.ndarray[Any, Any]]
    _inverse_encoding: Optional[np.ndarray[Any, Any]]
    _cnot_operations: Optional[Tuple[Tuple[int, int], ...]]
    _clifford_operations: Optional[np.ndarray[Any, Any]]
    estimated_bytes: int
    _native_plan: Any
    _locked: bool

    def __init__(
        self,
        mapping_name: str,
        n_modes: int,
        encoding: Optional[Tuple[Tuple[int, ...], ...]] = None,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
        _native_plan: Optional[Any] = None,
        _factory_token: object = None,
    ) -> None:
        if _factory_token is not _MAPPING_FACTORY_TOKEN:
            raise TypeError(
                "FermionQubitMapping instances must be created by a named factory"
            )
        n_modes = _exact_nonnegative(n_modes, "n_modes")
        if mapping_name not in {"jordan_wigner", "parity", "bravyi_kitaev"}:
            raise ValueError("unsupported fermion-to-qubit mapping")
        _validate_max_bytes(max_bytes)
        _check_allocation(
            _mapping_plan_upper_bound(n_modes),
            max_bytes,
            "fermion mapping plan",
        )
        if _native_plan is None:
            raise RuntimeError("mapping plans require a native handle")
        if encoding is not None:
            _validate_matrix(encoding, n_modes)
        estimated_bytes = max(
            int(_native_plan.estimated_bytes), _mapping_plan_upper_bound(n_modes)
        )
        object.__setattr__(self, "schema_version", _SCHEMA_VERSION)
        object.__setattr__(self, "_name", mapping_name)
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "nqubits", n_modes)
        object.__setattr__(self, "mode_ordering", "mode0_increasing")
        object.__setattr__(self, "basis_ordering", _BASIS_ORDERING)
        object.__setattr__(self, "convention", _CONVENTION)
        object.__setattr__(self, "_encoding", None)
        object.__setattr__(self, "_inverse_encoding", None)
        object.__setattr__(self, "_cnot_operations", None)
        object.__setattr__(self, "_clifford_operations", None)
        object.__setattr__(self, "estimated_bytes", estimated_bytes)
        object.__setattr__(self, "_native_plan", _native_plan)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("FermionQubitMapping is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def _build(
        cls,
        mapping_name: str,
        n_modes: int,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionQubitMapping":
        normalized_modes = _exact_nonnegative(n_modes, "n_modes")
        if mapping_name not in {"jordan_wigner", "parity", "bravyi_kitaev"}:
            raise ValueError("unsupported fermion-to-qubit mapping")
        _validate_max_bytes(max_bytes)
        _check_allocation(
            _mapping_plan_upper_bound(normalized_modes),
            max_bytes,
            "fermion mapping plan",
        )
        native_plan = _native.mapping_plan(
            mapping_name,
            normalized_modes,
            _effective_max_bytes(max_bytes),
        )
        return cls(
            mapping_name,
            normalized_modes,
            max_bytes=max_bytes,
            _native_plan=native_plan,
            _factory_token=_MAPPING_FACTORY_TOKEN,
        )

    @classmethod
    def jordan_wigner(
        cls,
        n_modes: int,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionQubitMapping":
        """Build the identity occupation encoding used by Jordan-Wigner.

        ``n_modes`` determines both the fermion-mode and qubit counts.
        ``max_bytes`` guards the mapping matrices and native plan workspace.
        """
        return cls._build("jordan_wigner", n_modes, max_bytes=max_bytes)

    @classmethod
    def parity(
        cls,
        n_modes: int,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionQubitMapping":
        """Build the prefix-parity occupation encoding."""
        return cls._build("parity", n_modes, max_bytes=max_bytes)

    @classmethod
    def bravyi_kitaev(
        cls,
        n_modes: int,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionQubitMapping":
        """Build the deterministic Fenwick-interval Bravyi-Kitaev encoding."""
        return cls._build("bravyi_kitaev", n_modes, max_bytes=max_bytes)

    @classmethod
    def from_name(
        cls,
        name: str,
        n_modes: int,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionQubitMapping":
        """Build a mapping plan from ``jordan_wigner``, ``parity``, or ``bravyi_kitaev``."""
        if not isinstance(name, str):
            raise TypeError("mapping name must be a string")
        return cls._build(name, n_modes, max_bytes=max_bytes)

    @property
    def name(self) -> str:
        """Return the stable mapping name."""
        return self._name

    @property
    def encoding_matrix(self) -> np.ndarray[Any, Any]:
        """Return the read-only binary matrix for ``q = B n (mod 2)``."""
        cached = self._encoding
        if cached is None:
            cached = np.asarray(
                self._native_plan.encoding_flat(), dtype=np.uint8
            ).reshape((self.n_modes, self.n_modes))
            cached.setflags(write=False)
            object.__setattr__(self, "_encoding", cached)
        return cached

    @property
    def inverse_encoding_matrix(self) -> np.ndarray[Any, Any]:
        """Return the read-only inverse transform from qubits to occupations."""
        cached = self._inverse_encoding
        if cached is None:
            cached = np.asarray(
                self._native_plan.inverse_encoding_flat(), dtype=np.uint8
            ).reshape((self.n_modes, self.n_modes))
            cached.setflags(write=False)
            object.__setattr__(self, "_inverse_encoding", cached)
        return cached

    @property
    def cnot_operations(self) -> Tuple[Tuple[int, int], ...]:
        """Return deterministic ``(control, target)`` CNOT provenance."""
        cached = self._cnot_operations
        if cached is None:
            cached = tuple(
                (int(control), int(target))
                for control, target in self._native_plan.cnot_operations
            )
            object.__setattr__(self, "_cnot_operations", cached)
        return cached

    @property
    def clifford_operations(self) -> np.ndarray[Any, Any]:
        """Return the read-only CNOT array in synthesis order."""
        cached = self._clifford_operations
        if cached is None:
            cached = np.asarray(self.cnot_operations, dtype=np.int64).reshape((-1, 2))
            cached = np.ascontiguousarray(cached)
            cached.setflags(write=False)
            object.__setattr__(self, "_clifford_operations", cached)
        return cached

    def encode_occupation(self, occupation: Sequence[int]) -> Tuple[int, ...]:
        """Encode one binary occupation vector with the frozen GF(2) convention.

        Args:
            occupation: Length-``n_modes`` sequence containing only ``0`` and
                ``1``.

        Returns:
            The encoded length-``nqubits`` binary vector.
        """
        values = tuple(occupation)
        if len(values) != self.n_modes or any(value not in (0, 1) for value in values):
            raise ValueError("occupation must be a binary vector of length n_modes")
        return tuple(
            int(value) for value in self._native_plan.encode_occupation(values)
        )

    def map_pauli(
        self,
        operator: PauliOperator,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> PauliOperator:
        """Conjugate a pure fermion-axis Pauli operator by the mapping CNOTs.

        The returned operator has ``n_modes`` qubits, exact conjugation signs,
        deterministic term ordering, and aggregated duplicate words.
        """
        if not isinstance(operator, PauliOperator):
            raise TypeError("map_pauli expects a PauliOperator")
        if operator.nqubits != self.n_modes:
            raise ValueError("mapping plan and Pauli qubit count are incompatible")
        _validate_max_bytes(max_bytes)
        _check_allocation(
            max(1, operator.term_count) * (self.n_modes + 4) * 16,
            max_bytes,
            "mapped Pauli operator",
        )
        if operator._native_handle is None:
            raise RuntimeError("PauliOperator must retain a native handle")
        result = self._native_plan.transform_pauli_handle(
            operator._native_handle,
            _effective_max_bytes(max_bytes),
        )
        return PauliOperator._from_native_handle(result)

    def map_fermion_operator(
        self, operator: Any, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> PauliOperator:
        """Map a pure ``FermionOperator`` to a canonical Pauli operator.

        Fermion ladder products are expanded through one batched Jordan-Wigner
        path and then conjugated by this mapping's encoding.

        Examples:
            >>> import tencirpauli as tcp
            >>> number = tcp.FermionOperator.from_terms(
            ...     1, [(((0, "create"), (0, "annihilate")), 1.0)]
            ... )
            >>> mapped = tcp.FermionQubitMapping.jordan_wigner(1).map_fermion_operator(number)
            >>> mapped.nqubits
            1
        """
        from .structured import FermionOperator

        if not isinstance(operator, FermionOperator):
            raise TypeError("mapping expects a FermionOperator")
        if operator.space.fermions != self.n_modes:
            raise ValueError("mapping plan and fermion mode counts are incompatible")
        jordan_wigner = operator.map_fermions("jordan_wigner", max_bytes=max_bytes)
        return self.map_pauli(jordan_wigner, max_bytes=max_bytes)

    def map_majorana_operator(
        self, operator: Any, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> PauliOperator:
        """Map a Majorana operator without materializing a fermion expansion.

        This is the preferred path for Majorana input because the native batch
        kernel aggregates mapped words directly and preserves exact signs.
        """
        from .majorana import MajoranaOperator

        if not isinstance(operator, MajoranaOperator):
            raise TypeError("mapping expects a MajoranaOperator")
        if operator.n_modes != self.n_modes:
            raise ValueError("mapping plan and Majorana mode counts are incompatible")
        _validate_max_bytes(max_bytes)
        if not isinstance(
            operator._native_handle, _native.NativeMajoranaOperatorHandle
        ):
            raise RuntimeError("MajoranaOperator must retain a native handle")
        result = self._native_plan.transform_majorana_handle(
            operator._native_handle,
            _effective_max_bytes(max_bytes),
        )
        return PauliOperator._from_native_handle(result)

    def map_hybrid(
        self, operator: HybridOperator, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> Union[HybridOperator, PauliOperator]:
        """Map only the fermion axes of a compatible hybrid operator.

        Boson, qubit, and qudit axes remain in their original operator-space
        ordering. A pure-qubit result is returned as :class:`PauliOperator`;
        mixed results remain :class:`HybridOperator`.
        """
        if not isinstance(operator, HybridOperator):
            raise TypeError("mapping expects a HybridOperator")
        if operator.space.fermions != self.n_modes:
            raise ValueError("mapping plan and fermion axis counts are incompatible")
        jordan_wigner = operator.map_fermions("jordan_wigner", max_bytes=max_bytes)
        if isinstance(jordan_wigner, PauliOperator):
            handle = jordan_wigner._native_handle
            if handle is None:
                raise RuntimeError("mapped PauliOperator must retain a native handle")
            result = self._native_plan.transform_pauli_handle_prefix(
                handle,
                self.n_modes,
                _effective_max_bytes(max_bytes),
            )
            return PauliOperator._from_native_handle(result)
        if not isinstance(
            jordan_wigner._native_handle, _native.NativeHybridOperatorHandle
        ):
            raise RuntimeError("mapped HybridOperator must retain a native handle")
        result = self._native_plan.transform_hybrid_handle(
            jordan_wigner._native_handle,
            _effective_max_bytes(max_bytes),
        )
        return _hybrid_from_native(jordan_wigner.space, result)

    def __repr__(self) -> str:
        return f"FermionQubitMapping({self.name!r}, n_modes={self.n_modes})"
