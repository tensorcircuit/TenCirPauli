"""Typed public API for Rust-native Pauli propagation."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, cast

import numpy as np

from . import _native
from .hamiltonian import DEFAULT_MAX_BYTES, _validate_max_bytes
from .pauli import PauliOperator


@dataclass(frozen=True)
class ZeroState:
    """The computational product state ``|0...0>``."""


@dataclass(frozen=True, init=False)
class ComputationalBasisState:
    """A computational-basis product state in qubit order."""

    bits: Tuple[int, ...]

    def __init__(self, bits: Iterable[int]) -> None:
        normalized = tuple(bits)
        if any(
            not isinstance(bit, int) or isinstance(bit, bool) or bit not in (0, 1)
            for bit in normalized
        ):
            raise ValueError("computational basis bits must be integers 0 or 1")
        object.__setattr__(self, "bits", normalized)


@dataclass(frozen=True, init=False)
class ProductBlochState:
    """Pure or mixed tensor-product single-qubit Bloch vectors."""

    bloch: np.ndarray[Any, Any]

    def __init__(self, bloch: Sequence[Sequence[float]] | np.ndarray[Any, Any]) -> None:
        array = np.asarray(bloch)
        if array.dtype.kind != "f" or array.dtype.itemsize != 8:
            raise TypeError("bloch must be a real float64 array")
        snapshot = np.array(array, dtype=np.float64, order="C", copy=True)
        if snapshot.ndim != 2 or snapshot.shape[1] != 3:
            raise ValueError("bloch must have shape (nqubits, 3)")
        if not np.isfinite(snapshot).all():
            raise ValueError("bloch entries must be finite")
        snapshot.flags.writeable = False
        object.__setattr__(self, "bloch", snapshot)


class GateTape:
    """Mutable typed builder for a Schrödinger-order gate tape."""

    def __init__(self, nqubits: int) -> None:
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("nqubits must be a non-negative integer")
        self.nqubits = nqubits
        self._operations: list[tuple[int, int, int, int, float, tuple[float, ...]]] = []

    def __len__(self) -> int:
        return len(self._operations)

    @property
    def nparameters(self) -> int:
        slots = [operation[3] for operation in self._operations if operation[3] >= 0]
        return max(slots, default=-1) + 1

    def _wire(self, wire: int) -> int:
        if not isinstance(wire, int) or isinstance(wire, bool):
            raise TypeError("wire must be an integer")
        if wire < 0 or wire >= self.nqubits:
            raise ValueError(f"wire {wire} is outside 0..{self.nqubits}")
        return wire

    def _two_wires(self, wire0: int, wire1: int) -> tuple[int, int]:
        first, second = self._wire(wire0), self._wire(wire1)
        if first == second:
            raise ValueError("two-qubit gate wires must differ")
        return first, second

    def _append_clifford(
        self, kind: int, wire0: int, wire1: Optional[int] = None
    ) -> None:
        if wire1 is None:
            first = self._wire(wire0)
            second = _WIRE_SENTINEL
        else:
            first, second = self._two_wires(wire0, wire1)
        self._operations.append((kind, first, second, -1, 0.0, ()))

    def x(self, wire: int) -> None:
        """Append an X Clifford gate."""
        self._append_clifford(0, wire)

    def y(self, wire: int) -> None:
        """Append a Y Clifford gate."""
        self._append_clifford(1, wire)

    def z(self, wire: int) -> None:
        """Append a Z Clifford gate."""
        self._append_clifford(2, wire)

    def h(self, wire: int) -> None:
        """Append a Hadamard gate."""
        self._append_clifford(3, wire)

    def s(self, wire: int) -> None:
        """Append an S gate."""
        self._append_clifford(4, wire)

    def sdg(self, wire: int) -> None:
        """Append an inverse-S gate."""
        self._append_clifford(5, wire)

    def cnot(self, control: int, target: int) -> None:
        """Append a directed CNOT gate."""
        self._append_clifford(6, control, target)

    def cz(self, wire0: int, wire1: int) -> None:
        """Append a controlled-Z gate."""
        self._append_clifford(7, wire0, wire1)

    def swap(self, wire0: int, wire1: int) -> None:
        """Append a SWAP gate."""
        self._append_clifford(8, wire0, wire1)

    def _slot(self, parameter: int) -> int:
        if (
            not isinstance(parameter, int)
            or isinstance(parameter, bool)
            or parameter < 0
        ):
            raise ValueError("parameter must be a non-negative integer slot")
        return parameter

    def _rotation(
        self,
        kind: int,
        wire0: int,
        wire1: Optional[int],
        *,
        angle: Optional[float],
        parameter: Optional[int],
    ) -> None:
        if (angle is None) == (parameter is None):
            raise ValueError("provide exactly one of angle or parameter")
        if wire1 is None:
            first = self._wire(wire0)
            second = _WIRE_SENTINEL
        else:
            first, second = self._two_wires(wire0, wire1)
        if parameter is not None:
            slot = self._slot(parameter)
            static_angle = 0.0
        else:
            assert angle is not None
            try:
                static_angle = float(angle)
            except (TypeError, ValueError) as error:
                raise TypeError("angle must be a real finite float") from error
            if not math.isfinite(static_angle):
                raise ValueError("angle must be finite")
            slot = -1
        self._operations.append((kind, first, second, slot, static_angle, ()))

    def rx(
        self,
        wire: int,
        *,
        angle: Optional[float] = None,
        parameter: Optional[int] = None,
    ) -> None:
        """Append ``exp(-i angle X / 2)`` or a parameterized RX gate."""
        self._rotation(9, wire, None, angle=angle, parameter=parameter)

    def ry(
        self,
        wire: int,
        *,
        angle: Optional[float] = None,
        parameter: Optional[int] = None,
    ) -> None:
        """Append a parameterized or static RY gate."""
        self._rotation(10, wire, None, angle=angle, parameter=parameter)

    def rz(
        self,
        wire: int,
        *,
        angle: Optional[float] = None,
        parameter: Optional[int] = None,
    ) -> None:
        """Append a parameterized or static RZ gate."""
        self._rotation(11, wire, None, angle=angle, parameter=parameter)

    def rxx(
        self,
        wire0: int,
        wire1: int,
        *,
        angle: Optional[float] = None,
        parameter: Optional[int] = None,
    ) -> None:
        """Append a two-qubit X-X rotation."""
        self._rotation(12, wire0, wire1, angle=angle, parameter=parameter)

    def ryy(
        self,
        wire0: int,
        wire1: int,
        *,
        angle: Optional[float] = None,
        parameter: Optional[int] = None,
    ) -> None:
        """Append a two-qubit Y-Y rotation."""
        self._rotation(13, wire0, wire1, angle=angle, parameter=parameter)

    def rzz(
        self,
        wire0: int,
        wire1: int,
        *,
        angle: Optional[float] = None,
        parameter: Optional[int] = None,
    ) -> None:
        """Append a two-qubit Z-Z rotation."""
        self._rotation(14, wire0, wire1, angle=angle, parameter=parameter)

    def ptm(
        self,
        wires: Sequence[int],
        matrix: np.ndarray[Any, Any],
        *,
        name: Optional[str] = None,
    ) -> None:
        """Append a real one- or two-qubit Pauli-transfer matrix."""
        del name  # Names are diagnostic metadata and do not affect semantics.
        normalized_wires = tuple(wires)
        if len(normalized_wires) not in (1, 2):
            raise ValueError("PTM wires must contain one or two distinct wires")
        first = self._wire(normalized_wires[0])
        second = (
            None
            if len(normalized_wires) == 1
            else self._two_wires(first, normalized_wires[1])[1]
        )
        array = np.asarray(matrix)
        if array.dtype.kind != "f" or array.dtype.itemsize != 8:
            raise TypeError("PTM matrix must be a real float64 array")
        dimension = 4 if second is None else 16
        if array.ndim != 2 or array.shape != (dimension, dimension):
            raise ValueError(f"PTM matrix must have shape ({dimension}, {dimension})")
        snapshot = np.ascontiguousarray(array, dtype=np.float64).copy()
        if not np.isfinite(snapshot).all():
            raise ValueError("PTM entries must be finite")
        self._operations.append(
            (
                15,
                first,
                _WIRE_SENTINEL if second is None else second,
                -1,
                0.0,
                tuple(float(value) for value in snapshot.ravel()),
            )
        )

    def _native_operations(
        self,
    ) -> tuple[tuple[int, int, int, int, float, tuple[float, ...]], ...]:
        return tuple(self._operations)


_WIRE_SENTINEL = 2 * sys.maxsize + 1
_DEFAULT_ZERO_STATE = ZeroState()


@dataclass(frozen=True)
class PropagationProfile:
    """Structural and timing metadata from one explicit profile call."""

    gate_count: int
    initial_terms: int
    final_terms: int
    peak_terms: int
    estimated_peak_bytes: int
    final_weight_counts: Tuple[int, ...]
    kernel_seconds: float


@dataclass(frozen=True)
class ProfiledExpectation:
    """Expectation value paired with a propagation profile."""

    value: float
    profile: PropagationProfile


class PropagationEngine:
    """Reusable Rust-native Heisenberg propagation handle."""

    def __init__(
        self,
        tape: GateTape,
        observable: PauliOperator,
        *,
        initial_state: (
            ZeroState | ComputationalBasisState | ProductBlochState | str
        ) = _DEFAULT_ZERO_STATE,
        max_weight: Optional[int] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        if not isinstance(tape, GateTape):
            raise TypeError("tape must be a GateTape")
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        if observable.nqubits != tape.nqubits:
            raise ValueError("tape and observable must use the same nqubits")
        if max_weight is not None and (
            not isinstance(max_weight, int)
            or isinstance(max_weight, bool)
            or max_weight < 0
        ):
            raise ValueError("max_weight must be a non-negative integer or None")
        _validate_max_bytes(max_bytes)
        kind, bits, values = _state_payload(initial_state, tape.nqubits)
        self._native = _native.pauli_propagation_engine(
            tape.nqubits,
            tape._native_operations(),
            *observable._arrays(),
            kind,
            bits,
            values,
            max_weight,
            max_bytes,
        )
        self.nqubits = int(self._native.nqubits)
        self.nparameters = int(self._native.nparameters)
        self.max_weight = max_weight
        self.is_exact = bool(self._native.is_exact)

    def _parameters(
        self, parameters: Sequence[float] | np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        values = np.asarray(parameters, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != self.nparameters:
            raise ValueError(
                f"parameters must have shape ({self.nparameters},), got {values.shape}"
            )
        return cast(np.ndarray[Any, Any], np.ascontiguousarray(values))

    def expectation(self, parameters: Sequence[float] | np.ndarray[Any, Any]) -> float:
        """Propagate and return a scalar product-state expectation."""
        return float(self._native.expectation(self._parameters(parameters)))

    def propagate_operator(
        self, parameters: Sequence[float] | np.ndarray[Any, Any]
    ) -> PauliOperator:
        """Materialize the canonical propagated operator on the explicit path."""
        result = self._native.propagate_operator(self._parameters(parameters))
        return PauliOperator._from_native(self.nqubits, result)

    def profile(
        self, parameters: Sequence[float] | np.ndarray[Any, Any]
    ) -> ProfiledExpectation:
        """Return the scalar and explicit propagation diagnostics."""
        value, initial, final, peak, estimated, weights, seconds = self._native.profile(
            self._parameters(parameters)
        )
        profile = PropagationProfile(
            gate_count=int(self._native.gate_count),
            initial_terms=int(initial),
            final_terms=int(final),
            peak_terms=int(peak),
            estimated_peak_bytes=int(estimated),
            final_weight_counts=tuple(int(weight) for weight in weights),
            kernel_seconds=float(seconds),
        )
        return ProfiledExpectation(float(value), profile)


def _state_payload(
    state: ZeroState | ComputationalBasisState | ProductBlochState | str,
    nqubits: int,
) -> tuple[int, list[int], list[float]]:
    if state == "zero" or isinstance(state, ZeroState):
        return 0, [], []
    if isinstance(state, ComputationalBasisState):
        if len(state.bits) != nqubits:
            raise ValueError(f"bits must contain exactly {nqubits} values")
        return 1, list(state.bits), []
    if isinstance(state, ProductBlochState):
        if state.bloch.shape != (nqubits, 3):
            raise ValueError(f"bloch must have shape ({nqubits}, 3)")
        norms = np.linalg.norm(state.bloch, axis=1)
        if np.any(norms > 1.0 + 1e-12):
            raise ValueError("each Bloch vector norm must be at most 1 + 1e-12")
        return 2, [], [float(value) for value in state.bloch.ravel()]
    raise TypeError("initial_state must be 'zero' or a typed state descriptor")
