"""Typed public facade for the Rust-native stochastic Pauli-path engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, cast

import numpy as np

from . import _native
from .circuit import _coerce_parameters
from .hamiltonian import DEFAULT_MAX_BYTES, _validate_max_bytes
from .pauli import PauliOperator
from .propagation import (
    ComputationalBasisState,
    GateTape,
    ProductBlochState,
    ZeroState,
    _state_payload,
)


_DEFAULT_ZERO_STATE = ZeroState()


@dataclass(frozen=True)
class SPPSEstimate:
    """One fixed-budget or adaptive SPPS estimate.

    ``value_standard_error`` uses the usual finite-sample sample-variance
    estimator of the sampled path distribution (with an ``N-1`` denominator).
    """

    value: float
    gradient: np.ndarray[Any, Any]
    value_standard_error: float
    replicates: int
    samples_per_replicate: tuple[int, ...]
    total_paths: int
    seed: int
    gradient_error_proxy: Optional[float]
    term_gradient_error_proxies: Optional[tuple[float, ...]]
    converged: Optional[bool]


@dataclass(frozen=True)
class SPPSValueEstimate:
    """One value-only stochastic Pauli-path estimate.

    ``value_standard_error`` uses the usual finite-sample sample-variance
    estimator of the sampled path distribution (with an ``N-1`` denominator).
    """

    value: float
    value_standard_error: float
    replicates: int
    samples_per_replicate: tuple[int, ...]
    total_paths: int
    seed: int


class SPPSEngine:
    """Reusable Rust-native stochastic Pauli-path estimation handle.

    Fixed-budget estimates use independent replicates and report a standard
    error for the sampled path distribution. Adaptive estimates add samples in
    cumulative rounds until the requested gradient error proxy is reached or
    the maximum budget is exhausted.
    """

    def __init__(
        self,
        tape: GateTape,
        observable: PauliOperator,
        *,
        initial_state: (
            ZeroState | ComputationalBasisState | ProductBlochState | str
        ) = _DEFAULT_ZERO_STATE,
        smoothing: float = 0.01,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        if not isinstance(tape, GateTape):
            raise TypeError("tape must be a GateTape")
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        if observable.nqubits != tape.nqubits:
            raise ValueError("tape and observable must use the same nqubits")
        if isinstance(smoothing, bool):
            raise TypeError("smoothing must be a finite positive float")
        try:
            normalized_smoothing = float(smoothing)
        except (TypeError, ValueError) as error:
            raise TypeError("smoothing must be a finite positive float") from error
        if not math.isfinite(normalized_smoothing) or normalized_smoothing <= 0.0:
            raise ValueError("smoothing must be a finite positive float")
        _validate_max_bytes(max_bytes)
        kind, bits, values = _state_payload(initial_state, tape.nqubits)
        self._native = _native.pauli_spps_engine(
            tape.nqubits,
            tape._native_operations(),
            *observable._arrays(),
            kind,
            bits,
            values,
            normalized_smoothing,
            max_bytes,
        )
        self.nqubits = int(self._native.nqubits)
        self.nparameters = int(self._native.nparameters)
        self.gate_count = int(self._native.gate_count)
        self.observable_terms = int(self._native.observable_terms)
        self.smoothing = float(self._native.smoothing)

    def _parameters(
        self, parameters: Sequence[float] | np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        values = _coerce_parameters(parameters, self.nparameters)
        return cast(np.ndarray[Any, Any], np.ascontiguousarray(values))

    @staticmethod
    def _seed(seed: int) -> int:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative 64-bit integer")
        if seed > (1 << 64) - 1:
            raise OverflowError("seed must fit in an unsigned 64-bit integer")
        return seed

    @staticmethod
    def _budget(value: int, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
        return value

    def expectation(
        self,
        parameters: Sequence[float] | np.ndarray[Any, Any],
        *,
        samples_per_term: int,
        seed: int,
    ) -> SPPSValueEstimate:
        """Estimate the expectation value without computing a gradient.

        Args:
            parameters: Finite runtime values for all circuit parameter slots.
            samples_per_term: Fixed path budget per observable term; must be at
                least two.
            seed: Non-negative unsigned 64-bit random seed.

        Returns:
            A value estimate containing the standard error, replicate budgets,
            total sampled paths, and effective seed.
        """
        budget = self._budget(samples_per_term, "samples_per_term")
        normalized_seed = self._seed(seed)
        value, standard_error, replicates, budgets, total_paths, result_seed = (
            self._native.expectation(
                self._parameters(parameters), budget, normalized_seed
            )
        )
        return SPPSValueEstimate(
            value=float(value),
            value_standard_error=float(standard_error),
            replicates=int(replicates),
            samples_per_replicate=tuple(int(item) for item in budgets),
            total_paths=int(total_paths),
            seed=int(result_seed),
        )

    def value_and_grad(
        self,
        parameters: Sequence[float] | np.ndarray[Any, Any],
        *,
        samples_per_term: int,
        seed: int,
    ) -> SPPSEstimate:
        """Estimate the value and all parameter gradients with one fixed budget.

        The returned gradient is a read-only float64 vector indexed by runtime
        parameter slot. The estimate also includes a value standard error and
        a gradient error proxy when the native estimator provides one.
        """
        budget = self._budget(samples_per_term, "samples_per_term")
        normalized_seed = self._seed(seed)
        result = self._native.value_and_grad(
            self._parameters(parameters), budget, normalized_seed
        )
        return _estimate_from_native(result)

    def value_and_grad_adaptive(
        self,
        parameters: Sequence[float] | np.ndarray[Any, Any],
        *,
        initial_samples_per_term: int,
        max_samples_per_term: int,
        gradient_tolerance: float,
        seed: int,
    ) -> SPPSEstimate:
        """Estimate value and gradients with an adaptive sample budget.

        Sampling starts at ``initial_samples_per_term`` and grows up to
        ``max_samples_per_term`` until ``gradient_tolerance`` is met. The
        returned ``converged`` flag distinguishes tolerance convergence from
        exhaustion of the maximum budget.
        """
        initial = self._budget(initial_samples_per_term, "initial_samples_per_term")
        maximum = self._budget(max_samples_per_term, "max_samples_per_term")
        if maximum < initial:
            raise ValueError("max_samples_per_term must be >= initial_samples_per_term")
        if isinstance(gradient_tolerance, bool):
            raise TypeError("gradient_tolerance must be finite and positive")
        try:
            tolerance = float(gradient_tolerance)
        except (TypeError, ValueError) as error:
            raise TypeError("gradient_tolerance must be finite and positive") from error
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("gradient_tolerance must be finite and positive")
        normalized_seed = self._seed(seed)
        result = self._native.value_and_grad_adaptive(
            self._parameters(parameters),
            initial,
            maximum,
            tolerance,
            normalized_seed,
        )
        return _estimate_from_native(result)


def _estimate_from_native(result: tuple[Any, ...]) -> SPPSEstimate:
    (
        value,
        gradient,
        standard_error,
        replicates,
        budgets,
        total_paths,
        seed,
        gradient_proxy,
        term_proxies,
        converged,
    ) = result
    normalized_gradient = np.ascontiguousarray(np.asarray(gradient, dtype=np.float64))
    normalized_gradient.flags.writeable = False
    return SPPSEstimate(
        value=float(value),
        gradient=normalized_gradient,
        value_standard_error=float(standard_error),
        replicates=int(replicates),
        samples_per_replicate=tuple(int(item) for item in budgets),
        total_paths=int(total_paths),
        seed=int(seed),
        gradient_error_proxy=(
            None if gradient_proxy is None else float(gradient_proxy)
        ),
        term_gradient_error_proxies=(
            None
            if term_proxies is None
            else tuple(float(item) for item in term_proxies)
        ),
        converged=None if converged is None else bool(converged),
    )
