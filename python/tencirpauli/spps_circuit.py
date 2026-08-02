"""Unified Python facade for stochastic Pauli-path circuits."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Sequence, Union, cast

import numpy as np

from .circuit import Angle, _coerce_parameters, _evaluate_angle
from .hamiltonian import _validate_max_bytes
from .pauli import PauliOperator
from .propagation import (
    ComputationalBasisState,
    ProductBlochState,
    ZeroState,
)
from .propagation_circuit import _USE_DEFAULT_MAX_BYTES, PropagationCircuit
from .spps import SPPSEngine, SPPSEstimate, SPPSValueEstimate


SPPSState = Union[ZeroState, ComputationalBasisState, ProductBlochState, str]


class SPPSCircuitPlan:
    """Immutable compiled stochastic propagation facade."""

    def __init__(
        self,
        engine: SPPSEngine,
        dynamic_angles: tuple[Angle, ...],
        nparameters: int,
    ) -> None:
        self._engine = engine
        self._dynamic_angles = dynamic_angles
        self.nqubits = engine.nqubits
        self.nparameters = nparameters
        self.smoothing = engine.smoothing

    def _native_parameters(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        values = _coerce_parameters(parameters, self.nparameters)
        native: np.ndarray[Any, Any] = np.empty(
            len(self._dynamic_angles), dtype=np.float64
        )
        jacobian: np.ndarray[Any, Any] = np.empty(
            (len(self._dynamic_angles), self.nparameters), dtype=np.float64
        )
        for index, angle in enumerate(self._dynamic_angles):
            native[index], jacobian[index] = _evaluate_angle(
                angle, values, self.nparameters
            )
        return native, jacobian

    def expectation(
        self,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]],
        *,
        samples_per_term: int,
        seed: int,
    ) -> SPPSValueEstimate:
        native, _ = self._native_parameters(parameters)
        return self._engine.expectation(
            native, samples_per_term=samples_per_term, seed=seed
        )

    def value_and_grad(
        self,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]],
        *,
        samples_per_term: int,
        seed: int,
    ) -> SPPSEstimate:
        native, jacobian = self._native_parameters(parameters)
        result = self._engine.value_and_grad(
            native, samples_per_term=samples_per_term, seed=seed
        )
        gradient = np.ascontiguousarray(jacobian.T @ result.gradient)
        gradient.flags.writeable = False
        return replace(result, gradient=gradient)

    def value_and_grad_adaptive(
        self,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]],
        *,
        initial_samples_per_term: int,
        max_samples_per_term: int,
        gradient_tolerance: float,
        seed: int,
    ) -> SPPSEstimate:
        native, jacobian = self._native_parameters(parameters)
        result = self._engine.value_and_grad_adaptive(
            native,
            initial_samples_per_term=initial_samples_per_term,
            max_samples_per_term=max_samples_per_term,
            gradient_tolerance=gradient_tolerance,
            seed=seed,
        )
        gradient = np.ascontiguousarray(jacobian.T @ result.gradient)
        gradient.flags.writeable = False
        return replace(result, gradient=gradient)


class SPPSCircuit(PropagationCircuit):
    """TensorCircuit-style builder for stochastic Pauli-path estimation."""

    def compile(  # type: ignore[override]
        self,
        observable: PauliOperator,
        *,
        initial_state: Optional[SPPSState] = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> SPPSCircuitPlan:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        state = self.initial_state if initial_state is None else initial_state
        budget = (
            self.max_bytes
            if max_bytes is _USE_DEFAULT_MAX_BYTES
            else cast(Optional[int], max_bytes)
        )
        _validate_max_bytes(budget)
        key = (
            self._generation,
            id(observable),
            id(state),
            float(smoothing),
            budget,
        )
        if self._cached_plan is not None and self._cached_plan[:5] == key:
            return cast(SPPSCircuitPlan, self._cached_plan[5])
        tape, dynamic = self._native_tape()
        engine = SPPSEngine(
            tape,
            observable,
            initial_state=state,
            smoothing=smoothing,
            max_bytes=budget,
        )
        plan = SPPSCircuitPlan(engine, dynamic, self.nparameters)
        self._cached_plan = (*key, plan)
        return plan

    def expectation(  # type: ignore[override]
        self,
        observable: PauliOperator,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        samples_per_term: int,
        seed: int,
        initial_state: Optional[SPPSState] = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> SPPSValueEstimate:
        return self.compile(
            observable,
            initial_state=initial_state,
            smoothing=smoothing,
            max_bytes=max_bytes,
        ).expectation(parameters, samples_per_term=samples_per_term, seed=seed)

    def value_and_grad(  # type: ignore[override]
        self,
        observable: PauliOperator,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        samples_per_term: int,
        seed: int,
        initial_state: Optional[SPPSState] = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> SPPSEstimate:
        return self.compile(
            observable,
            initial_state=initial_state,
            smoothing=smoothing,
            max_bytes=max_bytes,
        ).value_and_grad(parameters, samples_per_term=samples_per_term, seed=seed)

    def value_and_grad_adaptive(
        self,
        observable: PauliOperator,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        initial_samples_per_term: int,
        max_samples_per_term: int,
        gradient_tolerance: float,
        seed: int,
        initial_state: Optional[SPPSState] = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> SPPSEstimate:
        return self.compile(
            observable,
            initial_state=initial_state,
            smoothing=smoothing,
            max_bytes=max_bytes,
        ).value_and_grad_adaptive(
            parameters,
            initial_samples_per_term=initial_samples_per_term,
            max_samples_per_term=max_samples_per_term,
            gradient_tolerance=gradient_tolerance,
            seed=seed,
        )


__all__ = ["SPPSCircuit", "SPPSCircuitPlan"]
