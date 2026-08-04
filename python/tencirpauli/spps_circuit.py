"""Unified Python facade for stochastic Pauli-path circuits."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Sequence, Union, cast

import numpy as np

from .circuit import Angle, _coerce_parameters, _evaluate_angle, _evaluate_angle_value
from .hamiltonian import _validate_max_bytes
from .pauli import PauliOperator
from .propagation import (
    ComputationalBasisState,
    ProductBlochState,
    ZeroState,
)
from .propagation_circuit import _USE_DEFAULT_MAX_BYTES, _CircuitBuilder
from .spps import SPPSEngine, SPPSEstimate, SPPSValueEstimate


SPPSState = Union[ZeroState, ComputationalBasisState, ProductBlochState, str]


class SPPSCircuitPlan:
    """Immutable compiled stochastic propagation facade.

    The plan caches the native tape and applies the symbolic-angle Jacobian
    when returning gradients with respect to the circuit's public parameters.
    """

    __slots__ = (
        "_dynamic_angles",
        "_engine",
        "_locked",
        "nparameters",
        "nqubits",
        "smoothing",
    )

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
        self._locked = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("SPPSCircuitPlan is immutable")
        object.__setattr__(self, name, value)

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

    def _native_values(
        self, parameters: Optional[Sequence[float] | np.ndarray[Any, Any]]
    ) -> np.ndarray[Any, Any]:
        values = _coerce_parameters(parameters, self.nparameters)
        native: np.ndarray[Any, Any] = np.empty(
            len(self._dynamic_angles), dtype=np.float64
        )
        for index, angle in enumerate(self._dynamic_angles):
            native[index] = _evaluate_angle_value(angle, values, self.nparameters)
        return native

    def expectation(
        self,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        samples_per_term: int,
        seed: int,
    ) -> SPPSValueEstimate:
        """Estimate one real observable expectation with a fixed path budget.

        The result is an :class:`SPPSValueEstimate`, not an exact scalar, and
        non-Hermitian observables raise ``ValueError`` at compilation.
        """
        native = self._native_values(parameters)
        return self._engine.expectation(
            native, samples_per_term=samples_per_term, seed=seed
        )

    def value_and_grad(
        self,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        samples_per_term: int,
        seed: int,
    ) -> SPPSEstimate:
        """Estimate value and gradient for an exactly Hermitian observable.

        The result is an :class:`SPPSEstimate`, not an exact scalar, and
        non-Hermitian observables raise ``ValueError`` at compilation.
        """
        native, jacobian = self._native_parameters(parameters)
        result = self._engine.value_and_grad(
            native, samples_per_term=samples_per_term, seed=seed
        )
        gradient = np.ascontiguousarray(jacobian.T @ result.gradient)
        gradient.flags.writeable = False
        return replace(result, gradient=gradient)

    def value_and_grad_adaptive(
        self,
        parameters: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        *,
        initial_samples_per_term: int,
        max_samples_per_term: int,
        gradient_tolerance: float,
        seed: int,
    ) -> SPPSEstimate:
        """Estimate value and gradient with adaptive sampling.

        The result is an :class:`SPPSEstimate`, not an exact scalar, and
        non-Hermitian observables raise ``ValueError`` at compilation.
        """
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


class SPPSCircuit(_CircuitBuilder):
    """TensorCircuit-style builder for stochastic Pauli-path estimation.

    Gate construction follows :class:`PropagationCircuit`; execution requires
    explicit sampling budgets and a random seed through the SPPS methods.

    Examples:
        >>> import tencirpauli as tcp
        >>> circuit = tcp.SPPSCircuit(1)
        >>> observable = tcp.PauliOperator.from_terms(1, [("Z", 1.0)])
        >>> plan = circuit.compile(observable)
        >>> plan.nqubits
        1
    """

    def compile(
        self,
        observable: PauliOperator,
        *,
        initial_state: Optional[SPPSState] = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> SPPSCircuitPlan:
        """Compile and cache an SPPS plan for one observable.

        ``smoothing`` must be a finite positive value. The plan is invalidated
        whenever the circuit is mutated or its observable/state configuration
        changes. Non-Hermitian observables raise ``ValueError`` at this
        boundary because every SPPS terminal is a scalar estimator.
        """
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
        # Retain the key objects as well as their ids; otherwise CPython may
        # reuse an id after garbage collection and return a stale native plan.
        self._cached_plan = (*key, plan, observable, state)
        return plan

    def expectation(
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
        """Compile if needed and return a fixed-budget value estimate."""
        return self.compile(
            observable,
            initial_state=initial_state,
            smoothing=smoothing,
            max_bytes=max_bytes,
        ).expectation(parameters, samples_per_term=samples_per_term, seed=seed)

    def value_and_grad(
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
        """Compile if needed and return a fixed-budget value-and-gradient estimate."""
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
        """Compile if needed and return an adaptive value-and-gradient estimate."""
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
