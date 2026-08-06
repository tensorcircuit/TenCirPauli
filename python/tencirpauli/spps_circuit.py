"""Public fixed-budget stochastic Pauli-path circuit facade."""

from __future__ import annotations

from typing import Any, Optional, Union, cast

import numpy as np

from .hamiltonian import _validate_max_bytes
from .pauli import PauliOperator
from .propagation import ComputationalBasisState, ProductBlochState, ZeroState
from .propagation_circuit import (
    _USE_DEFAULT_MAX_BYTES,
    _CircuitBuilder,
)
from .spps import SPPSEngine, SPPSEstimate, SPPSValueEstimate


SPPSState = Union[ZeroState, ComputationalBasisState, ProductBlochState, str]


class SPPSCircuit(_CircuitBuilder):
    """TensorCircuit-style builder for fixed-budget stochastic estimation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._spps_objective_cache: Optional[tuple[tuple[Any, ...], SPPSEngine]] = None

    def _append(self, *args: Any, **kwargs: Any) -> None:
        super()._append(*args, **kwargs)
        self._spps_objective_cache = None

    def _spps_engine(
        self,
        observable: PauliOperator,
        *,
        initial_state: SPPSState,
        smoothing: float,
        max_bytes: Optional[int],
        gradient: bool,
    ) -> SPPSEngine:
        key = (
            self._generation,
            id(observable),
            id(initial_state),
            float(smoothing),
            max_bytes,
            gradient,
        )
        if (
            self._spps_objective_cache is not None
            and self._spps_objective_cache[0] == key
        ):
            return self._spps_objective_cache[1]
        engine = SPPSEngine(
            self._native_tape(gradient),
            observable,
            initial_state=initial_state,
            smoothing=smoothing,
            max_bytes=max_bytes,
        )
        self._spps_objective_cache = (key, engine)
        return engine

    def _budget(self, max_bytes: object) -> Optional[int]:
        budget = self.max_bytes if max_bytes is _USE_DEFAULT_MAX_BYTES else max_bytes
        _validate_max_bytes(cast(Optional[int], budget))
        return cast(Optional[int], budget)

    def expectation(
        self,
        observable: PauliOperator,
        *,
        samples_per_term: int,
        seed: int,
        initial_state: Optional[SPPSState] = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> SPPSValueEstimate:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        engine = self._spps_engine(
            observable,
            initial_state=(
                self.initial_state if initial_state is None else initial_state
            ),
            smoothing=smoothing,
            max_bytes=self._budget(max_bytes),
            gradient=False,
        )
        return engine.expectation(
            np.empty(0, dtype=np.float64),
            samples_per_term=samples_per_term,
            seed=seed,
        )

    def value_and_grad(
        self,
        observable: PauliOperator,
        *,
        samples_per_term: int,
        seed: int,
        initial_state: Optional[SPPSState] = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> SPPSEstimate:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        engine = self._spps_engine(
            observable,
            initial_state=(
                self.initial_state if initial_state is None else initial_state
            ),
            smoothing=smoothing,
            max_bytes=self._budget(max_bytes),
            gradient=True,
        )
        return engine.value_and_grad(
            self._angle_values(), samples_per_term=samples_per_term, seed=seed
        )

    def expectation_jax(
        self,
        observable: PauliOperator,
        *,
        samples_per_term: int,
        seed: int,
        initial_state: Optional[SPPSState] = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> Any:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        engine = self._spps_engine(
            observable,
            initial_state=(
                self.initial_state if initial_state is None else initial_state
            ),
            smoothing=smoothing,
            max_bytes=self._budget(max_bytes),
            gradient=True,
        )
        from .jax_support import native_expectation_jax

        class _Objective:
            def value_and_grad(
                self,
                parameters: np.ndarray[Any, Any],
                *,
                checkpoint_interval: Optional[int] = None,
            ) -> SPPSEstimate:
                del checkpoint_interval
                return engine.value_and_grad(
                    parameters, samples_per_term=samples_per_term, seed=seed
                )

        return native_expectation_jax(
            tuple(
                operation.theta
                for operation in self._operations
                if operation.theta is not None
            ),
            _Objective(),
        )

    def value_and_grad_adaptive(
        self,
        observable: PauliOperator,
        *,
        initial_samples_per_term: int,
        max_samples_per_term: int,
        gradient_tolerance: float,
        seed: int,
        initial_state: Optional[SPPSState] = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_DEFAULT_MAX_BYTES,
    ) -> SPPSEstimate:
        if not isinstance(observable, PauliOperator):
            raise TypeError("observable must be a PauliOperator")
        engine = self._spps_engine(
            observable,
            initial_state=(
                self.initial_state if initial_state is None else initial_state
            ),
            smoothing=smoothing,
            max_bytes=self._budget(max_bytes),
            gradient=True,
        )
        return engine.value_and_grad_adaptive(
            self._angle_values(),
            initial_samples_per_term=initial_samples_per_term,
            max_samples_per_term=max_samples_per_term,
            gradient_tolerance=gradient_tolerance,
            seed=seed,
        )


__all__ = ["SPPSCircuit"]
