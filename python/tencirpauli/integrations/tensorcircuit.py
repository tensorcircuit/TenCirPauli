"""Optional TensorCircuit backend-plan adapter.

TensorCircuit is imported only when an adapter is explicitly requested. The
Rust core and public top-level package remain independent of that dependency.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional, Sequence

import numpy as np

from ..hamiltonian import (
    DEFAULT_MAX_BYTES,
    BackendMVPPlan,
    _check_allocation,
    _dimension,
)


def require_tensorcircuit() -> Any:
    """Import TensorCircuit or fail with an actionable optional-dependency error."""
    try:
        return importlib.import_module("tensorcircuit")
    except ImportError as error:
        raise ImportError(
            "TensorCircuit integration requires the optional 'tensorcircuit-ng' "
            "dependency; install tencirpauli[tensorcircuit]"
        ) from error


def backend_mvp(
    plan: BackendMVPPlan,
    coefficients: Optional[Sequence[complex]] = None,
    backend: Any = None,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> Any:
    """Return a TensorCircuit-backend MVP callable for a pure-array plan.

    The returned callable accepts a flat ``2**n`` state or a rank-``n`` tensor.
    Plan structure is fixed before tracing; coefficients may be replaced by a
    backend tensor for a differentiable parameter buffer.
    """
    tensorcircuit = require_tensorcircuit()
    runtime_backend = backend if backend is not None else tensorcircuit.backend
    dimension = _dimension(plan.nqubits)
    estimated_bytes = dimension * ((len(plan.coefficients) + 1) * 8 + 48)
    _check_allocation(estimated_bytes, max_bytes, "TensorCircuit MVP adapter")
    if coefficients is None:
        coefficient_values = runtime_backend.convert_to_tensor(plan.coefficients)
    elif hasattr(coefficients, "shape"):
        coefficient_values = coefficients
    else:
        coefficient_values = runtime_backend.convert_to_tensor(coefficients)
    if len(plan.coefficients) != int(
        runtime_backend.shape_tuple(coefficient_values)[0]
    ):
        raise ValueError("coefficient buffer length does not match backend MVP plan")

    term_masks: list[Any] = []
    flip_axes: list[tuple[int, ...]] = []
    y_counts: list[int] = []
    for term_index in range(len(plan.coefficients)):
        z_mask: np.ndarray[Any, Any] = np.ones((1,) * plan.nqubits, dtype=np.float64)
        flips = []
        y_count = 0
        for qubit in range(plan.nqubits):
            code = _plan_code(plan, term_index, qubit)
            if code in (2, 3):
                local = np.array([1.0, -1.0]).reshape(
                    (1,) * qubit + (2,) + (1,) * (plan.nqubits - qubit - 1)
                )
                z_mask = z_mask * local
            if code in (1, 2):
                flips.append(qubit)
            if code == 2:
                y_count += 1
        term_masks.append(runtime_backend.convert_to_tensor(z_mask))
        flip_axes.append(tuple(flips))
        y_counts.append(y_count)

    def mvp(state: Any) -> Any:
        state_shape = runtime_backend.shape_tuple(state)
        if len(state_shape) == 1:
            if state_shape[0] != 2**plan.nqubits:
                raise ValueError("flat state has incompatible length")
            tensor_state = runtime_backend.reshape(state, (2,) * plan.nqubits)
            flat = True
        elif tuple(state_shape) == (2,) * plan.nqubits:
            tensor_state = state
            flat = False
        else:
            raise ValueError("state must be flat or rank-nqubits with binary axes")
        dtype = runtime_backend.dtype(tensor_state)
        total = runtime_backend.zeros_like(tensor_state)
        for term_index, mask in enumerate(term_masks):
            term_state = tensor_state * runtime_backend.cast(mask, dtype)
            if flip_axes[term_index]:
                slices = tuple(
                    (
                        slice(None, None, -1)
                        if axis in flip_axes[term_index]
                        else slice(None)
                    )
                    for axis in range(plan.nqubits)
                )
                term_state = term_state[slices]
            weight = coefficient_values[term_index] * (1j ** y_counts[term_index])
            total = total + term_state * weight
        return runtime_backend.reshape(total, (-1,)) if flat else total

    return mvp


def _plan_code(plan: BackendMVPPlan, term_index: int, qubit: int) -> int:
    x = (int(plan.x_words[term_index, qubit // 64]) >> (qubit % 64)) & 1
    z = (int(plan.z_words[term_index, qubit // 64]) >> (qubit % 64)) & 1
    return {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}[(x, z)]
