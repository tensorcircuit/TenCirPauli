"""SciPy matrix-free interoperability for reusable MVP plans."""

from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]

from ..hamiltonian import DEFAULT_MAX_BYTES, _validate_max_bytes


def to_scipy_linear_operator(
    plan: Any,
    *,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> LinearOperator:
    """Wrap a plan with ``apply`` and ``apply_adjoint`` as a LinearOperator.

    Both operations use the plan's basis and allocation limit. Adjoint
    application is the conjugate transpose, including for complex,
    non-Hermitian operators. SciPy supplies column-wise ``matmat`` and
    ``rmatmat`` from these vector operations.
    """
    _validate_max_bytes(max_bytes)
    if not callable(getattr(plan, "apply", None)) or not callable(
        getattr(plan, "apply_adjoint", None)
    ):
        raise TypeError("SciPy interop requires a plan with apply and apply_adjoint")
    dimension = int(plan.dimension)

    def apply(vector: Any, *, adjoint: bool) -> np.ndarray[Any, Any]:
        values = np.asarray(vector, dtype=np.complex128)
        if values.ndim == 2:
            if values.shape != (dimension, 1):
                raise ValueError(
                    f"LinearOperator vector must have shape ({dimension},) or "
                    f"({dimension}, 1), got {values.shape}"
                )
            values = values[:, 0]
        elif values.ndim != 1 or values.shape != (dimension,):
            raise ValueError(
                f"LinearOperator vector must have shape ({dimension},) or "
                f"({dimension}, 1), got {values.shape}"
            )
        operation = plan.apply_adjoint if adjoint else plan.apply
        return cast(
            np.ndarray[Any, Any],
            np.asarray(
                operation(np.ascontiguousarray(values), max_bytes=max_bytes),
                dtype=np.complex128,
                order="C",
            ),
        )

    return LinearOperator(
        shape=(dimension, dimension),
        matvec=lambda vector: apply(vector, adjoint=False),
        rmatvec=lambda vector: apply(vector, adjoint=True),
        dtype=np.complex128,
    )


__all__ = ["to_scipy_linear_operator"]
