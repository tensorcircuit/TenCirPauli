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
    """Wrap one immutable MVP plan without materializing its matrix."""
    _validate_max_bytes(max_bytes)
    dimension = int(plan.dimension)

    def matvec(vector: Any) -> np.ndarray[Any, Any]:
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
        return cast(
            np.ndarray[Any, Any],
            np.asarray(
                plan.apply(np.ascontiguousarray(values), max_bytes=max_bytes),
                dtype=np.complex128,
                order="C",
            ),
        )

    return LinearOperator(
        shape=(dimension, dimension),
        matvec=matvec,
        dtype=np.complex128,
    )


__all__ = ["to_scipy_linear_operator"]
