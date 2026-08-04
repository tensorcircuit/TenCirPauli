"""Small shared validators for public Python modules."""

from __future__ import annotations

import operator
from typing import SupportsIndex, cast

import numpy as np


def normalize_pauli_code(value: object) -> int:
    """Normalize one public Pauli code using Python ``__index__`` semantics."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Pauli code must support __index__; bool is not accepted")
    try:
        normalized = operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise TypeError("Pauli code must support __index__") from error
    if normalized < 0 or normalized >= 4:
        raise ValueError("Pauli code must be in the half-open range 0..4")
    return int(normalized)


def validate_nonnegative_int(value: object, name: str) -> int:
    """Return an exact non-negative Python integer or raise ``ValueError``."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
