"""Small shared validators for public Python modules."""

from __future__ import annotations


def validate_nonnegative_int(value: object, name: str) -> int:
    """Return an exact non-negative Python integer or raise ``ValueError``."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
