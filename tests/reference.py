"""Independent dense NumPy reference for small Pauli systems.

This module deliberately uses only the public Pauli matrices and Kronecker
products. It must not import or call the Rust extension so that differential
tests have an independent oracle.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


PAULI_MATRICES = (
    np.array([[1, 0], [0, 1]], dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
)

_LOCAL_PRODUCT_TABLE = (
    ((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)),
    ((1, 1.0), (0, 1.0), (3, 1j), (2, -1j)),
    ((2, 1.0), (3, -1j), (0, 1.0), (1, 1j)),
    ((3, 1.0), (2, 1j), (1, -1j), (0, 1.0)),
)


def validate_codes(codes: Sequence[int]) -> Tuple[int, ...]:
    """Return codes as a tuple and reject values outside ``0..3``."""
    result = tuple(int(code) for code in codes)
    if any(code not in range(4) for code in result):
        raise ValueError(f"Pauli codes must be in 0..3, got {result}")
    return result


def codes_to_symplectic(codes: Sequence[int]) -> Tuple[int, int]:
    """Convert external I/X/Y/Z codes to independent packed integer masks."""
    x_mask = 0
    z_mask = 0
    for qubit, code in enumerate(validate_codes(codes)):
        if code in (1, 2):
            x_mask |= 1 << qubit
        if code in (2, 3):
            z_mask |= 1 << qubit
    return x_mask, z_mask


def symplectic_to_codes(x_mask: int, z_mask: int, nqubits: int) -> Tuple[int, ...]:
    """Convert independent packed integer masks to external Pauli codes."""
    if nqubits < 0:
        raise ValueError("nqubits must be non-negative")
    if x_mask < 0 or z_mask < 0 or x_mask.bit_length() > nqubits:
        raise ValueError("x mask has bits outside nqubits")
    if z_mask.bit_length() > nqubits:
        raise ValueError("z mask has bits outside nqubits")
    return tuple(
        (1 if x_mask >> qubit & 1 else 0) + (2 if z_mask >> qubit & 1 else 0)
        for qubit in range(nqubits)
    )


def codes_to_dense(codes: Sequence[int]) -> np.ndarray:
    """Construct a dense operator using MSB qubit ordering and ``np.kron``."""
    result = np.array([[1]], dtype=np.complex128)
    for code in validate_codes(codes):
        result = np.kron(result, PAULI_MATRICES[code])
    return result


def product_single(left: int, right: int) -> Tuple[int, complex]:
    """Return the independent local Pauli product and exact phase."""
    left_code = validate_codes((left,))[0]
    right_code = validate_codes((right,))[0]
    result, phase = _LOCAL_PRODUCT_TABLE[left_code][right_code]
    return int(result), complex(phase)


def multiply_codes(
    left: Sequence[int], right: Sequence[int]
) -> Tuple[Tuple[int, ...], complex]:
    """Multiply two equal-length phase-free Pauli words."""
    left_codes = validate_codes(left)
    right_codes = validate_codes(right)
    if len(left_codes) != len(right_codes):
        raise ValueError("Pauli words must have equal lengths")
    result = []
    phase = 1.0 + 0.0j
    for left_code, right_code in zip(left_codes, right_codes):
        code, local_phase = product_single(left_code, right_code)
        result.append(code)
        phase *= local_phase
    return tuple(result), phase


def dense_operator(
    nqubits: int, structures: Iterable[Sequence[int]], coefficients: Sequence[complex]
) -> np.ndarray:
    """Build a dense operator by independently summing Kronecker products."""
    if nqubits < 0:
        raise ValueError("nqubits must be non-negative")
    structures_tuple = tuple(validate_codes(structure) for structure in structures)
    coefficients_array = np.asarray(coefficients, dtype=np.complex128)
    if coefficients_array.ndim != 1 or len(structures_tuple) != len(coefficients_array):
        raise ValueError("structures and coefficients must have matching 1-D lengths")
    if any(len(structure) != nqubits for structure in structures_tuple):
        raise ValueError("every structure must contain exactly nqubits codes")
    dimension = 1 << nqubits
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for structure, coefficient in zip(structures_tuple, coefficients_array):
        result += coefficient * codes_to_dense(structure)
    return result


def support(codes: Sequence[int]) -> Tuple[int, ...]:
    """Return non-identity qubit indices in canonical ascending order."""
    return tuple(index for index, code in enumerate(validate_codes(codes)) if code)


def commutes(left: Sequence[int], right: Sequence[int]) -> bool:
    """Check commutation from the independent local anticommutation rule."""
    left_codes = validate_codes(left)
    right_codes = validate_codes(right)
    if len(left_codes) != len(right_codes):
        raise ValueError("Pauli words must have equal lengths")
    anticommutations = 0
    for left_code, right_code in zip(left_codes, right_codes):
        if left_code and right_code and left_code != right_code:
            anticommutations += 1
    return anticommutations % 2 == 0
