"""Independent dense oracle for Phase 3 propagation tests."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from reference import PAULI_MATRICES, codes_to_dense


def single_matrix(code: int) -> np.ndarray:
    return PAULI_MATRICES[code]


def embedded_one(nqubits: int, wire: int, matrix: np.ndarray) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for qubit in range(nqubits):
        result = np.kron(result, matrix if qubit == wire else PAULI_MATRICES[0])
    return result


def embedded_two(
    nqubits: int, wire0: int, wire1: int, matrix: np.ndarray
) -> np.ndarray:
    dimension = 1 << nqubits
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        bits = [(column >> (nqubits - 1 - qubit)) & 1 for qubit in range(nqubits)]
        local_column = 2 * bits[wire0] + bits[wire1]
        for local_row in range(4):
            output = bits.copy()
            output[wire0], output[wire1] = divmod(local_row, 2)
            row = 0
            for bit in output:
                row = (row << 1) | bit
            result[row, column] = matrix[local_row, local_column]
    return result


def gate_matrix(nqubits: int, operation: tuple[object, ...]) -> np.ndarray:
    kind = operation[0]
    wire0 = int(operation[1])
    if kind == "x":
        return embedded_one(nqubits, wire0, single_matrix(1))
    if kind == "y":
        return embedded_one(nqubits, wire0, single_matrix(2))
    if kind == "z":
        return embedded_one(nqubits, wire0, single_matrix(3))
    if kind == "h":
        return embedded_one(
            nqubits,
            wire0,
            np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
        )
    if kind in ("s", "sdg"):
        phase = 1j if kind == "s" else -1j
        return embedded_one(nqubits, wire0, np.diag([1.0, phase]))
    if kind in ("cnot", "cz", "swap"):
        wire1 = int(operation[2])
        if kind == "cnot":
            local = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=np.complex128,
            )
        elif kind == "cz":
            local = np.diag([1, 1, 1, -1]).astype(np.complex128)
        else:
            local = np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=np.complex128,
            )
        return embedded_two(nqubits, wire0, wire1, local)
    if kind in ("rx", "ry", "rz", "rxx", "ryy", "rzz"):
        theta = float(operation[-1])
        axis = {"rx": 1, "ry": 2, "rz": 3, "rxx": 1, "ryy": 2, "rzz": 3}[kind]
        generator = single_matrix(axis)
        if kind in ("rx", "ry", "rz"):
            local_generator = generator
            return embedded_one(
                nqubits,
                wire0,
                np.cos(theta / 2) * PAULI_MATRICES[0]
                - 1j * np.sin(theta / 2) * local_generator,
            )
        wire1 = int(operation[2])
        local_generator = np.kron(generator, generator)
        return embedded_two(
            nqubits,
            wire0,
            wire1,
            np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * local_generator,
        )
    raise ValueError(f"unsupported reference operation {kind!r}")


def pauli_decompose(matrix: np.ndarray, nqubits: int) -> dict[tuple[int, ...], complex]:
    scale = float(1 << nqubits)
    result: dict[tuple[int, ...], complex] = {}
    for codes in np.ndindex(*(4,) * nqubits) if nqubits else [()]:
        coefficient = np.trace(codes_to_dense(codes) @ matrix) / scale
        if abs(coefficient) > 1e-12:
            result[tuple(codes)] = complex(coefficient)
    return result


def propagate_dense(
    nqubits: int,
    structures: Iterable[Sequence[int]],
    coefficients: Sequence[complex],
    operations: Sequence[tuple[object, ...]],
    max_weight: int | None = None,
) -> dict[tuple[int, ...], complex]:
    matrix = sum(
        (
            complex(coefficient) * codes_to_dense(structure)
            for structure, coefficient in zip(structures, coefficients)
        ),
        np.zeros((1 << nqubits, 1 << nqubits), dtype=np.complex128),
    )
    if max_weight is not None and max_weight < nqubits:
        coefficients_by_word = pauli_decompose(matrix, nqubits)
        matrix = sum(
            (
                coefficient * codes_to_dense(codes)
                for codes, coefficient in coefficients_by_word.items()
                if sum(code != 0 for code in codes) <= max_weight
            ),
            np.zeros_like(matrix),
        )
    for operation in reversed(operations):
        unitary = gate_matrix(nqubits, operation)
        matrix = unitary.conj().T @ matrix @ unitary
        if max_weight is not None and max_weight < nqubits:
            coefficients_by_word = pauli_decompose(matrix, nqubits)
            matrix = sum(
                (
                    coefficient * codes_to_dense(codes)
                    for codes, coefficient in coefficients_by_word.items()
                    if sum(code != 0 for code in codes) <= max_weight
                ),
                np.zeros_like(matrix),
            )
    return pauli_decompose(matrix, nqubits)


def product_expectation(
    matrix: np.ndarray, state: str | Sequence[int] | np.ndarray
) -> float:
    if isinstance(state, str) and state == "zero":
        vector = np.zeros(matrix.shape[0], dtype=np.complex128)
        vector[0] = 1.0
        return float(np.vdot(vector, matrix @ vector).real)
    if isinstance(state, np.ndarray):
        density = np.array([[1.0 + 0.0j]])
        for x, y, z in state:
            density = np.kron(
                density,
                (
                    PAULI_MATRICES[0]
                    + x * PAULI_MATRICES[1]
                    + y * PAULI_MATRICES[2]
                    + z * PAULI_MATRICES[3]
                )
                / 2,
            )
        return float(np.trace(density @ matrix).real)
    bits = tuple(state)
    index = 0
    for bit in bits:
        index = (index << 1) | int(bit)
    vector = np.zeros(matrix.shape[0], dtype=np.complex128)
    vector[index] = 1.0
    return float(np.vdot(vector, matrix @ vector).real)
