"""Small independent legal-path enumerator used by SPPS correctness tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from propagation_reference import gate_matrix, pauli_decompose
from reference import codes_to_dense


@dataclass(frozen=True)
class Path:
    probability: float
    factors: tuple[tuple[float, float], ...]
    word: tuple[int, ...]
    sign: float


def _local_product(left: int, right: int) -> tuple[int, complex]:
    table = {
        (0, 0): (0, 1),
        (0, 1): (1, 1),
        (0, 2): (2, 1),
        (0, 3): (3, 1),
        (1, 0): (1, 1),
        (2, 0): (2, 1),
        (3, 0): (3, 1),
        (1, 1): (0, 1),
        (2, 2): (0, 1),
        (3, 3): (0, 1),
        (1, 2): (3, 1j),
        (2, 1): (3, -1j),
        (1, 3): (2, -1j),
        (3, 1): (2, 1j),
        (2, 3): (1, 1j),
        (3, 2): (1, -1j),
    }
    code, phase = table[(left, right)]
    return code, complex(phase)


def _generator_product(
    word: Sequence[int], axis: int, wire0: int, wire1: Optional[int]
) -> tuple[tuple[int, ...], complex]:
    output = list(word)
    phase = 1.0 + 0.0j
    for wire in (wire0,) if wire1 is None else (wire0, wire1):
        output[wire], local_phase = _local_product(axis, word[wire])
        phase *= local_phase
    return tuple(output), phase


def _is_anticommuting(phase: complex) -> bool:
    return abs(phase.imag) > 0.5


def _clifford_step(
    nqubits: int, word: tuple[int, ...], operation: tuple[object, ...]
) -> tuple[tuple[int, ...], float]:
    matrix = gate_matrix(nqubits, operation)
    transformed = matrix.conj().T @ codes_to_dense(word) @ matrix
    decomposition = pauli_decompose(transformed, nqubits)
    if len(decomposition) != 1:
        raise AssertionError("reference Clifford did not map to one Pauli word")
    result, coefficient = next(iter(decomposition.items()))
    return result, float(np.real_if_close(coefficient))


def enumerate_paths(
    nqubits: int,
    observable_word: Sequence[int],
    operations: Sequence[tuple[object, ...]],
    angles: Sequence[float],
    *,
    smoothing: float = 0.01,
) -> tuple[Path, ...]:
    """Enumerate proposal paths for one observable term without native calls."""
    resolved = []
    for operation, angle in zip(operations, angles):
        resolved.append((operation, float(angle)))

    paths = [Path(1.0, (), tuple(observable_word), 1.0)]
    for operation, angle in reversed(resolved):
        kind = str(operation[0])
        if kind not in {"rx", "ry", "rz", "rxx", "ryy", "rzz"}:
            updated = []
            for path in paths:
                word, multiplier = _clifford_step(nqubits, path.word, operation)
                updated.append(
                    Path(path.probability, path.factors, word, path.sign * multiplier)
                )
            paths = updated
            continue
        axis = {"rx": 1, "ry": 2, "rz": 3}[kind]
        wire0 = int(operation[1])
        wire1 = int(operation[2]) if kind in {"rxx", "ryy", "rzz"} else None
        next_paths: list[Path] = []
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        q = (abs(cosine) + smoothing) / (abs(cosine) + abs(sine) + 2.0 * smoothing)
        for path in paths:
            product, phase = _generator_product(path.word, axis, wire0, wire1)
            if not _is_anticommuting(phase):
                next_paths.append(path)
                continue
            next_paths.append(
                Path(
                    path.probability * q,
                    (*path.factors, (cosine, -sine)),
                    path.word,
                    path.sign,
                )
            )
            sine_sign = -1.0 if phase == 1j else 1.0
            next_paths.append(
                Path(
                    path.probability * (1.0 - q),
                    (*path.factors, (sine_sign * sine, sine_sign * cosine)),
                    product,
                    path.sign,
                )
            )
        paths = next_paths
    return tuple(paths)


def exact_value_and_gradient(
    nqubits: int,
    observable_word: Sequence[int],
    operations: Sequence[tuple[object, ...]],
    angles: Sequence[float],
    state: str = "zero",
    *,
    smoothing: float = 0.01,
) -> tuple[float, float]:
    """Sum the unweighted legal path expansion and its local PAD derivative."""
    paths = enumerate_paths(
        nqubits, observable_word, operations, angles, smoothing=smoothing
    )
    value = 0.0
    gradient = 0.0
    for path in paths:
        terminal = product_expectation_for_word(path.word, state)
        factors = [factor for factor, _ in path.factors]
        value += path.sign * terminal * np.prod(factors)
        for index, (_, derivative) in enumerate(path.factors):
            gradient += (
                path.sign
                * terminal
                * derivative
                * np.prod(factors[:index] + factors[index + 1 :])
            )
    return float(value), float(gradient)


def product_expectation_for_word(word: Sequence[int], state: str) -> float:
    if state != "zero":
        raise NotImplementedError("the focused oracle currently uses ZeroState")
    return float(
        np.prod([1.0 if code in (0, 3) else 0.0 for code in word], dtype=np.float64)
    )
