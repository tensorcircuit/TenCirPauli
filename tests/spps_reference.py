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
    factors: tuple[tuple[float, float, Optional[int]], ...]
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
    parameter_slots: Sequence[Optional[int]] | None = None,
) -> tuple[Path, ...]:
    """Enumerate proposal paths for one observable term without native calls."""
    if parameter_slots is None:
        parameter_slots = (0,) * len(angles)
    if len(parameter_slots) != len(operations):
        raise ValueError("parameter_slots must match operations")
    resolved = []
    for operation, angle, slot in zip(operations, angles, parameter_slots):
        resolved.append((operation, float(angle), slot))

    paths = [Path(1.0, (), tuple(observable_word), 1.0)]
    for operation, angle, slot in reversed(resolved):
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
        axis = {
            "rx": 1,
            "ry": 2,
            "rz": 3,
            "rxx": 1,
            "ryy": 2,
            "rzz": 3,
        }[kind]
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
                    (*path.factors, (cosine, -sine, slot)),
                    path.word,
                    path.sign,
                )
            )
            sine_sign = -1.0 if phase == 1j else 1.0
            next_paths.append(
                Path(
                    path.probability * (1.0 - q),
                    (*path.factors, (sine_sign * sine, sine_sign * cosine, slot)),
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
    parameter_slots: Sequence[Optional[int]] | None = None,
) -> tuple[float, float]:
    """Sum the unweighted legal path expansion and its local PAD derivative."""
    paths = enumerate_paths(
        nqubits,
        observable_word,
        operations,
        angles,
        smoothing=smoothing,
        parameter_slots=parameter_slots,
    )
    value = 0.0
    gradient = 0.0
    for path in paths:
        terminal = product_expectation_for_word(path.word, state)
        factors = [factor for factor, _, _ in path.factors]
        value += path.sign * terminal * np.prod(factors)
        for index, (_, derivative, slot) in enumerate(path.factors):
            if slot is None:
                continue
            gradient += (
                path.sign
                * terminal
                * derivative
                * np.prod(factors[:index] + factors[index + 1 :])
            )
    return float(value), float(gradient)


def path_value_and_gradient(
    path: Path,
    state: str | Sequence[int] | np.ndarray = "zero",
    *,
    coefficient: float = 1.0,
    nparameters: int = 1,
) -> tuple[float, np.ndarray]:
    """Return one path's value contribution and slot-scattered PAD gradient."""
    terminal = product_expectation_for_word(path.word, state)
    factors = [factor for factor, _, _ in path.factors]
    value = (
        coefficient * path.sign * terminal * float(np.prod(factors, dtype=np.float64))
    )
    gradient = np.zeros(nparameters, dtype=np.float64)
    for index, (_, derivative, slot) in enumerate(path.factors):
        if slot is not None:
            gradient[slot] += (
                coefficient
                * path.sign
                * terminal
                * derivative
                * float(
                    np.prod(factors[:index] + factors[index + 1 :], dtype=np.float64)
                )
            )
    return float(value), gradient


def exact_value_and_gradient_slots(
    nqubits: int,
    observable_word: Sequence[int],
    operations: Sequence[tuple[object, ...]],
    angles: Sequence[float],
    state: str | Sequence[int] | np.ndarray = "zero",
    *,
    coefficient: float = 1.0,
    smoothing: float = 0.01,
    parameter_slots: Sequence[Optional[int]] | None = None,
) -> tuple[float, np.ndarray]:
    """Sum exact legal paths while scattering derivatives into shared slots."""
    normalized_slots = (
        tuple(parameter_slots) if parameter_slots is not None else (0,) * len(angles)
    )
    paths = enumerate_paths(
        nqubits,
        observable_word,
        operations,
        angles,
        smoothing=smoothing,
        parameter_slots=normalized_slots,
    )
    nparameters = (
        max((slot for slot in normalized_slots if slot is not None), default=-1) + 1
    )
    value = 0.0
    gradient = np.zeros(nparameters, dtype=np.float64)
    for path in paths:
        path_value, path_gradient = path_value_and_gradient(
            path,
            state,
            coefficient=coefficient,
            nparameters=nparameters,
        )
        value += path_value
        gradient += path_gradient
    return float(value), gradient


def product_expectation_for_word(
    word: Sequence[int], state: str | Sequence[int] | np.ndarray
) -> float:
    if isinstance(state, str):
        if state != "zero":
            raise ValueError(f"unsupported reference state {state!r}")
        local = [1.0 if code in (0, 3) else 0.0 for code in word]
    elif isinstance(state, np.ndarray):
        local = [
            1.0 if code == 0 else float(state[qubit, code - 1])
            for qubit, code in enumerate(word)
        ]
    else:
        bits = tuple(state)
        if len(bits) != len(word):
            raise ValueError("computational state width must match word")
        local = [
            1.0 if code == 0 else ((-1.0 if bits[qubit] else 1.0) if code == 3 else 0.0)
            for qubit, code in enumerate(word)
        ]
    return float(np.prod(local, dtype=np.float64))
