"""Shared workloads and an independent dict implementation for Lie closure."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Codes = Tuple[int, ...]
SparseOperator = Dict[str, complex]


_CHAR_TO_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_LOCAL_PRODUCTS = (
    ((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)),
    ((1, 1.0), (0, 1.0), (3, 1.0j), (2, -1.0j)),
    ((2, 1.0), (3, -1.0j), (0, 1.0), (1, 1.0j)),
    ((3, 1.0), (2, 1.0j), (1, -1.0j), (0, 1.0)),
)


def word_codes(word: str) -> Codes:
    """Convert a public Pauli string into independent integer codes."""
    return tuple(_CHAR_TO_CODE[character] for character in word)


GeneratorSpec = Tuple[Tuple[Tuple[str, complex], ...], ...]


def generator_terms(case: str, nqubits: int) -> Tuple[GeneratorSpec, int]:
    """Return anti-Hermitian generators and their effective width."""
    if case == "su2":
        return ((("X", -1.0j),), (("Z", -1.0j),)), 1
    if case == "su4":
        return (
            (("XI", -1.0j),),
            (("ZI", -1.0j),),
            (("IX", -1.0j),),
            (("IZ", -1.0j),),
            (("XX", -1.0j),),
        ), 2
    if case == "sum2":
        return (
            (("XI", -1.0j), ("IX", -0.75j)),
            (("ZZ", -1.0j), ("XX", -0.5j)),
            (("YI", -0.6j), ("IY", -0.4j)),
        ), 2
    if case == "chain":
        if nqubits < 2:
            raise ValueError("chain requires at least two qubits")
        terms: List[Tuple[str, complex]] = []
        for qubit in range(nqubits):
            x = ["I"] * nqubits
            z = ["I"] * nqubits
            x[qubit] = "X"
            z[qubit] = "Z"
            terms.extend((("".join(x), -1.0j), ("".join(z), -1.0j)))
        for qubit in range(nqubits - 1):
            xx = ["I"] * nqubits
            xx[qubit] = "X"
            xx[qubit + 1] = "X"
            terms.append(("".join(xx), -1.0j))
        return tuple(((term,) for term in terms)), nqubits
    raise ValueError(f"unknown closure case {case!r}")


def from_terms(terms: Iterable[Tuple[str, complex]]) -> SparseOperator:
    """Aggregate string terms into the independent dict representation."""
    result: SparseOperator = {}
    for word, coefficient in terms:
        result[word] = result.get(word, 0.0j) + coefficient
    return {word: value for word, value in result.items() if value != 0.0j}


def add(left: SparseOperator, right: SparseOperator) -> SparseOperator:
    """Add two independent sparse operators."""
    result = dict(left)
    for word, coefficient in right.items():
        result[word] = result.get(word, 0.0j) + coefficient
    return {word: value for word, value in result.items() if value != 0.0j}


def scale(operator: SparseOperator, scalar: complex) -> SparseOperator:
    """Scale an independent sparse operator."""
    return {
        word: scalar * coefficient
        for word, coefficient in operator.items()
        if scalar * coefficient != 0.0j
    }


def multiply_words(left: str, right: str) -> Tuple[str, complex]:
    """Multiply two Pauli strings with the exact local phase table."""
    if len(left) != len(right):
        raise ValueError("Pauli words must have equal widths")
    result: List[str] = []
    phase = 1.0 + 0.0j
    for left_code, right_code in zip(word_codes(left), word_codes(right)):
        code, local_phase = _LOCAL_PRODUCTS[left_code][right_code]
        result.append("IXYZ"[code])
        phase *= local_phase
    return "".join(result), phase


def multiply(left: SparseOperator, right: SparseOperator) -> SparseOperator:
    """Multiply sparse Pauli sums and aggregate equal words."""
    result: SparseOperator = {}
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            word, phase = multiply_words(left_word, right_word)
            result[word] = (
                result.get(word, 0.0j) + left_coefficient * right_coefficient * phase
            )
    return {word: value for word, value in result.items() if value != 0.0j}


def commutator(left: SparseOperator, right: SparseOperator) -> SparseOperator:
    """Return the sparse commutator."""
    return add(multiply(left, right), scale(multiply(right, left), -1.0))


def operator_norm(operator: SparseOperator) -> float:
    """Return the maximum absolute coefficient of a sparse operator."""
    return max((abs(value) for value in operator.values()), default=0.0)


def coordinate_matrix(
    basis: Sequence[SparseOperator], candidate: SparseOperator
) -> np.ndarray:
    """Build real coordinates in the anti-Hermitian ``-iP`` basis."""
    words = sorted({word for operator in (*basis, candidate) for word in operator})
    matrix = np.zeros((len(words), len(basis) + 1), dtype=np.float64)
    for column, operator in enumerate((*basis, candidate)):
        for row, word in enumerate(words):
            coefficient = 1.0j * operator.get(word, 0.0j)
            if abs(coefficient.imag) > 1.0e-10:
                raise ValueError("closure operators must be anti-Hermitian")
            matrix[row, column] = coefficient.real
    return matrix


def independent(
    basis: Sequence[SparseOperator], candidate: SparseOperator, tolerance: float
) -> bool:
    """Return whether a candidate increases the real anti-Hermitian rank."""
    if not candidate or operator_norm(candidate) <= tolerance:
        return False
    matrix = coordinate_matrix(basis, candidate)
    before = np.linalg.matrix_rank(matrix[:, :-1], tol=tolerance) if basis else 0
    after = np.linalg.matrix_rank(matrix, tol=tolerance)
    return after > before


def closure(
    generators: Sequence[SparseOperator],
    mode: str,
    max_dimension: int,
    tolerance: float,
) -> Tuple[List[SparseOperator], int, bool, float]:
    """Compute a deterministic bounded Lie closure with a Jacobi check."""
    basis: List[SparseOperator] = []
    words: set[str] = set()
    for generator in generators:
        if mode == "word":
            word = next(iter(generator))
            if word not in words:
                words.add(word)
                basis.append(generator)
        elif independent(basis, generator, tolerance):
            basis.append(generator)
        else:
            continue
        if len(basis) >= max_dimension:
            return basis, 0, False, 0.0

    candidate_count = 0
    index = 0
    while index < len(basis):
        left = basis[index]
        for right in tuple(basis):
            candidate_count += 1
            candidate = commutator(left, right)
            if mode == "word":
                if not candidate:
                    continue
                if len(candidate) != 1:
                    raise AssertionError("word closure produced a non-word bracket")
                word = next(iter(candidate))
                if word in words:
                    continue
                words.add(word)
                basis.append({word: -1.0j})
            elif independent(basis, candidate, tolerance):
                basis.append(candidate)
            else:
                continue
            if len(basis) >= max_dimension:
                return basis, candidate_count, False, 0.0
        index += 1

    jacobi = 0.0
    if len(generators) >= 3:
        first, second, third = generators[:3]
        jacobi_operator = add(
            add(
                commutator(first, commutator(second, third)),
                commutator(second, commutator(third, first)),
            ),
            commutator(third, commutator(first, second)),
        )
        jacobi = operator_norm(jacobi_operator)
    return basis, candidate_count, True, jacobi
