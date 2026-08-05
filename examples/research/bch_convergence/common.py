"""Shared workloads and an independent Python Pauli reference for BCH."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple


Codes = Tuple[int, ...]
Term = Tuple[Codes, complex]
SparseOperator = Dict[Codes, complex]


_LOCAL_PRODUCTS = (
    ((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)),
    ((1, 1.0), (0, 1.0), (3, 1.0j), (2, -1.0j)),
    ((2, 1.0), (3, -1.0j), (0, 1.0), (1, 1.0j)),
    ((3, 1.0), (2, 1.0j), (1, -1.0j), (0, 1.0)),
)


def make_terms(nqubits: int, count: int, seed: int) -> Tuple[Term, ...]:
    """Return deterministic, distinct full-width Hermitian Pauli terms."""
    if nqubits < 1:
        raise ValueError("nqubits must be positive")
    if count < 1:
        raise ValueError("count must be positive")
    generator = random.Random(seed)
    terms: List[Term] = []
    seen: set[Codes] = set()
    while len(terms) < count:
        word = tuple(generator.randrange(4) for _ in range(nqubits))
        if not any(word) or word in seen:
            continue
        seen.add(word)
        coefficient = 0.25 + 0.03125 * ((len(terms) * 7 + seed) % 17)
        terms.append((word, complex(coefficient)))
    return tuple(terms)


def scaled_terms(terms: Iterable[Term], scalar: complex) -> Tuple[Term, ...]:
    """Scale terms while preserving their deterministic order."""
    return tuple((word, scalar * coefficient) for word, coefficient in terms)


def multiply_words(left: Codes, right: Codes) -> Tuple[Codes, complex]:
    """Multiply two phase-free Pauli words using the local product table."""
    if len(left) != len(right):
        raise ValueError("Pauli words must have equal widths")
    result: List[int] = []
    phase = 1.0 + 0.0j
    for left_code, right_code in zip(left, right):
        code, local_phase = _LOCAL_PRODUCTS[left_code][right_code]
        result.append(code)
        phase *= local_phase
    return tuple(result), phase


def from_terms(terms: Iterable[Term]) -> SparseOperator:
    """Aggregate a term sequence into the independent dict reference."""
    result: SparseOperator = {}
    for word, coefficient in terms:
        result[word] = result.get(word, 0.0j) + coefficient
    return {word: value for word, value in result.items() if value != 0.0j}


def add(left: SparseOperator, right: SparseOperator) -> SparseOperator:
    """Add and aggregate two independent sparse operators."""
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


def multiply(left: SparseOperator, right: SparseOperator) -> SparseOperator:
    """Multiply sparse operators and aggregate duplicate Pauli words."""
    result: SparseOperator = {}
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            word, phase = multiply_words(left_word, right_word)
            result[word] = (
                result.get(word, 0.0j) + left_coefficient * right_coefficient * phase
            )
    return {word: value for word, value in result.items() if value != 0.0j}


def commutator(left: SparseOperator, right: SparseOperator) -> SparseOperator:
    """Return the independent sparse commutator ``left*right-right*left``."""
    return add(multiply(left, right), scale(multiply(right, left), -1.0))


def bch_series(
    operator_a: SparseOperator, operator_b: SparseOperator
) -> Tuple[SparseOperator, ...]:
    """Return fixed-order BCH truncations through total degree four."""
    ab = commutator(operator_a, operator_b)
    order_one = add(operator_a, operator_b)
    order_two = add(order_one, scale(ab, 0.5))
    aa = commutator(operator_a, ab)
    bb = commutator(operator_b, scale(ab, -1.0))
    order_three = add(order_two, scale(add(aa, bb), 1.0 / 12.0))
    fourth = commutator(operator_b, aa)
    order_four = add(order_three, scale(fourth, -1.0 / 24.0))
    return order_one, order_two, order_three, order_four
