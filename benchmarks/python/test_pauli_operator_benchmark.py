"""Local canonicalization and operator-algebra benchmarks."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tencirpauli import PauliOperator


def make_terms(count: int) -> Tuple[Tuple[Tuple[int, ...], complex], ...]:
    """Make a deterministic duplicate-heavy eight-qubit workload."""
    return tuple(
        (
            tuple((index + qubit) % 4 for qubit in range(8)),
            complex((index % 7) - 3, (index % 5) - 2),
        )
        for index in range(count)
    )


@pytest.mark.parametrize("count", (1_000, 10_000, 100_000))
def test_public_operator_canonicalization(
    benchmark: BenchmarkFixture,
    count: int,
) -> None:
    """Measure input conversion, one native canonicalization, and output conversion."""
    terms = make_terms(count)
    expected = PauliOperator.from_terms(8, terms)
    result = benchmark(PauliOperator.from_terms, 8, terms)
    assert result == expected


def make_array_terms(count: int, nqubits: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic high-cardinality code-array workload."""
    indices = np.arange(count, dtype=np.uint64)[:, None]
    qubits = np.arange(nqubits, dtype=np.uint64)[None, :]
    structures = ((indices >> (2 * (qubits % 16))) + qubits) % 4
    coefficients = (indices[:, 0] % 7).astype(np.float64) - 3.0
    return structures.astype(np.uint8), coefficients.astype(np.complex128)


def python_code_tuple_canonicalization(
    structures: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize the same arrays through conventional Python tuple keys."""
    aggregate: dict[tuple[int, ...], list[tuple[int, complex]]] = {}
    for index, (structure, coefficient) in enumerate(zip(structures, coefficients)):
        key = tuple(int(code) for code in structure)
        aggregate.setdefault(key, []).append((index, complex(coefficient)))
    ordered = sorted(aggregate)
    mapping = np.empty(len(coefficients), dtype=np.uintp)
    values = np.empty(len(ordered), dtype=np.complex128)
    for canonical_index, key in enumerate(ordered):
        contributions = aggregate[key]
        for input_index, _ in contributions:
            mapping[input_index] = canonical_index
        values[canonical_index] = sum(
            (coefficient for _, coefficient in contributions), 0.0j
        )
    return np.asarray(ordered, dtype=np.uint8), values, mapping


@pytest.mark.parametrize("count", (10_000, 100_000))
def test_public_code_array_plan_canonicalization(
    benchmark: BenchmarkFixture,
    count: int,
) -> None:
    """Measure contiguous backend-plan canonicalization without Python terms."""
    structures, coefficients = make_array_terms(count)
    expected = PauliOperator.canonicalize_code_arrays_numpy(structures, coefficients)
    result = benchmark(
        PauliOperator.canonicalize_code_arrays_numpy, structures, coefficients
    )
    np.testing.assert_array_equal(
        result.canonical_structures, expected.canonical_structures
    )
    np.testing.assert_array_equal(result.coefficients, expected.coefficients)
    np.testing.assert_array_equal(
        result.input_to_canonical, expected.input_to_canonical
    )


@pytest.mark.parametrize("count", (10_000, 100_000))
def test_python_code_tuple_canonicalization_baseline(
    benchmark: BenchmarkFixture,
    count: int,
) -> None:
    """Measure a matched conventional Python tuple/dict implementation."""
    structures, coefficients = make_array_terms(count)
    expected = PauliOperator.canonicalize_code_arrays_numpy(structures, coefficients)
    canonical_structures, values, mapping = benchmark(
        python_code_tuple_canonicalization, structures, coefficients
    )
    np.testing.assert_array_equal(canonical_structures, expected.canonical_structures)
    np.testing.assert_array_equal(values, expected.coefficients)
    np.testing.assert_array_equal(mapping, expected.input_to_canonical)
