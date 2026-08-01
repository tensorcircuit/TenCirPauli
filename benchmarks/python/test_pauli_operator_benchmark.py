"""Local canonicalization and operator-algebra benchmarks."""

from __future__ import annotations

from typing import Tuple

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
