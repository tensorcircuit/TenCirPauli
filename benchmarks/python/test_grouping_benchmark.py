"""Local deterministic grouping benchmarks."""

from __future__ import annotations

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tencirpauli import PauliOperator


def make_operator(count: int) -> PauliOperator:
    terms = tuple(
        (
            tuple((index // (qubit + 1)) % 4 for qubit in range(8)),
            1.0,
        )
        for index in range(count)
    )
    return PauliOperator.from_terms(8, terms)


@pytest.mark.parametrize("count", (128, 1_024))
def test_public_qwc_grouping(benchmark: BenchmarkFixture, count: int) -> None:
    """Measure the complete native graph coloring and result conversion path."""
    operator = make_operator(count)
    expected = operator.group_commuting()
    result = benchmark(operator.group_commuting)
    assert result == expected
