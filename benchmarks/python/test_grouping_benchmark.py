"""Local deterministic grouping benchmarks."""

from __future__ import annotations

import numpy as np
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


def python_qwc_largest_first(
    structures: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Match the deterministic native coloring with conventional Python lists."""
    size = len(structures)
    incompatible = [[False] * size for _ in range(size)]
    for left in range(size):
        for right in range(left + 1, size):
            compatible = all(
                a == 0 or b == 0 or a == b
                for a, b in zip(structures[left], structures[right])
            )
            incompatible[left][right] = not compatible
            incompatible[right][left] = not compatible
    degrees = [sum(row) for row in incompatible]
    groups: list[list[int]] = []
    for vertex in sorted(range(size), key=lambda index: (-degrees[index], index)):
        for group in groups:
            if all(not incompatible[vertex][other] for other in group):
                group.append(vertex)
                break
        else:
            groups.append([vertex])
    result = [tuple(sorted(group)) for group in groups]
    return tuple(sorted(result, key=lambda group: group[0]))


@pytest.mark.parametrize("count", (128, 1_024))
def test_public_qwc_grouping(benchmark: BenchmarkFixture, count: int) -> None:
    """Measure the complete native graph coloring and result conversion path."""
    operator = make_operator(count)
    expected = operator.group_commuting()
    result = benchmark(operator.group_commuting)
    benchmark.extra_info["nqubits"] = operator.nqubits
    benchmark.extra_info["support_density"] = float(
        np.count_nonzero(operator._arrays()[0])
        / (operator.term_count * operator.nqubits)
    )
    benchmark.extra_info["group_count"] = result.group_count
    assert result.groups == expected.groups
    assert result.bases == expected.bases


@pytest.mark.parametrize("count", (128, 1_024))
def test_python_qwc_grouping_baseline(benchmark: BenchmarkFixture, count: int) -> None:
    """Measure the same coloring contract through conventional Python lists."""
    operator = make_operator(count)
    structures = operator._arrays()[0]
    expected = operator.group_commuting().groups
    result = benchmark(python_qwc_largest_first, structures)
    assert result == expected


def _reconstruction_workload() -> tuple[PauliOperator, object, np.ndarray]:
    nqubits, term_count, shots = 64, 96, 2_000
    codes = np.zeros((term_count, nqubits), dtype=np.uint8)
    for term_index in range(term_count):
        for qubit in range(nqubits):
            if (term_index * 17 + qubit * 5) % 11 < 3:
                codes[term_index, qubit] = 3
        codes[term_index, term_index % nqubits] = 3
    operator = PauliOperator.from_code_arrays(codes, np.ones(term_count))
    grouping = operator.group_commuting()
    bits = np.random.default_rng(20260806).integers(
        0, 2, size=(shots, nqubits), dtype=np.int8
    )
    return operator, grouping, bits


def test_public_qwc_reconstruction(benchmark: BenchmarkFixture) -> None:
    operator, grouping, bits = _reconstruction_workload()
    result = benchmark(grouping.reconstruct, 0, bits)
    support_density = float(
        np.count_nonzero(operator._arrays()[0])
        / (operator.term_count * operator.nqubits)
    )
    benchmark.extra_info.update(
        {
            "nqubits": operator.nqubits,
            "shots": bits.shape[0],
            "group_size": len(grouping.groups[0]),
            "support_density": support_density,
            "group_count": grouping.group_count,
            "output_bytes": result.nbytes,
        }
    )
    assert result.shape == (bits.shape[0], len(grouping.groups[0]))
