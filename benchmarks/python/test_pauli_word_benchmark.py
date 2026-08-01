"""Local integration benchmarks for the public Python API."""

from typing import Sequence

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tencirpauli import PauliWord


SIZES = (64, 1_024, 16_384)


def make_word(nqubits: int, x_pattern: int, z_pattern: int) -> PauliWord:
    """Construct a deterministic packed Pauli word."""
    nwords = (nqubits + 63) // 64
    return PauliWord(
        nqubits=nqubits,
        x_words=(x_pattern,) * nwords,
        z_words=(z_pattern,) * nwords,
    )


@pytest.mark.parametrize("nqubits", SIZES)
def test_public_api_weight(benchmark: BenchmarkFixture, nqubits: int) -> None:
    """Measure Python validation, PyO3 conversion, and the native weight kernel."""
    word = make_word(nqubits, 0xAAAA_AAAA_AAAA_AAAA, 0x1111_1111_1111_1111)
    expected = word.weight
    result = benchmark(lambda: word.weight)
    assert result == expected


@pytest.mark.parametrize("nqubits", SIZES)
def test_public_api_commutation(benchmark: BenchmarkFixture, nqubits: int) -> None:
    """Measure the complete public commutation call path."""
    left = make_word(nqubits, 0xAAAA_AAAA_AAAA_AAAA, 0x1111_1111_1111_1111)
    right = make_word(nqubits, 0xCCCC_CCCC_CCCC_CCCC, 0x0101_0101_0101_0101)
    expected = left.commutes_with(right)
    result = benchmark(left.commutes_with, right)
    assert result == expected


def evaluate_workload(words: Sequence[PauliWord]) -> int:
    """Run a small end-to-end workload through repeated public API calls."""
    total = 0
    for index, word in enumerate(words):
        total += word.weight
        total += int(word.commutes_with(words[(index + 1) % len(words)]))
    return total


def test_public_api_workload(benchmark: BenchmarkFixture) -> None:
    """Track aggregate wrapper and FFI cost until a batched API is available."""
    words = tuple(
        make_word(
            256,
            0xAAAA_AAAA_AAAA_AAAA ^ index,
            0x1111_1111_1111_1111 ^ (index << 1),
        )
        for index in range(256)
    )
    expected = evaluate_workload(words)
    result = benchmark(evaluate_workload, words)
    assert result == expected
