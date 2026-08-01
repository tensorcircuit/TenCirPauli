"""Local Hamiltonian target construction and apply benchmarks."""

from __future__ import annotations

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture

from tencirpauli import PauliOperator


def make_operator() -> PauliOperator:
    terms = tuple(
        (
            tuple((index + qubit) % 4 for qubit in range(10)),
            complex(index + 1, -index / 5),
        )
        for index in range(64)
    )
    return PauliOperator.from_terms(10, terms)


def test_dense_target(benchmark: BenchmarkFixture) -> None:
    operator = make_operator()
    expected = operator.dense()
    result = benchmark(operator.dense)
    np.testing.assert_allclose(result, expected)


def test_coo_and_csr_targets(benchmark: BenchmarkFixture) -> None:
    operator = make_operator()
    expected = operator.coo()
    result = benchmark(operator.coo)
    np.testing.assert_array_equal(result.row, expected.row)
    np.testing.assert_array_equal(result.column, expected.column)
    np.testing.assert_array_equal(result.data, expected.data)
    csr = operator.csr()
    assert csr.shape == expected.shape


def test_native_mvp_target(benchmark: BenchmarkFixture) -> None:
    operator = make_operator()
    state = np.arange(1 << 10, dtype=np.float64) + 1j * np.arange(1 << 10)
    expected = operator.mvp(state)
    result = benchmark(operator.mvp, state)
    np.testing.assert_allclose(result, expected)


def test_backend_plan_construction(benchmark: BenchmarkFixture) -> None:
    operator = make_operator()
    expected = operator.backend_mvp_plan()
    result = benchmark(operator.backend_mvp_plan)
    assert result.schema_version == expected.schema_version
    np.testing.assert_array_equal(result.x_words, expected.x_words)
    np.testing.assert_array_equal(result.z_words, expected.z_words)
    np.testing.assert_array_equal(result.coefficients, expected.coefficients)
