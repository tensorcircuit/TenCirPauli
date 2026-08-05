"""Release benchmarks for Phase 9 handle-native producer boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


def _pauli_workload() -> tcp.PauliOperator:
    structures = np.asarray(
        [[(index // (3**qubit)) % 4 for qubit in range(10)] for index in range(128)],
        dtype=np.uint8,
    )
    return tcp.PauliOperator.from_code_arrays(structures, np.ones(128))


def test_flat_pauli_construction(benchmark: BenchmarkFixture) -> None:
    structures = np.asarray(
        [[(index // (3**qubit)) % 4 for qubit in range(10)] for index in range(128)],
        dtype=np.uint8,
    )
    coefficients = np.ones(128, dtype=np.complex128)
    result = benchmark(tcp.PauliOperator.from_code_arrays, structures, coefficients)
    benchmark.extra_info["input_term_count"] = 128
    benchmark.extra_info["nqubits"] = 10
    assert result.term_count > 0


def test_handle_native_mapping(benchmark: BenchmarkFixture) -> None:
    operator = tcp.FermionOperator.from_terms(
        8,
        tuple(
            (((mode, "create"), ((mode + 1) % 8, "annihilate")), 0.5)
            for mode in range(8)
        ),
    )
    result = benchmark(operator.map_fermions, "parity")
    benchmark.extra_info["input_term_count"] = operator.term_count
    benchmark.extra_info["output_term_count"] = result.term_count
    assert result.term_count > 0


def test_handle_native_majorana_conversion(benchmark: BenchmarkFixture) -> None:
    operator = tcp.FermionOperator.from_terms(
        6,
        tuple((((mode, "create"),), 0.25) for mode in range(6)),
    )
    result = benchmark(operator.to_majorana)
    benchmark.extra_info["input_term_count"] = operator.term_count
    benchmark.extra_info["output_term_count"] = result.term_count
    assert result._native_handle is not None


@pytest.mark.parametrize("target", ("dense", "coo", "csr", "native_mvp"))
def test_handle_native_terminal_compilation(
    benchmark: BenchmarkFixture, target: str
) -> None:
    operator = _pauli_workload()
    result = benchmark(operator.compile, target)
    benchmark.extra_info["input_term_count"] = operator.term_count
    benchmark.extra_info["nqubits"] = operator.nqubits
    assert result is not None
