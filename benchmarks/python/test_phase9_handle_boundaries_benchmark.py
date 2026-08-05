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
    benchmark.extra_info["output_term_count"] = result.term_count
    benchmark.extra_info["materialized"] = False
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
    benchmark.extra_info["materialized"] = False
    assert result.term_count > 0


def test_handle_native_majorana_conversion(benchmark: BenchmarkFixture) -> None:
    operator = tcp.FermionOperator.from_terms(
        6,
        tuple((((mode, "create"),), 0.25) for mode in range(6)),
    )
    result = benchmark(operator.to_majorana)
    benchmark.extra_info["input_term_count"] = operator.term_count
    benchmark.extra_info["output_term_count"] = result.term_count
    benchmark.extra_info["materialized"] = False
    assert result._native_handle is not None


def test_handle_native_grouping(benchmark: BenchmarkFixture) -> None:
    operator = _pauli_workload()
    result = benchmark(operator.group_commuting)
    benchmark.extra_info["input_term_count"] = operator.term_count
    benchmark.extra_info["output_group_count"] = result.group_count
    benchmark.extra_info["materialized"] = True
    assert result.term_count == operator.term_count


def test_handle_native_u1_restriction(benchmark: BenchmarkFixture) -> None:
    terms = []
    for index in range(1, 12):
        terms.extend(
            (
                (f"X{'I' * (index - 1)}X{'I' * (11 - index)}", 0.5),
                (f"Y{'I' * (index - 1)}Y{'I' * (11 - index)}", 0.5),
            )
        )
    operator = tcp.PauliOperator.from_terms(12, terms)
    result = benchmark(operator.restrict_charge, tcp.U1Sector(12, 2))
    benchmark.extra_info["input_term_count"] = operator.term_count
    benchmark.extra_info["output_dimension"] = result.dimension
    benchmark.extra_info["materialized"] = False
    assert result.dimension == 66


@pytest.mark.parametrize("target", ("dense", "coo", "csr", "native_mvp"))
def test_handle_native_terminal_compilation(
    benchmark: BenchmarkFixture, target: str
) -> None:
    operator = _pauli_workload()
    result = benchmark(operator.compile, target)
    benchmark.extra_info["input_term_count"] = operator.term_count
    benchmark.extra_info["nqubits"] = operator.nqubits
    benchmark.extra_info["output_dimension"] = 1 << operator.nqubits
    benchmark.extra_info["materialized"] = target != "native_mvp"
    assert result is not None
