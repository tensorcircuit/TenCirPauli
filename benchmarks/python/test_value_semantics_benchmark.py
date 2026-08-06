"""Release benchmarks for native equality and hashing on large values."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


def _workload(family: str) -> tuple[Any, Any]:
    if family == "pauli":
        codes = np.asarray(
            [
                [(index // (4**qubit)) % 4 for qubit in range(16)]
                for index in range(512)
            ],
            dtype=np.uint8,
        )
        left = tcp.PauliOperator.from_code_arrays(codes, np.ones(len(codes)))
        right = tcp.PauliOperator.from_code_arrays(codes, np.ones(len(codes)))
        return left, right
    if family == "majorana":
        terms = [((index, index + 1), 0.5 + 0.25j) for index in range(0, 128, 2)]
        return (
            tcp.MajoranaOperator.from_terms(128, terms),
            tcp.MajoranaOperator.from_terms(128, terms),
        )
    if family == "fermion":
        terms = [
            (((index, "create"), ((index + 1) % 128, "annihilate")), 0.5 + 0.25j)
            for index in range(128)
        ]
        return (
            tcp.FermionOperator.from_terms(128, terms),
            tcp.FermionOperator.from_terms(128, terms),
        )
    if family == "boson":
        terms = [
            (((index, "create"), ((index + 1) % 32, "annihilate")), 0.5 + 0.25j)
            for index in range(32)
        ]
        return (
            tcp.BosonOperator.from_terms(32, terms),
            tcp.BosonOperator.from_terms(32, terms),
        )
    space = tcp.OperatorSpace(qubits=64)
    terms = [space.qubit.z(index) * (0.5 + 0.25j) for index in range(64)]
    left = terms[0]
    right = terms[0]
    for term in terms[1:]:
        left = left + term
        right = right + term
    return left, right


@pytest.mark.parametrize("family", ("pauli", "majorana", "fermion", "boson", "hybrid"))
def test_large_native_equality_and_hash(
    benchmark: BenchmarkFixture, family: str
) -> None:
    left, right = _workload(family)
    result = benchmark(lambda: (left == right, hash(left), hash(right)))
    assert result[0]
    assert result[1] == result[2]
    benchmark.extra_info.update(
        {
            "family": family,
            "left_term_count": left.term_count,
            "right_term_count": right.term_count,
            "materialized": False,
        }
    )
