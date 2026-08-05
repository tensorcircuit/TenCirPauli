"""Benchmarks for native-backed Pauli algebra and lazy term materialization."""

from __future__ import annotations

import random
from typing import Tuple

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tencirpauli import PauliOperator


Workload = Tuple[PauliOperator, PauliOperator]


def make_workload(nqubits: int, terms: int) -> Workload:
    """Build deterministic high-cardinality BCH generators outside timing."""
    rng = random.Random(20260805 + nqubits * 101 + terms)

    def generator(real_period: int, imaginary_period: int) -> tuple:
        return tuple(
            (
                tuple(rng.randrange(4) for _ in range(nqubits)),
                complex(
                    (index % real_period) - real_period // 2,
                    (index % imaginary_period) - imaginary_period // 2,
                ),
            )
            for index in range(terms)
        )

    left = generator(7, 5)
    right = generator(11, 3)
    return PauliOperator.from_terms(nqubits, left), PauliOperator.from_terms(
        nqubits, right
    )


def eager_bch(left: PauliOperator, right: PauliOperator) -> Tuple[PauliOperator, ...]:
    """Build fourth-order BCH truncations with eager Python term results."""

    def eager(result: PauliOperator) -> PauliOperator:
        _ = result.terms
        return result

    ab = eager(left.commutator(right))
    order_one = eager(left.add(right))
    order_two = eager(order_one.add(eager(ab.scale(0.5))))
    aa = eager(left.commutator(ab))
    bb = eager(right.commutator(eager(ab.scale(-1.0))))
    order_three = eager(order_two.add(eager(aa.add(bb).scale(1.0 / 12.0))))
    fourth = eager(right.commutator(aa))
    order_four = eager(order_three.add(eager(fourth.scale(-1.0 / 24.0))))
    return order_one, order_two, order_three, order_four


def lazy_bch(left: PauliOperator, right: PauliOperator) -> Tuple[PauliOperator, ...]:
    """Build fourth-order BCH truncations with native-backed results."""
    ab = left.commutator(right)
    order_one = left.add(right)
    order_two = order_one.add(ab.scale(0.5))
    aa = left.commutator(ab)
    bb = right.commutator(ab.scale(-1.0))
    order_three = order_two.add(aa.add(bb).scale(1.0 / 12.0))
    fourth = right.commutator(aa)
    order_four = order_three.add(fourth.scale(-1.0 / 24.0))
    return order_one, order_two, order_three, order_four


@pytest.mark.parametrize("nqubits,terms", ((8, 16), (16, 32)))
def test_eager_pauli_bch(benchmark: BenchmarkFixture, nqubits: int, terms: int) -> None:
    """Measure BCH with ordinary Python term materialization at every node."""
    left, right = make_workload(nqubits, terms)
    result = benchmark(eager_bch, left, right)
    assert all(isinstance(operator.terms, tuple) for operator in result)


@pytest.mark.parametrize("nqubits,terms", ((8, 16), (16, 32)))
def test_native_backed_pauli_bch(
    benchmark: BenchmarkFixture, nqubits: int, terms: int
) -> None:
    """Measure BCH while retaining only native-backed PauliOperator shells."""
    left, right = make_workload(nqubits, terms)
    result = benchmark(lazy_bch, left, right)
    assert all(operator._terms is None for operator in result)
    assert result[-1].term_count > 0


@pytest.mark.parametrize("nqubits,terms", ((8, 16), (16, 32)))
def test_native_backed_pauli_plain_export(
    benchmark: BenchmarkFixture, nqubits: int, terms: int
) -> None:
    """Measure native-backed BCH plus plain string/weight export."""
    left, right = make_workload(nqubits, terms)

    def run() -> Tuple[dict[str, complex], ...]:
        return tuple(operator.to_dict() for operator in lazy_bch(left, right))

    result = benchmark(run)
    assert result[-1]


@pytest.mark.parametrize("nqubits,terms", ((8, 16), (16, 32)))
def test_native_backed_pauli_term_materialization(
    benchmark: BenchmarkFixture, nqubits: int, terms: int
) -> None:
    """Measure the explicitly requested Python term-object materialization."""
    left, right = make_workload(nqubits, terms)

    def run() -> Tuple[Tuple[object, ...], ...]:
        return tuple(tuple(operator.terms) for operator in lazy_bch(left, right))

    result = benchmark(run)
    assert result[-1]
