"""Benchmarks for native-backed Fermion and Boson BCH algebra."""

from __future__ import annotations

import random
from typing import Any, Tuple

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tencirpauli import BosonOperator, FermionOperator


Operator = Any


def make_workload(family: str) -> Tuple[Operator, Operator]:
    rng = random.Random(20260805 + (0 if family == "fermion" else 1))

    def terms() -> tuple:
        return tuple(
            (
                tuple(
                    (
                        rng.randrange(4),
                        "create" if rng.randrange(2) == 0 else "annihilate",
                    )
                    for _ in range(2 + index % 2)
                ),
                complex(0.2 + 0.03 * index, 0.01 * (index % 3)),
            )
            for index in range(8)
        )

    left, right = terms(), terms()
    constructor = FermionOperator if family == "fermion" else BosonOperator
    return constructor.from_terms(4, left), constructor.from_terms(4, right)


def lazy_bch(left: Operator, right: Operator) -> Tuple[Operator, ...]:
    ab = left.commutator(right)
    order_one = left.add(right)
    order_two = order_one.add(ab.scale(0.5))
    aa = left.commutator(ab)
    bb = right.commutator(ab.scale(-1.0))
    order_three = order_two.add(aa.add(bb).scale(1.0 / 12.0))
    fourth = right.commutator(aa)
    return order_one, order_two, order_three, order_three.add(fourth.scale(-1.0 / 24.0))


def eager_materialized_bch(left: Operator, right: Operator) -> Tuple[Operator, ...]:
    """Use the same algebra while explicitly materializing every result node."""

    def materialize(operator: Operator) -> Operator:
        _ = operator.terms
        return operator

    ab = materialize(left.commutator(right))
    order_one = materialize(left.add(right))
    order_two = materialize(order_one.add(materialize(ab.scale(0.5))))
    aa = materialize(left.commutator(ab))
    bb = materialize(right.commutator(materialize(ab.scale(-1.0))))
    order_three = materialize(order_two.add(materialize(aa.add(bb).scale(1.0 / 12.0))))
    fourth = materialize(right.commutator(aa))
    return (
        order_one,
        order_two,
        order_three,
        materialize(order_three.add(materialize(fourth.scale(-1.0 / 24.0)))),
    )


@pytest.mark.parametrize("family", ("fermion", "boson"))
def test_native_structured_bch(benchmark: BenchmarkFixture, family: str) -> None:
    left, right = make_workload(family)
    result = benchmark(lazy_bch, left, right)
    assert all(operator._terms is None for operator in result)
    assert result[-1].term_count > 0


@pytest.mark.parametrize("family", ("fermion", "boson"))
def test_native_structured_term_materialization(
    benchmark: BenchmarkFixture, family: str
) -> None:
    left, right = make_workload(family)

    def run() -> Tuple[Tuple[object, ...], ...]:
        return tuple(tuple(operator.terms) for operator in lazy_bch(left, right))

    result = benchmark(run)
    assert result[-1]


@pytest.mark.parametrize("family", ("fermion", "boson"))
def test_eager_materialized_structured_bch(
    benchmark: BenchmarkFixture, family: str
) -> None:
    left, right = make_workload(family)
    result = benchmark(eager_materialized_bch, left, right)
    benchmark.extra_info["input_term_count"] = left.term_count + right.term_count
    benchmark.extra_info["output_term_count"] = result[-1].term_count
    assert result[-1].terms
