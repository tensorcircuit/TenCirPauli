"""Release benchmarks for Phase 7.5 algebra, mappings, and charge sectors."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


def _majorana_workload() -> tcp.MajoranaOperator:
    return tcp.MajoranaOperator.from_terms(
        8,
        [((2 * index, 2 * index + 1), 0.5) for index in range(7)]
        + [((2 * index + 1, 2 * index + 2), -0.25j) for index in range(7)],
    )


def _fermion_workload() -> tcp.FermionOperator:
    terms = []
    for index in range(7):
        terms.extend(
            (
                (((index, "create"), (index + 1, "annihilate")), 0.5),
                (((index + 1, "create"), (index, "annihilate")), 0.5),
            )
        )
    return tcp.FermionOperator.from_terms(8, terms)


def _charge_workload() -> tuple[tcp.ChargeSector, tcp.ChargeRestrictedOperator]:
    space = tcp.OperatorSpace(fermions=8)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(8)})
    operator = _fermion_workload()
    sector = charge.sector(4)
    return sector, operator.restrict_charge(sector)


def _record(benchmark: BenchmarkFixture, **metadata: object) -> None:
    benchmark.extra_info.update(metadata)


def test_phase75_majorana_construction(benchmark: BenchmarkFixture) -> None:
    _record(benchmark, input_terms=14, n_modes=8)
    benchmark(_majorana_workload)


def test_phase75_majorana_fermion_conversion(benchmark: BenchmarkFixture) -> None:
    operator = _majorana_workload()
    _record(benchmark, input_terms=operator.term_count, n_modes=operator.n_modes)
    benchmark(operator.to_fermion)


@pytest.mark.parametrize("name", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_phase75_mapping(benchmark: BenchmarkFixture, name: str) -> None:
    operator = _fermion_workload()
    mapping = tcp.FermionQubitMapping.from_name(name, 8)
    _record(
        benchmark,
        mapping=name,
        input_terms=operator.term_count,
        n_modes=8,
        cnot_count=len(mapping.cnot_operations),
    )
    benchmark(operator.map_fermions, mapping)


def test_phase75_sector_setup(benchmark: BenchmarkFixture) -> None:
    space = tcp.OperatorSpace(fermions=12)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(12)})
    _record(benchmark, n_modes=12, sector_value=6)
    benchmark(charge.sector, 6)


def test_phase75_restricted_setup(benchmark: BenchmarkFixture) -> None:
    sector, _ = _charge_workload()
    operator = _fermion_workload()
    _record(
        benchmark,
        n_modes=8,
        sector_dimension=sector.dimension,
        input_terms=operator.term_count,
    )
    benchmark(operator.restrict_charge, sector)


def test_phase75_restricted_apply(benchmark: BenchmarkFixture) -> None:
    sector, restricted = _charge_workload()
    state = np.arange(sector.dimension, dtype=np.complex128)
    _record(
        benchmark,
        n_modes=8,
        sector_dimension=sector.dimension,
        transitions=restricted.mvp_plan().transition_count,
    )
    benchmark(restricted.apply, state)
