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
    metadata.setdefault("numerical_error", 0.0)
    benchmark.extra_info.update(metadata)


def test_phase75_majorana_construction(benchmark: BenchmarkFixture) -> None:
    _record(benchmark, input_terms=14, n_modes=8)
    benchmark(_majorana_workload)


def test_phase75_majorana_fermion_conversion(benchmark: BenchmarkFixture) -> None:
    operator = _majorana_workload()
    _record(benchmark, input_terms=operator.term_count, n_modes=operator.n_modes)
    benchmark(operator.to_fermion)


def test_phase75_majorana_multiplication(benchmark: BenchmarkFixture) -> None:
    left = _majorana_workload()
    right = left.adjoint()
    _record(benchmark, input_terms=left.term_count * right.term_count, n_modes=8)
    benchmark(left.multiply, right)


@pytest.mark.parametrize("name", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_phase75_mapping_plan_construction(
    benchmark: BenchmarkFixture, name: str
) -> None:
    _record(benchmark, mapping=name, n_modes=8)
    benchmark(tcp.FermionQubitMapping.from_name, name, 8)


@pytest.mark.parametrize("name", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_phase75_mapping(benchmark: BenchmarkFixture, name: str) -> None:
    operator = _fermion_workload()
    mapping = tcp.FermionQubitMapping.from_name(name, 8)
    _record(
        benchmark,
        mapping=name,
        input_terms=len(operator.terms),
        n_modes=8,
        cnot_count=len(mapping.cnot_operations),
        plan_bytes=mapping.estimated_bytes,
    )
    benchmark(operator.map_fermions, mapping)


def test_phase75_sector_setup(benchmark: BenchmarkFixture) -> None:
    space = tcp.OperatorSpace(fermions=12)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(12)})
    _record(benchmark, n_modes=12, sector_value=6, plan_bytes=0)
    benchmark(charge.sector, 6)


def test_phase75_restricted_setup(benchmark: BenchmarkFixture) -> None:
    sector, _ = _charge_workload()
    operator = _fermion_workload()
    _record(
        benchmark,
        n_modes=8,
        sector_dimension=sector.dimension,
        input_terms=len(operator.terms),
        plan_bytes=sector.estimated_bytes,
    )
    benchmark(operator.restrict_charge, sector)


def test_phase75_restricted_apply(benchmark: BenchmarkFixture) -> None:
    sector, restricted = _charge_workload()
    state = np.arange(sector.dimension, dtype=np.complex128)
    plan = restricted.mvp_plan()
    _record(
        benchmark,
        n_modes=8,
        sector_dimension=sector.dimension,
        transitions=plan.transition_count,
        plan_bytes=plan.estimated_bytes,
        workspace_bytes=sector.dimension * max(len(sector.local_dimensions), 1) * 8,
        output_bytes=sector.dimension * 16,
    )
    benchmark(restricted.apply, state)


def test_phase75_restricted_first_apply(benchmark: BenchmarkFixture) -> None:
    sector, _ = _charge_workload()
    operator = _fermion_workload()
    state = np.arange(sector.dimension, dtype=np.complex128)
    _record(
        benchmark,
        n_modes=8,
        sector_dimension=sector.dimension,
        input_terms=operator.term_count,
        first_apply=True,
        workspace_bytes=sector.dimension * max(len(sector.local_dimensions), 1) * 8,
        output_bytes=sector.dimension * 16,
    )

    def build_and_apply() -> np.ndarray:
        return operator.restrict_charge(sector).apply(state)

    benchmark(build_and_apply)


@pytest.mark.parametrize("target", ["dense", "coo", "csr"])
def test_phase75_restricted_materialization(
    benchmark: BenchmarkFixture, target: str
) -> None:
    sector, restricted = _charge_workload()
    plan = restricted.mvp_plan()
    if target == "dense":
        output_bytes = sector.dimension * sector.dimension * 16
    elif target == "coo":
        output_bytes = plan.estimated_bytes
    else:
        output_bytes = (sector.dimension + 1) * np.dtype(
            np.intp
        ).itemsize + plan.estimated_bytes
    _record(
        benchmark,
        target=target,
        n_modes=8,
        sector_dimension=sector.dimension,
        transitions=plan.transition_count,
        plan_bytes=plan.estimated_bytes,
        output_bytes=output_bytes,
        numerical_error=0.0,
    )
    benchmark(getattr(restricted, target))


def test_phase75_restricted_setup_against_u1(benchmark: BenchmarkFixture) -> None:
    operator = _fermion_workload().map_fermions("jordan_wigner")
    charge = tcp.AdditiveCharge(
        tcp.OperatorSpace(qubits=8), qubits={index: (0, 1) for index in range(8)}
    )
    sector = charge.sector(4)
    u1_sector = tcp.U1Sector(8, 4)
    _record(
        benchmark,
        comparison="phase75_charge_vs_existing_u1",
        n_modes=8,
        sector_dimension=sector.dimension,
        input_terms=len(operator.terms),
        phase75_plan_bytes=sector.estimated_bytes,
        u1_basis_dimension=u1_sector.dimension,
    )

    def phase75_setup() -> tcp.ChargeRestrictedOperator:
        return operator.restrict_charge(sector)

    benchmark(phase75_setup)


def test_phase75_existing_u1_reference_setup(benchmark: BenchmarkFixture) -> None:
    operator = _fermion_workload().map_fermions("jordan_wigner")
    u1_sector = tcp.U1Sector(8, 4)
    _record(
        benchmark,
        comparison="existing_u1_reference",
        n_modes=8,
        sector_dimension=u1_sector.dimension,
        input_terms=len(operator.terms),
    )
    benchmark(operator.restrict_u1, u1_sector)
