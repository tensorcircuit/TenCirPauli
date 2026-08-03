"""Release benchmarks for Phase 7.5 algebra, mappings, and charge sectors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp
from tencirpauli import charge as charge_module
from tencirpauli.majorana import _guard_expansion
from tencirpauli.mapping import _mapping_matrix


def _majorana_to_fermion_python_reference(
    operator: tcp.MajoranaOperator,
) -> tcp.FermionOperator:
    branches = sum(1 << term.word.degree for term in operator.terms)
    _guard_expansion(branches, tcp.DEFAULT_MAX_BYTES, "Majorana-to-fermion expansion")
    raw_terms = []
    for term in operator.terms:
        current = [((), term.coefficient)]
        for index in term.word.indices:
            mode, component = divmod(index, 2)
            options = (
                ((mode, "create"), 1.0 + 0j),
                ((mode, "annihilate"), 1.0 + 0j),
            )
            if component:
                options = (
                    ((mode, "create"), 1.0j),
                    ((mode, "annihilate"), -1.0j),
                )
            current = [
                ((*factors, factor), coefficient * local)
                for factors, coefficient in current
                for factor, local in options
            ]
        raw_terms.extend(current)
    return tcp.FermionOperator.from_terms(operator.n_modes, raw_terms)


def _mapping_plan_python_reference(
    mapping: str, n_modes: int
) -> tcp.FermionQubitMapping:
    return tcp.FermionQubitMapping(mapping, n_modes, _mapping_matrix(mapping, n_modes))


def _mapping_ab_workload(n_modes: int, term_count: int) -> tcp.PauliOperator:
    terms = []
    for term_index in range(term_count):
        code_number = term_index
        codes = []
        for _ in range(n_modes):
            codes.append(code_number % 4)
            code_number //= 4
        terms.append((codes, complex(1.0 + 0.001 * term_index, -0.01)))
    operator = tcp.PauliOperator.from_terms(n_modes, terms)
    assert len(operator.terms) == term_count
    return operator


def _map_pauli_python_reference(
    plan: tcp.FermionQubitMapping, operator: tcp.PauliOperator
) -> tcp.PauliOperator:
    return tcp.PauliOperator.from_terms(
        plan.n_modes,
        (
            (transformed, term.coefficient * phase)
            for term in operator.terms
            for transformed, phase in (
                plan._transform_codes_with_phase(term.word.to_codes()),
            )
        ),
    )


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


def _majorana_ab_workload(
    n_modes: int, term_count: int, degree: int
) -> tcp.MajoranaOperator:
    terms = []
    for start in range(term_count):
        indices = tuple(
            sorted(
                2 * ((start + offset) % n_modes) + (offset & 1)
                for offset in range(degree)
            )
        )
        terms.append((indices, complex(1.0 + 0.01 * start, -0.02 * (start % 3))))
    return tcp.MajoranaOperator.from_terms(n_modes, terms)


def _charge_workload() -> tuple[tcp.ChargeSector, tcp.ChargeRestrictedOperator]:
    space = tcp.OperatorSpace(fermions=8)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(8)})
    operator = _fermion_workload()
    sector = charge.sector(4)
    return sector, operator.restrict_charge(sector)


def _restricted_scaling_workload(
    n_modes: int, particle_number: int
) -> tuple[tcp.FermionOperator, tcp.ChargeSector]:
    space = tcp.OperatorSpace(fermions=n_modes)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(n_modes)})
    terms = [
        (((index, "create"), (index + 1, "annihilate")), 1.0)
        for index in range(n_modes - 1)
    ] + [
        (((index + 1, "create"), (index, "annihilate")), 1.0)
        for index in range(n_modes - 1)
    ]
    return tcp.FermionOperator.from_terms(n_modes, terms), charge.sector(
        particle_number
    )


def _simultaneous_spin_workload(
    spin_orbitals_per_sector: int,
) -> tuple[tcp.FermionOperator, tcp.ChargeSector]:
    n_modes = 2 * spin_orbitals_per_sector
    space = tcp.OperatorSpace(fermions=n_modes)
    number = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(n_modes)})
    spin = tcp.AdditiveCharge(
        space,
        fermions={
            index: (1 if index < spin_orbitals_per_sector else -1)
            for index in range(n_modes)
        },
    )
    terms = []
    for offset in (0, spin_orbitals_per_sector):
        for index in range(spin_orbitals_per_sector - 1):
            terms.extend(
                (
                    (
                        (
                            (offset + index, "create"),
                            (offset + index + 1, "annihilate"),
                        ),
                        1.0,
                    ),
                    (
                        (
                            (offset + index + 1, "create"),
                            (offset + index, "annihilate"),
                        ),
                        1.0,
                    ),
                )
            )
    return tcp.FermionOperator.from_terms(n_modes, terms), tcp.ChargeSector(
        ((number, spin_orbitals_per_sector), (spin, 0))
    )


def _bose_hubbard_workload(
    n_modes: int, total_occupation: int
) -> tuple[tcp.BosonOperator, tcp.ChargeSector]:
    space = tcp.OperatorSpace(bosons=n_modes)
    number = tcp.AdditiveCharge(space, bosons={index: 1 for index in range(n_modes)})
    terms = []
    for index in range(n_modes - 1):
        terms.extend(
            (
                (((index, "create"), (index + 1, "annihilate")), 1.0),
                (((index + 1, "create"), (index, "annihilate")), 1.0),
            )
        )
    return tcp.BosonOperator.from_terms(n_modes, terms), number.sector(total_occupation)


def _mixed_excitation_workload() -> tuple[tcp.HybridOperator, tcp.ChargeSector]:
    space = tcp.OperatorSpace(fermions=2, bosons=1)
    charge = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1}, bosons={0: 1})
    operator = (
        space.fermion.create(0) * space.boson.annihilate(0)
        + space.fermion.annihilate(0) * space.boson.create(0)
        + space.fermion.create(1) * space.boson.annihilate(0)
        + space.fermion.annihilate(1) * space.boson.create(0)
    )
    return operator, charge.sector(1)


def _charge_setup_workload(n_modes: int) -> tcp.AdditiveCharge:
    space = tcp.OperatorSpace(fermions=n_modes)
    return tcp.AdditiveCharge(space, fermions={index: 1 for index in range(n_modes)})


def _python_charge_dispatch(*args: object, **kwargs: object) -> None:
    del args, kwargs
    return None


def _balanced_occupation(n_modes: int) -> tuple[int, ...]:
    return tuple(index % 2 for index in range(n_modes))


def _assert_basis_boundary(
    basis: np.ndarray[Any, Any], n_modes: int, particle_number: int
) -> None:
    assert basis.shape == (math.comb(n_modes, particle_number), n_modes)
    np.testing.assert_array_equal(
        basis[0], np.asarray([0] * (n_modes - particle_number) + [1] * particle_number)
    )
    np.testing.assert_array_equal(
        basis[-1], np.asarray([1] * particle_number + [0] * (n_modes - particle_number))
    )


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


def test_phase75_fermion_majorana_conversion(benchmark: BenchmarkFixture) -> None:
    operator = _fermion_workload()
    expected = operator.to_majorana()
    _record(
        benchmark,
        operation="fermion_to_majorana",
        input_terms=operator.term_count,
        output_terms=expected.term_count,
        n_modes=operator.space.fermions,
    )
    benchmark(operator.to_majorana)


@pytest.mark.parametrize(
    ("scale", "n_modes", "term_count", "degree"),
    [
        ("small", 8, 8, 4),
        ("medium", 24, 16, 6),
        ("large", 64, 32, 8),
    ],
)
def test_phase75_majorana_conversion_ab_python(
    benchmark: BenchmarkFixture,
    scale: str,
    n_modes: int,
    term_count: int,
    degree: int,
) -> None:
    operator = _majorana_ab_workload(n_modes, term_count, degree)
    expected = _majorana_to_fermion_python_reference(operator)
    _record(
        benchmark,
        path="python_reference",
        scale=scale,
        n_modes=n_modes,
        input_terms=term_count,
        degree=degree,
        branches=term_count * (1 << degree),
        output_terms=len(expected.terms),
    )
    benchmark(_majorana_to_fermion_python_reference, operator)


@pytest.mark.parametrize(
    ("scale", "n_modes", "term_count", "degree"),
    [
        ("small", 8, 8, 4),
        ("medium", 24, 16, 6),
        ("large", 64, 32, 8),
    ],
)
def test_phase75_majorana_conversion_ab_native(
    benchmark: BenchmarkFixture,
    scale: str,
    n_modes: int,
    term_count: int,
    degree: int,
) -> None:
    operator = _majorana_ab_workload(n_modes, term_count, degree)
    expected = _majorana_to_fermion_python_reference(operator)
    actual = operator.to_fermion()
    assert actual.terms == expected.terms
    _record(
        benchmark,
        path="rust_native",
        scale=scale,
        n_modes=n_modes,
        input_terms=term_count,
        degree=degree,
        branches=term_count * (1 << degree),
        output_terms=len(actual.terms),
    )
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


@pytest.mark.parametrize(
    ("mapping", "scale", "n_modes"),
    [
        ("parity", "small", 8),
        ("parity", "medium", 32),
        ("parity", "large", 128),
        ("bravyi_kitaev", "small", 8),
        ("bravyi_kitaev", "medium", 32),
        ("bravyi_kitaev", "large", 128),
    ],
)
def test_phase75_mapping_plan_ab_python(
    benchmark: BenchmarkFixture, mapping: str, scale: str, n_modes: int
) -> None:
    plan = _mapping_plan_python_reference(mapping, n_modes)
    _record(
        benchmark,
        path="python_reference",
        mapping=mapping,
        scale=scale,
        n_modes=n_modes,
        cnot_count=len(plan.cnot_operations),
        plan_bytes=plan.estimated_bytes,
    )
    benchmark(_mapping_plan_python_reference, mapping, n_modes)


@pytest.mark.parametrize(
    ("mapping", "scale", "n_modes"),
    [
        ("parity", "small", 8),
        ("parity", "medium", 32),
        ("parity", "large", 128),
        ("bravyi_kitaev", "small", 8),
        ("bravyi_kitaev", "medium", 32),
        ("bravyi_kitaev", "large", 128),
    ],
)
def test_phase75_mapping_plan_ab_native(
    benchmark: BenchmarkFixture, mapping: str, scale: str, n_modes: int
) -> None:
    expected = _mapping_plan_python_reference(mapping, n_modes)
    actual = tcp.FermionQubitMapping.from_name(mapping, n_modes)
    np.testing.assert_array_equal(actual.encoding_matrix, expected.encoding_matrix)
    np.testing.assert_array_equal(
        actual.inverse_encoding_matrix, expected.inverse_encoding_matrix
    )
    assert actual.cnot_operations == expected.cnot_operations
    _record(
        benchmark,
        path="rust_native",
        mapping=mapping,
        scale=scale,
        n_modes=n_modes,
        cnot_count=len(actual.cnot_operations),
        plan_bytes=actual.estimated_bytes,
    )
    benchmark(tcp.FermionQubitMapping.from_name, mapping, n_modes)


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


@pytest.mark.parametrize("degree", [4, 8, 16, 32])
def test_phase75_direct_majorana_mapping_scaling(
    benchmark: BenchmarkFixture, degree: int
) -> None:
    operator = tcp.MajoranaOperator.from_terms(64, [(tuple(range(degree)), 1.0)])
    mapping = tcp.FermionQubitMapping.parity(64)
    expected = mapping.map_majorana_operator(operator)
    assert len(expected.terms) == 1
    _record(
        benchmark,
        operation="direct_majorana_mapping",
        degree=degree,
        n_modes=64,
        input_terms=operator.term_count,
        output_terms=len(expected.terms),
        mapped_weight=expected.terms[0].word.weight,
        plan_bytes=mapping.estimated_bytes,
    )
    benchmark(operator.map_fermions, mapping)


def test_phase75_charge_analysis_setup(benchmark: BenchmarkFixture) -> None:
    space = tcp.OperatorSpace(fermions=8)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(8)})
    operator = _fermion_workload()
    analysis = operator.analyze_charge(charge)
    assert analysis.is_conserved
    _record(
        benchmark,
        operation="charge_analysis",
        n_modes=8,
        input_terms=operator.term_count,
        output_terms=analysis.commutator_term_count,
    )
    benchmark(operator.analyze_charge, charge)


@pytest.mark.parametrize(
    ("mapping", "scale", "n_modes", "term_count"),
    [
        ("parity", "small", 8, 16),
        ("parity", "medium", 32, 64),
        ("parity", "large", 64, 128),
        ("bravyi_kitaev", "small", 8, 16),
        ("bravyi_kitaev", "medium", 32, 64),
        ("bravyi_kitaev", "large", 64, 128),
    ],
)
def test_phase75_mapping_ab_python(
    benchmark: BenchmarkFixture,
    mapping: str,
    scale: str,
    n_modes: int,
    term_count: int,
) -> None:
    plan = tcp.FermionQubitMapping.from_name(mapping, n_modes)
    operator = _mapping_ab_workload(n_modes, term_count)
    assert len(operator.terms) == term_count
    expected = _map_pauli_python_reference(plan, operator)
    _record(
        benchmark,
        path="python_reference",
        mapping=mapping,
        scale=scale,
        n_modes=n_modes,
        input_terms=term_count,
        output_terms=len(expected.terms),
        cnot_count=len(plan.cnot_operations),
    )
    benchmark(_map_pauli_python_reference, plan, operator)


@pytest.mark.parametrize(
    ("mapping", "scale", "n_modes", "term_count"),
    [
        ("parity", "small", 8, 16),
        ("parity", "medium", 32, 64),
        ("parity", "large", 64, 128),
        ("bravyi_kitaev", "small", 8, 16),
        ("bravyi_kitaev", "medium", 32, 64),
        ("bravyi_kitaev", "large", 64, 128),
    ],
)
def test_phase75_mapping_ab_native(
    benchmark: BenchmarkFixture,
    mapping: str,
    scale: str,
    n_modes: int,
    term_count: int,
) -> None:
    plan = tcp.FermionQubitMapping.from_name(mapping, n_modes)
    operator = _mapping_ab_workload(n_modes, term_count)
    assert len(operator.terms) == term_count
    expected = _map_pauli_python_reference(plan, operator)
    actual = plan.map_pauli(operator)
    assert actual.terms == expected.terms
    _record(
        benchmark,
        path="rust_native",
        mapping=mapping,
        scale=scale,
        n_modes=n_modes,
        input_terms=term_count,
        output_terms=len(actual.terms),
        cnot_count=len(plan.cnot_operations),
    )
    benchmark(plan.map_pauli, operator)


def test_phase75_sector_setup(benchmark: BenchmarkFixture) -> None:
    space = tcp.OperatorSpace(fermions=12)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(12)})
    _record(benchmark, n_modes=12, sector_value=6, plan_bytes=0)
    benchmark(charge.sector, 6)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [
        ("small", 8, 4),
        ("medium", 12, 6),
        ("large", 16, 8),
        ("xlarge", 20, 10),
    ],
)
def test_phase75_charge_sector_setup_ab_python(
    benchmark: BenchmarkFixture,
    monkeypatch: pytest.MonkeyPatch,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    monkeypatch.setattr(
        charge_module._native, "charge_sector_plan_compact", _python_charge_dispatch
    )
    sector = charge.sector(particle_number)
    assert sector.dimension == math.comb(n_modes, particle_number)
    _record(
        benchmark,
        path="python_reference",
        operation="setup",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
    )
    benchmark(charge.sector, particle_number)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [
        ("small", 8, 4),
        ("medium", 12, 6),
        ("large", 16, 8),
        ("xlarge", 20, 10),
    ],
)
def test_phase75_charge_sector_setup_ab_native(
    benchmark: BenchmarkFixture,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    sector = charge.sector(particle_number)
    assert sector.dimension == math.comb(n_modes, particle_number)
    _record(
        benchmark,
        path="rust_native",
        operation="setup",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
    )
    benchmark(charge.sector, particle_number)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [
        ("small", 8, 4),
        ("medium", 12, 6),
        ("large", 16, 8),
        ("xlarge", 20, 10),
    ],
)
def test_phase75_charge_sector_rank_ab_python(
    benchmark: BenchmarkFixture,
    monkeypatch: pytest.MonkeyPatch,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    monkeypatch.setattr(
        charge_module._native, "charge_sector_plan_compact", _python_charge_dispatch
    )
    sector = charge.sector(particle_number)
    occupation = _balanced_occupation(n_modes)
    expected = sector.rank(occupation)
    assert sector.unrank(expected) == occupation
    _record(
        benchmark,
        path="python_reference",
        operation="rank",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
    )
    benchmark(sector.rank, occupation)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [
        ("small", 8, 4),
        ("medium", 12, 6),
        ("large", 16, 8),
        ("xlarge", 20, 10),
    ],
)
def test_phase75_charge_sector_rank_ab_native(
    benchmark: BenchmarkFixture,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    sector = charge.sector(particle_number)
    occupation = _balanced_occupation(n_modes)
    expected = sector.rank(occupation)
    assert sector.unrank(expected) == occupation
    _record(
        benchmark,
        path="rust_native",
        operation="rank",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
    )
    benchmark(sector.rank, occupation)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [
        ("small", 8, 4),
        ("medium", 12, 6),
        ("large", 16, 8),
        ("xlarge", 20, 10),
    ],
)
def test_phase75_charge_sector_unrank_ab_python(
    benchmark: BenchmarkFixture,
    monkeypatch: pytest.MonkeyPatch,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    monkeypatch.setattr(
        charge_module._native, "charge_sector_plan_compact", _python_charge_dispatch
    )
    sector = charge.sector(particle_number)
    index = sector.dimension // 2
    occupation = sector.unrank(index)
    assert sector.rank(occupation) == index
    _record(
        benchmark,
        path="python_reference",
        operation="unrank",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
    )
    benchmark(sector.unrank, index)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [
        ("small", 8, 4),
        ("medium", 12, 6),
        ("large", 16, 8),
        ("xlarge", 20, 10),
    ],
)
def test_phase75_charge_sector_unrank_ab_native(
    benchmark: BenchmarkFixture,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    sector = charge.sector(particle_number)
    index = sector.dimension // 2
    occupation = sector.unrank(index)
    assert sector.rank(occupation) == index
    _record(
        benchmark,
        path="rust_native",
        operation="unrank",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
    )
    benchmark(sector.unrank, index)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [("small", 8, 4), ("medium", 16, 8), ("large", 20, 10)],
)
def test_phase75_charge_sector_basis_ab_python(
    benchmark: BenchmarkFixture,
    monkeypatch: pytest.MonkeyPatch,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    monkeypatch.setattr(
        charge_module._native, "charge_sector_plan_compact", _python_charge_dispatch
    )
    sector = charge.sector(particle_number)
    basis = sector.basis_states()
    _assert_basis_boundary(basis, n_modes, particle_number)
    _record(
        benchmark,
        path="python_reference",
        operation="basis_states",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
        output_bytes=basis.nbytes,
    )
    benchmark(sector.basis_states)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [("small", 8, 4), ("medium", 16, 8), ("large", 20, 10)],
)
def test_phase75_charge_sector_basis_ab_native(
    benchmark: BenchmarkFixture,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    sector = charge.sector(particle_number)
    basis = sector.basis_states()
    _assert_basis_boundary(basis, n_modes, particle_number)
    _record(
        benchmark,
        path="rust_native",
        operation="basis_states",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
        output_bytes=basis.nbytes,
    )
    benchmark(sector.basis_states)


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


@pytest.mark.parametrize(
    ("n_modes", "particle_number"),
    [(8, 4), (12, 6), (16, 8), (20, 10)],
)
def test_phase75_restricted_setup_scaling(
    benchmark: BenchmarkFixture, n_modes: int, particle_number: int
) -> None:
    operator, sector = _restricted_scaling_workload(n_modes, particle_number)
    expected = operator.restrict_charge(sector)
    _record(
        benchmark,
        operation="restricted_setup",
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        input_terms=operator.term_count,
        transitions=expected.mvp_plan().transition_count,
        plan_bytes=expected.estimated_bytes,
    )
    benchmark(operator.restrict_charge, sector)


@pytest.mark.parametrize(
    "workload", ["simultaneous_spin", "bose_hubbard", "mixed_excitation"]
)
def test_phase75_restricted_domain_workloads(
    benchmark: BenchmarkFixture, workload: str
) -> None:
    if workload == "simultaneous_spin":
        operator, sector = _simultaneous_spin_workload(6)
    elif workload == "bose_hubbard":
        operator, sector = _bose_hubbard_workload(3, 4)
    else:
        operator, sector = _mixed_excitation_workload()
    expected = operator.restrict_charge(sector)
    _record(
        benchmark,
        operation="restricted_domain_setup",
        workload=workload,
        sector_dimension=sector.dimension,
        input_terms=operator.term_count,
        transitions=expected.mvp_plan().transition_count,
        plan_bytes=expected.estimated_bytes,
        workspace_bytes=sector.estimated_bytes,
    )
    benchmark(operator.restrict_charge, sector)


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
