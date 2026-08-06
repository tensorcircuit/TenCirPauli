"""Release benchmarks for Majorana algebra, mappings, and charge sectors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp
from tencirpauli.majorana import _guard_expansion


_PAULI_PRODUCT = (
    ((0, 1), (1, 1), (2, 1), (3, 1)),
    ((1, 1), (0, 1), (3, 1j), (2, -1j)),
    ((2, 1), (3, -1j), (0, 1), (1, 1j)),
    ((3, 1), (2, 1j), (1, -1j), (0, 1)),
)


def _independent_cnot_conjugate(
    codes: tuple[int, ...], operations: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, ...], complex]:
    result = list(codes)
    phase = 1.0 + 0j
    control_images = ((0, 0), (1, 1), (2, 1), (3, 0))
    target_images = ((0, 0), (0, 1), (3, 2), (3, 3))
    for control, target in operations:
        control_code, target_code = control_images[result[control]]
        image_control, image_target = target_images[result[target]]
        result[control], local_phase = _PAULI_PRODUCT[control_code][image_control]
        phase *= local_phase
        result[target], local_phase = _PAULI_PRODUCT[target_code][image_target]
        phase *= local_phase
    if phase not in (1.0 + 0j, -1.0 + 0j):
        raise AssertionError("CNOT conjugation produced a non-real phase")
    return tuple(result), phase


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
    return tcp.FermionQubitMapping.from_name(mapping, n_modes)


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
    operations = plan.cnot_operations
    return tcp.PauliOperator.from_terms(
        plan.n_modes,
        (
            (transformed, term.coefficient * phase)
            for term in operator.terms
            for transformed, phase in (
                _independent_cnot_conjugate(term.word.to_codes(), operations),
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


def _balanced_occupation(n_modes: int) -> tuple[int, ...]:
    return tuple(index % 2 for index in range(n_modes))


def _combination_rank_reference(occupation: tuple[int, ...]) -> int:
    n_modes = len(occupation)
    particles = sum(occupation)
    rank = 0
    remaining = particles
    for position, value in enumerate(occupation):
        if value:
            rank += math.comb(n_modes - position - 1, remaining)
            remaining -= 1
    return rank


def _combination_unrank_reference(
    n_modes: int, particles: int, index: int
) -> tuple[int, ...]:
    result = []
    remaining = particles
    for position in range(n_modes):
        zero_count = math.comb(n_modes - position - 1, remaining)
        if index < zero_count:
            result.append(0)
        else:
            result.append(1)
            index -= zero_count
            remaining -= 1
    return tuple(result)


def _combination_basis_reference(n_modes: int, particles: int) -> np.ndarray:
    return np.asarray(
        [
            _combination_unrank_reference(n_modes, particles, index)
            for index in range(math.comb(n_modes, particles))
        ],
        dtype=np.uint8,
    )


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


def test_majorana_charge_majorana_construction(benchmark: BenchmarkFixture) -> None:
    _record(benchmark, input_terms=14, n_modes=8)
    benchmark(_majorana_workload)


def test_majorana_charge_majorana_fermion_conversion(
    benchmark: BenchmarkFixture,
) -> None:
    operator = _majorana_workload()
    _record(benchmark, input_terms=operator.term_count, n_modes=operator.n_modes)
    benchmark(operator.to_fermion)


def test_majorana_charge_fermion_majorana_conversion(
    benchmark: BenchmarkFixture,
) -> None:
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
def test_majorana_charge_majorana_conversion_ab_python(
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
def test_majorana_charge_majorana_conversion_ab_native(
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


def test_majorana_charge_majorana_multiplication(benchmark: BenchmarkFixture) -> None:
    left = _majorana_workload()
    right = left.adjoint()
    _record(benchmark, input_terms=left.term_count * right.term_count, n_modes=8)
    benchmark(left.multiply, right)


@pytest.mark.parametrize("name", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_majorana_charge_mapping_plan_construction(
    benchmark: BenchmarkFixture, name: str
) -> None:
    _record(benchmark, mapping=name, n_modes=8)
    benchmark(tcp.FermionQubitMapping.from_name, name, 8)


@pytest.mark.parametrize("mapping", ["parity", "bravyi_kitaev"])
def test_majorana_charge_mapping_plan_construction_scale(
    benchmark: BenchmarkFixture, mapping: str
) -> None:
    n_modes = 512
    _record(benchmark, mapping=mapping, n_modes=n_modes, scale="xlarge")
    benchmark(tcp.FermionQubitMapping.from_name, mapping, n_modes)


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
def test_majorana_charge_mapping_plan_ab_python(
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
def test_majorana_charge_mapping_plan_ab_native(
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
def test_majorana_charge_mapping(benchmark: BenchmarkFixture, name: str) -> None:
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
def test_majorana_charge_direct_majorana_mapping_scaling(
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


def test_majorana_charge_charge_analysis_setup(benchmark: BenchmarkFixture) -> None:
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
def test_majorana_charge_mapping_ab_python(
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
def test_majorana_charge_mapping_ab_native(
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


@pytest.mark.parametrize("mapping", ["parity", "bravyi_kitaev"])
def test_majorana_charge_mapping_scale_native(
    benchmark: BenchmarkFixture, mapping: str
) -> None:
    n_modes = 512
    term_count = 1024
    plan = tcp.FermionQubitMapping.from_name(mapping, n_modes)
    operator = _mapping_ab_workload(n_modes, term_count)
    actual = plan.map_pauli(operator)
    assert len(operator.terms) == term_count
    assert len(actual.terms) == term_count
    _record(
        benchmark,
        path="rust_native",
        mapping=mapping,
        scale="xlarge",
        n_modes=n_modes,
        raw_requested_terms=term_count,
        input_terms=len(operator.terms),
        output_terms=len(actual.terms),
        cnot_count=len(plan.cnot_operations),
        plan_bytes=plan.estimated_bytes,
    )
    benchmark(plan.map_pauli, operator)


def test_majorana_charge_mapping_long_parity_word(benchmark: BenchmarkFixture) -> None:
    n_modes = 512
    plan = tcp.FermionQubitMapping.parity(n_modes)
    codes = np.zeros((1, n_modes), dtype=np.uint8)
    codes[0, 0] = 1
    operator = tcp.PauliOperator.from_code_arrays(codes, [1.0])
    mapped = plan.map_pauli(operator)
    assert len(mapped.terms) == 1
    _record(
        benchmark,
        path="rust_native",
        operation="long_parity_word",
        mapping="parity",
        n_modes=n_modes,
        raw_requested_terms=1,
        input_terms=len(operator.terms),
        output_terms=len(mapped.terms),
        mapped_weight=mapped.terms[0].word.weight,
        cnot_count=len(plan.cnot_operations),
        plan_bytes=plan.estimated_bytes,
    )
    benchmark(plan.map_pauli, operator)


def test_majorana_charge_sector_setup(benchmark: BenchmarkFixture) -> None:
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
def test_majorana_charge_charge_sector_setup_ab_python(
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
        path="combinatorial_reference",
        operation="setup",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
    )
    expected = math.comb(n_modes, particle_number)
    assert expected == sector.dimension
    benchmark(math.comb, n_modes, particle_number)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [
        ("small", 8, 4),
        ("medium", 12, 6),
        ("large", 16, 8),
        ("xlarge", 20, 10),
    ],
)
def test_majorana_charge_charge_sector_setup_ab_native(
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
def test_majorana_charge_charge_sector_rank_ab_python(
    benchmark: BenchmarkFixture,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    sector = charge.sector(particle_number)
    occupation = _balanced_occupation(n_modes)
    expected = _combination_rank_reference(occupation)
    assert sector.rank(occupation) == expected
    _record(
        benchmark,
        path="combinatorial_reference",
        operation="rank",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
    )
    benchmark(_combination_rank_reference, occupation)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [
        ("small", 8, 4),
        ("medium", 12, 6),
        ("large", 16, 8),
        ("xlarge", 20, 10),
    ],
)
def test_majorana_charge_charge_sector_rank_ab_native(
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
def test_majorana_charge_charge_sector_unrank_ab_python(
    benchmark: BenchmarkFixture,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    sector = charge.sector(particle_number)
    index = sector.dimension // 2
    expected = _combination_unrank_reference(n_modes, particle_number, index)
    assert sector.unrank(index) == expected
    _record(
        benchmark,
        path="combinatorial_reference",
        operation="unrank",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
    )
    benchmark(_combination_unrank_reference, n_modes, particle_number, index)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [
        ("small", 8, 4),
        ("medium", 12, 6),
        ("large", 16, 8),
        ("xlarge", 20, 10),
    ],
)
def test_majorana_charge_charge_sector_unrank_ab_native(
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
def test_majorana_charge_charge_sector_basis_ab_python(
    benchmark: BenchmarkFixture,
    scale: str,
    n_modes: int,
    particle_number: int,
) -> None:
    charge = _charge_setup_workload(n_modes)
    sector = charge.sector(particle_number)
    basis = _combination_basis_reference(n_modes, particle_number)
    native_basis = sector.basis_states()
    _assert_basis_boundary(basis, n_modes, particle_number)
    np.testing.assert_array_equal(native_basis, basis)
    _record(
        benchmark,
        path="combinatorial_reference",
        operation="basis_states",
        scale=scale,
        n_modes=n_modes,
        sector_dimension=sector.dimension,
        plan_bytes=sector.estimated_bytes,
        output_bytes=basis.nbytes,
    )
    benchmark(_combination_basis_reference, n_modes, particle_number)


@pytest.mark.parametrize(
    ("scale", "n_modes", "particle_number"),
    [("small", 8, 4), ("medium", 16, 8), ("large", 20, 10)],
)
def test_majorana_charge_charge_sector_basis_ab_native(
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


def test_majorana_charge_restricted_setup(benchmark: BenchmarkFixture) -> None:
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


def test_majorana_charge_restricted_eager_setup(benchmark: BenchmarkFixture) -> None:
    sector, _ = _charge_workload()
    operator = _fermion_workload()
    expected = operator.restrict_charge(sector, storage="eager")
    _record(
        benchmark,
        n_modes=8,
        sector_dimension=sector.dimension,
        input_terms=operator.term_count,
        storage="eager",
        first_apply=False,
        plan_bytes=expected.estimated_bytes,
    )
    result = benchmark(operator.restrict_charge, sector, storage="eager")
    assert result.storage == "eager"


def test_majorana_charge_restricted_apply(benchmark: BenchmarkFixture) -> None:
    sector, restricted = _charge_workload()
    state = np.arange(sector.dimension, dtype=np.complex128)
    plan = restricted.mvp_plan()
    _record(
        benchmark,
        n_modes=8,
        sector_dimension=sector.dimension,
        storage=plan.storage,
        strategy=plan.strategy,
        plan_bytes=plan.estimated_bytes,
        workspace_bytes=sector.dimension * max(len(sector.local_dimensions), 1) * 8,
        output_bytes=sector.dimension * 16,
    )
    benchmark(restricted.apply, state)


def test_majorana_charge_restricted_first_apply(benchmark: BenchmarkFixture) -> None:
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
def test_majorana_charge_restricted_materialization(
    benchmark: BenchmarkFixture, target: str
) -> None:
    sector, _ = _charge_workload()
    operator = _fermion_workload()
    plan = operator.restrict_charge(sector).mvp_plan(storage="eager")
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
        storage=plan.storage,
        strategy=plan.strategy,
        plan_bytes=plan.estimated_bytes,
        output_bytes=output_bytes,
        numerical_error=0.0,
    )
    benchmark(lambda: getattr(operator.restrict_charge(sector), target)())


def test_majorana_charge_restricted_setup_against_u1(
    benchmark: BenchmarkFixture,
) -> None:
    operator = _fermion_workload().map_fermions("jordan_wigner")
    charge = tcp.AdditiveCharge(
        tcp.OperatorSpace(qubits=8), qubits={index: (0, 1) for index in range(8)}
    )
    sector = charge.sector(4)
    u1_sector = tcp.U1Sector(8, 4)
    _record(
        benchmark,
        comparison="majorana_charge_charge_vs_existing_u1",
        n_modes=8,
        sector_dimension=sector.dimension,
        input_terms=len(operator.terms),
        majorana_charge_plan_bytes=sector.estimated_bytes,
        u1_basis_dimension=u1_sector.dimension,
    )

    def majorana_charge_setup() -> tcp.ChargeRestrictedOperator:
        return operator.restrict_charge(sector)

    benchmark(majorana_charge_setup)


@pytest.mark.parametrize(
    ("n_modes", "particle_number"),
    [(8, 4), (12, 6), (16, 8), (20, 10)],
)
def test_majorana_charge_restricted_setup_scaling(
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
        storage=expected.storage,
        plan_bytes=expected.estimated_bytes,
    )
    benchmark(operator.restrict_charge, sector)


@pytest.mark.parametrize(
    "workload", ["simultaneous_spin", "bose_hubbard", "mixed_excitation"]
)
def test_majorana_charge_restricted_domain_workloads(
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
        storage=expected.storage,
        plan_bytes=expected.estimated_bytes,
        workspace_bytes=sector.estimated_bytes,
    )
    benchmark(operator.restrict_charge, sector)


def test_majorana_charge_existing_u1_reference_setup(
    benchmark: BenchmarkFixture,
) -> None:
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
