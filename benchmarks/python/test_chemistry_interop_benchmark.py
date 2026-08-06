"""Small local benchmarks for PySCF conversion and SciPy MVP interop."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp
from tencirpauli.integrations.pyscf import (
    _fermion_from_integral_blocks,
    _transform_eri,
    from_molecule,
    from_scf,
)


try:
    import resource
except ImportError:  # pragma: no cover - Windows benchmark workers
    resource = None  # type: ignore[assignment]


def _peak_rss_bytes() -> int:
    if resource is None:
        return 0
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _closed_shell_determinant_index(n_spatial: int, occupied_orbitals: int) -> int:
    n_modes = 2 * n_spatial
    index = 0
    for orbital in range(occupied_orbitals):
        for spin in (0, 1):
            index |= 1 << (n_modes - 1 - (2 * orbital + spin))
    return index


@pytest.fixture(scope="module")
def h2_data() -> tuple[
    Any,
    Any,
    tuple[np.ndarray[Any, Any], ...],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    pytest.importorskip("pyscf")
    import pyscf.ao2mo as ao2mo
    from pyscf import gto, scf

    molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mean_field = scf.RHF(molecule).run()
    coefficients = np.asarray(mean_field.mo_coeff, dtype=np.complex128)
    one_body = np.ascontiguousarray(
        coefficients.conj().T @ mean_field.get_hcore() @ coefficients
    )
    eri = _transform_eri(
        ao2mo,
        molecule,
        coefficients,
        coefficients.shape[1],
        general=False,
        max_bytes=None,
    )
    return molecule, mean_field, (eri, eri, eri, eri), one_body, coefficients


@pytest.fixture(scope="module")
def medium_data() -> tuple[
    Any,
    Any,
    tuple[np.ndarray[Any, Any], ...],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    pytest.importorskip("pyscf")
    import pyscf.ao2mo as ao2mo
    from pyscf import gto, scf

    molecule = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g")
    mean_field = scf.RHF(molecule).run()
    coefficients = np.asarray(mean_field.mo_coeff, dtype=np.complex128)
    one_body = np.ascontiguousarray(
        coefficients.conj().T @ mean_field.get_hcore() @ coefficients
    )
    eri = _transform_eri(
        ao2mo,
        molecule,
        coefficients,
        coefficients.shape[1],
        general=False,
        max_bytes=None,
    )
    return molecule, mean_field, (eri, eri, eri, eri), one_body, coefficients


def test_pyscf_mo_transformation(
    benchmark: BenchmarkFixture,
    h2_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    molecule, _, _, _, coefficients = h2_data
    pytest.importorskip("pyscf")
    import pyscf.ao2mo as ao2mo

    def transform() -> np.ndarray[Any, Any]:
        return _transform_eri(
            ao2mo,
            molecule,
            coefficients,
            coefficients.shape[1],
            general=False,
            max_bytes=None,
        )

    result = benchmark(transform)
    benchmark.extra_info.update(
        {
            "stage": "pyscf_ao_to_mo",
            "integral_bytes": int(result.nbytes),
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    )


def test_native_compact_integral_ingestion(
    benchmark: BenchmarkFixture,
    h2_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    _, mean_field, eri_blocks, one_body, _ = h2_data

    def ingest() -> tcp.FermionOperator:
        return _fermion_from_integral_blocks(
            tcp.FermionOperator,
            2,
            one_body,
            one_body,
            *eri_blocks,
            spin_ordering="interleaved",
            constant=float(mean_field.mol.energy_nuc()),
            max_bytes=None,
        )

    result = benchmark(ingest)
    determinant = _closed_shell_determinant_index(2, 1)
    error = abs(
        result.compile("dense")[determinant, determinant].real - mean_field.e_tot
    )
    assert error < 1.0e-10
    benchmark.extra_info.update(
        {
            "stage": "native_integral_ingestion",
            "term_count": int(result.term_count),
            "peak_rss_bytes": _peak_rss_bytes(),
            "max_abs_error": float(error),
        }
    )


def test_complete_pyscf_conversion(
    benchmark: BenchmarkFixture,
    h2_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    _, mean_field, _, _, _ = h2_data
    result = benchmark(from_scf, mean_field, max_bytes=None)
    determinant = _closed_shell_determinant_index(2, 1)
    error = abs(
        result.compile("dense")[determinant, determinant].real - mean_field.e_tot
    )
    assert error < 1.0e-10
    benchmark.extra_info.update(
        {
            "stage": "complete_pyscf_conversion",
            "term_count": int(result.term_count),
            "peak_rss_bytes": _peak_rss_bytes(),
            "max_abs_error": float(error),
        }
    )


def test_medium_native_ingestion(
    benchmark: BenchmarkFixture,
    medium_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    _, mean_field, eri_blocks, one_body, _ = medium_data

    def ingest() -> tcp.FermionOperator:
        return _fermion_from_integral_blocks(
            tcp.FermionOperator,
            6,
            one_body,
            one_body,
            *eri_blocks,
            spin_ordering="interleaved",
            constant=float(mean_field.mol.energy_nuc()),
            max_bytes=None,
        )

    result = benchmark(ingest)
    determinant = _closed_shell_determinant_index(6, 2)
    error = abs(
        result.compile("dense")[determinant, determinant].real - mean_field.e_tot
    )
    assert error < 1.0e-9
    benchmark.extra_info.update(
        {
            "stage": "medium_native_integral_ingestion",
            "spatial_orbitals": 6,
            "term_count": int(result.term_count),
            "peak_rss_bytes": _peak_rss_bytes(),
            "max_abs_error": float(error),
        }
    )


def test_medium_complete_pyscf_conversion(
    benchmark: BenchmarkFixture,
    medium_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    _, mean_field, _, _, _ = medium_data
    result = benchmark(from_scf, mean_field, max_bytes=None)
    determinant = _closed_shell_determinant_index(6, 2)
    error = abs(
        result.compile("dense")[determinant, determinant].real - mean_field.e_tot
    )
    assert error < 1.0e-9
    benchmark.extra_info.update(
        {
            "stage": "medium_complete_pyscf_conversion",
            "spatial_orbitals": 6,
            "term_count": int(result.term_count),
            "peak_rss_bytes": _peak_rss_bytes(),
            "max_abs_error": float(error),
        }
    )


def test_medium_from_molecule_conversion(
    benchmark: BenchmarkFixture,
    medium_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    molecule, mean_field, _, _, _ = medium_data
    result = benchmark(from_molecule, molecule, max_bytes=None)
    determinant = _closed_shell_determinant_index(6, 2)
    error = abs(
        result.compile("dense")[determinant, determinant].real - mean_field.e_tot
    )
    assert error < 1.0e-9
    benchmark.extra_info.update(
        {
            "stage": "medium_from_molecule_end_to_end",
            "workload": "LiH sto-3g",
            "spatial_orbitals": 6,
            "term_count": int(result.term_count),
            "peak_rss_bytes": _peak_rss_bytes(),
            "max_abs_error": float(error),
        }
    )


def test_medium_native_plan_construction(
    benchmark: BenchmarkFixture,
    medium_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    _, mean_field, _, _, _ = medium_data
    hamiltonian = from_scf(mean_field, max_bytes=None).map_fermions()
    plan = benchmark(hamiltonian.compile, "native_mvp")
    benchmark.extra_info.update(
        {
            "stage": "medium_native_plan_construction",
            "spatial_orbitals": 6,
            "dimension": int(plan.dimension),
            "term_count": int(plan.term_count),
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    )


def test_medium_dense_materialization(
    benchmark: BenchmarkFixture,
    medium_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    _, mean_field, _, _, _ = medium_data
    hamiltonian = from_scf(mean_field, max_bytes=None).map_fermions()
    dense = benchmark(hamiltonian.dense)
    assert dense.shape == (4096, 4096)
    benchmark.extra_info.update(
        {
            "stage": "medium_dense_materialization",
            "spatial_orbitals": 6,
            "dimension": 4096,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    )


def test_scipy_first_and_steady_mvp(
    benchmark: BenchmarkFixture,
    h2_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    _, mean_field, _, _, _ = h2_data
    plan = from_scf(mean_field, max_bytes=None).map_fermions().compile("native_mvp")
    linear = plan.to_scipy_linear_operator(max_bytes=None)
    state = np.arange(plan.dimension, dtype=np.complex128)
    expected = plan.apply(state, max_bytes=None)
    result = benchmark.pedantic(linear.matvec, args=(state,), rounds=5, iterations=1)
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update(
        {
            "stage": "steady_scipy_matvec",
            "dimension": int(plan.dimension),
            "state_bytes": int(state.nbytes),
            "peak_rss_bytes": _peak_rss_bytes(),
            "max_abs_error": float(np.max(np.abs(result - expected))),
        }
    )


def test_scipy_first_mvp(
    benchmark: BenchmarkFixture,
    h2_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    _, mean_field, _, _, _ = h2_data
    plan = from_scf(mean_field, max_bytes=None).map_fermions().compile("native_mvp")
    linear = plan.to_scipy_linear_operator(max_bytes=None)
    state = np.arange(plan.dimension, dtype=np.complex128)
    result = benchmark.pedantic(linear.matvec, args=(state,), rounds=1, iterations=1)
    benchmark.extra_info.update(
        {
            "stage": "first_scipy_matvec",
            "dimension": int(plan.dimension),
            "state_bytes": int(state.nbytes),
            "peak_rss_bytes": _peak_rss_bytes(),
            "max_abs_error": float(
                np.max(np.abs(result - plan.apply(state, max_bytes=None)))
            ),
        }
    )


def test_native_steady_mvp(
    benchmark: BenchmarkFixture,
    h2_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    _, mean_field, _, _, _ = h2_data
    plan = from_scf(mean_field, max_bytes=None).map_fermions().compile("native_mvp")
    state = np.arange(plan.dimension, dtype=np.complex128)
    expected = plan.apply(state, max_bytes=None)
    result = benchmark.pedantic(plan.apply, args=(state,), rounds=5, iterations=1)
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update(
        {
            "stage": "steady_native_matvec",
            "dimension": int(plan.dimension),
            "state_bytes": int(state.nbytes),
            "peak_rss_bytes": _peak_rss_bytes(),
            "max_abs_error": float(np.max(np.abs(result - expected))),
        }
    )


def test_native_first_mvp(
    benchmark: BenchmarkFixture,
    h2_data: tuple[
        Any,
        Any,
        tuple[np.ndarray[Any, Any], ...],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
) -> None:
    _, mean_field, _, _, _ = h2_data
    plan = from_scf(mean_field, max_bytes=None).map_fermions().compile("native_mvp")
    state = np.arange(plan.dimension, dtype=np.complex128)
    result = benchmark.pedantic(plan.apply, args=(state,), rounds=1, iterations=1)
    expected = plan.apply(state, max_bytes=None)
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update(
        {
            "stage": "first_native_matvec",
            "dimension": int(plan.dimension),
            "state_bytes": int(state.nbytes),
            "peak_rss_bytes": _peak_rss_bytes(),
            "max_abs_error": float(np.max(np.abs(result - expected))),
        }
    )
