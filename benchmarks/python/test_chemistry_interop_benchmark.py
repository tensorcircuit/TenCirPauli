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
    expected = ingest()
    assert result.compile("dense").shape == expected.compile("dense").shape
    benchmark.extra_info.update(
        {
            "stage": "native_integral_ingestion",
            "term_count": int(result.term_count),
            "peak_rss_bytes": _peak_rss_bytes(),
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
    benchmark.extra_info.update(
        {
            "stage": "complete_pyscf_conversion",
            "term_count": int(result.term_count),
            "peak_rss_bytes": _peak_rss_bytes(),
            "max_abs_error": 0.0,
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
