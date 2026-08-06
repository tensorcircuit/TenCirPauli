"""Optional PySCF molecular-Hamiltonian ingestion."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Optional, Tuple, cast

import numpy as np

from ..hamiltonian import DEFAULT_MAX_BYTES, _check_allocation, _validate_max_bytes
from ..structured import FermionOperator, _fermion_from_integral_blocks


def _require_pyscf() -> Tuple[Any, Any, Any, Any]:
    """Load PySCF only when the chemistry adapter is actually called."""
    try:
        pyscf = importlib.import_module("pyscf")
        ao2mo = importlib.import_module("pyscf.ao2mo")
        gto = importlib.import_module("pyscf.gto")
        scf = importlib.import_module("pyscf.scf")
    except ImportError as error:
        raise ImportError(
            "PySCF is required for the chemistry adapter; install it with "
            'pip install "tencirpauli[chemistry]"'
        ) from error
    return pyscf, ao2mo, gto, scf


def _validate_spin_ordering(spin_ordering: str) -> None:
    if spin_ordering not in {"interleaved", "alpha_then_beta"}:
        raise ValueError(
            "spin_ordering must be exactly 'interleaved' or 'alpha_then_beta'"
        )


def _coefficient_pair(
    mo_coeff: object, family: str
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if family == "rhf":
        values = np.asarray(mo_coeff)
        if values.ndim != 2:
            raise ValueError("RHF mo_coeff must be a two-dimensional array")
        return values, values
    if not isinstance(mo_coeff, (tuple, list)) or len(mo_coeff) != 2:
        values = np.asarray(mo_coeff)
        if values.ndim != 3 or values.shape[0] != 2:
            raise ValueError("UHF mo_coeff must contain alpha and beta arrays")
        return values[0], values[1]
    alpha, beta = (np.asarray(value) for value in mo_coeff)
    return alpha, beta


def _validate_orbitals(
    alpha: np.ndarray[Any, Any],
    beta: np.ndarray[Any, Any],
    overlap: object,
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    alpha = np.asarray(alpha, dtype=np.complex128)
    beta = np.asarray(beta, dtype=np.complex128)
    metric = np.asarray(overlap, dtype=np.complex128)
    if alpha.ndim != 2 or beta.ndim != 2:
        raise ValueError("SCF orbitals must be two-dimensional")
    if alpha.shape != beta.shape:
        raise ValueError("alpha and beta orbital matrices must have equal shapes")
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("AO overlap must be a square matrix")
    if alpha.shape[0] != metric.shape[0]:
        raise ValueError("SCF orbitals and AO overlap have incompatible shapes")
    for name, values in (("alpha", alpha), ("beta", beta), ("overlap", metric)):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} orbital data must be finite")
    identity = np.eye(alpha.shape[1], dtype=np.complex128)
    for name, values in (("alpha", alpha), ("beta", beta)):
        error = values.conj().T @ metric @ values - identity
        if np.max(np.abs(error), initial=0.0) > 1.0e-8:
            raise ValueError(f"{name} orbitals are not orthonormal under AO overlap")
    return alpha, beta, metric


def _as_square_hcore(value: object, name: str) -> np.ndarray[Any, Any]:
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return cast(np.ndarray[Any, Any], result)


def _transform_eri(
    ao2mo: Any,
    mol: Any,
    coefficients: Any,
    n_spatial: int,
    *,
    general: bool,
    max_bytes: Optional[int],
) -> np.ndarray[Any, Any]:
    expected = n_spatial**4
    _check_allocation(
        expected * np.dtype(np.complex128).itemsize,
        max_bytes,
        "PySCF MO two-body integral block",
    )
    if _has_nonzero_imaginary_part(coefficients):
        nao = int(mol.nao_nr())
        _check_allocation(
            nao**4 * np.dtype(np.float64).itemsize,
            max_bytes,
            "PySCF AO two-body integral workspace",
        )
        ao_integrals = mol.intor("int2e", aosym="s1")
        raw = ao2mo.incore.general(
            ao_integrals,
            _as_coefficient_tuple(coefficients),
            compact=False,
        )
    else:
        coefficients = _pyscf_coefficients(coefficients)
        raw = (
            ao2mo.general(mol, coefficients, compact=False)
            if general
            else ao2mo.kernel(mol, coefficients, compact=False)
        )
    values = np.asarray(raw)
    if values.size != expected:
        raise ValueError("PySCF AO-to-MO transformation returned an unexpected shape")
    values = np.asarray(values, dtype=np.complex128)
    return cast(
        np.ndarray[Any, Any], np.ascontiguousarray(values.reshape((n_spatial,) * 4))
    )


def _pyscf_coefficients(value: Any) -> Any:
    """Keep real MO coefficients real for PySCF's standard AO2MO path."""
    if isinstance(value, tuple):
        return tuple(_pyscf_coefficients(item) for item in value)
    values = np.asarray(value)
    if np.iscomplexobj(values) and np.all(values.imag == 0.0):
        return np.asarray(values.real, dtype=np.float64, order="C")
    return value


def _as_coefficient_tuple(value: Any) -> Tuple[np.ndarray[Any, Any], ...]:
    if isinstance(value, tuple):
        return tuple(np.asarray(item, dtype=np.complex128) for item in value)
    array = np.asarray(value, dtype=np.complex128)
    return (array, array, array, array)


def _has_nonzero_imaginary_part(value: Any) -> bool:
    if isinstance(value, tuple):
        return any(_has_nonzero_imaginary_part(item) for item in value)
    array = np.asarray(value)
    return bool(np.iscomplexobj(array) and np.any(array.imag != 0.0))


def from_scf(
    mf: Any,
    *,
    spin_ordering: str = "interleaved",
    include_nuclear_repulsion: bool = True,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> FermionOperator:
    """Import a converged PySCF RHF or UHF result as a FermionOperator."""
    _, ao2mo, gto, scf = _require_pyscf()
    rhf_type = scf.hf.RHF
    uhf_type = scf.uhf.UHF
    _validate_max_bytes(max_bytes)
    _validate_spin_ordering(spin_ordering)
    if not isinstance(include_nuclear_repulsion, bool):
        raise TypeError("include_nuclear_repulsion must be a bool")
    if type(mf).__module__.startswith("pyscf.dft."):
        raise NotImplementedError("PySCF DFT objects are not supported by this adapter")
    if isinstance(mf, rhf_type):
        family = "rhf"
    elif isinstance(mf, uhf_type):
        family = "uhf"
    else:
        raise NotImplementedError("PySCF adapter supports only RHF and UHF objects")
    if getattr(mf, "converged", None) is not True:
        raise ValueError("PySCF SCF object must be converged")
    mol = getattr(mf, "mol", None)
    if mol is None or not isinstance(mol, gto.Mole):
        raise ValueError("PySCF SCF object must provide a molecule")
    mo_coeff = getattr(mf, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("PySCF SCF object must provide mo_coeff")

    alpha_coeff, beta_coeff = _coefficient_pair(mo_coeff, family)
    overlap = mf.get_ovlp()
    alpha_coeff, beta_coeff, overlap = _validate_orbitals(
        alpha_coeff, beta_coeff, overlap
    )
    hcore_value = mf.get_hcore()
    if isinstance(hcore_value, (tuple, list)) and len(hcore_value) == 2:
        hcore_alpha = _as_square_hcore(hcore_value[0], "alpha core Hamiltonian")
        hcore_beta = _as_square_hcore(hcore_value[1], "beta core Hamiltonian")
    else:
        hcore_alpha = _as_square_hcore(hcore_value, "core Hamiltonian")
        hcore_beta = hcore_alpha
    if hcore_alpha.shape != overlap.shape or hcore_beta.shape != overlap.shape:
        raise ValueError("core Hamiltonian and AO overlap have incompatible shapes")
    one_alpha = np.ascontiguousarray(alpha_coeff.conj().T @ hcore_alpha @ alpha_coeff)
    one_beta = np.ascontiguousarray(beta_coeff.conj().T @ hcore_beta @ beta_coeff)
    n_spatial = alpha_coeff.shape[1]
    if n_spatial != beta_coeff.shape[1]:
        raise ValueError("UHF alpha and beta orbital counts must be equal")

    if family == "rhf":
        eri = _transform_eri(
            ao2mo,
            mol,
            alpha_coeff,
            n_spatial,
            general=False,
            max_bytes=max_bytes,
        )
        eri_blocks = (eri, eri, eri, eri)
    else:
        eri_blocks = (
            _transform_eri(
                ao2mo,
                mol,
                (alpha_coeff, alpha_coeff, alpha_coeff, alpha_coeff),
                n_spatial,
                general=True,
                max_bytes=max_bytes,
            ),
            _transform_eri(
                ao2mo,
                mol,
                (alpha_coeff, alpha_coeff, beta_coeff, beta_coeff),
                n_spatial,
                general=True,
                max_bytes=max_bytes,
            ),
            _transform_eri(
                ao2mo,
                mol,
                (beta_coeff, beta_coeff, alpha_coeff, alpha_coeff),
                n_spatial,
                general=True,
                max_bytes=max_bytes,
            ),
            _transform_eri(
                ao2mo,
                mol,
                (beta_coeff, beta_coeff, beta_coeff, beta_coeff),
                n_spatial,
                general=True,
                max_bytes=max_bytes,
            ),
        )
    nuclear = float(mol.energy_nuc()) if include_nuclear_repulsion else 0.0
    if not np.isfinite(nuclear):
        raise ValueError("nuclear-repulsion energy must be finite")
    return _fermion_from_integral_blocks(
        FermionOperator,
        n_spatial,
        one_alpha,
        one_beta,
        *eri_blocks,
        spin_ordering=spin_ordering,
        constant=nuclear,
        max_bytes=max_bytes,
    )


def from_molecule(
    mol: Any,
    *,
    method: str = "rhf",
    scf_kwargs: Optional[Mapping[str, object]] = None,
    spin_ordering: str = "interleaved",
    include_nuclear_repulsion: bool = True,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> FermionOperator:
    """Run an explicitly selected PySCF RHF or UHF calculation and import it."""
    _, _, gto, scf = _require_pyscf()
    if not isinstance(mol, gto.Mole):
        raise TypeError("mol must be a pyscf.gto.Mole")
    if method not in {"rhf", "uhf"}:
        raise ValueError("method must be exactly 'rhf' or 'uhf'")
    if scf_kwargs is None:
        kwargs: dict[str, object] = {}
    elif isinstance(scf_kwargs, Mapping):
        kwargs = dict(scf_kwargs)
    else:
        raise TypeError("scf_kwargs must be a mapping or None")
    mf = scf.RHF(mol) if method == "rhf" else scf.UHF(mol)
    mf.run(**kwargs)
    return from_scf(
        mf,
        spin_ordering=spin_ordering,
        include_nuclear_repulsion=include_nuclear_repulsion,
        max_bytes=max_bytes,
    )


__all__ = ["from_molecule", "from_scf"]
