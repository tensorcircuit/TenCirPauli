"""Optional PySCF adapter tests and small-molecule numerical references."""

from __future__ import annotations

import numpy as np
import pytest

from tencirpauli.integrations.pyscf import from_molecule, from_scf


def test_pyscf_dependency_is_lazy_and_reports_install_command(monkeypatch) -> None:
    import tencirpauli.integrations.pyscf as adapter

    original_import = adapter.importlib.import_module

    def missing_pyscf(name: str):
        if name.startswith("pyscf"):
            raise ModuleNotFoundError("No module named 'pyscf'")
        return original_import(name)

    monkeypatch.setattr(adapter.importlib, "import_module", missing_pyscf)
    with pytest.raises(ImportError, match=r"PySCF.*tencirpauli\[chemistry\]"):
        from_scf(object())


def _h2_rhf():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    molecule = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
    )
    mean_field = scf.RHF(molecule).run()
    return pyscf, molecule, mean_field


def _fermionic_permutation(n_modes: int, permutation: tuple[int, ...]) -> np.ndarray:
    dimension = 1 << n_modes
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for source in range(dimension):
        occupations = tuple((source >> (n_modes - 1 - i)) & 1 for i in range(n_modes))
        mapped = tuple(occupations[i] for i in np.argsort(permutation))
        target = sum(bit << (n_modes - 1 - i) for i, bit in enumerate(mapped))
        occupied_images = [permutation[i] for i, bit in enumerate(occupations) if bit]
        inversions = sum(
            left > right
            for index, left in enumerate(occupied_images)
            for right in occupied_images[index + 1 :]
        )
        result[target, source] = (-1.0) ** inversions
    return result


def test_rhf_h2_constant_mapping_and_determinant_energy() -> None:
    _, molecule, mean_field = _h2_rhf()
    from tencirpauli.integrations.pyscf import from_scf

    fermion = from_scf(mean_field)
    assert fermion.space.fermions == 4
    identity = next(term for term in fermion.terms if term.word.is_identity)
    assert identity.coefficient == pytest.approx(molecule.energy_nuc(), abs=1.0e-12)
    matrix = fermion.compile("dense")
    determinant_index = int("1100", 2)
    assert matrix[determinant_index, determinant_index].real == pytest.approx(
        mean_field.e_tot, abs=1.0e-8
    )
    mapped = fermion.map_fermions("jordan_wigner")
    np.testing.assert_allclose(mapped.dense(), matrix, rtol=1.0e-10, atol=1.0e-10)


def test_rhf_orderings_are_related_by_fermionic_mode_permutation() -> None:
    _, _, mean_field = _h2_rhf()
    interleaved = from_scf(mean_field, spin_ordering="interleaved").compile("dense")
    alpha_then_beta = from_scf(mean_field, spin_ordering="alpha_then_beta").compile(
        "dense"
    )
    transform = _fermionic_permutation(4, (0, 2, 1, 3))
    np.testing.assert_allclose(
        alpha_then_beta,
        transform @ interleaved @ transform.conj().T,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_from_molecule_preserves_kwargs_and_rejects_implicit_methods() -> None:
    pytest.importorskip("pyscf")
    from pyscf import gto

    molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    kwargs = {"conv_tol": 1.0e-9}
    operator = from_molecule(molecule, method="rhf", scf_kwargs=kwargs)
    assert kwargs == {"conv_tol": 1.0e-9}
    assert operator.space.fermions == 4
    with pytest.raises(ValueError, match="method"):
        from_molecule(molecule, method="auto")


def test_from_scf_rejects_unconverged_and_dft_objects_before_integrals() -> None:
    pytest.importorskip("pyscf")
    from pyscf import dft, gto, scf

    molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    with pytest.raises(ValueError, match="converged"):
        from_scf(scf.RHF(molecule))
    with pytest.raises(NotImplementedError, match="DFT"):
        from_scf(dft.RKS(molecule))


def test_uhf_open_shell_uses_separate_alpha_beta_orbitals() -> None:
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        charge=0,
        spin=2,
        unit="Angstrom",
    )
    mean_field = scf.UHF(molecule).run()
    assert mean_field.converged is True
    assert not np.allclose(mean_field.mo_coeff[0], mean_field.mo_coeff[1])
    operator = from_scf(mean_field, spin_ordering="alpha_then_beta")
    assert operator.space.fermions == 4
    determinant = int("1100", 2)
    assert operator.compile("dense")[determinant, determinant].real == pytest.approx(
        mean_field.e_tot, abs=1.0e-7
    )
