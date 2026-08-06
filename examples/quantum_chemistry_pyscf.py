"""Optional PySCF-to-Pauli TensorCircuit backend-MVP example."""

from __future__ import annotations

import numpy as np
import tensorcircuit as tc

from tencirpauli.integrations.pyscf import from_scf
from tencirpauli.integrations.tensorcircuit import backend_mvp


def main() -> None:
    tc.set_backend("numpy")
    tc.set_dtype("complex128")
    from pyscf import gto, scf

    molecule = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
    )
    mean_field = scf.RHF(molecule).run()
    fermion_hamiltonian = from_scf(mean_field)
    pauli_hamiltonian = fermion_hamiltonian.map_fermions("jordan_wigner")
    backend_plan = pauli_hamiltonian.backend_mvp_plan()

    circuit = tc.Circuit(pauli_hamiltonian.nqubits)
    circuit.h(0)
    circuit.ry(1, theta=0.23)
    state = np.asarray(circuit.state(), dtype=np.complex128)
    apply_backend_mvp = backend_mvp(backend_plan)
    energy = np.vdot(state, np.asarray(apply_backend_mvp(state)))
    dense_energy = np.vdot(state, pauli_hamiltonian.dense() @ state)
    np.testing.assert_allclose(energy, dense_energy, rtol=1.0e-11, atol=1.0e-11)
    print(f"H2 fixed-parameter backend-MVP energy: {energy.real:.10f}")


if __name__ == "__main__":
    main()
