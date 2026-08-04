"""Small U(1)-restricted VQE using the unified public facade."""

from __future__ import annotations

import numpy as np

import tencirpauli as tcp


def main() -> None:
    theta = tcp.Parameter(0)
    circuit = tcp.U1Circuit(nqubits=2, particle_number=1, occupied=[0])
    circuit.iswap(0, 1, theta=theta)
    hamiltonian = tcp.PauliOperator.from_terms(
        2,
        (("ZI", 0.5), ("IZ", -0.5), ("XX", 0.25), ("YY", 0.25)),
    )

    parameters = np.asarray([0.2], dtype=np.float64)
    for _ in range(8):
        result = circuit.value_and_grad(hamiltonian, parameters=parameters)
        parameters -= 0.2 * result.gradient

    energy = circuit.expectation(hamiltonian, parameters=parameters)
    print(f"U1 VQE energy: {energy.real:.8f}; theta={parameters[0]:.8f}")


if __name__ == "__main__":
    main()
