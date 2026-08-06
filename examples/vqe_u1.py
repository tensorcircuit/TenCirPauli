"""Small U(1)-restricted VQE using the unified public facade."""

from __future__ import annotations

import tencirpauli as tcp


def main() -> None:
    circuit = tcp.U1Circuit(nqubits=2, particle_number=1, occupied=[0])
    hamiltonian = tcp.PauliOperator.from_terms(
        2,
        (("ZI", 0.5), ("IZ", -0.5), ("XX", 0.25), ("YY", 0.25)),
    )

    theta = 0.2
    for _ in range(8):
        circuit = tcp.U1Circuit(nqubits=2, particle_number=1, occupied=[0])
        circuit.iswap(0, 1, theta=theta)
        result = circuit.value_and_grad(hamiltonian)
        theta -= 0.2 * result.gradient[0]

    circuit = tcp.U1Circuit(nqubits=2, particle_number=1, occupied=[0])
    circuit.iswap(0, 1, theta=theta)
    energy = circuit.expectation(hamiltonian)
    print(f"U1 VQE energy: {energy.real:.8f}; theta={theta:.8f}")


if __name__ == "__main__":
    main()
