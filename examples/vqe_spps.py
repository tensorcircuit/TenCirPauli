"""Stochastic Pauli-path VQE using the unified public facade."""

from __future__ import annotations

import tencirpauli as tcp


def main() -> None:
    hamiltonian = tcp.PauliOperator.from_terms(1, (("Z", 1.0),))

    theta = 0.2
    for iteration in range(6):
        circuit = tcp.SPPSCircuit(nqubits=1, initial_state=tcp.ZeroState())
        circuit.ry(0, theta=theta)
        estimate = circuit.value_and_grad(
            hamiltonian,
            samples_per_term=512,
            seed=iteration,
        )
        theta -= 0.2 * estimate.gradient[0]

    circuit = tcp.SPPSCircuit(nqubits=1, initial_state=tcp.ZeroState())
    circuit.ry(0, theta=theta)
    energy = circuit.expectation(
        hamiltonian,
        samples_per_term=1024,
        seed=100,
    )
    print(
        "SPPS VQE energy: "
        f"{energy.value:.8f} +/- {energy.value_standard_error:.8f}; "
        f"theta={theta:.8f}"
    )


if __name__ == "__main__":
    main()
