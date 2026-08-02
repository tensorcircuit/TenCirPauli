"""Stochastic Pauli-path VQE using the unified public facade."""

from __future__ import annotations

import numpy as np

import tencirpauli as tcp


def main() -> None:
    theta = tcp.Parameter(0)
    circuit = tcp.SPPSCircuit(nqubits=1, initial_state=tcp.ZeroState())
    circuit.ry(0, theta=theta)
    hamiltonian = tcp.PauliOperator.from_terms(1, (("Z", 1.0),))

    parameters = np.asarray([0.2], dtype=np.float64)
    for iteration in range(6):
        estimate = circuit.value_and_grad(
            hamiltonian,
            parameters=parameters,
            samples_per_term=512,
            seed=iteration,
        )
        parameters -= 0.2 * estimate.gradient

    energy = circuit.expectation(
        hamiltonian,
        parameters=parameters,
        samples_per_term=1024,
        seed=100,
    )
    print(
        "SPPS VQE energy: "
        f"{energy.value:.8f} +/- {energy.value_standard_error:.8f}; "
        f"theta={parameters[0]:.8f}"
    )


if __name__ == "__main__":
    main()
