"""Deterministic Pauli-propagation VQE using the unified public facade."""

from __future__ import annotations

import numpy as np

import tencirpauli as tcp


def main() -> None:
    theta0 = tcp.Parameter(0)
    theta1 = tcp.Parameter(1)
    circuit = tcp.PropagationCircuit(nqubits=2, initial_state=tcp.ZeroState())
    circuit.ry(0, theta=theta0)
    circuit.ry(1, theta=theta1)
    circuit.cnot(0, 1)
    hamiltonian = tcp.PauliOperator.from_terms(
        2,
        (("ZI", 0.5), ("IZ", 0.5), ("ZZ", 0.2), ("XX", 0.3)),
    )

    parameters = np.asarray([0.2, -0.1], dtype=np.float64)
    for _ in range(12):
        result = circuit.value_and_grad(hamiltonian, parameters=parameters)
        parameters -= 0.15 * result.gradient

    energy = circuit.expectation(hamiltonian, parameters=parameters)
    print(f"Propagation VQE energy: {energy:.8f}; parameters={parameters.tolist()}")


if __name__ == "__main__":
    main()
