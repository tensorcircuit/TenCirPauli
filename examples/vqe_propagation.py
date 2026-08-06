"""Deterministic Pauli-propagation VQE using the unified public facade."""

from __future__ import annotations

import tencirpauli as tcp


def main() -> None:
    hamiltonian = tcp.PauliOperator.from_terms(
        2,
        (("ZI", 0.5), ("IZ", 0.5), ("ZZ", 0.2), ("XX", 0.3)),
    )

    parameters = [0.2, -0.1]
    for _ in range(12):
        circuit = tcp.PropagationCircuit(nqubits=2, initial_state=tcp.ZeroState())
        circuit.ry(0, theta=parameters[0])
        circuit.ry(1, theta=parameters[1])
        circuit.cnot(0, 1)
        result = circuit.value_and_grad(hamiltonian)
        parameters = [
            value - 0.15 * gradient
            for value, gradient in zip(parameters, result.gradient)
        ]

    energy = circuit.expectation(hamiltonian)
    print(f"Propagation VQE energy: {energy:.8f}; parameters={parameters}")


if __name__ == "__main__":
    main()
