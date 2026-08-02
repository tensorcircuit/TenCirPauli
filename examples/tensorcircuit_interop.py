"""Convert a TensorCircuit circuit through the public classmethod boundary."""

from __future__ import annotations

import tensorcircuit as tc

import tencirpauli as tcp


def main() -> None:
    tc.set_backend("numpy")
    tc.set_dtype("complex128")
    tensor_circuit = tc.Circuit(2)
    tensor_circuit.h(0)
    tensor_circuit.cnot(0, 1)
    tensor_circuit.rz(1, theta=0.23)

    circuit = tcp.PropagationCircuit.from_circuit(tensor_circuit)
    observable = tcp.PauliOperator.from_terms(2, (("ZZ", 1.0),))
    print(f"Converted TensorCircuit expectation: {circuit.expectation(observable):.8f}")


if __name__ == "__main__":
    main()
