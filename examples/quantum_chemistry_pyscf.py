"""Optional PySCF-to-Pauli TensorCircuit/JAX VQE example."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

from tencirpauli.integrations.pyscf import from_scf
from tencirpauli.integrations.tensorcircuit import backend_mvp


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
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
    apply_backend_mvp = backend_mvp(backend_plan)

    def state_from_parameters(parameters: jax.Array) -> jax.Array:
        circuit = tc.Circuit(pauli_hamiltonian.nqubits)
        circuit.x(0)
        circuit.x(1)
        for layer in range(parameters.shape[0]):
            for qubit in range(pauli_hamiltonian.nqubits):
                circuit.ry(qubit, theta=parameters[layer, qubit])
            for qubit in range(pauli_hamiltonian.nqubits - 1):
                circuit.cnot(qubit, qubit + 1)
        return circuit.state()

    def energy(parameters: jax.Array) -> jax.Array:
        state = state_from_parameters(parameters)
        transformed = apply_backend_mvp(state)
        return jnp.real(jnp.vdot(state, transformed))

    parameters = jnp.linspace(
        -0.18,
        0.18,
        2 * pauli_hamiltonian.nqubits,
        dtype=jnp.float64,
    ).reshape((2, pauli_hamiltonian.nqubits))
    value_and_grad = jax.jit(jax.value_and_grad(energy))
    optimizer = optax.adam(learning_rate=0.08)
    optimizer_state = optimizer.init(parameters)

    @jax.jit
    def adam_step(
        current_parameters: jax.Array, current_optimizer_state: optax.OptState
    ) -> tuple[jax.Array, optax.OptState, jax.Array, jax.Array]:
        current_energy, gradient = value_and_grad(current_parameters)
        updates, next_optimizer_state = optimizer.update(
            gradient, current_optimizer_state, current_parameters
        )
        next_parameters = optax.apply_updates(current_parameters, updates)
        return next_parameters, next_optimizer_state, current_energy, gradient

    initial_energy, _ = value_and_grad(parameters)
    for _ in range(100):
        parameters, optimizer_state, _, _ = adam_step(parameters, optimizer_state)
    final_energy, final_gradient = value_and_grad(parameters)

    dense_hamiltonian = jnp.asarray(pauli_hamiltonian.dense(), dtype=jnp.complex128)
    final_state = state_from_parameters(parameters)
    dense_energy = jnp.real(jnp.vdot(final_state, dense_hamiltonian @ final_state))
    if not bool(jnp.allclose(final_energy, dense_energy, rtol=1.0e-11, atol=1.0e-11)):
        raise AssertionError("backend MVP energy disagrees with the dense reference")

    print(f"H2 JAX VQE: {float(initial_energy):.10f} -> {float(final_energy):.10f}")
    print(f"final gradient norm: {float(jnp.linalg.norm(final_gradient)):.3e}")
    print(f"final parameters: {parameters}")


if __name__ == "__main__":
    main()
