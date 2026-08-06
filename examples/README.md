# Examples

These scripts are executable smoke examples for the public circuit facade API. Each one constructs a circuit, creates a Pauli Hamiltonian, evaluates an energy, and (where applicable) applies a short gradient-based update. They are intentionally small so an Agent can run them as contract checks.

Run them from the repository root after installing TenCirPauli:

```bash
python examples/vqe_u1.py
python examples/vqe_propagation.py
python examples/vqe_spps.py
python examples/tensorcircuit_interop.py
python examples/structured_algebra.py
python examples/majorana_charge.py
```

`U1Circuit`, `PropagationCircuit`, and `SPPSCircuit` have the same user-level construction, `theta=`, `expectation()`, and `value_and_grad()` shape. Their result contracts remain different where the underlying executor is different: deterministic circuits return scalar/gradient results, while SPPS returns an estimator with sampling metadata.

`structured_algebra.py` demonstrates finite boson compilation, matrix-free native application, and direct uniform-qudit backend MVP execution.

`majorana_charge.py` demonstrates exact Majorana conversion, a reusable Bravyi–Kitaev plan, additive particle-number sectors, and restricted matrix-free execution.

`quantum_chemistry_pyscf.py` is an optional PySCF example. It imports an RHF H2 Hamiltonian, maps it explicitly to Pauli form, and runs a TensorCircuit JAX VQE with gradient descent through the backend MVP plan. Install PySCF with `pip install "tencirpauli[chemistry]"` before running it.

Heavier, manually run end-to-end studies live under [`research/`](research/). They are not part of the CI example execution list; the first study compares a restricted Fermi–Hubbard MVP with QuSpin's `quantum_LinearOperator`.
