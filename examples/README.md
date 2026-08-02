# Examples

These scripts are executable smoke examples for the public Phase Alpha API. Each one constructs a circuit, creates a Pauli Hamiltonian, evaluates an energy, and (where applicable) applies a short gradient-based update. They are intentionally small so an Agent can run them as contract checks.

Run them from the repository root after installing TenCirPauli:

```bash
python examples/vqe_u1.py
python examples/vqe_propagation.py
python examples/vqe_spps.py
python examples/tensorcircuit_interop.py
```

`U1Circuit`, `PropagationCircuit`, and `SPPSCircuit` have the same user-level construction, `theta=`, `expectation()`, and `value_and_grad()` shape. Their result contracts remain different where the underlying executor is different: deterministic circuits return scalar/gradient results, while SPPS returns an estimator with sampling metadata.
