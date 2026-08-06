# Quickstart

This is the shortest useful path through the public API. Install the wheel, build one Pauli operator, then choose whether the next operation belongs in a native plan or a TensorCircuit backend.

## Install

```bash
python -m pip install tencirpauli
```

Released wheels include the Rust extension. A local Rust toolchain is only needed when installing from source.

## Build a Hamiltonian

```python
import tencirpauli as tcp

h = tcp.PauliOperator.from_terms(
    2,
    [("XX", 0.5), ("YY", 0.5), ("ZI", -0.2)],
)

print(h.nqubits)
print(len(h.terms))
print(h.dense().shape)
```

`PauliOperator.from_terms()` accepts strings, code sequences, and `PauliWord` objects. Construction canonicalizes duplicate words and preserves deterministic term order.

SciPy is a required runtime dependency. Use `plan.to_scipy_linear_operator()` or `h.to_scipy_linear_operator()` for iterative solvers without materializing a matrix. PySCF is optional: install `tencirpauli[chemistry]` to use `tencirpauli.integrations.pyscf.from_scf()` or `from_molecule()`. The chemistry adapter uses orthonormal canonical orbitals, the fixed spin-orbital orderings `interleaved` or `alpha_then_beta`, and the documented spin-orbital integral convention; it does not require OpenFermion.

## Group measurements

```python
groups = h.group_commuting(mode="qubit_wise")
print(len(groups.groups))
print(groups.bases)
```

QWC groups include the local basis information needed to rotate computational-basis samples back into Pauli-term values.

## Propagate an observable

```python
circuit = tcp.PropagationCircuit(
    nqubits=2,
    initial_state=tcp.ZeroState(),
)
circuit.h(0)
circuit.cnot(0, 1)

observable = tcp.PauliOperator.from_terms(2, [("ZZ", 1.0)])
print(circuit.expectation(observable))
```

For a parameterized circuit, pass concrete values with `theta=` and call `value_and_grad(observable)`. The returned gradient is ordered by gate occurrence. For an execution that must stay inside JAX or another TensorCircuit backend, use `expectation_jax()` for circuit differentiation or compile an operator backend MVP plan instead of calling a native circuit engine.

## Choose the next layer

| If you need to... | Start with... |
| --- | --- |
| Inspect or transform Pauli terms | `PauliWord`, `PauliOperator` |
| Materialize or apply a Hamiltonian | `dense()`, `coo()`, `csr()`, `compile("native_mvp")` |
| Keep JAX/TensorCircuit active | `backend_mvp_plan()` and `backend_mvp()` |
| Run a native circuit observable | `PropagationCircuit`, `U1Circuit`, or `SPPSCircuit` |
| Work with structured fermions or charges | `OperatorSpace`, `FermionQubitMapping`, `AdditiveCharge` |

The [API reference](api.md) is generated from the public Python package and its docstrings.
