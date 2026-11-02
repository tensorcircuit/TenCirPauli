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

## Matrix-free time evolution with SciPy

Native plans expose `apply_adjoint(state)`, which computes the conjugate transpose in the same basis as `apply(state)`. Their SciPy adapters support `rmatvec`, `.H`, and column-wise `matmat`/`rmatmat` for complex operators, including non-Hermitian ones. Pauli, structured/Weyl, U1, and general charge plans support this in both lazy and eager storage modes. Compile mapped fermion operators with the desired mapping before adapting the plan.

```python
import numpy as np
from scipy.sparse.linalg import expm_multiply

linear = h.to_scipy_linear_operator()
state = np.array([1, 0, 0, 0], dtype=np.complex128)
evolved = expm_multiply(-0.2j * linear, state, traceA=0.0)
```

Here `h` is the traceless Hamiltonian above. In general, supply the trace of the scaled generator (`-1j * time * trace(H)`), not an unconditional zero. For an evenly spaced time grid, pass `-1j * linear`, `start`, `stop`, and `num`, with `traceA=-1j * trace(H)`.

[SciPy's `expm_multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html) uses norm estimation that can call the adjoint in both single-time and time-grid execution. Passing `traceA` avoids trace estimation but does not remove that adjoint requirement. Other SciPy routines that call `rmatvec` or `.H`, such as `onenormest`, `lsqr`, `lsmr`, and `svds`, use the same adjoint. By contrast, explicit ODE stepping of `dstate/dt = -1j * (linear @ state)` and Hermitian `eigsh` only need forward multiplication.

Adjoint calls keep the plan's storage, mapping, basis ordering, and allocation limits unchanged. They allocate an output vector without building a dense matrix, a transposed sparse graph, or a second plan. CSR adjoints traverse the retained graph serially; native batched matrix application is not yet provided. TensorCircuit backend plans are not native SciPy plans.

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

## Estimate with classical shadows

```python
import numpy as np
import tencirpauli as tcp

state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
snapshots = tcp.Snapshots.sample(
    state, nsettings=256, protocol="pauli", seed=7
)
print(snapshots.expectation(tcp.PauliWord.from_string("X")))
```

`Snapshots.sample()` also accepts `StabilizerState.zero_state(nqubits)` and supports `protocol="clifford"` for global-Clifford snapshots. Use `estimate_many()` for a batch of Pauli targets, `energy()` for a Hermitian Hamiltonian, and `rdm()`, `renyi_entropy()`, `energy_variance()`, `fidelity()`, or `fidelity_statevector()` for post-processing. To analyze measurements collected elsewhere, use `Snapshots.from_pauli_measurements()` or `Snapshots.from_clifford_measurements()`.

## Choose the next layer

| If you need to... | Start with... |
| --- | --- |
| Inspect or transform Pauli terms | `PauliWord`, `PauliOperator` |
| Materialize or apply a Hamiltonian | `dense()`, `coo()`, `csr()`, `compile("native_mvp")` |
| Keep JAX/TensorCircuit active | `backend_mvp_plan()` and `backend_mvp()` |
| Estimate randomized measurements | `Snapshots` and `StabilizerState` |
| Run a native circuit observable | `PropagationCircuit`, `U1Circuit`, or `SPPSCircuit` |
| Work with structured fermions or charges | `OperatorSpace`, `FermionQubitMapping`, `AdditiveCharge` |

The [API reference](api.md) is generated from the public Python package and its docstrings.
