# Concepts

TenCirPauli is a Python-first interface over a compact Rust implementation for structured Pauli workloads. The useful distinction is not between “Python features” and “Rust features”; it is between the public objects you compose and the execution plan chosen for the workload.

## The four verbs

### Represent

`PauliWord` and `PauliOperator` provide canonical, deterministic objects for Pauli algebra. Duplicate terms are combined, phases are explicit, and the same representation can feed Hamiltonian construction, grouping, symmetry analysis, and propagation.

Structured operators extend the same idea to fermions, bosons, qudits, Majorana words, and hybrid spaces when the workflow needs more than qubit Pauli strings.

### Compile

An operator can be compiled for the result you actually need:

| Target | Use it when |
| --- | --- |
| Dense | The system is small and you want a direct matrix. |
| COO / CSR | A sparse matrix is the interface to the next tool. |
| Native MVP | The system is too large for a matrix and should stay on CPU. |
| Backend plan | NumPy, JAX, TensorCircuit, JIT, or backend autodiff must remain active. |

The compilation boundary is coarse-grained: Python validates friendly inputs, then one batched native call builds the structure or plan.

### Analyze

The analysis layer turns structure into a useful execution choice. It includes qubit-wise and general commuting groups, Z₂ tapering, explicit U(1) sectors, additive charge sectors, and fermion-to-qubit mappings.

These tools are deliberately explicit about their contracts. A qubit-wise commuting group exposes local measurement bases; a general commuting group is not silently presented as the same thing. A restricted sector validates leakage instead of assuming conservation.

### Execute

The circuit facades cover three different numerical contracts: fixed-particle-number native circuits, deterministic Pauli propagation, and stochastic Pauli-path estimation. They share a Python-level shape while keeping their native executors separate.

These facades store concrete gate angles and expose `expectation(observable)`, `value_and_grad(observable)`, and `expectation_jax(observable)`. The direct gradient has one entry per supported gate occurrence. There is no public circuit compilation or symbolic parameter object; compile targets remain an operator API.

## Two execution paths

<div class="tp-path-grid">
  <div class="tp-path tp-path-native">
    <p class="tp-card-index">NATIVE CPU</p>
    <h3>Keep the computation compact</h3>
    <p>Use Rust-native plans and propagation when the workload is discrete, CPU-oriented, and does not need to be traced by a backend.</p>
    <code>operator.native_mvp_plan()</code>
  </div>
  <div class="tp-path tp-path-backend">
    <p class="tp-card-index">BACKEND PLAN</p>
    <h3>Keep the tensor graph alive</h3>
    <p>Use a stable structural plan when TensorCircuit, JAX, JIT, or backend autodiff should own the numerical execution.</p>
    <code>operator.backend_mvp_plan()</code>
  </div>
</div>

## What it is not

TenCirPauli is not a general-purpose statevector simulator, tensor-network engine, or replacement for TensorCircuit backends. It prepares and executes the structured Pauli work around those systems, and it fails explicitly when a target would require an unbounded or unsupported expansion.

## Where to go next

- [Quickstart](quickstart.md) for a small Hamiltonian and propagation example.
- [API reference](api.md) for public classes and methods generated from the Python source docstrings.
- [GitHub repository](https://github.com/tensorcircuit/TenCirPauli) for benchmarks, design notes, and contribution details.
