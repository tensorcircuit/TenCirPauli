# TenCirPauli

TenCirPauli is TensorCircuit's Rust-native companion for Pauli algebra, Hamiltonian construction, measurement grouping, symmetry reduction, U(1) restricted circuits, deterministic Pauli propagation, and stochastic Pauli-path estimation. The Python package targets TensorCircuit users; TensorCircuit is a required runtime dependency, while the Rust core remains independent of both Python and TensorCircuit.

## Install

```bash
pip install tencirpauli
```

Released wheels target CPython 3.9+ on Linux x86_64/aarch64, macOS x86_64/aarch64, and Windows x64. A matching wheel does not require a local Rust toolchain. Source builds require Rust 1.85+, Cargo, and maturin.

## Architecture

```text
TensorCircuit / Python facade
        │
        ├── PauliOperator, grouping, symmetry, backend MVP
        ├── U1Circuit
        ├── PropagationCircuit       (deterministic native facade)
        └── SPPSCircuit              (stochastic native facade)
        │
        ▼
PyO3 batch boundary
        │
        ├── Rust U(1) restricted-state executor
        ├── Rust deterministic Heisenberg propagation executor
        └── Rust stochastic Pauli-path executor
```

The three circuit facades share Python-level construction, parameter and objective conventions. Their native executors remain independent because they implement different numerical contracts.

## Core conventions

External Pauli codes are `0=I`, `1=X`, `2=Y`, `3=Z`. Internal packed words use qubit zero as the least-significant bit. Matrix and TensorCircuit computational-basis interfaces use qubit zero as the most-significant bit. Coefficients are complex128-compatible, duplicate Pauli terms are aggregated deterministically, and public arrays are returned read-only where the API promises immutable results.

## Main objects

| Task | Entry point |
| --- | --- |
| Pauli algebra | `PauliWord`, `PauliOperator` |
| Hamiltonian targets | `.dense()`, `.coo()`, `.csr()`, `.mvp()` |
| Reusable native MVP | `.native_mvp_plan()` |
| TensorCircuit backend MVP | `.backend_mvp_plan()`, `backend_mvp()` |
| QWC/general grouping | `.group_commuting()` |
| Z2 symmetry/tapering | `.find_z2_symmetries()`, `.taper_z2()` |
| Fixed-particle-number operator | `U1Sector`, `.restrict_u1()` |
| Fixed-particle-number circuit | `U1Circuit` |
| Deterministic Pauli propagation | `PropagationCircuit` (low-level `GateTape`/`PropagationEngine` remains available) |
| Stochastic Pauli-path estimation | `SPPSCircuit` (low-level `SPPSEngine` remains available) |

## Common circuit facade

Phase Alpha defines the target Python contract for the three circuit classes. The circuit structure is built once; runtime values are supplied as a parameter vector.

```python
import tencirpauli as tcp

p0 = tcp.Parameter(0)
p1 = tcp.Parameter(1)

circuit = tcp.U1Circuit(nqubits=4, k=2, filled=[0, 1])
circuit.iswap(0, 1, theta=p0)
circuit.rzz(1, 2, theta=2.0 * p1 + 0.1)

hamiltonian = tcp.PauliOperator.from_terms(
    4,
    (("XXII", 0.5), ("YYII", 0.5), ("ZIZI", -0.2)),
)

result = circuit.value_and_grad(
    hamiltonian,
    parameters=[0.2, -0.3],
)
energy = circuit.expectation(hamiltonian, parameters=[0.2, -0.3])
```

The same high-level shape is used by the implemented `PropagationCircuit` and `SPPSCircuit` facades:

```python
circuit = tcp.PropagationCircuit(nqubits=4, initial_state=tcp.ZeroState())
circuit.ry(0, theta=p0)
circuit.cnot(0, 1)
circuit.rz(1, theta=p1)

result = circuit.value_and_grad(hamiltonian, parameters=[0.2, -0.3])
energy = circuit.expectation(hamiltonian, parameters=[0.2, -0.3])
```

`Parameter` reuse means shared differentiation. A static `theta=0.2` is not a differentiable parameter. Concrete NumPy arrays, Python sequences, and concrete JAX arrays are converted to host contiguous float64 vectors for native calls. Native circuit calls are not JAX-traceable; use the backend MVP path when the computation must remain inside a JAX graph.

The current implementation contract and rollout status are recorded in [`docs/vibe/phase-alpha-spec.md`](docs/vibe/phase-alpha-spec.md).

## Backend MVP

Use the TensorCircuit backend path when JAX/JIT/autodiff or a TensorCircuit backend tensor must remain active:

```python
import numpy as np
import tensorcircuit as tc
import tencirpauli as tcp

tc.set_backend("numpy")
tc.set_dtype("complex128")

h = tcp.PauliOperator.from_terms(2, (("XY", 0.5), ("ZI", -1.25j)))
plan = h.backend_mvp_plan()
state = np.arange(4, dtype=np.complex128)
result = tcp.backend_mvp(plan)(state)
```

The plan structure is static; coefficients may be supplied as backend tensors where the plan API permits it. This path is distinct from native deterministic or stochastic circuit gradients.

## Existing low-level propagation API

The low-level API remains available when an Agent needs explicit tape or engine control:

```python
tape = tcp.GateTape(3)
tape.h(0)
tape.cnot(0, 1)
tape.rz(1, parameter=0)

observable = tcp.PauliOperator.from_terms(3, (("ZII", 1.0),))
engine = tcp.PropagationEngine(tape, observable, max_weight=3)
result = engine.value_and_grad([0.125])
```

`PropagationEngine` propagates the observable in reverse Heisenberg order. `max_weight=None` or a cutoff at least as large as `nqubits` is exact; a finite cutoff applies deterministic Pauli-weight projection after same-word contributions have been aggregated. The gradient is for the executed frozen sparse trace, not a dense derivative at support-change points.

The Phase Alpha facade adds the corresponding value-only form `circuit.expectation(observable, parameters=...)`; it must agree with `value_and_grad(...).value` for deterministic execution without computing a gradient.

`SPPSEngine` provides seeded stochastic value-and-gradient estimates with fixed or adaptive per-term sample budgets. Its result includes standard-error and stopping-proxy metadata and must not be interpreted as a deterministic gradient result.

## U(1) semantics

`U1Sector` and `U1Circuit` use TensorCircuit computational-basis integer ordering. `U1Circuit` stores and executes only the fixed-Hamming-weight sector; `to_dense()` and `probability_full()` are explicit full-space terminals. The native restricted implementation supports arbitrary-width packed occupation limbs, while full-space materialization remains subject to the public `DEFAULT_MAX_BYTES` guard.

## TensorCircuit conversion

TensorCircuit is the required ecosystem dependency. User-facing conversion uses target-type classmethods:

```python
native_u1 = tcp.U1Circuit.from_circuit(tc_u1_circuit)
native_propagation = tcp.PropagationCircuit.from_circuit(tc_circuit)
```

Low-level QIR restoration remains available through `from_qir()`. Numeric QIR produces static gates; direct symbolic references produce parameter slots. TensorCircuit gate objects are normalized at the boundary to a static logical payload, especially for `diagonal` gates.

## Development

The local quality gate is:

```bash
python scripts/check.py --benchmark smoke
```

The full release checks include Rust formatting, Clippy, Rust tests, Black, Ruff, strict mypy, release maturin installation, Python tests, and benchmark harness smoke tests. TensorCircuit differential tests use the supported TensorCircuit installation and compare ordering, gate conventions, state/observable values, backend MVP results, and native conversion behavior.

See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/vibe/phase-alpha-spec.md`](docs/vibe/phase-alpha-spec.md), [`docs/vibe/semantics.md`](docs/vibe/semantics.md), and [`docs/vibe/releasing.md`](docs/vibe/releasing.md).

## License

Apache License 2.0.
