# TenCirPauli

TenCirPauli is a Rust-native Pauli algebra, deterministic measurement grouping, Hamiltonian compiler, and Heisenberg Pauli propagation engine with a typed Python API compatible with TensorCircuit's Pauli codes and qubit-ordering conventions.

The public surface includes phase-free `PauliWord`, canonical `PauliOperator`, QWC and general-commuting grouping results, dense/COO/CSR Hamiltonian targets, native matrix-free MVP, a versioned backend MVP plan, Pauli Z2 symmetry/tapering plans, explicit U(1) restricted-sector operators, and Rust-native deterministic and stochastic value-and-gradient engines built on `GateTape`.

## Architecture

```text
tencir-pauli-core        Pure Rust algebra, grouping, and Hamiltonian algorithms
        │
        ▼
tencirpauli-native       Thin PyO3 batch facade, private tencirpauli._native module
        │
        ▼
tencirpauli              Typed public Python package
        └── integrations.tensorcircuit   Optional lazy backend-plan adapter
```

The Rust core uses external codes `0=I`, `1=X`, `2=Y`, `3=Z`, packed qubit zero as LSB, and exact four-valued multiplication phases. Matrix targets explicitly map qubit zero to the MSB, matching TensorCircuit's computational-basis ordering. Native numeric coefficients are complex128-compatible and duplicate terms are aggregated deterministically before exact-zero removal.

## Installation

Install a released wheel or source distribution with `pip install tencirpauli`. For local development, create an environment containing Python, Rust/Cargo, maturin, NumPy, pytest, and the quality tools, then run `maturin develop --release`.

TensorCircuit integration is optional: `pip install 'tencirpauli[tensorcircuit]'`. If it is not installed, importing `tencirpauli` still works; explicitly requesting the adapter raises an actionable `ImportError`.

## Python example

```python
import numpy as np
import tencirpauli as tcp

from tencirpauli import PauliOperator, PauliWord, U1Sector

word = PauliWord.from_string("XYZ")
product = PauliWord.from_string("X").multiply(PauliWord.from_string("Y"))
assert product.word.to_string() == "Z"

hamiltonian = PauliOperator.from_terms(
    3,
    (("ZZI", 1.0), ("IZZ", 0.5), ("XII", 0.25)),
)
groups = hamiltonian.group_commuting(mode="qubit_wise")
matrix = hamiltonian.dense()
state = np.ones(2**3, dtype=np.complex128)
np.testing.assert_allclose(hamiltonian.mvp(state), matrix @ state)
plan = hamiltonian.backend_mvp_plan()
np.testing.assert_allclose(plan.apply(state), matrix @ state)
native_plan = hamiltonian.native_mvp_plan()
np.testing.assert_allclose(native_plan.apply(state), matrix @ state)

analysis = hamiltonian.find_z2_symmetries()
if analysis.rank:
    tapered = analysis.tapering_plan((1,) * analysis.rank).transform_operator(
        hamiltonian
    )

number_conserving = PauliOperator.from_terms(
    3, (("XXI", 0.5), ("YYI", 0.5))
)
sector = U1Sector(3, 1)
restricted = number_conserving.restrict_u1(sector)
next_state = restricted.apply(np.ones(sector.dimension, dtype=np.complex128))

tape = tcp.GateTape(3)
tape.h(0)
tape.cnot(0, 1)
tape.rz(1, parameter=0)
observable = PauliOperator.from_terms(3, (("ZII", 1.0),))
engine = tcp.PropagationEngine(tape, observable, max_weight=3)
value = engine.expectation([0.125])
materialized = engine.propagate_operator([0.125])
```

Explicit `dense`, `coo`, `csr`, MVP targets, and propagation engines use the public `DEFAULT_MAX_BYTES` best-effort guard, currently 16 GiB. Pass `max_bytes` per call to reject cheaply estimated major outputs or workspaces, pass `None` to disable that guard, or raise it when the host has enough RAM. This is not an exact peak-RSS limit: allocator overhead, FFI conversion and transient buffers may exceed the estimate, so an operating-system out-of-memory failure remains possible.

Use `native_mvp_plan()` when applying the same static Hamiltonian repeatedly. It precomputes phase structure in Rust, releases the GIL during application, and avoids rebuilding the operator on every statevector call. Use `backend_mvp_plan()` when the calculation must remain inside a TensorCircuit backend and JAX autodiff/JIT is required.

`PauliOperator.canonicalize_batch()` is the dynamic/backend-facing batch form: it returns canonical structures, aggregated coefficients including exact-zero keys, `input_to_canonical`, and exact phase multipliers. Static `PauliOperator.from_terms()` keeps its faster exact-zero-dropping path.

For large numeric batches, use `PauliOperator.from_code_arrays()` to construct a static operator or `PauliOperator.canonicalize_code_arrays_numpy()` to keep canonical structures, coefficients, mappings, and phases in contiguous read-only NumPy arrays without materializing Python objects per term.

`PauliOperator.group_commuting(mode="general")` returns an explicitly algebraic prototype with `measurement_ready=False`; it must not be used as a local single-qubit measurement plan. QWC reconstruction uses the returned group masks and rotated measurement bitstrings.

## Rust-native propagation

`GateTape` records gates in Schrödinger execution order. `PropagationEngine` traverses that tape in reverse for Heisenberg propagation, using exact recurrence when `max_weight=None` or the cutoff is at least `nqubits`, and applying a deterministic Pauli-weight projection after each aggregated gate when a finite cutoff is selected. The supported built-ins are `X/Y/Z/H/S/Sdg/CNOT/CZ/SWAP`, `RX/RY/RZ/RXX/RYY/RZZ`, and finite real one- or two-qubit PTMs.

The default `expectation(parameters)` path stays in Rust and returns one scalar for `ZeroState`, `ComputationalBasisState`, or `ProductBlochState`. Use `propagate_operator()` only when the full canonical operator is needed; use `profile()` for explicit structural and timing metadata. Parameter slots are immutable after engine construction, while a mutable `GateTape` can continue to be used to build other engines.

`PropagationEngine.value_and_grad()` returns a read-only `float64` gradient for the executed sparse trace. Its zero-branch, duplicate-aggregation, and `max_weight` support decisions are frozen to that forward call; it is not a dense/fixed-basis derivative at support-change points. `SPPSEngine.value_and_grad()` provides seeded fixed-budget stochastic value-and-gradient estimates with positive smoothing, stable PAD, and per-term sample budgets; `value_and_grad_adaptive()` adds the two-replicate empirical stopping proxy. Both calls are coarse-grained native operations and release the GIL.

Use `PropagationBatch(tape, observables)` when independent observable rows must be retained. It shares the immutable compiled tape/state program, returns `(B,)` expectations or `(B, nparameters)` frozen-support gradients, and parallelizes only sufficiently heavy observable batches while preserving input-row order and deterministic per-row results.

## Symmetry and restricted sectors

Z2 analysis is exposed as `analysis = h.find_z2_symmetries(max_bytes=...)`. It returns `analysis.generators`, `analysis.rank`, and `analysis.constraint_rank`; select a sector with `plan = analysis.tapering_plan((+1, -1, ...))`, then call `plan.transform_operator(h)` or reuse the same plan for compatible observables. The transformed result is an ordinary smaller `PauliOperator`, so its existing `.dense()`, `.coo()`, `.csr()`, `.mvp()`, `.native_mvp_plan()`, and `.backend_mvp_plan()` targets remain available. `h.taper_z2(sector=...)` is the one-shot convenience form.

U1 is explicit rather than automatically inferred: `sector = U1Sector(nqubits=h.nqubits, particle_number=k)`. Use `sector.dimension`, `sector.rank(bitstring)`, `sector.unrank(index)`, and `sector.basis_words()` for the fixed-Hamming-weight basis. `restricted = h.restrict_u1(sector)` validates sector preservation after duplicate-transition aggregation; it exposes reduced-space `.apply(state)`, `.mvp_plan().apply(state)`, `.dense()`, `.coo()`, and `.csr()`. The U1 API never materializes a full-space dense/COO matrix or a `2**n` statevector. Native restricted Hamiltonian/MVP/CSR now uses arbitrary-width packed `uint64` occupation limbs while restricted logical and public sparse indices remain checked bounded indices; wide `basis_words()` returns a read-only `(dimension, word_count)` packed array. U1 circuit execution and time evolution remain Phase 6 scope.

```python
analysis = h.find_z2_symmetries()
if analysis.rank:
    tapered = analysis.tapering_plan((1,) * analysis.rank).transform_operator(h)
    tapered_plan = tapered.native_mvp_plan()

sector = U1Sector(nqubits=h.nqubits, particle_number=2)
restricted = h.restrict_u1(sector)
restricted_plan = restricted.mvp_plan()
next_state = restricted_plan.apply(np.ones(sector.dimension, dtype=np.complex128))
restricted_csr = restricted.csr()
```

## Development checks

Run `python scripts/check.py --fix --benchmark smoke` while editing. The repository hook uses this fast smoke mode by default: it runs the full correctness/quality checks, compiles all Rust benchmark targets, and runs only Python benchmark cases outside the `performance_large` marker. The complete release benchmark record is intentionally opt-in; run `python scripts/check.py --benchmark record` or use `TENCIRPAULI_PRE_COMMIT_BENCHMARK=record git commit ...` when a performance checkpoint is needed. `--benchmark skip` is available for code-only iteration after the regular smoke gate. Results are stored under the ignored `.benchmarks/` directory; use `python benchmarks/run.py compare <baseline-label>` for same-machine comparisons.

The independent dense NumPy oracle and fixed P0 regression vectors are documented in [`docs/vibe/reference-vectors.md`](docs/vibe/reference-vectors.md). The durable implementation evidence and next milestone are tracked in [`docs/vibe/implementation-status.md`](docs/vibe/implementation-status.md).

## License

Apache License 2.0.
