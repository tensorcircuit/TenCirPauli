# TenCirPauli

TenCirPauli is a Rust-native Pauli algebra, deterministic measurement grouping, and Hamiltonian compiler with a typed Python API compatible with TensorCircuit's Pauli codes and qubit-ordering conventions.

The public surface includes phase-free `PauliWord`, canonical `PauliOperator`, QWC and general-commuting grouping results, dense/COO/CSR Hamiltonian targets, native matrix-free MVP, a versioned backend MVP plan, Pauli Z2 symmetry/tapering plans, and explicit U(1) restricted-sector operators. GateTape, Pauli propagation, and native gradients remain outside the current scope.

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
```

Explicit `dense`, `coo`, `csr`, and MVP targets use the public `DEFAULT_MAX_BYTES` best-effort guard, currently 4 GiB. Pass `max_bytes` per call to reject cheaply estimated major outputs or workspaces, or raise it when the host has enough RAM. This is not an exact peak-RSS limit: allocator overhead, FFI conversion and transient buffers may exceed the estimate, so an operating-system out-of-memory failure remains possible.

Use `native_mvp_plan()` when applying the same static Hamiltonian repeatedly. It precomputes phase structure in Rust, releases the GIL during application, and avoids rebuilding the operator on every statevector call. Use `backend_mvp_plan()` when the calculation must remain inside a TensorCircuit backend and JAX autodiff/JIT is required.

`PauliOperator.canonicalize_batch()` is the dynamic/backend-facing batch form: it returns canonical structures, aggregated coefficients including exact-zero keys, `input_to_canonical`, and exact phase multipliers. Static `PauliOperator.from_terms()` keeps its faster exact-zero-dropping path.

For large numeric batches, use `PauliOperator.from_code_arrays()` to construct a static operator or `PauliOperator.canonicalize_code_arrays_numpy()` to keep canonical structures, coefficients, mappings, and phases in contiguous read-only NumPy arrays without materializing Python objects per term.

`PauliOperator.group_commuting(mode="general")` returns an explicitly algebraic prototype with `measurement_ready=False`; it must not be used as a local single-qubit measurement plan. QWC reconstruction uses the returned group masks and rotated measurement bitstrings.

## Symmetry and restricted sectors

Z2 analysis is exposed as `analysis = h.find_z2_symmetries(max_bytes=...)`. It returns `analysis.generators`, `analysis.rank`, and `analysis.constraint_rank`; select a sector with `plan = analysis.tapering_plan((+1, -1, ...))`, then call `plan.transform_operator(h)` or reuse the same plan for compatible observables. The transformed result is an ordinary smaller `PauliOperator`, so its existing `.dense()`, `.coo()`, `.csr()`, `.mvp()`, `.native_mvp_plan()`, and `.backend_mvp_plan()` targets remain available. `h.taper_z2(sector=...)` is the one-shot convenience form.

U1 is explicit rather than automatically inferred: `sector = U1Sector(nqubits=h.nqubits, particle_number=k)`. Use `sector.dimension`, `sector.rank(bitstring)`, `sector.unrank(index)`, and `sector.basis_words()` for the fixed-Hamming-weight basis. `restricted = h.restrict_u1(sector)` validates sector preservation after duplicate-transition aggregation; it exposes reduced-space `.apply(state)`, `.mvp_plan().apply(state)`, `.dense()`, `.coo()`, and `.csr()`. The U1 API never materializes a full-space dense/COO matrix or a `2**n` statevector. Python basis helpers support packed multiword states, while the current native restricted Hamiltonian/MVP/CSR path uses one `usize` computational-basis index and explicitly rejects `nqubits >= usize` bit width; multiword native restriction is planned for Phase 5.

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

Run `python scripts/check.py --fix --benchmark smoke` while editing and `python scripts/check.py` before a local commit. The repository also records release-mode Rust Criterion and Python pytest-benchmark results under the ignored `.benchmarks/` directory; use `python benchmarks/run.py compare <baseline-label>` for same-machine comparisons.

The independent dense NumPy oracle and fixed P0 regression vectors are documented in [`docs/vibe/reference-vectors.md`](docs/vibe/reference-vectors.md). The durable implementation evidence and next milestone are tracked in [`docs/vibe/implementation-status.md`](docs/vibe/implementation-status.md).

## License

Apache License 2.0.
