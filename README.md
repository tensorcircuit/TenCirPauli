# TenCirPauli

TenCirPauli is a Rust-native Pauli algebra, deterministic measurement grouping, and Hamiltonian compiler with a typed Python API compatible with TensorCircuit's Pauli codes and qubit-ordering conventions.

The Phase 1 public surface includes phase-free `PauliWord`, canonical `PauliOperator`, QWC and general-commuting grouping results, dense/COO/CSR Hamiltonian targets, native matrix-free MVP, and a versioned backend MVP plan. Symmetry analysis, GateTape, Pauli propagation, and native gradients are intentionally outside Phase 1.

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

from tencirpauli import PauliOperator, PauliWord

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
```

Explicit `dense`, `coo`, `csr`, and MVP targets use the public `DEFAULT_MAX_BYTES` budget, currently 4 GiB. Pass `max_bytes` per call to lower the safety budget for a memory-constrained job or raise it when the host has enough RAM; a `MemoryError` reports the estimated request before a large allocation is attempted.

Use `native_mvp_plan()` when applying the same static Hamiltonian repeatedly. It precomputes phase structure in Rust, releases the GIL during application, and avoids rebuilding the operator on every statevector call. Use `backend_mvp_plan()` when the calculation must remain inside a TensorCircuit backend and JAX autodiff/JIT is required.

`PauliOperator.canonicalize_batch()` is the dynamic/backend-facing batch form: it returns canonical structures, aggregated coefficients including exact-zero keys, `input_to_canonical`, and exact phase multipliers. Static `PauliOperator.from_terms()` keeps its faster exact-zero-dropping path.

`PauliOperator.group_commuting(mode="general")` returns an explicitly algebraic prototype with `measurement_ready=False`; it must not be used as a local single-qubit measurement plan. QWC reconstruction uses the returned group masks and rotated measurement bitstrings.

## Development checks

Run `python scripts/check.py --fix --benchmark smoke` while editing and `python scripts/check.py` before a local commit. The repository also records release-mode Rust Criterion and Python pytest-benchmark results under the ignored `.benchmarks/` directory; use `python benchmarks/run.py compare <baseline-label>` for same-machine comparisons.

The independent dense NumPy oracle and fixed P0 regression vectors are documented in [`docs/vibe/reference-vectors.md`](docs/vibe/reference-vectors.md). The durable implementation evidence and next milestone are tracked in [`docs/vibe/implementation-status.md`](docs/vibe/implementation-status.md).

## License

Apache License 2.0.
