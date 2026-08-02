# TenCirPauli Repository Guide for AI Agents

## Mission

TenCirPauli is a standalone Rust-native library for Pauli algebra, measurement grouping, Hamiltonian construction, symmetry analysis, and Pauli propagation. Keep the Rust core independent from Python and TensorCircuit while providing a stable Python API and an optional TensorCircuit adapter.

## Engineering Priorities

1. Numerical and semantic correctness is the first priority. Algebraic conventions, phases, qubit ordering, truncation semantics, gradients, and public results must be verified against trusted references. Never trade correctness, reproducibility, or explicit error behavior for benchmark numbers.
2. Performance is the second priority and the central reason this project uses Rust. Once correctness is protected by tests, actively push for the fastest practical implementation rather than accepting code that is merely faster than Python. Optimize end-to-end latency, throughput, peak memory, and scaling on representative scientific workloads.

Treat algorithmic complexity, compact data representation, cache locality, allocation behavior, batching, parallelism, and FFI overhead as first-class design concerns. Prefer established high-performance scientific-computing techniques and idiomatic, zero-cost Rust abstractions. Every material performance claim or optimization must be supported by reproducible release-mode benchmarks and profiling; include input conversion and Python/Rust boundary costs rather than timing only an isolated kernel.

Correctness tests are the gate for optimization. Maintain dense or otherwise trusted small-system references, property and differential tests, numerical tolerances, and deterministic regression cases. After that gate passes, benchmark realistic term counts and qubit counts, identify the dominant bottleneck, optimize it, and retain the benchmark to prevent regressions.

## Architecture

- `crates/tencir-pauli-core/` contains pure Rust algorithms and must not depend on PyO3, Python, NumPy, or TensorCircuit.
- `crates/tencirpauli-native/` contains the thin PyO3 binding and builds the private Python module `tencirpauli._native`.
- `python/tencirpauli/` contains the single public Python package.
- `python/tencirpauli/integrations/tensorcircuit.py` is the only place that may directly import TensorCircuit.
- `docs/vibe/architecture.md` is the architecture and scope source of truth.
- `docs/vibe/` contains experimental specifications, design decisions, release notes, and other vibe-coding working documents. Keep this material out of the repository root and maintain `docs/vibe/README.md` as its index.

## Non-Negotiable Rules

- Keep FFI calls coarse-grained. Never cross PyO3 once per Pauli term, gate, or matrix element in a hot path.
- Use a canonical binary symplectic representation and test phase, qubit ordering, and endianness against dense references.
- Keep public outputs deterministic. Hash-map iteration order must not leak into serialized operators, grouping results, or tests.
- Rust propagation supports exact dynamic operators and Pauli-weight projection. Do not reproduce fixed-buffer top-k sparse propagation in the Rust engine.
- Apply weight projection only after contributions with the same Pauli word have been aggregated.
- Native gradients use explicit local derivative/VJP rules. A parameter-dependent coefficient cutoff must not silently enter a gradient-supported path.
- Fail fast for unsupported gates, invalid dimensions, incompatible word lengths, obviously excessive major allocations, and missing optional integrations.
- Keep changes minimal and avoid speculative abstractions that are not required by the current milestone.
- Treat public `max_bytes` values as best-effort guards for cheaply estimated major outputs and workspaces, not as exact peak-RSS guarantees. Checked dimension/arithmetic overflow remains mandatory, but do not add complex allocator, FFI, or transient-buffer accounting solely to make `max_bytes` exact.

## Rust Standards

- Use stable Rust and keep the minimum supported Rust version declared in the workspace manifest.
- Prefer safe Rust. Any `unsafe` block requires a local safety argument and dedicated tests.
- Design hot paths around compact contiguous storage, borrowed slices, preallocated capacity, and reusable scratch buffers. Eliminate unnecessary cloning, temporary collections, per-term allocation, and repeated canonicalization.
- Prefer asymptotic and data-layout improvements before micro-optimization. Use specialized hashing, Rayon, explicit SIMD, or architecture-specific kernels only when profiling identifies the bottleneck and benchmarks demonstrate a representative end-to-end gain.
- Measure performance with optimized release builds. Debug-build timings are not evidence, and microbenchmarks must not replace end-to-end workload benchmarks.
- Track runtime, throughput, peak memory, allocation behavior, and scaling where relevant. Compare against the best applicable TensorCircuit/Python/JAX baseline with equivalent semantics and accuracy.
- Run `cargo fmt --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and `cargo test --workspace`.
- Add property tests for algebraic identities and deterministic regression tests for canonical ordering.
- Release the GIL around long-running native computation and avoid hidden global mutable state.

## Python Standards

- Keep the public API in Python modules and treat `_native` as private implementation detail.
- Use type hints for public Python APIs and provide concise public docstrings.
- Format Python with Black, lint it with Ruff, and type-check the public package with strict mypy. Do not add Pylint alongside Ruff unless a documented gap justifies the duplicate tool.
- Validate friendly input forms in Python, then make one batched native call.
- Keep TensorCircuit imports lazy and optional.
- Run `black --check python tests benchmarks scripts`, `ruff check python tests benchmarks scripts`, and `mypy` for Python quality validation.
- Run `maturin develop --release` followed by `pytest` for integration validation.

## Packaging and Compatibility

- The public distribution and import package are both named `tencirpauli`.
- The Rust core may be published independently as `tencir-pauli-core`; the binding crate is not a second Python distribution.
- Common platforms receive wheels, and an sdist remains available for source builds.
- Do not compile public wheels with `target-cpu=native`.
- Keep local paths and environment details out of tracked files. Read `AGENTS.local.md` when it exists, but never commit it.

## Testing Priorities

- Test Pauli multiplication, commutation, adjoint, support, weight, and canonical round trips.
- Differentially test Hamiltonian dense/COO/CSR/MVP targets on small systems.
- Test QWC and general-commuting groups separately, including basis-change reconstruction.
- Compare native propagation values and gradients with the same weight-truncated recurrence in a trusted reference.
- Separate cold build/setup, first execution, steady execution, memory, and accuracy in benchmarks.
- Keep performance tracking local and informational unless the project explicitly changes this policy. Do not add wall-time CI gates or external benchmark services by default.
- Add stable Rust microbenchmarks and Python/FFI integration benchmarks for every material hot path. Record and compare them with `python benchmarks/run.py` on the same machine.
- Never commit `.benchmarks/` results; preserve benchmark source, workload definitions, and documentation in the repository.
- Before a local commit, run `python scripts/check.py` using the project environment. Use `--fix` before staging when formatting changes are expected; the Git hook itself must remain check-only so staged content cannot silently diverge from formatter output.
