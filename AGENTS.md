# TenCirPauli Repository Guide for AI Agents

## Mission

TenCirPauli is TensorCircuit's Rust-native companion for Pauli algebra, measurement grouping, Hamiltonian construction, symmetry analysis, and Pauli propagation. Keep the Rust core independent from Python and TensorCircuit while providing a stable TensorCircuit-facing Python API.

## Engineering Priorities

1. Numerical and semantic correctness is the first priority. Algebraic conventions, phases, qubit ordering, truncation semantics, gradients, and public results must be verified against trusted references. Never trade correctness, reproducibility, or explicit error behavior for benchmark numbers.
2. Performance is the second priority and the central reason this project uses Rust. Once correctness is protected by tests, actively push for the fastest practical implementation rather than accepting code that is merely faster than Python. Optimize end-to-end latency, throughput, peak memory, and scaling on representative scientific workloads.

Treat algorithmic complexity, compact data representation, cache locality, allocation behavior, batching, parallelism, and FFI overhead as first-class design concerns. Every material performance claim must be supported by reproducible release-mode benchmarks that include input conversion and Python/Rust boundary costs, not just an isolated kernel.

Correctness tests are the gate for optimization. Maintain trusted small-system references, property and differential tests, and deterministic regression cases. After that gate passes, benchmark realistic sizes, identify the dominant bottleneck, optimize it, and retain the benchmark to prevent regressions.

Judge against representative scientific workloads. Do not over-engineer for implausible numerical extremes — ordinary `f64`/`complex128` overflow or underflow outside the practical range may fail naturally. Do not add arbitrary-precision fallbacks, log-domain rescue paths, or elaborate recovery branches solely for huge coefficients, extreme powers, or similarly unrealistic cases. This does not relax algebraic correctness on supported workloads, nor the checked arithmetic required for dimensions, indices, FFI buffer lengths, and major allocations.

## Architecture

- `crates/tencir-pauli-core/` contains pure Rust algorithms and must not depend on PyO3, Python, NumPy, or TensorCircuit.
- `crates/tencirpauli-native/` contains the thin PyO3 binding and builds the private Python module `tencirpauli._native`.
- `python/tencirpauli/` contains the single public Python package.
- `tensorcircuit-ng` is a required Python runtime dependency; `python/tencirpauli/integrations/tensorcircuit.py` remains the boundary that may directly import TensorCircuit.
- `docs/vibe/architecture.md` is the architecture and scope source of truth.
- `docs/vibe/` contains experimental specifications, design decisions, release notes, and other vibe-coding working documents. Keep this material out of the repository root and maintain `docs/vibe/README.md` as its index.

## Non-Negotiable Rules

- Keep the active milestone label (currently Phase 8.5) out of formal implementation artifacts: filenames and contents under `tests/`, `benchmarks/`, `python/`, `examples/`, and production source files must use capability or behavior names, not labels such as `phase85` or `Phase 8.5`. Milestone labels are allowed in `docs/vibe/` review, specification, status, and archival documents.
- Keep FFI calls coarse-grained. Never cross PyO3 once per Pauli term, gate, or matrix element in a hot path.
- Keep Python wrappers thin. Python owns the public API, friendly input normalization, small control metadata, explicit result materialization, and external-framework integration; Rust owns work that scales with operator terms, gates, term pairs, groups, symmetry rows, charge transitions, or basis states. Prefer lazy Python facades backed by native handles, keep handle-to-handle pipelines in Rust, and never materialize complete native data into Python merely to pass it back into Rust.
- Use a canonical binary symplectic representation and test phase, qubit ordering, and endianness against dense references.
- Keep public outputs deterministic. Hash-map iteration order must not leak into serialized operators, grouping results, or tests.
- Rust propagation supports exact dynamic operators and Pauli-weight projection. Do not reproduce fixed-buffer top-k sparse propagation in the Rust engine.
- Apply weight projection only after contributions with the same Pauli word have been aggregated.
- Native gradients use explicit local derivative/VJP rules. A parameter-dependent coefficient cutoff must not silently enter a gradient-supported path.
- Fail fast for unsupported gates, invalid dimensions, incompatible word lengths, obviously excessive major allocations, and missing required runtime dependencies.
- Keep changes minimal and avoid speculative abstractions that are not required by the current milestone.
- Do not add defensive complexity for situations absent from representative workloads. Explicitly invalid public inputs may be rejected once at the boundary, but trusted native handles must not be repeatedly rescanned or wrapped in recovery machinery for hypothetical edge cases. Treat unnecessary defensive branches, fallbacks, and extreme-only tests as maintainability problems, not improvements.
- Treat public `max_bytes` values as best-effort guards for cheaply estimated major outputs and workspaces, not as exact peak-RSS guarantees. Checked dimension/arithmetic overflow remains mandatory, but do not add complex allocator, FFI, or transient-buffer accounting solely to make `max_bytes` exact.

## Anti-patterns observed in AI-authored code

Recurring failure modes from audit rounds. Non-negotiable; override the "just complete the task" instinct.

- **Reuse before writing.** Before implementing any operation, grep both sides of the boundary for an existing primitive that already does it. Re-wiring an existing native method is strongly preferred over re-deriving logic from lower-level ops. Silently divergent duplicates are correctness hazards, not conveniences.
- **Delete the path you abandoned.** When a pivot leaves a representation, branch, or helper unused, delete it in the same change — including helpers, stubs, and dead branches. Do not silence the type checker to keep a dead branch compiling; a suppressive cast is evidence the path is unused and should be removed. Dead parallel representations must be reasoned about on every future read, so they are the most expensive debt.
- **Numeric tests must assert numeric correctness, not shape.** A numerical method is not covered by a test that only asserts shape, laziness, replay equality, or metadata counts. Every native entry point that produces a value or gradient must have at least one test binding its output to a known-correct number. "Looks like a test" is not a test.
- **Profile before optimizing, and quantify the win against the dominant term.** Never propose a performance fix without first identifying the dominant cost of the path and the fix's share of that cost. A redundancy that saves a handful of operations on a path dominated by millions of unavoidable operations is not worth the refactor risk. State the magnitude explicitly; if the claimed savings are off by orders of magnitude, the finding is wrong.
- **Fix the pattern, not the first occurrence.** When fixing a bug or applying a review comment, grep for the same pattern and fix all instances in one change. Leaving siblings unfixed creates inconsistent behavior and guarantees a re-audit on the same issue. Before declaring done, ask "where else does this pattern appear?"
- **Use one read-back ABI.** All native handle read-back paths must return flat NumPy arrays or accept a handle directly — never deeply nested Python sequences that the wrapper must rebuild into typed objects and re-feed into another FFI call. Prefer handle-accepting FFI when the wrapper only needs the result back in Rust. Adding a new nested-list read-back variant is a regression.

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
- Keep complete O(n), O(term-pair count), mapping, compilation, conversion, and execution preparation inside the GIL-released Rust section; releasing the GIL only after Python has serialized or cloned the full workload does not satisfy this rule.

## Python Standards

- Keep the public API in Python modules and treat `_native` as private implementation detail.
- Use type hints for public Python APIs and provide concise public docstrings.
- Format Python with Black, lint it with Ruff, and type-check the public package with strict mypy. Do not add Pylint alongside Ruff unless a documented gap justifies the duplicate tool.
- Validate friendly input forms in Python, then make one batched native call.
- Store scalable symbolic state in private native handles and materialize terms or arrays only when the public API explicitly requests them. Do not retain parallel Python storage, production reference kernels, compatibility probes, or silent Python fallbacks after the required native path exists.
- Keep TensorCircuit imports inside the Python boundary; Rust core code must never import or depend on TensorCircuit.
- Run `black --check python tests benchmarks scripts examples`, `ruff check python tests benchmarks scripts examples`, and `mypy` for Python quality validation.
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
