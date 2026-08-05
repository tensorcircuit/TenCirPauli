# Deep-Scan Checklist & Prompt

> **Purpose.** A reusable, prompt-shaped checklist for deep audits of this codebase. Use it two ways: (1) as a finder prompt skeleton when launching an audit workflow (each `###` group maps to a finder dimension); (2) as a human reminder of what to point an agent at. This file is the *lens*; the per-round audit reports (`audit-report-*-round*.md`) are the *findings*.
>
> **How to use as a prompt.** Pick the groups relevant to the round, prepend the exclusion ledger from the last audit (so the agent does not re-raise fixed/closed items), and ask for structured findings with `file:lines` evidence and an adversarial self-check. Require each finding to state the magnitude/quantified benefit before accepting it.

---

## Layer 1 — Generic software defects (always scan)

### Correctness & numerical
- Off-by-one in rank/unrank/index arithmetic; sector/basis indexing mismatches between construction and application.
- Phase/sign errors in algebraic products, commutators, adjoints, Weyl/qudit products, Majorana reordering.
- Integer overflow in non-checked paths (estimates, sizes, strides); sentinel values (`usize::MAX`, etc.) merged into later checked arithmetic.
- `f64`/`complex128` precision loss: large-integer-to-float casts before mod reduction; accumulation order; cancellation.
- NaN/Inf propagation; division-by-zero on degenerate inputs; empty-collection edge cases (0 terms, 0 qubits, dimension 0).
- Thread-safety: hidden `static mut`, shared `RefCell`/`Mutex` across `py.allow_threads`, `Rc`/`Arc` cycles, non-`Sync` types captured by `allow_threads` closures.
- Determinism: hash-map iteration order leaking into serialized output, grouping results, or tests.

### API & contract
- Public methods whose return type lets callers construct invalid internal state; missing input validation at the boundary.
- Error handling: panics instead of `Result`; `unwrap`/`expect` on FFI inputs or sizes; errors that swallow context.
- Stale/inaccurate docstrings, `.pyi` stubs diverging from the native implementation, default values missing in stubs.

### Robustness & edge cases
- Empty / single-element / max-size inputs on every public entry point.
- Mismatched operand dimensions/shapes/encodings that should fail fast but silently corrupt.
- `max_bytes` / budget guards: enforced inside the engine independent of any reporting/profiling output?

---

## Layer 2 — Performance & architecture (always scan)

### Rust core
- Per-call scratch allocation in hot loops (matrix-free `apply_into`, reverse-AD maps, hashmap rebuilds) where a reusable buffer would do.
- `.clone()` of large/heap-owning types (Vec, enum-with-Vec variants) on hot per-term/per-gate paths.
- O(n²) where hashing/sorting/prefix-aggregation gives O(n log n) or O(n); recomputation of invariants per apply that a one-time plan compile would cache.
- Repeated canonicalization / re-validation of trusted handles on every call.
- GIL not released around long native computation; `py.allow_threads` that only wraps serialization rather than the real work.

### Cross-boundary (the highest-yield area for this project)
- Python loops over native data that then feed back into another native call (the "materialize → rebuild Python objects → reconvert" anti-pattern).
- Per-term / per-gate FFI crossings where one batched opaque call would do.
- Handle churn: create handle → immediately `materialize()` to Python → re-feed into Rust.
- Nested `Sequence[Sequence[...]]` returns from FFI that force per-element Python int/tuple allocation; check whether a flat-numpy or handle-accepting variant exists and should be preferred.
- Inconsistent ABI across sibling methods (some return `Complex128` arrays, others split re/im; some accept numpy, others nested lists).

### Memory & scaling
- Transient allocation proportional to output size purely for format shuffling (e.g. split-then-rezip complex values that are layout-identical to the source).
- Retained Python-side parallel storage of data already held in a native handle.
- Scaling cliffs: algorithms that work at bench size but blow up at 2× term count or +1 qubit.

---

## Layer 3 — Test coverage (always scan)

- Hot native entry points (engines, plans, apply methods, estimators) with **no numerical correctness test** — only shape/metadata/laziness/replay assertions. "Looks like a test" is not a test.
- Branches gated by size/count thresholds (e.g. `if len >= PARALLEL_THRESHOLD`) where the only exerciser is a shape-only benchmark outside `tests/`.
- Adaptive / convergence / budget-growth logic whose defining behavior (per-term divergent budgets, convergence comparisons) is never asserted.
- Diagnostic/profile return fields that are computed but never validated by any test.
- Gradient paths covered only via the value path's tests (shared aggregator ≠ independent coverage — verify the shared code is actually identical, not assumed).
- Methods covered only indirectly through a facade, where the plan-level wrapper is a separate public surface with its own delegation wiring.

---

## Layer 4 — AI-authored-code failure modes (this project's recurring debt)

These are the failure patterns seen across audit rounds 1–3; weight them heavily because AI agents repeat them.

- **Re-derivation instead of reuse.** An operation implemented from lower-level primitives when an existing native method already does it (grep failed before writing). Symptom: duplicated logic that silently diverges from the canonical implementation. *Check: for every Python-side or Rust-side operation, is there already a native primitive doing it?*
- **Abandoned paths left alive.** After a design pivot, the old representation/helpers/branches remain, often with `cast(Any, ...)` or type-suppression keeping them compiling. *Check: grep for `cast(Any`, `# type: ignore`, `#[allow(...)]`; grep for `is not None` branches on fields that are always None on real instances.*
- **Shape-only tests on numerical methods.** A test asserts `.shape`, `is None`, laziness, or metadata but never a known-correct number. *Check: for each native entry point, does any test bind its output to an exact reference / dense oracle / finite-difference?*
- **Unquantified "optimizations."** A performance finding that names a redundancy but never estimates its share of the path's dominant cost. *Check: require the finder to state the magnitude and the dominant term; reject if off by orders of magnitude.*
- **First-occurrence-only fixes.** A bug fixed at one site while sibling occurrences of the same pattern remain. *Check: after any fix, grep the codebase for the pattern; ask "where else does this appear?"*
- **Inconsistent read-back ABI.** A new FFI returns nested Python lists when siblings already use flat numpy or handle-accepting. *Check: is the new method consistent with the canonical read-back ABI?*

---

## Layer 5 — Project-specific hot spots (TenCirPauli)

Prioritize these files/paths; they carry the most traffic and the most prior debt. Listed at module granularity so the guidance stays valid as internals shift; consult the latest audit reports for the concrete issues current at scan time.

- `crates/tencir-pauli-core/src/propagation.rs` — forward/reverse AD, per-transition key cost, reverse-pass index maps, batch worker storage estimation, max_weight projection.
- `crates/tencir-pauli-core/src/charge.rs` — generic charge apply scratch reuse, fermion parity computation, non-conserved aggregation keys, parallel-CSR branch coverage.
- `crates/tencir-pauli-core/src/structured.rs` — qudit/Weyl phase handling, hybrid product canonical invariants, large-dimension apply paths.
- `crates/tencir-pauli-core/src/u1_circuit.rs` — gate schedule, pair-map symmetry, expectation/gradient correctness, parameter-program evaluation.
- `crates/tencir-pauli-core/src/spps.rs` — adaptive convergence logic, per-term budget allocation, cross-term aggregation, parallelism between fixed and adaptive paths.
- `crates/tencirpauli-native/src/` (operator, structured, majorana, charge_sector, convert) — read-back ABI, GIL release around clone/split, dead/orphaned FFI duplicated by array variants.
- `python/tencirpauli/` (pauli, structured, charge, symmetry, mapping, majorana, propagation_circuit, spps_circuit, u1_circuit) — wrapper thickness, per-term materialization loops, equality/hash/hermiticity native paths, dead or abandoned storage representations.
- `python/tencirpauli/_native.pyi` — stubs matching the actual native surface (arity, defaults, types vs bare `object`).

### Known-open ledger
Items formally open at the time of writing live in the most recent round's audit report and any completion-review follow-up. Before filing a finding, read the latest `audit-report-*-round*.md` and that round's "Dropped findings" section to avoid re-raising closed or already-tracked items. Do not re-raise an open item unless you are showing it is newly incomplete in a way not already recorded.

---

## Output contract for a scan

Each finding must include:
- `file:lines` anchored to the current tree (verified, not assumed — line numbers are for the current scan only and must be re-resolved, never copied from a prior report).
- Concrete failure scenario: what input/state → what wrong/slow outcome.
- Quantified benefit and the path's dominant cost (reject unquantified "this is slow").
- Confirmation it is genuinely new (not in the closed ledger) and that the fix's benefit exceeds its risk.
- For correctness findings: a concrete failing input or reproduction.
- For test-gap findings: the specific untested branch/method, not a vague "more tests needed."
