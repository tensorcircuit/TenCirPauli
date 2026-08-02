# Phase 3 Implementation Review (Archived)

Review date: 2026-08-02

Review scope: Phase 3 Rust-native Pauli propagation changes from `84d4a5e..HEAD`, with `docs/vibe/phase-3-spec.md` as the acceptance contract. The review covers numerical correctness, error behavior, memory guards, hot-path performance, FFI/materialization, test coverage, benchmarks, and documentation accuracy.

No source files were changed. This report is the only file added by the review.

## Executive conclusion

Phase 3 has a sound basic architecture and the ordinary exact/projected recurrence appears numerically correct. A focused remediation is justified before treating its performance claims as complete: the scalar expectation path performs work explicitly reserved for operator materialization/profile calls, the measured local transition hotspot remains avoidably full-width, and a rare but supported finite-input overflow can silently return zero. Several other findings below are specification-compliance debt or profile-gated opportunities rather than release blockers; they should not be implemented mechanically.

## Pragmatic priority and anti-overengineering triage

This table supersedes the raw adversarial severity labels when planning work. The original issue sections remain detailed evidence, but not every contract mismatch deserves implementation effort.

| Priority | Findings | Practical decision |
| --- | --- | --- |
| P0 — genuinely important | MAJOR-0, MAJOR-1, the local-kernel portion of MAJOR-2, and the minimal subset of MAJOR-5 | Fix now. These cover silent wrong output, the central scalar hot-path contract, a profile-confirmed bottleneck, and the tests needed to protect those changes. |
| P1 — cheap, bounded improvements | MAJOR-4, a simplified wide-key payload fix from MAJOR-3, and MINOR-3 | Fix in the same remediation if straightforward. Normalize exact cutoff, include known wide mask bytes without attempting exact RSS accounting, and make status documentation truthful. |
| P2 — profile-gated, not acceptance blockers | Cross-call scratch reuse from MAJOR-2, MAJOR-7, and most of MAJOR-6 | Defer until measurement shows material benefit. Do not build a scratch pool, parallel engine, or new packed public representation merely to satisfy the original report. |
| P3 — specification cleanup or opportunistic hardening | MINOR-1, MINOR-2, MINOR-4, exhaustive portions of MAJOR-5/6 | Usually update/narrow the specification or fix only while touching adjacent code. These do not justify standalone engineering work. |

The smallest high-value remediation is therefore: checked coefficient arithmetic; a direct packed scalar expectation path; O(1) local Clifford/rotation updates; exact-cutoff normalization; a small set of targeted regression/property tests; and corrected status documentation. Everything else requires new profiling evidence or an explicit owner choice to retain the very broad frozen acceptance matrix.

The earlier demand for detailed simultaneous-buffer/hash/FFI accounting was too strict and would conflict with the repository rule that `max_bytes` is only a cheap best-effort guard. The useful fix is limited to obviously omitted, cheaply known wide-key payload; exact allocator or transient accounting should not be built.

## Compliance checklist

| Area | Status | Evidence |
| --- | --- | --- |
| Pure Rust core / thin PyO3 / typed Python facade | PASS | Module placement and dependency boundaries follow the repository architecture. |
| Reverse Heisenberg order, built-in gate conventions, PTM orientation | PASS | Fixed tests pass; an additional review run of 80 seeded random exact cases passed against the independent dense oracle. |
| Per-gate weight projection semantics on ordinary finite workloads | PASS | Existing cases plus an additional review run of 60 seeded random projected cases passed after independently applying the initial projection. |
| Deterministic canonical public operator output | PASS | Aggregation is followed by canonical sorting and no parallel nondeterministic reduction is currently used. |
| Explicit non-finite/error behavior | FAIL | Post-aggregation overflow is silently removed rather than reported; see MAJOR-0. |
| Scalar expectation hot-path contract | FAIL | `expectation()` invokes full public-term conversion, weight statistics, and sorting; see MAJOR-1. |
| 16 GiB / `max_bytes` best-effort guard contract | FAIL | Wide-key heap storage and major simultaneously live buffers/output are not included; see MAJOR-3 and MINOR-1. |
| Reusable scratch and specialized local kernels | FAIL | Per-call/per-gate allocation remains and the recorded dominant key transition is still asymptotically avoidable; see MAJOR-2. |
| Required committed correctness/property matrix | FAIL | The dense oracle and tests omit several frozen cases, including isolated initial projection and randomized committed differential coverage; see MAJOR-5. |
| Required release benchmark/microbenchmark matrix | FAIL | Required duplicate-heavy, thread-scaling, kernel, profile, storage, and boundary splits are absent; see MAJOR-6. |
| Formatting, linting, typing, build, and current regression suite | PASS | `python scripts/check.py --benchmark smoke` passed: 12 Rust tests, 92 Python tests passed with 2 optional skips, and 61 benchmark smoke cases passed with 36 optional skips. |
| Documentation accurately states completion status | FAIL | `implementation-status.md` claims P0-P7 completion and reusable aggregation despite unresolved REQUIRED items; see MINOR-3. |

In this checklist, FAIL means “does not satisfy the original frozen specification,” not automatically “must be engineered now.” The pragmatic triage above determines whether the right action is code, a narrower test, profiling, documentation correction, or an owner-approved scope revision.

## CRITICAL

No high-probability catastrophic defect was found in ordinary scientific workloads. The finite-overflow issue originally placed here is reclassified as MAJOR-0: it is a real correctness bug and cheap to fix, but its `~1e308` trigger makes it a poor reason for a broad architectural rewrite.

## MAJOR

### MAJOR-0: Finite supported inputs can silently collapse to an empty operator after aggregation overflow

In `crates/tencir-pauli-core/src/propagation.rs:648-658`, each incoming contribution is checked for finiteness before insertion, but `and_modify(|current| *current += coefficient)` can overflow two finite values to infinity. The final filter at `crates/tencir-pauli-core/src/propagation.rs:661-668` then treats a non-finite aggregate as a false predicate and silently drops the term instead of returning `NonFiniteCoefficient`.

The issue is reproducible with a valid real PTM that maps both `X` and `Y` to `I`, and an observable containing finite coefficients `1e308 * X + 1e308 * Y`: `propagate_operator([])` returns an empty operator and `expectation([])` returns `-0.0`. The mathematically evaluated coefficient overflows and must fail explicitly; returning zero is silent scientific data corruption.

Resolution: validate the result immediately after every deterministic coefficient addition, return a typed non-finite arithmetic error, and add regressions for aggregation overflow through PTM collisions and rotation collisions. The error must be shared by scalar expectation, profile, and materialization paths.

### MAJOR-1: The scalar expectation path performs operator materialization/statistics work forbidden by the Phase 3 contract

`PropagationEngine::expectation()` calls `self.propagate()` at `crates/tencir-pauli-core/src/propagation.rs:158-168`. `propagate()` converts every `PackedKey` to a heap-backed public `PauliWord`, computes full weight counts, constructs `PauliTerm` values, and sorts them at `crates/tencir-pauli-core/src/propagation.rs:240-253`; only afterward does expectation scan those public terms at `crates/tencir-pauli-core/src/propagation.rs:866-898`.

This conflicts with `docs/vibe/phase-3-spec.md:125`, which reserves canonical sorting and public-word construction for operator materialization/profile calls, and with `docs/vibe/phase-3-spec.md:247-251`, which defines scalar expectation as the main Rust-only hot path. It adds allocation, wide-key cloning, conversion, weight counting, and `O(T log T)` sorting to every scalar evaluation.

Resolution: split propagation into an internal dynamic-term result and optional diagnostics. Evaluate the product-state expectation directly from `PackedKey` plus coefficients; only `propagate_operator()` should convert/sort public words, and only `profile()` should compute full structural statistics and timing. Add a benchmark that compares scalar, profile, Rust propagation-only, and full Python materialization boundaries on the same workload.

### MAJOR-2: The dominant local transition remains avoidably `O(nqubits)`, and reusable scratch is not implemented

`PackedKey::multiply()` scans every qubit to accumulate phase at `crates/tencir-pauli-core/src/propagation.rs:378-415`. Two-qubit Clifford mapping invokes full-key multiplication twice at `crates/tencir-pauli-core/src/propagation.rs:724-748`, even though the map changes only two local codes. Rotation creates a full generator key and uses the same full scan at `crates/tencir-pauli-core/src/propagation.rs:590-614`, although commutation/sign can be derived from one or two local codes or packed parity operations.

The engine stores no scratch state at `crates/tencir-pauli-core/src/propagation.rs:49-60`; each Clifford gate creates a new vector at `crates/tencir-pauli-core/src/propagation.rs:561-588`, and every branching gate creates fresh contribution vectors and a fresh hash map at `crates/tencir-pauli-core/src/propagation.rs:596-671`. This contradicts the reusable current/next storage requirement in `docs/vibe/phase-3-spec.md:306-312`.

The project profile already reports `PackedKey::multiply` in 3,236 of 3,868 native samples at `docs/vibe/implementation-status.md:107`, yet the known bottleneck was deferred without the Phase 3 structural optimization required by `docs/vibe/phase-3-spec.md:461-468`. The same release record reports the matched 12-qubit native steady path at about 3.92 ms versus JAX at 0.74 ms (`docs/vibe/implementation-status.md:105`); there is no fixed speed ratio gate, but the frozen process requires continued bottleneck-driven optimization.

Resolution: replace two-qubit Clifford composition with a 16-entry `(output codes, sign)` table and direct bit updates; implement local rotation commutation/product/sign without a full-width identity generator. Re-profile before doing anything else. Introduce capacity reuse within a call only if allocations remain material; do not introduce a cross-call scratch pool without separate evidence.

### MAJOR-3: The propagation memory guard does not cover cheap-to-estimate wide-key or simultaneous major storage

Engine construction estimates every observable term as `size_of::<PackedKey>() + size_of::<Complex64>()` at `crates/tencir-pauli-core/src/propagation.rs:91-101`, but a `PackedKey::Wide` owns two heap vectors whose size grows with `ceil(nqubits/64)`. Runtime accounting similarly takes the maximum of individual vector sizes at `crates/tencir-pauli-core/src/propagation.rs:196-237` instead of accounting for simultaneously live current terms, contributions, aggregation payload, ordered output, and public materialization. Final wide `PauliWord` masks and code output are also omitted.

This is not allocator-overhead exactness: the omitted packed vector lengths and major output arrays are explicitly and cheaply known. A 10,000-qubit, one-term engine successfully constructs and propagates with `max_bytes=1000`; `profile()` reports 72 bytes even though the key alone owns 314 `u64` mask words (2,512 bytes) and operator materialization creates a 10,000-byte code row. This violates `docs/vibe/phase-3-spec.md:259-261` and the public promise in `README.md:76`.

Resolution: the bounded P1 fix is to make wide-key storage representation-aware using `packed_word_count(nqubits)` and add one small-budget regression. More complete simultaneous-buffer or materialized-output accounting is optional and should remain simple; hash bucket overhead, allocator fragmentation, PyO3 objects, and exact RSS must stay excluded.

### MAJOR-4: `max_weight >= nqubits` is semantically exact but still pays projection/weight work

`propagate()` computes `exact` but retains the raw `self.max_weight` as `cutoff` at `crates/tencir-pauli-core/src/propagation.rs:177-180`, then passes that cutoff into every operation at `crates/tencir-pauli-core/src/propagation.rs:208-228`. Consequently Clifford and aggregated paths still call `weight()` at `crates/tencir-pauli-core/src/propagation.rs:568-584` and `crates/tencir-pauli-core/src/propagation.rs:661-668` when `max_weight == nqubits` or larger.

This violates the exact-path requirement at `docs/vibe/phase-3-spec.md:298-304` and `docs/vibe/phase-3-spec.md:434-441`. A review microbenchmark with one 128-qubit term and 1,000 H gates measured about 18.36 microseconds per call for `max_weight=None` and 20.22 microseconds for `max_weight=128`, roughly 10% avoidable overhead.

Resolution: derive an effective cutoff of `None` whenever `is_exact()` is true while preserving the public configured value for introspection, and add a regression/benchmark asserting that `None`, `nqubits`, and larger cutoffs select the same no-projection kernel.

### MAJOR-5: The committed reference and correctness matrix do not prove several REQUIRED claims

The dense oracle constructs the full initial matrix and applies projection only inside the reversed gate loop at `tests/propagation_reference.py:106-133`. It therefore does not implement initial projection for an empty tape. The test named `test_projection_is_initial_and_per_gate_after_aggregation` at `tests/test_propagation.py:90-105` includes a gate and does not isolate initial projection or create a duplicate collision that proves aggregation-before-projection.

The committed suite has no seeded randomized exact/projected property loop required by `docs/vibe/phase-3-spec.md:344-353`, no exhaustive correctness test for `RXX/RYY/RZZ`, no random two-qubit unitary-derived PTM test, no post-aggregation overflow case, no propagation-specific small-budget wide-key test, and no test that demonstrates another Python thread progresses while a long native call releases the GIL. The Rust workspace reports only the 12 pre-Phase-3-style core tests and contains no propagation unit/property test, despite `docs/vibe/phase-3-spec.md:416-423` requiring Rust local-rule tests.

As an audit cross-check, 80 additional seeded random exact cases and 60 additional seeded random projected cases passed against the dense oracle after the reviewer independently applied initial projection. This reduces concern about ordinary recurrence correctness but does not replace durable regression coverage.

Resolution: correct the oracle's initial step; add isolated initial projection, duplicate/cancellation/overflow vectors, direct RXX/RYY/RZZ coverage, and a small seeded exact/projected differential loop. Add Rust tests only for new local tables and checked arithmetic. The larger PTM, hash-seed, GIL-progress, exhaustive boundary, and duplicated Rust/Python matrices are optional coverage, not blockers.

### MAJOR-6: The performance evidence does not satisfy the frozen benchmark contract

The Rust propagation Criterion file contains only one rotation case and three whole-tape cases at `crates/tencir-pauli-core/benches/propagation.rs:81-156`. It does not separately cover inline key hash/equality/weight, Clifford updates, rotation commute versus branch, custom PTM apply, duplicate aggregation, finite projection, or product-state expectation as required by `docs/vibe/phase-3-spec.md:399-403`.

The ten Python benchmark functions at `benchmarks/python/test_propagation_benchmark.py:45-209` omit explicit `profile()` overhead, major storage/RSS, a duplicate-heavy workload, `max_weight=2/3/4` and exact-small scans, sparse one-qubit versus sparse/dense two-qubit PTM execution, thread-count scaling, and separate Rust propagation/packed-return/Python-object materialization boundaries. The operator materialization case measures the combined public call only. The full manifest records `RAYON_NUM_THREADS=unset`, so the required 1/2/4/fixed-max thread matrix was not run.

Resolution: benchmark the paths actually changed and re-profile the 12q/100q representative workloads. Implement the full matrix from `docs/vibe/phase-3-spec.md:384-403` only if the owner explicitly retains it as an acceptance requirement; thread scaling is meaningless while propagation is sequential.

### MAJOR-7: Explicit operator materialization uses nested code vectors and repacks every term in Python instead of the specified packed-array boundary

The native materializer converts every public word to a fresh `Vec<u8>` at `crates/tencirpauli-native/src/propagation.rs:246-255`. Python then copies those nested rows to tuples and repacks every code back into x/z words at `python/tencirpauli/pauli.py:874-909`. This is a double conversion with per-term/per-qubit Python work, not the contiguous packed/code array plus complex128 buffer required by `docs/vibe/phase-3-spec.md:247-251`.

Resolution: first benchmark a representative large final operator. Only if this conversion dominates should native return contiguous packed x/z and complex128 buffers and a private packed constructor be added. Do not redesign `PauliOperator` or make `.terms` lazy based on the current evidence alone.

## MINOR

### MINOR-1: Public `max_bytes=0` is accepted although the frozen contract says positive integer or `None`

`python/tencirpauli/hamiltonian.py:167-176` rejects only negative integers, so zero is accepted across migrated public APIs. `docs/vibe/phase-3-spec.md:255-261` requires a positive integer or `None`.

Resolution: no standalone code change is recommended. Decide whether zero usefully means “reject all nonzero allocations”; either document it or adjust the specification when adjacent validation is already being changed.

### MINOR-2: An internal impossible Clifford phase degrades to a zero coefficient in release builds

`phase_sign()` at `crates/tencir-pauli-core/src/propagation.rs:472-483` uses a `debug_assert!` for an unexpected imaginary phase and returns `0.0`. If a future local mapping violates the invariant, release builds will silently erase a term.

Resolution: fix opportunistically when the Clifford lookup is rewritten, preferably by making the table total. This does not justify a new error subsystem.

### MINOR-3: Completion documentation overstates implemented performance mechanisms and acceptance status

`docs/vibe/implementation-status.md:101-107` declares P0-P7 complete and says reusable dynamic aggregation addresses the design target, while the implementation allocates new vectors/maps for each call and the frozen tests/benchmarks are incomplete. `docs/vibe/implementation-status.md:113-116` moves the dominant transition and aggregation work to a later phase despite the Phase 3 P6 requirement.

Resolution: mark Phase 3 as focused remediation-required until MAJOR-0, MAJOR-1, the local-kernel portion of MAJOR-2, and the agreed minimal tests are closed; replace the reusable-scratch claim with the actual allocation model. List broader benchmark/test items as deferred or remove them from the acceptance contract with owner approval.

### MINOR-4: Several validation failures use semantically imprecise exception categories/messages

Invalid parameter-slot coverage is represented as `InvalidClifford` at `crates/tencir-pauli-core/src/propagation.rs:79-89`, and native core errors other than memory/overflow are uniformly mapped to `ValueError` at `crates/tencirpauli-native/src/convert.rs:34-40`. Some Python type errors, such as a non-integer parameter slot, also raise `ValueError` at `python/tencirpauli/propagation.py:133-140`.

Resolution: defer unless users depend on precise exception categories. Clear `ValueError`, `OverflowError`, and `MemoryError` behavior is sufficient for the current private boundary; avoid a large error-enum taxonomy for cosmetic purity.

## OBSERVATIONS

- The architecture boundary is clean: the core crate remains free of PyO3/NumPy/TensorCircuit dependencies, the PyO3 call is coarse-grained, and long native calls use `allow_threads`.
- The public builder snapshots tape/PTM/Bloch data, parameter slots are compiled once, PTM exact-nonzero transitions are precompiled, and public operator output is deterministically sorted.
- Static rotation sine/cosine values are cached at construction, and runtime parameter sine/cosine values are resolved once per gate per execution.
- The existing release benchmark records and synchronized JAX call use are useful, but they should be treated as partial evidence rather than completion evidence until the frozen matrix is filled.
- The working tree had no tracked source modifications before the review. The prescribed smoke workflow passed after review diagnostics.

## RECOMMENDED IMPROVEMENTS

1. Fix MAJOR-0 and add one regression; this is a bounded arithmetic fix, not a reason to redesign the engine.
2. Split the scalar dynamic-term path from profile/materialization (MAJOR-1) and benchmark the real scalar boundary.
3. Replace full-width local Clifford/rotation multiplication (the important part of MAJOR-2), then re-profile the existing 12q/100q workloads.
4. Normalize exact cutoffs (MAJOR-4), correct the dense oracle's initial projection, and add a small targeted differential set.
5. Add only the cheap wide-key payload estimate from MAJOR-3 unless stronger memory accounting is demonstrably needed.
6. Correct completion documentation. Treat scratch pooling, parallelism, packed public materialization, exhaustive tests/benchmarks, zero-budget semantics, and exception taxonomy as conditional or deferred.

Recommended acceptance decision: focused remediation required. Treat MAJOR-0, MAJOR-1, the local-kernel portion of MAJOR-2, MAJOR-4, and a minimal targeted subset of MAJOR-5 as the default acceptance work. Do not require cross-call scratch pooling, parallel propagation, exhaustive benchmark matrices, packed public-operator redesign, strict exception taxonomy, or exact memory accounting unless fresh profiling or an owner decision justifies them.

## SELF-CONTAINED REMEDIATION EXECUTION PLAN

This section is an implementation handoff and a catalogue of possible remediation. It must not be executed wholesale. The default high-value path is R0 (targeted tests/status), R1, R2, R3, the minimal subset of R6, and the documentation portion of R8. From R4, implement only simple representation-aware wide-key accounting unless allocation profiling proves scratch reuse worthwhile. R5 and the broad R7 matrix are conditional. Read `AGENTS.md`, `AGENTS.local.md`, `docs/vibe/semantics.md`, and `docs/vibe/phase-3-spec.md` before editing, keep each chosen slice independently testable, and do not combine Phase 4 gradient work.

Default scoped sequence:

1. Add four targeted failures: overflow, direct-scalar/materialization equivalence, two-qubit rotation differential, and isolated initial projection.
2. Fix checked coefficient arithmetic.
3. Separate packed dynamic propagation from public materialization and reduce scalar expectation directly.
4. Replace full-width local Clifford/rotation composition and normalize the exact cutoff.
5. Run focused end-to-end benchmarks and profile the same workloads that motivated the changes.
6. Add the simple wide-key byte term if it remains a public-contract priority, then correct documentation and stop.

Do not add a scratch pool, Rayon propagation, lazy public operator, new Cargo feature, dozens of benchmark cases, or detailed exception hierarchy in the default sequence.

### Frozen behavior that must not change

- Tape operations remain appended in Schrödinger order and executed in reverse for `U† O U`.
- Pauli rotations remain `exp(-i theta P / 2)` with the existing sign convention.
- Initial projection occurs after canonical input aggregation; finite-weight projection occurs after every gate's duplicate aggregation; `None` or `max_weight >= nqubits` is exact.
- There is no coefficient cutoff, top-k truncation, fixed-size sparse buffer, or discarded-norm approximation.
- Custom PTMs remain finite real `float64` one-/two-qubit maps with `R[out, in]` orientation and caller-provided wire order.
- Public outputs remain deterministic and canonically ordered. A performance change must not make coefficient reduction depend on hash iteration or worker completion order.
- `expectation()` continues to require an exactly Hermitian input observable and returns a Python `float`; `propagate_operator()` continues to allow general complex coefficients.
- The Rust core remains independent of Python/PyO3/NumPy/TensorCircuit. The public API remains in `python/tencirpauli/`; `_native` remains private.
- Preserve concurrency safety and GIL release. Do not introduce a single long-held lock around propagation.
- Public `max_bytes` remains a best-effort guard, not an exact RSS promise, but all cheap major allocations must be represented and arithmetic must remain checked.

### Slice R0: Establish a failing-test baseline and correct status tracking

Purpose: turn the review reproductions into durable failures before changing implementation.

Files to change:

- `tests/test_propagation.py`
- `tests/propagation_reference.py`
- `crates/tencir-pauli-core/src/tests.rs` or a new `crates/tencir-pauli-core/src/propagation/tests.rs`
- `docs/vibe/implementation-status.md`

Required changes:

1. Mark Phase 3 as `remediation active` in `implementation-status.md`; retain historical benchmark facts, but remove the claim that reusable aggregation is already implemented.
2. Fix the dense oracle so initial canonical terms are projected before the reversed gate loop. The oracle should decompose the initial dense matrix, remove words above the cutoff, reconstruct the dense matrix, and then continue per-gate decomposition/projection. Keep the numerical oracle threshold explicit and separate from production exact-zero semantics.
3. Add the two currently failing public regressions below.

```python
def test_ptm_duplicate_aggregation_overflow_fails_explicitly() -> None:
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 1] = 1.0
    matrix[0, 2] = 1.0
    tape = tcp.GateTape(1)
    tape.ptm((0,), matrix)
    observable = tcp.PauliOperator(1, [((1,), 1e308), ((2,), 1e308)])
    engine = tcp.PropagationEngine(tape, observable)
    with pytest.raises(ValueError, match="finite|overflow"):
        engine.propagate_operator([])
    with pytest.raises(ValueError, match="finite|overflow"):
        engine.expectation([])
    with pytest.raises(ValueError, match="finite|overflow"):
        engine.profile([])


def test_wide_key_memory_estimate_rejects_small_budget() -> None:
    nqubits = 10_000
    codes = [0] * nqubits
    codes[-1] = 3
    observable = tcp.PauliOperator(nqubits, [(codes, 1.0)])
    with pytest.raises(MemoryError):
        tcp.PropagationEngine(tcp.GateTape(nqubits), observable, max_bytes=1_000)
```

4. Add an empty-tape initial-projection test: one identity term and one overweight nonidentity term under `max_weight=0`; only identity may survive.
5. Add a collision/cancellation PTM test in which two distinct input words map to the same output word with opposite coefficients; assert exact removal and deterministic empty output.

R0 acceptance:

- The new overflow and wide-memory tests fail against the current implementation for the reasons documented here.
- Existing tests remain green except for those intentional failures.
- No production code has changed yet.

### Slice R1: Make coefficient arithmetic fail explicitly instead of silently dropping non-finite aggregates

Purpose: close MAJOR-0 without changing ordinary finite recurrence results.

Files to change:

- `crates/tencir-pauli-core/src/error.rs`
- `crates/tencir-pauli-core/src/propagation.rs`
- `crates/tencirpauli-native/src/convert.rs` only if a new error mapping is needed
- Rust and Python propagation tests

Recommended implementation:

1. Add a dedicated core error such as `NonFiniteArithmetic { context: &'static str, contribution: usize }` or reuse `NonFiniteCoefficient` with a deterministic contribution index. Prefer a distinct arithmetic error because the input coefficients were valid and the failure occurred during recurrence.
2. Replace the unchecked `and_modify` addition in `aggregate()` with an explicit `Entry` match. After every addition, check both real and imaginary components and return the typed error immediately if either is non-finite.
3. Do not retain the final `is_finite()` filter as a deletion mechanism. At the final collection point, a non-finite coefficient must be unreachable or must return an error.
4. Apply the same post-operation finiteness policy to coefficient multiplication in Clifford, rotation, and PTM paths. A single finite multiplication can overflow before aggregation and should report the same arithmetic error.

Suggested core shape:

```rust
fn ensure_finite_coefficient(
    value: Complex64,
    context: &'static str,
    index: usize,
) -> Result<Complex64, PauliError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(PauliError::NonFiniteArithmetic { context, index })
    }
}
```

Do not use saturating arithmetic, coefficient clipping, or a magnitude cutoff; those would change the frozen recurrence.

R1 tests:

- PTM collision addition overflow fails on expectation, profile, and materialization.
- Direct PTM multiplication overflow fails.
- Large but finite non-overflowing coefficients still propagate correctly.
- Exact cancellation of large finite opposite coefficients returns the empty operator without error.
- Existing ordinary differential tests remain bitwise/tolerance equivalent.

R1 acceptance commands:

```bash
conda run -p .conda cargo test --locked --workspace
conda run -p .conda python -m pytest tests/test_propagation.py -q
```

### Slice R2: Separate internal dynamic propagation from scalar expectation, profile, and public materialization

Purpose: close MAJOR-1 and create the structure needed for later optimization.

Primary file:

- `crates/tencir-pauli-core/src/propagation.rs`

Boundary files likely affected:

- `crates/tencirpauli-native/src/propagation.rs`
- `python/tencirpauli/propagation.py` only if return shapes change

Recommended internal design:

```rust
struct DynamicPropagationResult {
    terms: Vec<DynamicTerm>,
    initial_terms: usize,
    peak_terms: usize,
    estimated_peak_bytes: usize,
}

impl PropagationEngine {
    fn execute_dynamic(
        &self,
        parameters: &[f64],
        collect_stats: bool,
        scratch: &mut PropagationScratch,
    ) -> Result<DynamicPropagationResult, PauliError>;

    pub fn expectation(&self, parameters: &[f64]) -> Result<f64, PauliError>;
    pub fn propagate(&self, parameters: &[f64]) -> Result<PropagationResult, PauliError>;
}
```

Implementation requirements:

1. `expectation()` validates Hermiticity and parameters, executes the dynamic recurrence, and reduces directly over `DynamicTerm { PackedKey, coefficient }`.
2. Implement `expectation_from_dynamic_terms()` using packed masks/local codes. It must not allocate `PauliWord`, `PauliTerm`, weight-count vectors, or a sorted public-term vector.
3. `propagate()` converts dynamic terms to public words and sorts only for the explicit operator path.
4. `profile()` may request statistics and canonical sorting because it is an explicit diagnostic call, but do not make ordinary expectation pay those costs.
5. Validate identical results between direct scalar reduction and public materialization for all product-state descriptors.
6. Ensure the direct scalar path retains deterministic floating reduction order. If dynamic terms are not already in canonical order, either keep their deterministic vector order or sort keys without converting to public words; do not iterate a hash map directly.

Product-state packed reduction guidance:

- Zero/computational states can reject a term immediately when `(x_mask != 0)` because any X/Y local factor has an x bit; surviving Z parity can be computed by packed operations.
- Computational-state signs can be obtained from parity of `z_mask & state_one_bits` for surviving diagonal terms.
- Bloch states still require visiting support codes, but should read them directly from `PackedKey`; it should not construct public words.
- Preserve the zero-qubit empty-product result.

R2 benchmark additions:

- `test_propagation_expectation_steady`
- `test_propagation_profile_overhead`
- a Rust `expectation_only` case and a separate `public_materialization` case using the same engine/workload

R2 acceptance:

- A code audit confirms no call from `expectation()` reaches `PackedKey::to_word`, `PauliWord::from_words`, public-term sorting, or final weight-count construction.
- Scalar values remain within existing tolerances.
- The steady scalar benchmark improves or is at least neutral; a regression requires investigation before continuing.

### Slice R3: Normalize the exact cutoff and replace full-width local gate composition

Purpose: close MAJOR-4 and the highest-confidence portion of MAJOR-2 before adding allocation complexity.

Files to change:

- `crates/tencir-pauli-core/src/propagation.rs`
- optionally `crates/tencir-pauli-core/src/gate.rs` for compiled lookup tables
- Rust propagation tests and Criterion benchmark

Exact-cutoff change:

```rust
let effective_cutoff = if self.is_exact() { None } else { self.max_weight };
```

Use `effective_cutoff` for initial and per-gate projection. Keep the configured `self.max_weight` unchanged for the public getter.

Clifford change:

1. Precompute or declare constant lookup tables indexed by one local code for one-qubit Clifford gates and by `4 * first + second` for two-qubit gates.
2. A two-qubit table entry should contain `(output_first, output_second, sign)` for the frozen Heisenberg convention.
3. `map_clifford2()` should clone/move the input key once, set two local codes, and multiply the scalar coefficient by the table sign. It must not create identity keys or invoke `PackedKey::multiply()`.
4. Exhaustively compare all 16 inputs for CNOT/CZ/SWAP against the independent dense matrices. Include both CNOT wire directions and nonadjacent/cross-word wires.

Rotation change:

1. For a one-/two-wire Pauli generator, determine commutation from the local symplectic parity only.
2. For an anticommuting input, update the generator wires directly and derive the exact `i P Q` sign from the local code multiplication table. Do not create a full-width generator or scan all qubits.
3. Keep sine/cosine evaluation once per gate execution.

R3 required microbenchmarks:

- one- and two-qubit Clifford local update at 2, 64, 100, 128, and 129 qubits
- rotation commute and branch at the same widths
- exact `None` versus `nqubits` versus `nqubits + 1`
- the existing 100q Clifford and 128q deep near-Clifford workloads

R3 acceptance:

- `PackedKey::multiply()` is no longer present in built-in Clifford or Pauli-rotation hot stacks.
- `None`, `nqubits`, and larger cutoffs use the same internal no-weight-check kernel.
- The review's 128q/1,000-H comparison no longer shows systematic overhead for `max_weight=128` relative to `None`.
- Release end-to-end results improve on the profile workload and do not regress the 12q projected workload.

### Slice R4: Optional allocation reuse; bounded wide-key memory-accounting fix

Priority: mixed. The simple wide-key payload estimate is P1 and worthwhile. Cross-call scratch pooling and detailed simultaneous-buffer accounting are P2 and must be attempted only after allocation profiling shows they materially affect representative end-to-end runtime or safety.

Purpose: close the cheap, clear part of MAJOR-3 first; address the allocation/reuse portion of MAJOR-2 only with evidence.

Files to change:

- `crates/tencir-pauli-core/src/propagation.rs`
- core errors if checked-estimate contexts need expansion
- Python propagation tests

Potential scratch structure if profiling justifies it:

```rust
struct PropagationScratch {
    current: Vec<DynamicTerm>,
    next: Vec<DynamicTerm>,
    contributions: Vec<(PackedKey, Complex64)>,
    aggregate: FxHashMap<PackedKey, Complex64>,
    ordered: Vec<(PackedKey, Complex64)>,
}
```

Concurrency-safe reuse options if cross-call allocation is proven material, in preference order:

1. Store a short-lock scratch pool in the engine, for example `Arc<Mutex<Vec<PropagationScratch>>>`. Acquire/pop before computation, release the mutex, run propagation, clear retained buffers, then return the scratch under a short lock using an RAII guard so errors do not leak it. Concurrent calls create additional scratch only when the pool is empty; they must not serialize the kernel.
2. If measurements show the pool lock or retained peak memory is undesirable, use per-thread scratch with a documented bounded policy. Do not use unbounded global mutable state.
3. A single `Mutex<PropagationScratch>` held for the entire propagation is unacceptable because it serializes supported concurrent calls.

If scratch reuse is implemented, requirements are:

- `clear()` vectors/maps while retaining capacity between steady calls.
- Move/swap `current` and `next`; do not allocate a new vector for every Clifford gate.
- Pre-reserve contribution capacity using checked `term_count * branch_factor`.
- Reuse the aggregation map and ordered buffer.
- Cap retained scratch capacity if one pathological call would otherwise permanently pin a very large allocation; document the cap policy.

Minimum recommended memory-estimate formula:

```text
word_count = ceil(nqubits / 64)
wide_key_payload = 0                                  when nqubits <= 128
wide_key_payload = 2 * word_count * sizeof(u64)      otherwise
dynamic_term_major = sizeof(DynamicTerm) + wide_key_payload
contribution_major = sizeof((PackedKey, Complex64)) + wide_key_payload
packed_public_term = 2 * word_count * sizeof(u64) + sizeof(complex128)
```

Use checked multiplication/addition. The default remediation only needs to include known wide mask payload and known explicit output; it does not need a simulated allocator or exact hash-map peak model. Sum additional simultaneously live buffers only when the estimate is already available and simple. It is acceptable to exclude hash control bytes, allocator fragmentation, PyO3 object overhead, and OS RSS if the docs continue to say so.

Stats requirements:

- `estimated_peak_bytes` should include documented cheap major allocations. Do not pursue exact peak-RSS equivalence.
- Engine construction must include the observable native snapshot and compiled PTM transition payload.
- `propagate_operator()` must guard its packed/code output allocation before creating it.
- `None` disables only the budget comparison, not checked arithmetic.

R4 tests:

- Inline one-term engines pass/fail around a deterministic small budget.
- 129q and 10,000q wide keys include the mask payload.
- A branching PTM/rotation case includes simultaneous current/contribution/aggregate storage.
- A materialization-specific budget can allow scalar expectation but reject explicit operator output when appropriate.
- `max_bytes=None` permits the same modest test case while overflow checks remain active.
- Four concurrent calls return identical results and are not serialized by a long-held scratch lock.

R4 acceptance:

- The R0 10,000q/1,000-byte regression raises `MemoryError` before propagation allocation.
- Profile estimates are at least the sum of documented major live buffers for deterministic fixture cases.
- Repeated steady calls show reduced allocations in a profiler or allocator counter.

### Slice R5: Conditional materialization-boundary optimization

Priority: P2. Do not implement this slice unless a representative large-output benchmark shows that nested code-vector conversion materially dominates `propagate_operator()`. The current recorded materialization path is explicit rather than the main scalar hot path, so a public-operator representation redesign may cost more complexity than it saves.

Purpose when justified: close MAJOR-7 while preserving the existing public `PauliOperator` behavior.

Files to change:

- `crates/tencirpauli-native/src/propagation.rs`
- `crates/tencirpauli-native/src/convert.rs`
- `python/tencirpauli/_native.pyi`
- `python/tencirpauli/pauli.py`
- `python/tencirpauli/propagation.py`
- tests and materialization benchmarks

Minimum acceptable implementation:

1. Return one contiguous `uint8` code buffer with `(term_count, nqubits)` metadata and one contiguous `complex128` coefficient buffer from native code.
2. Add a private `PauliOperator._from_native_arrays()` constructor that consumes these arrays without a nested `Vec<Vec<u8>>` conversion.
3. Ensure the native buffers are canonical and read-only from the public object's perspective.

Preferred implementation for large/wide operators:

1. Return contiguous packed x/z `uint64` arrays with shape `(term_count, word_count)` plus a complex128 coefficient array.
2. Store packed arrays as the primary private operator representation and derive code rows only when an existing API explicitly needs them.
3. Consider lazy construction/caching of the public `.terms` tuple, but do not silently break dataclass equality, iteration, serialization, or existing API behavior. If lazy `.terms` is too broad for this remediation, keep eager construction but eliminate the native code-row round trip first and record the remaining Python-object cost.

Use NumPy/PyO3 array return types as existing Hamiltonian array paths do; do not build one Python object through PyO3 per term.

R5 benchmarks:

- Rust dynamic propagation only
- native packed/code buffer construction
- Python private constructor
- complete public `propagate_operator()`
- scaling by final terms and qubits, with output bytes recorded

R5 acceptance:

- The native propagation materializer contains no per-term `word.codes()` allocation for the preferred packed path.
- The Python boundary receives contiguous arrays rather than nested Python sequences.
- Existing `PauliOperator` equality and `.terms` tests pass.
- The explicit materialization benchmark improves or, if eager public objects dominate, the report identifies that cost accurately.

### Slice R6: Add minimal high-value tests; treat the exhaustive matrix as optional

Purpose: close the meaningful part of MAJOR-5 after the implementation structure stabilizes without creating a large, slow, redundant suite.

Files to change:

- `tests/propagation_reference.py`
- `tests/test_propagation.py`
- Rust propagation tests

Minimum required Python additions:

- Exhaustive 16-input correctness for RXX/RYY/RZZ at one nonspecial angle, because these gates currently lack direct correctness tests.
- A small deterministic randomized differential loop, for example 20 exact and 20 projected cases, rather than hundreds of parametrized tests.
- Isolated initial projection with an empty tape.
- PTM duplicate collision/cancellation and overflow.
- One 63/64 cross-word gate case and the existing 129q fallback case.
- Repeated parameter slot, non-finite runtime parameter, and direct scalar versus materialized expectation.
- The simple wide-key budget regression if MAJOR-3 is fixed.

The following broader groups are optional coverage ideas from the frozen specification, not default remediation requirements:

1. Exhaustive local basis: all 4 inputs for every one-qubit built-in and all 16 inputs for CNOT/CZ/SWAP/RXX/RYY/RZZ at fixed special and nonspecial angles.
2. Rotation vectors: `theta = 0, ±pi/2, pi` plus seeded random angles for RX/RY/RZ/RXX/RYY/RZZ.
3. Seeded random exact differential: at least 100 tapes across `n=1..5`, mixed Clifford/rotation gates, multiple real and complex observable coefficients, and explicit reverse-order comparison.
4. Seeded random projected differential: at least 100 tapes across `n=1..4`, multiple `max_weight` values including zero, with correct initial and per-gate oracle projection.
5. PTM: negative entries, exact tiny nonzero entries, sparse/dense 1q and 2q maps, random one-/two-qubit unitary-derived PTMs, reversed wire order, complex dtype, wrong shape, NaN/Inf, and duplicate wires.
6. Packed boundaries: gates crossing wires 63/64 and 64/65; structures at 64/65/100/128/129 qubits; independent Python packed rules for update, weight, commutation, and product-state expectation.
7. Parameters: repeated slots, static/slot mixture, holes, wrong vector length, non-finite values, bool rejection, and snapshot behavior.
8. States: qubit-0 bit ordering, pure/mixed Bloch vectors, norm boundary `1 + 1e-12`, invalid dtype/value/shape, and zero-qubit identity.
9. Determinism: repeat operator and profile results under multiple `PYTHONHASHSEED` subprocesses; when parallel propagation exists, compare supported thread counts bitwise.
10. Errors/memory: arithmetic overflow, checked count overflow through a core-only synthetic path, small inline/wide budgets, output-only materialization rejection, and `None` behavior.
11. GIL/concurrency: use one native call long enough for a Python heartbeat thread to make measurable progress while the call is active; separately test concurrent calls with distinct parameters and compare against serial results.

Minimum Rust additions should cover the new local tables, checked aggregation error, and exact-cutoff normalization. The broader Rust groups below are optional unless the implementation exposes a distinct untested invariant:

- Exhaustive local lookup-table correctness and sign/involution identities.
- Rotation local phase/sign table for all local codes.
- Exact cancellation, deterministic aggregation order, and non-finite arithmetic errors.
- Effective exact-cutoff normalization.
- Packed key inline/wide conversion, local update, weight, equality/hash, and cross-boundary behavior.
- Representation-aware checked memory estimates.

Testing discipline:

- Use deterministic seeds written in source.
- Keep dense tests small (`n <= 5`) and structural wide tests non-dense.
- Never derive expected lookup tables from the production lookup table itself.
- Test exact-zero production semantics separately from the numerical threshold used by dense decomposition.

R6 acceptance commands:

```bash
conda run -p .conda cargo test --locked --workspace
conda run -p .conda python -m pytest -q
conda run -p .conda python scripts/check.py --benchmark smoke
```

### Slice R7: Focused benchmark proof; broad matrix conditional on owner scope

Priority: the focused proof is P0/P1; the full frozen matrix is P2/P3. The practical goal is to demonstrate that the selected hot-path changes improve representative end-to-end workloads without correctness regression. If the owner no longer needs every workload in the oversized original specification, update that specification/status instead of generating benchmarks with no decision value.

Purpose: provide a valid performance conclusion for the changes actually made.

Files to change:

- `crates/tencir-pauli-core/benches/propagation.rs`
- `benchmarks/python/test_propagation_benchmark.py`
- `benchmarks/python/matched_jax_propagation.py` as required for matched cases
- `benchmarks/README.md`
- `docs/vibe/implementation-status.md`

Minimum Criterion coverage for this remediation:

- two-qubit Clifford local update before/after proxy
- rotation branch before/after proxy
- exact `None` versus `nqubits`
- existing 12q, 100q, and 128q whole-tape kernels

Additional Criterion coverage below is conditional on the corresponding code being optimized:

- inline key equality/hash/weight at 64/100/128 qubits and wide fallback at 129+
- one-/two-qubit Clifford local update
- rotation commute and branch
- sparse and dense custom PTM transition application
- duplicate-heavy aggregation with and without exact cancellation
- finite projection versus exact no-projection
- Zero/computational/Bloch product-state reduction
- full exact and projected tape kernels

Because `PackedKey` is private, do not copy its implementation into the benchmark. Either benchmark faithful engine-level one-gate proxies, or add a narrowly scoped `bench-internals` Cargo feature with `#[doc(hidden)]` wrappers and `required-features` on the Criterion target. Do not expose mutable internals in the normal public API.

Minimum Python coverage:

- 12q steady scalar and matched synchronized JAX
- 100q Clifford scalar
- explicit scalar versus profile versus materialization timing on one shared workload

Additional Python workload coverage below is optional unless retained as a formal owner requirement:

- friendly tape/observable construction
- native engine construction and PTM compilation
- first scalar call
- steady scalar call
- explicit profile overhead
- Rust propagation-only where a private benchmark hook is available
- native packed return
- Python public object construction
- full `propagate_operator()`
- output bytes, estimated peak, final/peak terms, accuracy, and thread count
- 12q PPE with `max_weight=2/3/4` plus exact-small control and matched synchronized complex128 JAX
- 4x4 Heisenberg with `max_weight=3/4`
- 100q Clifford and near-Clifford
- 128q deep native regression
- duplicate-heavy synthetic
- sparse 1q, sparse 2q, and dense 2q PTMs reported separately
- scaling scans over qubits, layers, input terms, cutoff, and final/peak terms

Thread controls are required only if propagation actually introduces Rayon or another parallel kernel. For the current sequential implementation they have no diagnostic value beyond confirming that thread count is irrelevant. If parallelism is introduced, use:

```bash
RAYON_NUM_THREADS=1 conda run -p .conda python benchmarks/run.py record --label phase3-remediation-t1
RAYON_NUM_THREADS=2 conda run -p .conda python benchmarks/run.py record --label phase3-remediation-t2
RAYON_NUM_THREADS=4 conda run -p .conda python benchmarks/run.py record --label phase3-remediation-t4
```

If the propagation kernel remains sequential, record that fact and show the controls; do not claim parallel speedup. Add Rayon only when a large representative workload shows a repeatable end-to-end gain after deterministic worker-local reduction. Any parallel merge must use a fixed chunk order and fixed canonical merge/reduction order so public coefficients remain bitwise reproducible across supported thread counts.

Profiling requirements:

1. Re-run the existing 100q Clifford profile after R3 and R4.
2. Profile the 12q projected workload where native was slower than matched JAX.
3. Profile a duplicate-heavy aggregation case and a large materialization case.
4. Record tool, commit, workload, dominant frames, allocation observations, thread configuration, before/after medians, accuracy, and conclusion in `implementation-status.md`; raw profiles remain untracked.

R7 focused acceptance:

- Existing JAX calls remain synchronized and use complex128/equivalent recurrence.
- The changed scalar/local-kernel paths have clean before/after end-to-end evidence on 12q and 100q representative workloads.
- Accuracy and final term counts remain unchanged within the documented tolerance.
- At least one structural optimization targeting the measured bottleneck improves representative end-to-end runtime, not only a microbenchmark.
- `.benchmarks/` remains ignored and untracked.

Only if the owner retains the full original benchmark contract must every workload in `phase-3-spec.md:384-403` receive a named stable benchmark and full boundary/thread/storage metadata.

### Slice R8: Update docs; validation cleanup only when locally useful

Purpose: make the handoff truthful. The validation cleanups are P3 and should not block acceptance.

Files to change:

- `python/tencirpauli/hamiltonian.py`
- public callers/tests using `max_bytes`
- `crates/tencir-pauli-core/src/error.rs`
- `crates/tencirpauli-native/src/convert.rs`
- `python/tencirpauli/propagation.py`
- `python/tencirpauli/_native.pyi`
- `README.md`, `CHANGELOG.md`, `benchmarks/README.md`, `docs/vibe/architecture.md`, `docs/vibe/implementation-status.md`

Required documentation changes:

1. Describe the actual scalar/materialization boundary, remaining allocation model, benchmark labels, and known limitations.
2. Record exact validation commands/results and remove claims for mechanisms that were not implemented.
3. Mark Phase 3 complete only against the newly agreed focused acceptance scope, or explicitly revise the overly broad original specification with owner approval.

Optional cleanup while adjacent files are already touched:

1. Decide whether `max_bytes=0` is useful as “reject every nonzero allocation”; if so, update the specification instead of rejecting it solely for textual consistency.
2. Add dedicated error variants only if users or tests rely on exception categories; current `ValueError` behavior is adequate for most private-boundary failures.
3. Remove the release-mode zero fallback in `phase_sign()` opportunistically when the local lookup is rewritten.

Final acceptance commands:

```bash
conda run -p .conda python scripts/check.py --fix --benchmark smoke
conda run -p .conda python scripts/check.py --benchmark smoke
conda run -p .conda cargo test --locked --workspace
conda run -p .conda cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
conda run -p .conda maturin develop --release --locked
conda run -p .conda python -m pytest -q
```

Record a full benchmark manifest only if the owner retains that acceptance requirement; otherwise run and record the focused propagation cases changed by this remediation. Do not spend hours populating unrelated benchmark cells solely for checklist completeness.

Run packaging smoke after the clean code commit:

```bash
conda run -p .conda maturin build --release --locked --out /private/tmp/tencirpauli-phase3-remediation-dist
conda run -p .conda maturin sdist --out /private/tmp/tencirpauli-phase3-remediation-dist
```

### Focused mandatory final acceptance checklist

- The overflow reproducer raises an explicit error on all three public execution paths.
- Scalar expectation does not convert to public words, build weight statistics, or sort public terms.
- `max_weight=None`, `nqubits`, and larger values select the same exact no-projection kernel.
- Two-qubit Clifford and Pauli-rotation local transitions no longer scan all qubits.
- Isolated initial projection, two-qubit rotations, arithmetic overflow, and a small seeded exact/projected differential set pass.
- Existing public semantics, deterministic output, concurrency behavior, and product-state results remain unchanged.
- A profile-backed structural optimization shows a representative end-to-end improvement, not merely an isolated microbenchmark gain.
- `python scripts/check.py` passes on the final clean implementation and packaging artifacts build.
- `implementation-status.md` contains exact commands/results and no longer claims mechanisms that are absent from code.

Conditional acceptance items, required only after separate evidence/owner choice: cross-call capacity reuse, detailed simultaneous-buffer accounting, contiguous packed public materialization, parallel/thread scaling, the exhaustive test matrix, and the full REQUIRED benchmark matrix.

### Stop/escalation conditions for the implementing model

Stop and request an owner decision rather than improvising if any proposed fix would change the frozen recurrence, public gate convention, PTM orientation/dtype, deterministic coefficient contract, public `PauliOperator` behavior, or the meaning of `max_bytes`. Also stop before adding coefficient pruning, top-k, approximate reduction, architecture-specific unsafe/SIMD, a new runtime dependency, or a long-held serialization lock. Performance work may change internal representation and private FFI shapes when public behavior and the required tests remain stable.

## Remediation addendum — 2026-08-02

The focused remediation is complete. MAJOR-0 is closed by checked coefficient scaling and checked deterministic aggregation; finite PTM and rotation collisions now return `NonFiniteCoefficient` through scalar expectation, profile, and operator materialization. MAJOR-1 is closed by keeping scalar expectation on the internal packed dynamic-term result, while public word conversion, final weight statistics, and sorting remain on the explicit propagation/profile path.

The local-kernel portion of MAJOR-2 is closed: two-qubit Clifford conjugation uses 16-entry local tables and Pauli rotations compute product/sign from their one or two active wires. Cross-call scratch reuse remains deferred because the current engine is sequential and the review did not establish that a scratch pool would produce a material end-to-end gain. MAJOR-4 is closed by using an effective `None` cutoff on exact paths while preserving the configured public value.

The bounded MAJOR-3 fix is closed by including the heap payload of Wide packed masks in the cheap propagation estimate. MAJOR-5's focused subset is closed by adding initial projection to the independent reference, seeded exact/projected differential cases, local Rust-rule tests, cancellation/overflow regressions, and a small-budget wide-key regression. MINOR-2 and MINOR-3 are also closed by removing the release-mode phase fallback and correcting the implementation status document.

MAJOR-6's complete benchmark matrix, MAJOR-7's packed public materialization redesign, cross-call scratch pooling, parallel propagation, and exhaustive optional correctness coverage remain explicitly profile-gated or owner-choice items. They are not represented as completed in `docs/vibe/implementation-status.md`.

Focused verification completed during remediation: `PATH="$PWD/.conda/bin:$PATH" cargo fmt --all`; `PATH="$PWD/.conda/bin:$PATH" cargo test -p tencir-pauli-core` (15 passed); `PATH="$PWD/.conda/bin:$PATH" maturin develop --release --skip-install`; and `PYTHONPATH="$PWD/python:$PWD/tests" ./.conda/bin/python -m pytest tests/test_propagation.py -q` (25 passed). The ordinary `maturin develop --release` install path was not usable on this workstation because the maturin binary selected `/opt/anaconda3/bin/python` and its pip lacked permission to write `/Users/shixin/.local`; the release extension itself built successfully with `--skip-install` and was tested through the project-local Python 3.11 environment.
