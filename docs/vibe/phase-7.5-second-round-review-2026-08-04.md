# Phase 7.5 second-round implementation and acceptance review

Review date: 2026-08-04

Reviewed commit: `53cd1972434034055a800311a7cd02b4cd7a063b` (`audit: apply review remediations and archive report`), including the Phase 7.5 remediation chain `4b56f18..df82de5` and the subsequent third-party audit remediation commit.

Scope: independent follow-up review of the Phase 7.5 C1–M5 closure, the adopted and deferred third-party audit findings, the current Rust/PyO3/Python architecture, boundary and resource contracts, correctness evidence, release benchmark coverage, and representative scaling risks. This review did not modify production code or benchmark results.

## Verdict

The Phase 7.5 numerical core is substantially correct and the first-round critical findings are genuinely remediated. Exact additive-charge decisions no longer pass through lossy complex128 charge generators; canonical Majorana words map directly to one Pauli word; packed native Majorana and mapping paths are active; compatible hybrid mapping is batched; and restricted compilation uses the reusable charge-sector plan rather than materializing the selected basis. The third-party audit's critical/high correctness fixes are also present in production paths and backed by regressions.

Phase 7.5 should nevertheless move from `accepted` to `implemented; second-round remediation open`. Two major design/resource issues and three medium contract/evidence issues remain. They do not invalidate the small and medium numerical results already recorded, but they prevent the stronger claim that the current phase has no obvious mismatch, scalability boundary, or resource-contract defect.

The recommended outcome is conditional acceptance of the implemented feature set, followed by a short scoped remediation. A broad architectural rewrite is not justified.

## Review and validation performed

- Inspected the implementation and documentation changes in `4b56f18`, `80be5fc`, `df82de5`, and `53cd197` against `phase-7.5-spec.md`, the first-round review, and the archived third-party audit ledger.
- Ran `conda run -p .conda python scripts/check.py --benchmark skip`: Rust formatting and Clippy, Black, Ruff, strict mypy, release `maturin develop --release --locked`, 38 Rust tests, and 292 Python tests passed.
- Ran `conda run -p .conda python scripts/check.py --benchmark smoke`: the same quality gate passed, all three Rust Criterion smoke suites passed, and 274 selected Python benchmark-smoke cases passed.
- Ran 720 randomized direct-Majorana mapping differentials against the expanded fermion path under Jordan–Wigner, parity, and Bravyi–Kitaev mappings; all passed.
- Ran 240 randomized exact Pauli-charge differentials against explicit canonical commutators; all passed.
- Measured public mapping-plan construction through 512 modes and exercised low-budget, wide-sector, per-term-FFI, and public `max_bytes` probes described below.
- The worktree remained clean after validation.

## Acceptance matrix

| Area | Result | Second-round assessment |
| --- | --- | --- |
| C1 exact integer charge decisions | PASS | Large integer selection rules and exact binary-float coefficient cancellation are implemented and pass deterministic plus randomized differentials. |
| C2 direct Majorana mapping | PASS | The production path does not call exponential Majorana-to-fermion conversion and agrees with the independent expanded path. |
| M1 packed Majorana representation and native conversion | PASS WITH MINOR DEBT | Operator multiplication and conversion use native packed aggregate keys; the single-word Python convenience path remains a lower-priority duplicate implementation. |
| M2 charge preflight and analysis budgets | PARTIAL | Compact charge metadata preflight is present, but the public budget contract remains inconsistent on other paths and requires the policy in this report. |
| M3 packed parity/BK execution and real term-count evidence | PARTIAL | Direct packed transforms and unique-term assertions are present, but the committed release matrix stops below the safe 512-mode/1024-term coverage requested by the first review. |
| M4 plan-driven restricted compiler | PARTIAL | Basis materialization and per-source-per-term occupation allocation are removed, but full-space mixed-radix integer limits still reject small valid sectors on wide layouts. |
| M5 delivery and closure evidence | PARTIAL | The complete local quality/smoke gate passes and representative domain workloads exist; the remaining scale and resource-contract gaps make the recorded unconditional acceptance too strong. |
| Third-party audit critical/high findings | PASS | H1–H7 are implemented; canonical hybrid identity, repeated QIR symbols, fermion embedding signs, cached arrays, and propagation hot paths have regression coverage. |
| Third-party audit medium/deferred findings | OPEN BY DECISION | U1 double-pass and marginal hash changes remain deferred; SPPS population-variance standard error is documented but unchanged. The audit is triaged, not equivalent to every confirmed item being fixed. |

## Findings

### SR1 — MAJOR: `ChargeSector` rejects small selected sectors when the full Hilbert space exceeds platform indexing

Locations: `python/tencirpauli/charge.py:429-441` and `crates/tencir-pauli-core/src/charge.rs:519-528`; contract: `docs/vibe/phase-7.5-spec.md:348-364,382,414`.

`_local_dimensions()` computes `math.prod(local_dimensions)` and compares the full finite Hilbert-space dimension with NumPy `intp` before the selected charge-sector dimension is known. On a 64-bit platform, a fixed-particle-number-one sector succeeds at 62 fermion modes but fails at 63 modes, even though the selected sector has only 63 states and its suffix-DP rank/unrank plan is small.

The remediated native transition compiler independently encodes every destination occupation as one mixed-radix `u64` key and preflights the product of all local dimensions. Removing only the Python check would therefore move the same full-space ceiling into Rust restriction.

This is a scalability and design mismatch, not a wrong numerical result: failure is explicit, but it rejects a valid and cheap restricted computation for an implementation-internal reason. It also conflicts with the architectural reason for working directly in the restricted basis.

Required resolution:

1. Validate only the selected sector dimension against platform/public sparse-index limits; do not require the full mixed-radix space to fit `intp`.
2. Aggregate candidate destinations with a key that does not encode the full Hilbert space into one `u64`. Acceptable scoped choices include borrowing occupation slices through a reusable scratch/key arena, packed small-axis words, or aggregate-then-rank logic against `ChargeSectorPlan`.
3. Preserve exact cancellation-before-leakage, deterministic final ordering, and the specialized arbitrary-width `U1Sector` implementation.
4. Add 62/63/64/65-mode fixed-number-one construction and restriction regressions, plus at least one wide simultaneous-charge case whose selected dimension is small.

Closure gate: a 65-mode small charge sector constructs, ranks/unranks, restricts a long-range hopping operator, and agrees with the specialized U1 reference without allocating or indexing the full space.

### SR2 — MAJOR: mapping-plan `estimated_bytes` and low-budget behavior omit predictable major retained storage

Locations: `python/tencirpauli/mapping.py:42-46,177-223` and `crates/tencir-pauli-core/src/mapping.rs:24-79,419-465`; contract: `docs/vibe/architecture.md:279` and `docs/vibe/phase-7.5-spec.md:185-192,401-418`.

The native estimate counts the public encoding matrices and canonical CNOT pairs but omits the retained packed X/Z transform matrices. The Python wrapper then retains additional encoding/inverse NumPy arrays, a Python tuple-of-tuples CNOT representation, a NumPy Clifford array, and the native plan containing the original Rust buffers. These are not allocator trivia or temporary RSS noise; several are predictable, long-lived, same-order copies created by the public plan itself.

A parity plan at 512 modes succeeds with `max_bytes=3_000_000` and reports `estimated_bytes=2_617_600`. Python `tracemalloc` observes approximately 14.75 MB of live Python-side storage and a 26.9 MB construction peak, excluding Rust allocations. This probe is diagnostic rather than a proposal to make `tracemalloc` an API test, but it demonstrates that the current estimate omits dominant retained components.

The right fix is not exact Python-object or peak-RSS accounting. Prefer reducing duplication: keep packed execution state native, expose large diagnostic matrices/provenance lazily, and materialize requested NumPy metadata under its own output guard. The plan constructor's estimate should then cover the cheaply known retained native payload plus any eagerly retained large public arrays.

Closure gate: the documented logical-major-byte estimate includes every eagerly retained large native/NumPy buffer, low budgets reject before those buffers are built, and tests validate the estimate formula rather than process RSS or allocator overhead.

### SR3 — MEDIUM: `PauliOperator.tensor_product(max_bytes=...)` validates but ignores the budget

Location: `python/tencirpauli/pauli.py:892-919`; contract: `docs/vibe/phase-7-spec.md:272` and the unified public memory policy.

The method calls `_validate_max_bytes(max_bytes)` but constructs the full Python pair list and calls unbounded `PauliOperator.from_terms()`. A four-output fixture succeeds with `max_bytes=1`.

Required resolution: perform a cheap checked pair-count/output preflight before building the list and route the canonicalization through a bounded batched native call or an equivalent running major-buffer guard. A one-byte limit must reject any nonempty material output.

### SR4 — MEDIUM: exact Pauli charge analysis reintroduces per-term PyO3 calls and scales in Python `Fraction` space

Locations: `python/tencirpauli/charge.py:1140-1209`; contract: `docs/vibe/phase-7.5-spec.md:433`.

The Pauli branch iterates `operator.terms` and invokes `term.word.to_codes()` once per term. A direct probe confirms one native call per canonical term. This violates the frozen no-per-term-FFI requirement and is inconsistent with the cached-array fixes applied to mapping, tensor product, and restricted compilation.

Representative exact-charge setup medians were approximately 11.1 ms for 256 eight-qubit terms, 68.7 ms for 1,024 twelve-qubit terms, and 372.8 ms for 4,096 sixteen-qubit terms. These timings combine legitimate exact `Fraction` aggregation with avoidable object and FFI overhead; they are not evidence that exact arithmetic itself should be weakened.

Required resolution: consume `PauliOperator._arrays()` once. If subsequent profiling still identifies exact analysis as material, add a coarse native fast path for weights/levels that fit the existing exact integer transport and retain the Python arbitrary-integer path as a correctness fallback. Do not convert accepted arbitrary-size charge integers to floating point.

### SR5 — MEDIUM: the clean Phase 7.5 release matrix does not cover the full first-review mapping scale request

Locations: `docs/vibe/phase-7.5-review-2026-08-03.md:241-251` and `benchmarks/python/test_majorana_mapping_charge_benchmark.py:340-521`.

The repaired benchmarks correctly assert their actual canonical term counts, but committed mapping-plan cases stop at 128 modes and mapping batches stop at 128 terms. The first review requested safe coverage through 512 modes and 1,024 unique terms, mapping-sensitive long parity strings, and first-plan versus reused-plan timing.

The omitted 512-mode scale is locally safe: public construction medians were approximately 76.9 ms for parity and 39.1 ms for Bravyi–Kitaev on the review machine. It should therefore be represented in the release manifest rather than treated as an unsafe optional diagnostic.

Closure gate: extend the release source and record a clean comparison that covers at least one 512-mode plan, one 1,024-unique-term reused mapping, a long parity-sensitive word, public conversion costs, canonical input/output counts, plan bytes under the revised policy, throughput, and numerical error.

### SR6 — DEFERRED NON-T7.5 DEBT: SPPS `value_standard_error` retains the population-variance convention

Location: `crates/tencir-pauli-core/src/spps.rs:987-1064`; audit ledger: `docs/vibe/audit-report-2026-08-04.md:720-808`.

The third-party audit recorded two conflicting recommendations and the implementation selected the compatibility-preserving branch: document the MLE/population-variance convention and defer the Bessel-corrected estimator until downstream consumers are audited. The result is still biased low by `sqrt((N-1)/N)` at finite sample count, including approximately 29% at two samples.

This item does not reopen Phase 7.5 and is no longer silent, but it means the third-party audit is triaged rather than fully fixed. Before a stable release, either rename the field/convention so it is not mistaken for an unbiased estimated standard error, or perform the consumer audit and change the formula with a release note.

## Recommended `max_bytes` owner policy

### Objective

`max_bytes` should remain a cheap fail-fast guard against obviously excessive dominant allocations. It should not become an exact memory quota, allocator model, peak-RSS promise, or proof that the operating system cannot terminate the process. Exact accounting across Rust containers, Python objects, NumPy ownership, PyO3 conversion, allocator capacity, threads, and temporary copies would be expensive, platform-dependent, and likely to make ordinary scientific calls slower without providing a reliable guarantee.

### Per-call incremental semantics

Interpret `max_bytes` as the budget for major new memory created by the current public call. Inputs and immutable plans that already existed before the call are not charged again. A constructor charges the major retained storage of the object it creates plus its dominant construction workspace. A materializer charges the new output plus its dominant workspace. An `apply()` call charges its new output and dominant scratch, not the already-retained plan or caller-owned input state.

This rule avoids both common failure modes: repeatedly subtracting pre-existing plans makes a small operation fail for memory it does not allocate, while ignoring a newly created returned plan or array allows a call to exceed its own advertised budget.

### Included categories

The implementation must count a category when its logical size is cheaply available and it can dominate the call:

- returned dense, COO, CSR, state, gradient, basis, or transition arrays;
- eagerly retained native plan buffers, including packed masks/matrices, coefficient buffers, index arrays, transition storage, and worker/checkpoint buffers;
- eagerly retained large NumPy mirrors or provenance arrays created by the public object;
- unavoidable construction scratch whose size follows directly from dimensions, term counts, transition counts, branch counts, or active worker count;
- conservative expansion bounds when they are simple checked arithmetic and do not require executing the algebra twice.

### Excluded categories

The public contract must explicitly exclude:

- Python/Rust object headers, hash-table bucket slack, `Vec` spare capacity, allocator metadata, alignment, fragmentation, and allocator caches;
- interpreter, imported-module, thread-stack, Rayon-pool, BLAS/runtime, and pre-existing process memory;
- PyO3 boxing and conversion temporaries unless a specific conversion is itself known to be a dominant repeatable buffer and can be removed or bounded cheaply;
- caller-owned inputs and reusable plans allocated before the current call;
- OS-level RSS, compressed/swap behavior, and backend copies outside TenCirPauli's direct allocation control.

### Implementation pattern

Use a two-stage guard only where both stages are naturally available. First, perform a cheap checked upper-bound preflight before the first dominant allocation. Second, after canonical term/transition counts are known, check the exact logical sizes of the returned or retained major buffers before allocating them. Do not add a symbolic dry run, traverse the same algebra solely to count memory, query allocators, or perform per-element budget bookkeeping in a hot loop.

Growing sparse/hash workloads may check capacity at natural batch, gate, source-state, or worker boundaries using a conservative bytes-per-entry constant. The estimate may be deliberately loose, but it must preserve the correct asymptotic variables and include each dominant concurrently live category once. A fixed safety multiplier is not a substitute for identifying a missing major buffer and should not be applied globally because it makes small and structurally sparse workloads arbitrarily over-strict.

Whenever public diagnostics create a large second representation, prefer lazy materialization or a compact array representation over estimating millions of Python objects. Reducing the allocation is more robust than attempting to model it.

### Test and benchmark policy

Correctness tests should assert that a deliberately tiny budget rejects before a known dominant allocation, that a budget at or above the documented logical estimate succeeds for a bounded fixture, that `max_bytes=None` disables the budget while retaining checked arithmetic, and that estimate metadata is monotonic in the dimensions that drive retained storage. Tests must not assert exact RSS or `tracemalloc` values.

Memory diagnostics may record process RSS or Python-traced peaks in isolated benchmarks to discover missing dominant categories, but those measurements remain informational and platform-specific. A discrepancy should lead either to adding a cheaply known major buffer to the formula or to removing avoidable duplication, not to allocator-level production instrumentation.

### Application to the second-round findings

- SR2 requires counting or eliminating eagerly retained mapping matrices, packed transforms, CNOT/Clifford arrays, but does not require counting tuple headers or construction peak RSS exactly.
- SR3 requires a pair-count/output preflight because it is cheap and directly controls the dominant tensor-product materialization.
- SR1 is primarily a representation problem, not a reason to reject a valid small sector using a full-space byte estimate.
- SR4 should remove per-term FFI for performance; its exact `Fraction` aggregate may retain a conservative entry estimate without simulating the commutator twice.

## Architecture assessment

The overall framework remains sound. The pure Rust core has no PyO3, Python, NumPy, or TensorCircuit dependency; the native crate is a thin binding layer; the public package owns friendly validation and object shaping; TensorCircuit imports remain isolated at the integration boundary; deterministic sorting is imposed before public emission; and the generic charge engine does not replace the mature arbitrary-width U1 implementation.

The second-round issues cluster at representation and accounting boundaries rather than indicating a failed architecture. The main corrective principle is to keep scalable canonical data native and compact, make large diagnostics lazy, cross PyO3 once per complete operation, and use selected-space rank/lookup rather than full-space integer encoding. These changes fit the existing modules and public API.

## Recommended remediation order

1. Fix SR1 and add wide small-sector regressions before claiming scalable generic charge restriction.
2. Adopt and document the `max_bytes` policy above, then repair SR2 and SR3 without attempting exact RSS accounting.
3. Remove the SR4 per-term FFI path and benchmark exact analysis at 256/1,024/4,096 canonical terms.
4. Extend the clean mapping manifest for SR5 after the plan-memory representation is stable.
5. Keep SR6 in the non-T7.5 release backlog with an explicit consumer-audit gate.
6. Re-run `python scripts/check.py --benchmark smoke`, record the focused Phase 7.5 release manifest on a clean commit, and restore `accepted` only after SR1–SR5 are closed or explicitly narrowed by an owner decision.

## Final acceptance statement

Current Phase 7.5 status: **implemented and numerically validated; conditional acceptance; second-round remediation required**.

The current code is suitable for continued development and bounded scientific use on the tested workloads. It is not yet justified to state that all present-stage acceptance goals, wide-layout scalability, public resource contracts, and third-party audit concerns are fully closed.
