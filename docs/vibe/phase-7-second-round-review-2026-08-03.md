# Phase 7 second-round remediation review

Review date: 2026-08-03

Reviewed commit: `2b0f0dc5818226389c16174db5559d42af9bf118` (`Remediate phase 7 structured operator review`).

Scope: the remediation of `docs/vibe/phase-7-review-2026-08-03.md`, with adversarial checks of partially mapped hybrid algebra, CAR/CCR expansion guards, direct-Weyl backend dimensions, the Phase 7 correctness matrix, benchmark records and metadata, and P0–P5 delivery against `docs/vibe/phase-7-spec.md`.

No implementation, test, benchmark, specification, or status source file was changed during this second-round review. This report is the only repository file added by the review.

## Verdict

The remediation is materially better but is not complete and Phase 7 must remain under acceptance review. The matrix-free native MVP, finite-boson amplitude recurrence, embedding validation, repeated-site Pauli builder semantics, uniform-qudit backend executor, domain-specific boson/qudit compile signatures, and coarse-FFI cleanup are real improvements. However, one CRITICAL silent algebra-corruption path remains because partially mapped hybrid multiplication is still order-dependent, the fermion expansion preflight rejects simple one-output inputs using a structure-insensitive `2**len` estimate, direct-Weyl plan dimensions can silently overflow platform indices, and the required correctness, metadata, typing, benchmark, and release-record handoff is still incomplete.

The current `implementation-status.md:7,16,56-58,206` correctly keeps Phase 7 under acceptance review and says that the full P0–P5 evidence matrix remains work. That status must not be changed to complete until the findings below are closed.

## Compliance checklist

| Requirement | Result | Evidence |
| --- | --- | --- |
| Original C2 embedding invariants | PASS | Explicit maps are validated before the fast path; map names, exact integer indices, complete coverage, injectivity, target ranges, qudit dimensions, canonical sorting, and fermionic permutation signs are handled in `python/tencirpauli/structured.py:585-751`. |
| Original M2 matrix-free native structured MVP | PASS | `StructuredMvpPlan` stores local operations and coefficients rather than COO transitions in `crates/tencir-pauli-core/src/structured.rs:1338-1479`; focused dense/MVP equivalence passes. |
| Original M3 finite-boson numeric consistency | PASS | Ladder amplitudes use representable square-root recurrences and dense/sparse targets agree at the reviewed `(b†)^171` boundary; `tests/test_phase7_structured.py:326-332` passes. |
| Original M5 repeated Pauli builder product | PASS | Local Pauli factors are multiplied in input order with their phase in `python/tencirpauli/structured.py:2152-2169`; the `X*Y=iZ` regression passes. |
| Original M6 production coarse-FFI path | PASS with cleanup observation | Adjoint and tensor product no longer cross PyO3 per term, and finite compilation dispatches to native kernels. An unused Python transition implementation remains as MINOR N1. |
| Original N1/N2 validation and signatures | PASS | `OperatorSpace` enforces `d <= u32::MAX`, and boson/qudit public compile methods reject irrelevant mapping keywords through explicit signatures. |
| Partially mapped algebra is closed and order-correct | FAIL | CRITICAL C1 reproduces a wrong dense operator for raw-left/mapped-right multiplication. |
| Expansion guards are useful as well as safe | FAIL | MAJOR M1 rejects an already canonical 40-creation word as an estimated 211 TB expansion even though the exact output has one term. |
| Direct-Weyl plan dimensions use checked arithmetic | FAIL | MAJOR M2: `np.prod(..., dtype=np.intp)` silently wraps an oversized dimension. |
| Frozen correctness/property matrix is complete | FAIL | MAJOR M3: the 18 focused tests do not cover multiple required algebra, backend, determinism, overflow, and property cases. |
| Representative release benchmark handoff is complete | FAIL | MAJOR M4: the official clean manifest is `failed`, required workloads are absent, and most required metadata fields are not recorded. |
| Public plan metadata, typing, and docstrings satisfy P0/P5 | FAIL | MAJOR M5: frozen plan metadata fields and strict public typing/docstring coverage remain incomplete. |
| Repository quality smoke | PASS | `python scripts/check.py --benchmark smoke` passed 31 Rust tests, 206 Python tests, and 158 selected benchmark-smoke cases; this is regression evidence, not release benchmark acceptance. |

## CRITICAL

### C1. Partially mapped hybrid multiplication still corrupts raw-left/mapped-right operator order

Locations: `crates/tencir-pauli-core/src/structured.rs:82-145,675-728` and `tests/test_phase7_structured.py:290-300`.

The remediation now preserves an existing mapped Pauli factor when `map_fermions()` expands a remaining raw fermion word, but it always evaluates the final product as `base_mapped * newly_mapped_raw` at `structured.rs:695-701`. Hybrid multiplication separately combines all raw fermion factors at `structured.rs:102-105` and all mapped factors at `structured.rs:110-111`; the term representation does not retain whether a mapped factor originally appeared before or after a raw fermion factor. It therefore cannot represent the general product `(M_left F_left)(M_right F_right)` as one unordered pair of `mapped_codes` and raw `fermion` data.

The added regression covers only mapped-left/raw-right:

```python
((create * boson_create).map_fermions() * annihilate).map_fermions()
```

That orientation now agrees with mapping after the complete product. The reverse orientation remains wrong:

```python
space = tcp.OperatorSpace(fermions=1, bosons=1)
create = space.fermion.create(0)
annihilate = space.fermion.annihilate(0)
boson_create = space.boson.create(0)

actual = (annihilate * (create * boson_create).map_fermions()).map_fermions()
expected = (annihilate * (create * boson_create)).map_fermions()
```

On the reviewed release extension, `max(abs(actual_dense - expected_dense))` is `1.4142135623730951`. The mapped-left/raw-right orientation gives `0.0`. This is a silent numerical error on accepted public algebra and remains release-blocking.

Resolution path:

1. Prefer the smallest closed semantic rule: if either operand contains `mapped_fermion` and either operand still contains a raw fermion factor, map the raw fermion content of each operand separately and multiply the resulting Pauli factors in original operand order. The result may become fully mapped; preserving a partially raw representation is not required.
2. Alternatively, reject every further multiplication, tensor product, commutator, anticommutator, or adjoint that would mix raw and mapped fermion factors. A clear `ValueError`/`NotImplementedError` is acceptable if partial-algebra closure is deliberately removed from the public contract; silently reordering is not.
3. Do not attempt to repair this only by swapping the operands in `jordan_wigner_hybrid_terms`; no single fixed order handles both orientations or nested products.
4. Add dense differentials for mapped-left/raw-right, raw-left/mapped-right, both operands containing raw-plus-mapped content, nested three-factor products, adjoint, commutator, and tensor-product paths. Each case must compare with “complete raw algebra first, then one Jordan–Wigner mapping.”

## MAJOR

### M1. The fermion expansion preflight is safe but structurally useless for simple valid inputs

Locations: `crates/tencir-pauli-core/src/structured.rs:859-910`; contract: `docs/vibe/phase-7-spec.md:25,279-285,444-455,499-503`.

`fermion_rewrite()` computes `2**sequence.len()` before inspecting the sequence and charges 192 bytes for every hypothetical branch. An already canonical sequence of 40 distinct creation operators has exactly one output term and requires no contraction branch, but the public constructor raises:

```text
MemoryError: requested 211106232532992 bytes exceeds memory limit 17179869184
```

The current guard prevents unbounded work, but it violates the practical purpose of a best-effort memory guard by rejecting obviously cheap inputs. It also leaves the adjacent-swap recursion unchanged at `structured.rs:868-910`, so the implementation still pays repeated sequence cloning and can revisit equivalent intermediate states.

More precise protection does add work, and this review does not assume that even an `O(n)` scan is free on the common one-/two-body and large-batch construction paths. Precision is not the goal. The goal is to avoid catastrophic or obviously excessive allocation without imposing a material end-to-end regression. Any replacement must therefore be implemented behind a release-mode A/B benchmark and profile decision, not accepted merely because its asymptotic bound looks better.

Owner performance decision for this remediation: do not add an expensive exact preflight. If a structure-aware preflight causes a clear common-path performance regression, simplify it or remove it. Retain mandatory checked dimension/arithmetic overflow and cheap guards for allocations whose size is already known. For symbolic CAR expansion whose eventual size cannot be estimated cheaply and usefully, it is acceptable to rely on low-overhead running checks or, if those are also a measured bottleneck, to leave pathological memory exhaustion to the caller. Public `max_bytes` remains a best-effort guard, not an exact allocator or peak-RSS guarantee.

Recommended implementation:

1. First profile the current `fermion_rewrite()` call sites and establish how much time is spent in preflight, sequence scanning, cloning, recursion, aggregation, and FFI conversion on representative release workloads. Do not optimize from the 40-factor reproducer alone.
2. Prototype an `O(n)` canonical/nilpotent fast path before any exponential estimate. Detect repeated equal-action/equal-mode factors, detect an already canonical sequence, and return zero or the single canonical key directly. An inversion-only sequence with no possible same-mode contraction should canonicalize deterministically with one sign and one output rather than receive a `2**n` estimate.
3. For multiplication of canonical fermion words, evaluate contraction-aware set/bitset dynamic programming. Only overlap between left annihilation modes and right creation modes can branch. Estimate from actual contraction opportunities only if that estimate is cheaper than the work it prevents.
4. For arbitrary raw `from_terms()` sequences, prefer incremental canonical aggregation with an `O(1)` maintained contribution/key counter. Do not rescan the full aggregate on every insertion, and do not perform a complete dry-run expansion solely to predict the complete real expansion.
5. Benchmark and profile at least three variants on the same commit and machine: the current coarse `2**len` preflight, the proposed structure-aware guard, and a no-symbolic-preflight variant that retains checked arithmetic plus actual-allocation/running guards. Include Rust-core and Python/PyO3 end-to-end boundaries.
6. Choose the fastest variant that preserves correctness and avoids the demonstrated false rejection. If the structure-aware guard has a clear measurable regression on representative common paths, remove or simplify it even if it predicts pathological inputs more accurately.
7. Preserve the project rule that `max_bytes` is a cheap best-effort guard. Do not add allocator accounting, exact peak-RSS prediction, or a second full expansion pass. Pathological caller input exhausting memory is not by itself evidence that a more expensive preflight belongs on the hot path.

Required regressions:

- 40 and 128 distinct canonical creation factors produce one term under a deliberately small but sufficient `max_bytes` limit.
- A long inversion-only word produces one signed term rather than an exponential preflight failure.
- Repeated identical creation or annihilation factors return zero without expansion.
- Small contraction-heavy words agree with an independent dense Fock reference and exercise zero, one, and multiple contraction branches.
- A bounded genuinely expansion-heavy case fails before unsafe growth when `max_bytes` is low and succeeds with the same deterministic result when the limit is sufficient.
- Release A/B benchmarks cover common one-/two-body bulk construction, duplicate-heavy canonicalization, hybrid builder input, canonical/no-contraction, inversion-only, and bounded contraction-heavy cases. The report must include core-kernel and Python/PyO3 end-to-end results.
- A profile identifies the actual cost center for each tested guard variant. If sampling tools are unavailable, use scoped release instrumentation and allocation counters; do not substitute debug timing.
- The remediation handoff states explicitly which variant was selected and why. A structure-aware guard is rejected if its overhead is material on representative common paths, even when it is more accurate on pathological inputs.

### M2. Direct-Weyl and generic local-dimension plan metadata can silently overflow

Locations: `python/tencirpauli/hamiltonian.py:125-130,224-229,240-256` and `python/tencirpauli/structured.py:2298-2345`; contract: `docs/vibe/phase-7-spec.md:17,116,274-285,395-397,451-453`.

Both `NativeMVPPlan.dimension` and `BackendMVPPlan.dimension` use `np.prod(self.local_dimensions, dtype=np.intp)`. NumPy integer multiplication wraps at platform width instead of raising. A reviewed direct-Weyl plan with dimension 3 and 50 sites reports `6048575297968530377`, while the exact value `3**50` exceeds the platform-index range. Subsequent memory checks and shape validation therefore operate on a false dimension.

Resolution path:

1. Compute local-dimension products with checked Python integers, for example `math.prod`, then reject values larger than `np.iinfo(np.intp).max` before conversion to an array shape or platform index.
2. Validate and cache the checked dimension during plan construction/`__post_init__`; do not recompute it through NumPy every time the property is accessed.
3. Validate that direct-Weyl `local_dimensions`, `qudit_dimension`, exponent-array width, and exponent ranges are mutually consistent.
4. Add no-allocation overflow tests for oversized `d**n`, boundary-success tests at representable products, and malformed public `BackendMVPPlan` construction tests.

### M3. The frozen Phase 7 correctness and property matrix remains incomplete

Locations: `tests/test_phase7_structured.py:81-353`; contract: `docs/vibe/phase-7-spec.md:440-455,493-521`.

The focused file has increased from 13 to 18 passing tests, but the additions are narrow regressions rather than the required acceptance matrix. The direct-Weyl parameterized test at `tests/test_phase7_structured.py:112-130` checks one monomial against dense/COO/CSR/native MVP. It does not test Weyl multiplication phases, adjoint, commutation, Hermiticity, modular properties, or coefficient aggregation. The backend test at `tests/test_phase7_structured.py:335-353` covers only TensorCircuit NumPy, `d=3`, one coefficient set, and one two-site layout; JAX, `d=4/5/6`, coefficient override, JIT-compatible execution, and invalid-dimension boundaries are absent.

Other missing gates include property-based CAR/CCR/Weyl identities, both partial-mapping operand orders, complete embedding permutation vectors, multiple-cutoff reuse, deterministic structural replay across supported Rayon thread counts, explicit `max_bytes=None`, plan-metadata invariants, and checked direct-Weyl dimension overflow.

Resolution path:

1. Add independent small-system property/differential tests that never call the implementation under test to construct the expected algebra.
2. Parameterize the uniform-Weyl backend differential over `d=3,4,5,6`, one and multiple sites, flat and rank-shaped states, default and overridden coefficients, and TensorCircuit NumPy/JAX backends. Enable JAX x64 for complex128 acceptance comparisons and use the documented tolerance.
3. Add exact structural assertions for multiplication phase exponents, adjoints, commutation, and Hermiticity, not only matrix-action comparisons.
4. Add thread-count replay tests only for paths that actually use supported parallel execution; compare deterministic structural arrays bitwise and numeric outputs at the specified tolerance.
5. Convert every reproduced second-round issue into a deterministic regression before changing implementation.

### M4. The release benchmark handoff is still not an accepted record and lacks required workloads/metadata

Locations: `benchmarks/python/test_phase7_structured_benchmark.py:15-338`, `.benchmarks/runs/phase7-remediation-20260803.json:1-25`, and `docs/vibe/phase-7-spec.md:457-472,523-531`.

A clean focused pytest-benchmark JSON exists for commit `2b0f0dc`, but the official harness manifest `phase7-remediation-20260803` is marked `"status": "failed"`. It therefore does not satisfy the P5 requirement for a completed clean release record through the benchmark harness.

The benchmark source still lacks a Holstein or spin-boson scaling workload and an aggregation-heavy duplicate fermion workload. It measures one uniform-Weyl point at `d=5` rather than several dimensions. Most cases record no `extra_info`; the recorded metadata does not provide the required input terms, canonical terms, generated contributions, nonzeros/transitions, output bytes, thread count, throughput, and numerical error. The source labels and docstrings also still refer to an “adaptive Python/Rust” split although structured finite targets now use the native path.

Resolution path:

1. Add the missing representative workloads without making them memory-dangerous: duplicate-heavy fermion canonicalization/mapping, Holstein or spin-boson mixed native MVP scaling, and uniform Weyl chains at several `d` values.
2. Populate every required metadata field from measured values or explicit workload construction. Do not leave correctness error implicit in an assertion; record it.
3. Separate construction, first apply, steady apply, and materialization where the frozen specification requires separate boundaries.
4. Run `python benchmarks/run.py record` on a clean commit and confirm `python benchmarks/run.py list` reports the label as `complete`, not `failed`. Keep `.benchmarks/` ignored.
5. Record the successful label, commit, workload coverage, thread environment, and known limitations in `implementation-status.md` only after the run succeeds.

### M5. Public plan metadata, typing, and docstring delivery is still below the frozen P0/P5 contract

Locations: `python/tencirpauli/hamiltonian.py:63-161`, `python/tencirpauli/structured.py:759-877,1104-1228,1884-2087,2224-2345`; contract: `docs/vibe/phase-7-spec.md:390-399,426-438,523-527`.

`NativeMVPPlan` exposes dimension, local dimensions, term count, estimated bytes, basis ordering, and strategy, but not the frozen target/schema version, mapping label, boson cutoffs, projected-Fock boundary label, or direct-Weyl convention where applicable. `BackendMVPPlan.required_operations` defaults to Pauli operations even for a `direct_weyl` plan because `_direct_weyl_backend_plan()` does not override it. Public factories, `terms`, arithmetic dunder methods, mapping, and compile returns still use `Any`, and `structured.py` has only 12 docstring blocks across roughly 138 class/property/method definitions. Passing mypy under these annotations does not demonstrate the strict public typing requested by P5.

Resolution path:

1. Add the compact frozen metadata fields to reusable plans, keeping domain-inapplicable fields explicit and immutable rather than inferred from private handles.
2. Set direct-Weyl `required_operations` to the actual versioned backend operation vocabulary and test it.
3. Replace public `Any` returns with concrete result types or overloads/Literal-target signatures. Internal private helpers may retain pragmatic types where justified.
4. Add concise docstrings to the exported constructors, algebra methods, compile targets, plan fields, and error semantics; private mechanical helpers do not need ceremonial documentation.
5. Add introspection/type-check tests only where they protect a stable public contract; do not introduce a second stub-only API that drifts from runtime signatures.

## MINOR

### N1. Dead Python finite-transition code and stale adaptive labels remain after the native-kernel cleanup

Locations: `python/tencirpauli/structured.py:2394-2550` and `benchmarks/python/test_phase7_structured_benchmark.py:106-116,181-182,211,241`.

`_finite_entries()` and its transition helpers are no longer called by production compilation, but the duplicate numerical implementation remains in the runtime module. Benchmark case names such as `small_python`, `medium_rust`, and descriptions of adaptive dispatch are now misleading because `native_mvp`, dense, and sparse structured targets use the native kernels. Remove the dead runtime implementation after preserving any needed logic as an explicitly independent test reference, and rename benchmark cases by workload size rather than an obsolete dispatch strategy.

## OBSERVATIONS

- The second-round review found no new defect in the matrix-free plan representation itself. `StructuredMvpPlan` is compact in the intended sense and no longer retains a complete COO matrix.
- The finite-boson overflow fix is appropriate: multiplying square-root ladder factors avoids an unrepresentable factorial intermediate while still rejecting a genuinely non-finite final complex128 result.
- The embedding remediation is substantially aligned with the original review. Further embedding work should be driven by missing frozen test vectors rather than another redesign.
- `scripts/check.py --benchmark smoke` passing is valuable regression evidence, but the repository correctly documents that benchmark smoke is not a comparable release record.

## Current P0–P5 assessment

| Slice | Result | Remaining gate |
| --- | --- | --- |
| P0 references/API alignment | PARTIAL | Complete reusable-plan metadata, checked local-dimension products, frozen metadata tests, and strict target return typing. |
| P1 fermion-to-Pauli | FAIL | C1 order-correct partial mapping, useful contraction-aware expansion protection, CAR/property coverage, and complete release construction/mapping evidence. |
| P2 boson/native finite | PARTIAL | Core numeric and matrix-free fixes pass; multiple-cutoff/property coverage and accepted release plan/apply evidence remain. |
| P3 hybrid native | FAIL | C1 remains a silent hybrid algebra error; Holstein/spin-boson differential and scaling evidence is absent. |
| P4 direct Weyl | PARTIAL | Backend execution exists, but checked plan dimension, algebra/property suite, formal NumPy/JAX matrix, and multi-`d` release evidence remain. |
| P5 delivery/handoff | FAIL | Smoke and the example pass, but strict typing/docstrings, full metadata, complete benchmark coverage, and a successful clean harness record remain absent. |

## RECOMMENDED IMPROVEMENTS

1. Close C1 before any release claim. Either eagerly map both operands when partial and raw fermion content meet, preserving operand order, or reject the operation explicitly. Add both-order and nested algebra regressions.
2. Profile and A/B test the CAR guard before selecting it. Compare the current blanket estimate, a canonical/no-contraction fast path with contraction-aware incremental DP, and a no-symbolic-preflight variant. Keep the refined guard only if release core and Python/PyO3 measurements show no material common-path regression; otherwise simplify or remove it while retaining checked arithmetic and cheap known-allocation guards.
3. Replace NumPy integer products in plan dimension metadata with one checked platform-index product and add no-allocation overflow regressions.
4. Complete the frozen correctness/property matrix, especially direct-Weyl algebra, TensorCircuit NumPy/JAX `d=3/4/5/6`, coefficient overrides, deterministic replay, and every second-round reproduction.
5. Complete public plan metadata, correct direct-Weyl required operations, and replace exported `Any` return types with concrete types/overloads plus concise public docstrings.
6. Add the missing duplicate-heavy, Holstein/spin-boson, and multi-`d` workloads; record all required metadata; then produce a clean benchmark-harness manifest with `status=complete`.
7. Rerun `python scripts/check.py --benchmark smoke`, the executable structured example, focused performance-large cases where applicable, and the clean release record before changing Phase 7 from under acceptance review.

## Validation performed

- `conda run -p .conda maturin develop --release --locked` — passed on the reviewed commit.
- `conda run -p .conda pytest -q tests/test_phase7_structured.py` — 18 passed.
- `conda run -p .conda python scripts/check.py --benchmark smoke` — passed formatting, Clippy, Ruff, strict mypy, 31 Rust tests, release build, 206 Python tests, Rust benchmark smoke, and 158 selected Python benchmark-smoke cases.
- `conda run -p .conda python examples/structured_algebra.py` — passed and printed `structured targets agree`.
- Independent read-only probes reproduced C1 with maximum dense error `1.4142135623730951`, reproduced the canonical 40-creation false memory rejection at 211,106,232,532,992 estimated bytes, and reproduced direct-Weyl platform-index wrap for `3**50`.
- Independent direct-Weyl probes executed TensorCircuit NumPy and JAX backends for `d=3,4,5,6`, flat states, and coefficient overrides; this supports the implementation direction but does not replace the missing committed acceptance tests.
