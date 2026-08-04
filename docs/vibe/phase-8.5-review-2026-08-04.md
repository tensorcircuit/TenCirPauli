# Phase 8.5 explicit MVP storage and reusable execution review

Review date: 2026-08-04

Reviewed commit: `2ca1d722115b2f1024bf3156aa3fefdfa06fe50e` (`perf: optimize packed U1 lazy and CSR execution`), including the Phase 8.5 implementation commit `a534e0d`.

Scope: independent implementation, correctness, resource-contract, and hot-path review against `phase-8.5-spec.md`. The review covered unrestricted Pauli and structured native MVP plans, generic charge-restricted lazy/eager execution, packed U1 dispatch, the spinful-fermion Hubbard path, strict `apply_into`, facade eager caching, memory guards, benchmark coverage, and the committed QuSpin research workflow. Production code was not modified by this review.

## Verdict

Phase 8.5 has a sound API and correctness foundation, but it is not complete against the frozen acceptance contract. Uniform lazy defaults, packed U1 lazy execution, deprecation of `restrict_u1()`, immutable plan metadata, basic eager-cache behavior, and small-system numerical differentials are implemented. The complete Python test suite passes.

The remaining gaps are concentrated in the paths that determine whether the milestone is usable at its intended scale. The spinful-fermion fast path allocates a hidden state-sized output and rebuilds its combinatorial index on every call; generic eager charge execution is not retained as a native CSR handle; several budget checks occur after allocation or cache mutation; allocating public MVP calls make avoidable full-output copies; U1 lazy `apply_into` ignores its scratch budget; and structured eager storage is currently metadata-only. The Phase 8.5 benchmark file also does not implement the frozen representative matrix.

Recommended status: **implementation checkpoint reached; major remediation open; performance acceptance pending**. The current implementation is suitable for bounded correctness work and small/medium experiments, but the 4x4 spinful-Hubbard target and the complete storage/resource contract are not yet accepted.

## Review and validation performed

- Inspected `phase-8.5-spec.md`, commits `a534e0d` and `2ca1d72`, and the current Rust core, PyO3, Python facade, tests, benchmarks, and research scripts along all Phase 8.5 execution paths.
- Ran the complete Python suite with the project environment: `317 passed` with `57` deprecation warnings in approximately 43 seconds.
- Ran 121 focused Phase 8.5, charge, Hamiltonian, structured, and symmetry tests; all passed.
- Ran `benchmarks/python/test_phase85_mvp_benchmark.py --benchmark-only`; all six committed benchmark cases passed.
- Ran targeted runtime probes for cached-plan budgets, failed materialization cache mutation, structured eager/lazy metadata, native-array ownership, and U1 `apply_into` scratch budgets. These probes reproduced R3, R4, R5, and R6 below.
- Ran the committed TenCirPauli Hubbard script locally. The 2x4 half-filled case measured approximately `0.846 ms` median allocating MVP. The 4x3 case measured approximately `217 ms` for allocating `apply()` and `216 ms` for `apply_into()`, showing that the current strict-buffer entry point does not yet avoid the dominant temporary in the spinful fast path.
- Built the documentation with MkDocs strict mode after archiving this report; the build passed.
- Rust workspace tests were not rerun because `cargo` was unavailable in the review environment. Existing repository status claims for Rust checks were not treated as independently revalidated evidence.

## Acceptance matrix

| Area | Result | Assessment |
| --- | --- | --- |
| Uniform lazy defaults | PASS | Pauli, structured, generic charge, and packed U1 public CPU-native entry points default to lazy. |
| Fixed plan metadata and immutability | PASS WITH GAPS | Public metadata is fixed, but structured eager metadata does not correspond to a distinct retained representation. |
| Restricted facade eager caching | PARTIAL | Cache reuse and lazy-plan stability work, but target preflight and cached-plan budget checks are incorrect. |
| Pauli native MVP | PASS WITH PERFORMANCE DEBT | Lazy/eager kernels are native and correct; allocating `apply()` makes a redundant full-output copy. |
| Structured native MVP | PARTIAL | Compact native execution exists, but eager storage is not implemented as a distinct bounded cache. |
| Generic charge lazy MVP | PARTIAL | Repeated term serialization is avoided, but positions and structural validations are rebuilt on every call and Python descriptor mirrors remain retained. |
| Generic charge eager MVP | FAIL | The facade retains COO-like NumPy triples rather than one immutable native destination-major CSR handle. |
| Packed U1 backend | PASS WITH RESOURCE GAP | Native lazy/eager paths and packed dispatch work, but lazy `apply_into` ignores scratch budgets and allocating fixed plans copy outputs. |
| Spinful-fermion backend | FAIL FOR TARGET SCALE | The current shortcut is numerically correct on bounded fixtures but does not cache its index/descriptors and allocates a hidden full state on every call. |
| Strict `apply_into` protocol | PARTIAL | Type, shape, contiguity, writeability, and overlap checks exist; the spinful path violates the no-hidden-output allocation objective. |
| Correctness tests | PASS FOR COVERED CASES | The full Python suite passes, but several frozen concurrency, retry, fallback, and cross-strategy cases are not covered. |
| Performance acceptance and QuSpin target | PENDING | Small/medium evidence exists; the complete matrix and approved 4x4 matched A/B are not complete. |

## Findings

### R1 — MAJOR: the spinful-fermion `apply_into` path allocates a hidden full state and rebuilds its index every call

Locations: `crates/tencir-pauli-core/src/charge.rs:390-477,562-642,1123-1139`; `crates/tencirpauli-native/src/charge_sector.rs:362-372`; contract: sections 5.5, 6, 7, 9, 13, and 16 of `phase-8.5-spec.md`.

`NativeChargeMvpPlan` retains the generic sector plan, raw term descriptors, and a Python-derived particle count, but it does not retain a `FastFermionSectorIndex` or specialized spinful term descriptors. Each call enters `try_apply_fast_fermion_mvp()`, constructs the combination masks and optional direct rank table again, scans every raw fermion operation through `apply_fast_fermion_term()`, allocates `vec![0; state.len()]`, and finally copies that vector into the caller-owned output.

This defeats both reasons for adding `apply_into`: the call still allocates one complete output-sized buffer and still pays a full memory write/copy. It also means the supposedly reusable plan repeats combinatorial table construction. On the approved 4x4 half-filled target, the restricted dimension is `C(16, 8)^2 = 165,636,900`, so one complex128 state is `2,650,190,400` bytes, approximately `2.47 GiB`. `apply_into()` therefore hides an additional approximately `2.47 GiB` allocation; allocating `apply()` concurrently holds the outer result and this inner temporary, for approximately `4.94 GiB` of output storage before the input and other process memory are counted.

The local 4x3 measurement is consistent with this implementation: `apply()` and `apply_into()` were approximately `217 ms` and `216 ms`, respectively. The reusable output buffer provides effectively no steady benefit because the core still allocates and copies the dominant state.

Required resolution:

1. Compile and validate eligible spinful terms once into compact descriptors at native plan construction. At minimum, distinguish diagonal density terms, quadratic hopping, and validated generic fermion fallback descriptors.
2. Construct the immutable `FastFermionSectorIndex` once per plan and share it with all calls. Cache tables only when their checked size fits the construction budget; retain the existing combinatorial fallback otherwise.
3. Change the spinful kernel to accept `&mut [Complex64]`, clear it, and write directly into caller-owned output. Do not return a temporary `Vec<Complex64>` from the fast helper.
4. Keep the initial kernel serial if necessary. Do not introduce source-parallel state-sized worker outputs. Evaluate destination-major parallelism only after a direct-output cached baseline is measured.
5. Move fast-path eligibility and descriptor validation into the native constructor rather than treating Python booleans as semantic authority.

### R1 memory/speed tradeoff assessment

The recommended cache is strongly favorable for the approved workload and does not require dimension-scale retained memory. For the 4x4 half-filled sector, one shared combination-mask table contains `C(16, 8) = 12,870` `u128` masks and occupies `205,920` bytes, approximately `0.196 MiB`. A direct rank table contains `2^16` `u32` entries and occupies `262,144` bytes, exactly `0.25 MiB`. Together they occupy approximately `0.446 MiB`; compact descriptors for the roughly 80 Hubbard terms add only a few KiB. This sub-MiB retained cache replaces a per-call approximately `2.47 GiB` temporary and repeated table construction.

The tradeoff must remain bounded rather than unconditional. The current internal thresholds imply a maximum combination-mask cache of approximately `64 MiB` and a maximum rank table of approximately `4 MiB`. Those limits are reasonable as upper eligibility bounds but must also obey the caller's remaining construction `max_bytes`. When either table does not fit, the plan should retain only the compact combinatorial metadata and use the existing checked rank/unrank fallback. `estimated_bytes` must include whichever tables were actually retained.

The cache should be stored once and shared, not duplicated for up/down species: the two species use the same `(sites, particles)` combinatorics in the approved balanced sector. No full basis, source table, destination table, output state, or eager transition graph is required. Therefore the proposed change simultaneously reduces peak memory, removes memory bandwidth from the output copy, and reduces steady CPU time; it is not a speed-for-large-memory exchange on the target workload.

### R2 — MAJOR: generic eager charge plans are not immutable native CSR handles

Locations: `python/tencirpauli/charge.py:810-931,1212-1293`; `crates/tencirpauli-native/src/charge.rs:107-225`; contract: sections 4, 5.3, 6, and 9 of `phase-8.5-spec.md`.

`ChargeMvpPlan` stores read-only NumPy `rows`, `columns`, and `coefficients` arrays and advertises `strategy="destination_major_csr"`. Repeated `apply()` and `apply_into()` call free PyO3 functions that borrow all arrays again, convert and bounds-check every row and column, clear the output, and perform a serial row scatter. No native eager plan object owns the validated transition graph, and no CSR `indptr` is retained. Calling `csr()` reconstructs `indptr` in Python every time.

This is a substantial mismatch with the frozen native-handle and authoritative-CSR design. It adds repeated validation and FFI array plumbing in the steady path, prevents direct destination-row gather and bounded Rayon parallelism, retains an unnecessary row index for every transition, and makes the `strategy` label stronger than the actual representation.

Required resolution: introduce a native eager charge plan with shared immutable `indptr`, `columns`, and `values`; validate indices and ordering once; expose direct `apply_into`; derive COO rows only when requested; and share the native storage between the facade and fixed eager plans. A serial gather threshold followed by destination-row Rayon is the natural performance path because workers own disjoint output rows without state-sized scratch.

### R3 — MAJOR: failed or low-budget materialization can populate caches before target validation

Location: `python/tencirpauli/charge.py:1212-1293`; contract: sections 4 and 11 of `phase-8.5-spec.md`.

`dense()` calls `_ensure_eager()` before checking the dense output size. `csr()` calls `_ensure_eager()`, allocates and fills `row_indices` and `indptr`, and only then checks the CSR budget. A targeted generic-sector probe called `dense(max_bytes=100_000)` on a 252-dimensional facade. The call correctly raised because the dense output required approximately `1,016,064` bytes, but the facade retained estimate increased from `6,880` to `9,120` bytes because an eager cache had already been installed.

Cached eager plans also bypass the plan-return budget: after constructing a 64-byte eager plan, `mvp_plan(storage="eager", max_bytes=0)` returned that cached plan instead of rejecting it. This contradicts both the method documentation and the fixed-plan budget contract.

Required resolution: perform checked target-size and combined incremental-memory preflight before `_ensure_eager()` when the cache is absent; do not mutate the facade on a target-budget failure; check cached fixed-plan retained bytes before returning; and check CSR output before allocating `indptr`. A failed eager build must continue to leave the cache empty so a later larger budget can retry.

### R4 — MEDIUM: allocating MVP wrappers make a redundant complete output copy

Locations: `python/tencirpauli/hamiltonian.py:257-286`; `python/tencirpauli/charge.py:882-911`; `python/tencirpauli/symmetry.py:496-523`; contract: sections 7, 11, and 13 of `phase-8.5-spec.md`.

The native functions already return an owned NumPy array backed by the Rust `Vec`. `NativeMVPPlan.apply()`, eager `ChargeMvpPlan.apply()`, and fixed `U1MvpPlan.apply()` then call `np.array(..., copy=True)`, allocating and copying a second complete output. The direct native result has a valid owning base and does not require this defensive copy.

On a local 20-qubit one-term Pauli workload, direct native allocation measured approximately `0.484 ms`, public allocating `apply()` approximately `0.763 ms`, and `apply_into()` approximately `0.304 ms`. The redundant wrapper copy therefore added approximately 58% over the direct native allocating call on this memory-bandwidth-sensitive case. It also makes the actual peak output memory exceed the single-output `max_bytes` estimate.

Required resolution: return the native PyArray directly after dtype/shape invariants are guaranteed, or allocate exactly one Python-owned output and dispatch through the shared `apply_into` kernel. Add ownership and peak-major-buffer regressions that distinguish a view from a safe owned native array without forcing a second copy.

### R5 — MEDIUM: U1 lazy plan and `apply_into` budget checks are incomplete

Locations: `python/tencirpauli/symmetry.py:330-354,525-545`; `crates/tencirpauli-native/src/symmetry.rs:334-350`; `crates/tencir-pauli-core/src/sector.rs:533-631`; contract: sections 6, 7, and 11 of `phase-8.5-spec.md`.

The U1 facade returns an existing lazy fixed plan without checking its retained estimate against the requested `max_bytes`. The native lazy `apply_into` binding explicitly discards `max_bytes`, even though the core allocates a basis iterator, active-qubit vector, two packed-word buffers, and a per-source aggregate vector.

A targeted probe created a lazy plan with `estimated_bytes=456`; `mvp_plan(max_bytes=0)` returned it, and `apply_into(max_bytes=0)` completed successfully. The scratch is normally small, but it can scale with packed width and X-group count, and the public contract explicitly assigns it to `apply_into`.

Required resolution: check retained bytes before returning fixed plans, thread a scratch budget into `U1LazyMvpPlan::apply_into`, preflight its cheaply known vectors/capacities, and keep zero-scratch eager `apply_into` valid with `max_bytes=0` where no new major allocation occurs.

### R6 — MEDIUM: structured eager storage is currently a metadata-only distinction

Locations: `python/tencirpauli/structured.py:2612-2651`; `crates/tencir-pauli-core/src/structured.rs:1443-1620`; contract: sections 3.1, 5.2, and 16 of `phase-8.5-spec.md`.

Both structured storage values call the same constructor with the same arguments and retain the same compact plan. The native layer never receives `storage`. A probe of a finite boson creation plan reported identical strategy and retained bytes for lazy and eager: `("lazy", "structured_mvp_native", 136)` versus `("eager", "structured_mvp_native", 136)`.

The specification permits the same kernel when bounded caches do not help, but it says distinct storage metadata should be reported only when retained data actually differ. The current result advertises an eager plan that has no eager representation.

Required resolution: profile and implement only bounded local tables that demonstrate a representative benefit, such as reusable boson destination/factor tables or Weyl phase/shift tables. If no table is retained, the public metadata must not claim a distinct eager representation without an explicit owner-approved contract change. Do not allocate a meaningless cache merely to make the labels differ.

### R7 — MEDIUM: the committed Phase 8.5 correctness and benchmark matrices do not cover the frozen gates

Locations: `tests/test_phase85_api.py:11-80`; `tests/test_charge.py:292-313`; `benchmarks/python/test_phase85_mvp_benchmark.py:21-65`; contract: sections 12, 13, and 16 of `phase-8.5-spec.md`.

The dedicated correctness file checks one cache creation, basic plan storage, dtype rejection, identity overlap, and one lazy/eager Pauli comparison. It does not freeze failed-build retry, concurrent first materialization, overlapping slices, non-contiguous and read-only outputs, unchanged input, repeated two-buffer alternation, concurrent calls on one plan, cached-plan budgets, or the spinful table/fallback boundaries. The spinful test named `matches_generic_and_eager` constructs both `eager` and `fast` with default lazy storage, so it never compares the eager graph.

The dedicated benchmark file has only six small cases. The two tests named `construction_and_apply` time construction only, and `test_u1_lazy_and_eager_apply` constructs and times only the default lazy plan. It has no structured case, no non-termwise generic aggregation case, no eager charge steady apply, no cache-construction/materialization split, no spinful 2x4 or 4x3 workload, and no allocation/peak-memory evidence.

Required resolution: implement the frozen correctness checklist and benchmark matrix with separate construction, first apply, steady allocating apply, steady `apply_into`, eager-cache build, materialization, retained bytes, scratch/output bytes, transition count, and numerical error. Keep wall times informational. The approved 4x4 QuSpin comparison remains manual and must be run only after R1 is closed.

### R8 — MINOR: defensive Python mirrors retain duplicate plan metadata and understate retained bytes

Locations: `python/tencirpauli/charge.py:789-807,947-1019`; `python/tencirpauli/symmetry.py:203-259,423-494`.

`ChargeLazyMvpPlan` stores `_inputs` after `compile_mvp()` has cloned the same descriptors into the native handle. Production execution never reads `_inputs`; the current test suite uses it as a private hook to disable the spinful shortcut. `estimated_bytes` counts the sector and coefficient array but not the retained nested Python lists or the native descriptor copy. U1 wrappers similarly retain source operators and Python `_terms` for compatibility fallbacks even when a native plan is always present in normal construction.

Required resolution: remove production retention that exists only to support a test or unreachable compatibility fallback, or isolate it behind an explicit advanced/debug path. Expose a native test selector or construct a genuinely non-eligible fixture rather than retaining duplicate descriptor graphs. Report the logical native descriptor storage that is actually retained; Python object headers and allocator metadata still need not be modeled exactly.

## Architecture and performance assessment

The core architecture remains appropriate: pure Rust algorithms are separated from PyO3, public plans are immutable, FFI calls are coarse, packed U1 remains specialized, and generic charge restriction does not materialize a full computational basis. The open issues do not require a new public abstraction or a broad rewrite.

The highest-value correction is to make retained native handles match the advertised storage strategies. Spinful lazy execution needs a cached compact index and direct output, while generic eager execution needs a true destination-major CSR handle. Both changes reduce work and memory simultaneously. They should be completed before adding more generalized cache layers, tunable public thresholds, allocator accounting, or parallel execution schemes.

The generic structured and charge kernels should not be prematurely over-engineered. Bounded tables, packed destination keys, and destination-major parallelism remain profile-gated. The current generic per-source occupation-vector aggregation is acceptable until a representative profile identifies it as material. In contrast, the full-state spinful temporary, repeated index construction, repeated eager-edge validation, and redundant public output copies are directly observable dominant costs and do not require additional profiling to justify removal.

## Recommended remediation order

1. Close R1 by caching the bounded spinful index/descriptors and writing directly into caller output. Re-run 2x4 and 4x3 first, then the approved 4x4 matched QuSpin A/B on a sufficiently provisioned machine.
2. Close R2 and R3 together by introducing one native shared generic eager CSR handle with correct preflight, cache publication, retry, and fixed-plan budget behavior.
3. Remove the R4 output copies and close the R5 U1 budget gaps; benchmark allocating and caller-buffer execution separately.
4. Resolve R6 only with measured bounded structured caches or an explicit owner contract adjustment.
5. Remove R8 defensive mirrors where practical, then complete the R7 correctness and benchmark matrices.
6. Run `python scripts/check.py --benchmark smoke`, strict documentation build, the complete release benchmark matrix, and the separate manual QuSpin comparison before changing Phase 8.5 to accepted.

## Closure checklist

- [ ] Spinful `apply_into` performs no state-sized internal allocation and does not rebuild combination/rank tables per call.
- [ ] Spinful cached tables obey construction `max_bytes`, are included in `estimated_bytes`, and fall back to bounded combinatorial rank/unrank when unavailable.
- [ ] Eligible spinful terms are compiled and validated once in the native constructor.
- [ ] Generic eager charge plans retain one shared native CSR handle and use destination-major gather.
- [ ] Failed dense/COO/CSR/eager requests do not populate caches before complete target preflight, and cached plans still obey call-level budget checks.
- [ ] Allocating Pauli, structured, generic eager, and packed U1 MVP calls create exactly one major output array.
- [ ] U1 lazy fixed-plan and `apply_into` budgets include retained data and newly allocated scratch.
- [ ] Structured eager metadata corresponds to a measured bounded retained cache or the contract is explicitly revised.
- [ ] The complete `apply_into` validation, overlap, overwrite, alternation, concurrency, retry, and cache-lifecycle matrix passes.
- [ ] The committed benchmark matrix includes Pauli crossover, packed U1, structured, generic aggregation, 2x4 spinful Hubbard, and 4x3 spinful Hubbard cases with construction/apply/materialization separation.
- [ ] A matched approved-machine 4x4 comparison shows TenCirPauli steady MVP above QuSpin beyond timing noise without an unbounded memory multiplier.
- [ ] The full Rust/Python quality gate and strict documentation build pass.

## Final acceptance statement

Current Phase 8.5 status: **implemented at API/correctness checkpoint; major hot-path and resource-contract remediation required; performance acceptance pending**.

The numerical results covered by the current Python suite are trustworthy, and the packed U1 work is directionally strong. The milestone should not be marked complete until the spinful hidden output and per-call index build are removed, generic eager storage becomes a real native CSR handle, materialization budgets are preflighted before cache mutation, and the frozen benchmark and QuSpin evidence are complete.

## Post-review remediation update (2026-08-04)

The remediation was implemented in the working tree and revalidated after the review. The active Phase 8.5 label is intentionally present only in `docs/vibe`; formal source, test, benchmark, and example artifacts use capability names. The original findings below remain historical observations; the table records their current disposition and the evidence used.

| Finding | Current disposition | Evidence and rationale |
| --- | --- | --- |
| R1 spinful hidden output and repeated index construction | Adopted | The native plan builds bounded combination/rank caches and compact validated descriptors once, then writes directly to caller output. The fast path accepts `apply_into(max_bytes=0)`, and the generic fallback remains available when eligibility or cache budgets fail. The focused spinful differential and the 2x4/4x3 release cases pass; the latest 4x3 `apply_into` median is about 203.0 ms on this machine. |
| R2 generic eager native CSR | Adopted | Eager execution retains one native destination-major CSR handle with checked indices, direct gather, and thresholded row parallelism. Python CSR/COO arrays are generated only for materialization or inspection and are not retained by `ChargeMvpPlan`; the regression checks the absence of those storage slots. The latest generic CSR A/B measured about 677.9 microseconds serial versus 449.0 microseconds parallel on the large graph, so the bounded row strategy is enabled only where it helped. |
| R3 cache and materialization budget ordering | Adopted | Dense, COO, CSR, and eager-plan paths preflight the missing cache plus target before publishing the cache; cached fixed plans still check call-level budgets. Failed dense and sparse requests leave the retained estimate unchanged and later larger-budget retries succeed. |
| R4 redundant allocating output copies | Adopted | Pauli, generic eager charge, and packed U1 wrappers now return the owning native array without a second complete copy. `apply_into` remains the single-output path, and ownership, overwrite, overlap, and concurrency regressions pass. |
| R5 U1 scratch budget | Adopted | Lazy U1 `apply_into` now receives and checks a conservative scratch estimate; fixed eager U1 `apply_into(max_bytes=0)` remains valid because it allocates no major scratch. The same caller-owned-buffer rule is enforced for generic eager CSR. The zero-budget lazy/eager regressions and U1 differential suite pass. |
| R6 structured eager representation | Not adopted after measurement | Repeated release A/B runs on a 4096-state four-boson workload fluctuated around 40.6–40.8 microseconds eager versus 41.1–42.2 microseconds lazy, a small difference within local benchmark noise. The proposed mixed-radix stride cache was therefore removed; eager and lazy deliberately retain the same compact representation and report the same strategy/bytes. This avoids keeping a cache without a demonstrated material gain. |
| R7 correctness and benchmark matrix | Adopted; matched 4x4 gate passed on this machine | The capability-named test and benchmark files cover cache retry, concurrent first materialization, strict buffers, sparse/dense target preflight, structured, generic aggregation, eager CSR, U1, first/steady apply, repeated buffers, and 2x4/4x3 spinful cases. The focused resource suite has 19 tests; the full Python release recording has 376 passing and 1 skipped optional case, and the Rust release recording completed. The matched 4x4 run with fixed BLAS/OpenMP thread settings measured TenCirPauli lazy `apply()` at 54.004 s and 5.363 GB peak RSS versus QuSpin `quantum_LinearOperator.dot` at 93.519 s and 11.585 GB peak RSS, with the same dimension and 2.650 GB input/output buffers. This is approximately 1.73x faster and 2.16x lower peak RSS on this machine; no repeated statistical series or cross-machine guarantee is claimed. |
| R8 duplicate Python mirrors | Adopted at the Python boundary | Charge lazy descriptors and U1 compatibility mirrors are no longer retained on normal native plans; eager charge public arrays are derived from the native handle on demand. The native generic descriptor batch is intentionally retained because the same immutable plan must support lazy generic fallback and later eager compilation; it is counted as plan-owned logical storage rather than copied into the Python facade. |

The current evidence shows no observed correctness regression or material performance regression in the measured representative cases: the full Python suite has 330 passing tests, the Rust workspace has 41 passing tests, and all 19 focused resource regressions pass. The positive measurements are the direct-output spinful path, the large destination-row CSR parallel path, and the matched 4x4 Hubbard run; the structured cache was explicitly rejected after its A/B was essentially neutral. The complete local release recordings are now present under ignored `.benchmarks/` (Python: 376 passed and 1 skipped optional case; Rust: all three Criterion suites completed). The 4x4 result is a single same-machine research measurement, so it supports the local positive direction and resource claim but is not a universal performance guarantee.
