# Phase 8.5 second-round implementation and acceptance review

Review date: 2026-08-04

Reviewed baseline: `2ca1d722115b2f1024bf3156aa3fefdfa06fe50e` (`perf: optimize packed U1 lazy and CSR execution`) plus the uncommitted Phase 8.5 first-review remediation in the working tree.

Scope: follow-up review of R1–R7 from `phase-8.5-review-2026-08-04.md`, focused on the production MVP routes, numerical correctness, high availability, end-to-end performance, resource contracts, and representative benchmark evidence. Over-defensive edge accounting and speculative redesign were intentionally excluded. This review did not modify production code, tests, benchmark sources, or recorded benchmark results.

## Verdict

The principal Phase 8.5 computational and performance repairs are real. The spinful-fermion plan now retains bounded combinatorial tables and compact descriptors once and writes directly into caller-owned output; generic eager charge execution retains a native destination-major CSR handle; public allocating wrappers no longer make the reviewed redundant full-output copies; and lazy packed-U1 `apply_into` now receives an execution scratch budget. Focused and full correctness gates pass, and the recorded 2x4, 4x3, large-CSR, and manual 4x4 measurements support the intended performance direction.

The first review is nevertheless not fully closed. Native generic eager compilation still holds the Python GIL for the complete transition build, low-budget materialization requests can still build and discard the complete eager graph before discovering that the requested output cannot fit, and the benchmark advertised as generic non-termwise aggregation actually dispatches to the packed U1 backend. These are scoped availability and evidence defects rather than a reason to redesign the MVP architecture.

Recommended status: **core implementation and primary performance route accepted; short high-availability and benchmark-evidence remediation open**. Do not describe the Phase 8.5 local performance gate as fully complete until SR1–SR3 below are closed and revalidated.

## Review and validation performed

- Inspected the first-round R1–R7 resolutions across the Rust core, PyO3 handles, Python facades, capability-named tests and benchmarks, local release records, implementation status, and manual Hubbard comparison scripts.
- Ran `conda run -p ./.conda python scripts/check.py --benchmark smoke`. Rust formatting, Clippy with warnings denied, Black, Ruff, strict mypy, `git diff --check`, release `maturin develop`, 41 Rust tests, 330 Python tests, 10 doctests, all three Rust Criterion smoke suites, and 297 non-large Python benchmark-smoke cases passed.
- Ran 133 focused charge, MVP-resource, structured, symmetry, and Hamiltonian tests; all passed.
- Ran additional spinful differentials across 2–6 sites, every nonzero filling including high-hole cases, complex hopping, diagonal terms, and supported generic quartic terms against a deliberately non-fast generic charge fixture; all outputs agreed exactly in the probes.
- Reproduced the generic eager GIL stall with an 823,680-transition graph: eager construction took approximately 218.5 ms and the independent Python observer thread experienced an approximately 219.1 ms scheduling gap.
- Reproduced delayed dense-budget failure on the same graph: `dense(max_bytes=100_000_000)` spent approximately 217 ms constructing the complete eager graph, then rejected the approximately 2.67 GB combined request and correctly left the facade cache empty. A subsequent eager-plan request repeated approximately 205 ms of construction.
- Inspected the saved full Python release record with 376 benchmark cases. The case named `test_generic_charge_aggregation_steady_apply` recorded `strategy="u1_lazy"`, confirming that it did not exercise generic charge aggregation.
- The working tree was already dirty and the new resource test, benchmark, and first-review report remained untracked during this review. No existing user changes were modified.

## First-review closure matrix

| Finding | Result | Second-round assessment |
| --- | --- | --- |
| R1 spinful hidden output and repeated index construction | PASS | The reusable native plan owns the bounded index and compact descriptors and writes directly into caller output. The reviewed main path has no state-sized hidden output, repeated table construction, or reproduced numerical defect. |
| R2 generic eager native CSR | PASS WITH AVAILABILITY GAP | The retained representation and steady apply path are correct native destination-major CSR. Construction still holds the GIL for the full transition compilation and conversion, recorded below as SR1. |
| R3 cache and materialization budget ordering | PARTIAL | Failed requests no longer publish a cache and cached-plan call budgets are checked, but target-budget rejection still occurs after an uncached eager graph has been constructed. SR2 retains the unresolved fail-fast requirement. |
| R4 redundant allocating output copies | PASS | Reviewed Pauli, generic eager charge, and packed-U1 wrappers return the owning native array without the former second full-output copy. |
| R5 U1 scratch budget | PASS FOR THE MAIN CONTRACT | Lazy native U1 `apply_into` receives a checked scratch budget; eager caller-owned-buffer execution remains valid with a zero scratch budget. No main-route resource regression was reproduced. |
| R6 structured eager representation | ACCEPTABLE PERFORMANCE DECISION; CONTRACT NOTE OPEN | The measured bounded cache had no material benefit and was correctly removed. Eager and lazy still expose different `storage` labels while retaining identical data and strategy; resolve this by an explicit contract clarification rather than by adding a meaningless cache. |
| R7 correctness and benchmark matrix | PARTIAL | The resource and strict-buffer regressions are materially broader and the 2x4/4x3/large-CSR cases are present, but the required generic non-termwise aggregation performance case is misrouted to U1. |

## Findings

### SR1 — MAJOR: generic eager charge compilation holds the Python GIL

Location: `crates/tencirpauli-native/src/charge_sector.rs:547-564`; contract: the repository GIL rule and sections 6, 9, and 16 of `phase-8.5-spec.md`.

`NativeChargeMvpPlan::compile_eager` calls `compile_charge_transitions_from_plan` and converts the returned COO graph into the native CSR handle without `Python::allow_threads`. Steady `apply` and `apply_into` correctly release the GIL, but the potentially much longer first eager materialization does not.

On the review machine, an 823,680-transition graph took approximately 218.5 ms to compile while an otherwise runnable Python observer thread stopped for approximately 219.1 ms. The stall scales with graph construction and can therefore freeze unrelated Python orchestration for seconds or longer on larger valid eager requests. This is a high-availability issue even though the resulting graph and numerical execution are correct.

Required resolution: accept a `Python<'_>` token in `compile_eager`, move transition compilation and the pure-Rust CSR validation/conversion under `allow_threads`, and keep Python-array creation outside the released section only where the Python API requires it. Re-run the concurrent first-materialization regression and retain a construction benchmark; do not add a flaky wall-time CI assertion.

### SR2 — MAJOR: uncached materialization still checks the target budget after building the complete eager graph

Locations: `python/tencirpauli/charge.py:1257-1332` and `python/tencirpauli/symmetry.py:380-429`; contract: sections 4 and 11 of `phase-8.5-spec.md`; first-review R3.

The remediation correctly prevents a failed request from publishing `_eager_plan`, but `ChargeRestrictedOperator._ensure_eager_for_target` calls `compile_eager` before computing `target_bytes`. The U1 sibling similarly constructs the eager restriction before checking the combined retained-plus-target size. For dense output the dominant target size is known entirely from `dimension`, so this ordering is unnecessary.

The review probe requested a 12,870-dimensional dense result with a 100 MB budget. The call spent approximately 217 ms building an 823,680-transition eager graph, then raised because the combined request was approximately 2.67 GB and discarded the graph. The cache remained empty as intended, but a later eager request repeated approximately 205 ms of work. On larger graphs the failed call can consume substantial CPU and transient memory before reporting a budget error.

Required resolution: preflight the exact dense output size before eager construction and reserve it from the construction budget. For CSR/COO, preflight the cheaply known dimension-dependent minimum before construction and check the exact transition-dependent combined size at the earliest point where the transition count is known, before allocating any avoidable materialization copy or publishing the cache. Apply the same ordering to generic charge and packed U1 facades.

### SR3 — MEDIUM: the required generic aggregation benchmark dispatches to packed U1

Location: `benchmarks/python/test_native_mvp_resources_benchmark.py:285-304`; contract: sections 10, 13, and 16 of `phase-8.5-spec.md`; first-review R7.

The benchmark builds a canonical qubit-number charge sector with `XX + YY`. Unified restriction correctly recognizes this as packed U1, and the saved release record reports `storage="lazy"` and `strategy="u1_lazy"`. The result is a valid U1 benchmark but does not measure the generic non-termwise-conserving destination-aggregation route required by the frozen matrix.

Required resolution: use a charge/layout fixture that cannot dispatch to canonical U1, such as a mixed-domain spectator or a genuinely noncanonical additive charge, while retaining cancellation-before-leakage semantics. Assert the expected generic plan type or `strategy == "term_direct"` so future dispatch changes cannot silently invalidate the benchmark. Re-record only the affected release evidence plus any comparison needed to show no representative regression.

## Non-blocking contract disposition for structured storage

The decision not to retain a neutral structured stride cache is sound and aligns with the repository's profile-first performance policy. Reintroducing storage merely to distinguish two labels would be over-design. However, `phase-8.5-spec.md` currently says distinct storage metadata should be reported only when retained data differ, while structured construction passes the caller's `storage="eager"` label onto the same native plan, strategy, and retained-byte estimate used by lazy construction.

Close this administratively with an explicit owner-approved specification clarification, or normalize/reject the unsupported distinction in the public API. It is not a computational or performance blocker and should not delay SR1–SR3.

## Recommended closure order

1. Release the GIL around generic eager transition compilation and pure-Rust CSR conversion.
2. Move exact dense target preflight ahead of eager construction and align the U1 sibling; preserve retryable cache publication semantics.
3. Replace the misrouted generic aggregation benchmark, assert its strategy, and record the corrected case.
4. Update the Phase 8.5 status and first-review post-remediation table so they distinguish accepted core performance from the remaining availability/evidence work.
5. Resolve the structured storage-label wording by owner decision without introducing an unmeasured cache.

## Acceptance recommendation

The spinful Hubbard and steady native CSR paths are suitable for continued scientific use and performance work. No reproduced algebraic, phase, ordering, or main-route output-allocation defect remains from R1, R2, R4, or R5. Phase 8.5 should remain a short remediation checkpoint rather than return to broad implementation work.

Closure requires SR1–SR3, the full local quality gate, the corrected focused benchmark record, and an updated status statement. The existing single-machine 4x4 TenCirPauli-versus-QuSpin result remains useful positive evidence but is not changed into a cross-machine or CI performance guarantee.

No production source, test, benchmark source, or benchmark result was modified by this review. Only this review report and its `docs/vibe/README.md` index entry were added as archival documentation.
