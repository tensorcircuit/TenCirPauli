# TenCirPauli Deep Audit — Round 4 (2026-08-06)

> **Scope.** An independent fourth-pass scan of the **current working tree** (commit `32bc7d8`, post-Phase 10), run after the Round-3 archive (`audit-report-2026-08-05-round3.md`). This pass covers the three feature phases that landed *after* Round 3 and were never independently audited as audit dimensions: **Phase 9** (Rust-native data plane & Python thinning), **Phase 9.5** (circuit differentiation boundary & JAX), and **Phase 10** (chemistry & SciPy interop) — together ~16k insertions across 151 files. It also re-verifies every Round-3 finding and the carried-forward R2-4 item against the current tree to produce an accurate open ledger.
>
> **Method.** Six finder dimensions were fanned out in parallel (Phase-10 chemistry correctness, Phase-9 wrapper/ABI/GIL compliance, Phase-9.5 JAX/differentiation, Rust-core correctness/perf, test-coverage gaps, and Round-3 ledger re-verification). Each finder re-read its cited `file:lines` in the current tree and checked the R1/R2/R3 + Phase-9/9.5-review exclusion ledger. I independently re-verified the load-bearing correctness claims (R2-4 arithmetic, ERI convention, GIL releases, scratch-reuse status, fused-commutator signs) before writing this report; verification notes are inline.
>
> **No code was changed.** This is a report only.
>
> **Working-tree re-verification (post-report).** After the initial report, uncommitted working-tree changes landed a Phase-10 chemistry refactor: Hermitian/finite validation moved from Python into a Rust-side `validate_hermitian_pair` (GIL-released, `rtol=1e-10`/`atol=1e-12`), the float→complex copy was removed (native FFI now accepts float64 *or* complex128 directly), and `BackendMVPPlan.to_scipy_linear_operator` was deleted (with a test asserting its absence). PySCF/UHF tests gained an independent `_independent_integral_reference` dense oracle with per-coefficient assertions, mapped-Pauli agreement, and dual-ordering coverage. §3.4 records which Round-4 findings these changes close or leave open. All Rust-core (R4-N1, R3-*), dead-FFI (R4-D1..D4), SPPS, propagation, and naming findings were re-verified unchanged against the current tree.

## 1. Executive Summary

**Headline.** Phase 9 closed the bulk of Round-3's wrapper/FFI debt *for real* — not a single false-closure claim was found among the 26 Round-3 items re-verified. The most consequential result is that **R2-4 (the projected-batch overflow that blocked Phase-8.5 closure across two prior rounds) is genuinely fixed**, with correct `max_weight`-projected semantics and a dedicated regression test. The one notable **new correctness regression** is in SPPS: Phase-9.5's removal of "over-defense" stripped the non-finite overflow guards from the SPPS path-product accumulator, so long circuits silently produce `NaN` estimators instead of raising (R4-N1). Everything else is low-severity test-coverage gap, dead/superseded-FFI cleanup residue, or already-known carried-forward perf items.

**Counts.**
- Round-3 items re-verified: 26 + R2-4. **Closed: 21. Still open: 5.** No false closures.
- New Round-4 findings: 13 after withdrawing R4-N2 (1 low correctness, 4 dead-FFI, 7 test-gap, 1 naming-residue). One originally-filed item (R4-N2) was reclassified as over-defense that Phase 9 correctly removed.

**Real-user-scenario vs over-defense.** Each finding is tagged `[REAL]` (representative workload), `[NARROW]` (real but uncommon), or `[OVER-DEFENSE]` (guards a magnitude/hypothetical future no real workload produces — not recommended for re-implementation). The owner's stated preference: avoid layered intermediate checks, input-scanning, and "test that X never happens" patterns that hurt readability for no representative-workload benefit. Over-defense items are explicitly marked **do not re-implement**.

**Themes new in this pass.**
1. **Phase 9's thinning is real and verified.** Flat-NumPy read-back ABI (`NumpyPauliPackedOutput` etc.), handle-accepting terminals (`pauli_dense_handle`, `compile_mvp_*_handle`), fused native commutators, GIL release on QWC reconstruction and charge compilation, `_native_data` deletion, dead Pauli FFI removal, `ParameterExpr` removal — all confirmed in the current tree by re-reading the code, not by trusting the closure docs.
2. **Phase 9's over-defense removal was mostly correct.** The `checked_scale`/`checked_add`/`is_finite`-per-term removal on the propagation path (trig-bounded → finite×finite stays finite) and the operator/majorana aggregation finiteness scan (R4-N2, only triggers near `f64::MAX`) were correctly removed as over-defense. The one exception is SPPS (R4-N1): its path-product is *unbounded*, so a single exit check is warranted — but only an exit check, not the layered per-branch checks that were (correctly) removed.
3. **Dead/superseded FFI residue from incomplete slice-7 cleanup.** Phase 9's slice 7 ("remove legacy storage, dead private FFI") left behind a handful of array-input FFI entry points whose handle variants now make them unreachable: `NativeChargeSectorPlan.compile_mvp` (no GIL release, nested `Vec<Vec<...>>` input), four non-handle `expectation`/`value_and_grad` methods on `NativeU1CircuitPlan`/`NativeU1FinalState`, and two non-flat `pauli_canonicalize_batch*` variants. None are called by production, but they remain registered and typed in `_native.pyi`.
4. **Phase-10 chemistry is numerically faithful.** The notoriously bug-prone ERI chemist↔spin-orbital transpose was verified end-to-end against PySCF's `ao2mo` index convention and numerically (Hermitian-averaging double-count cancelled exactly by the 0.5 prefactor; ratio = 1.0). The remaining Phase-10 findings are test-coverage gaps (no Hermiticity/particle-number assertion, no independent dense-matrix oracle, UHF missing mapped-Pauli agreement), not math errors.

**Prioritized action list**
1. **Correctness (narrow, minimal fix):** add one exit finiteness check in SPPS `combine_fixed` (R4-N1, `[NARROW]`). ~3 lines, no layered checks. Do **not** restore the removed per-branch checks.
2. **Cheap real-coverage tests:** signed-zero hash regression test (R4-N3 `[REAL, trivial]`); SciPy "compiles once" counter (R4-TG1 `[REAL]`, only the compile-count part — the rest of R4-TG1 and the PySCF oracle R4-TG3/TG4 are now closed in the working tree).
3. **Dead-FFI / redundant-field cleanup (zero complexity, no defense):** remove `NativeChargeSectorPlan.compile_mvp` (R4-D1), the four non-handle U1 `expectation`/`value_and_grad` (R4-D2), consolidate the two non-flat `pauli_canonicalize_batch*` variants (R4-D3), and drop the redundant `reconstruction_masks` field + `masks` return from `pauli_qwc_group_handle` / `QWCGroupingResult` (R4-D4 — API not yet frozen, field provably unused by `reconstruct`, remove outright while it is still cheap to do).
4. **Explicitly do NOT do (over-defense):** R4-N2 (already correctly removed — do not re-add operator aggregation finiteness scans); the no-materialization and single-adapter monkeypatch tests in R4-TG1; the residency monkeypatch + `isinstance(np.ndarray)` tests in R4-TG6.
5. **Carried-forward perf (low priority, documented):** R3-11/12 scratch reuse (deferred per Phase-9.5 §8 with recorded decision), R3-16 non-conserved charge Vec clone, R3-14 PackedKey clone (narrow), R3-23 SPPS adaptive per-term parallelism.

---

## 2. Round-3 / R2-4 Re-Verification Ledger

Re-verified against the current tree (`git diff ab311ad..HEAD`), re-reading cited lines. **No false closures found.**

| ID | Title | Status | Current evidence |
|---|---|---|---|
| **R2-4** | Projected batch estimate overflow | **CLOSED** | `propagation.rs:916-1052` `projected_pauli_word_bound` truncates `propagated_terms` by `sum_{w≤k} C(n,w)·3^w` in `u128` checked arithmetic. Repro (64q/64rot/`max_weight=1`/`max_bytes=None`) confirmed constructible by two independent agents; `max_weight=None` still correctly raises on genuine unbounded growth. Test `test_wide_weight_projected_batch_uses_projected_storage_bound` exists. |
| R3-1 | restrict_charge 3 Python passes | CLOSED | `charge.py:923-992,1430-1447` — handle-compile dispatch, single `termwise_conserves_charge` loop. |
| R3-2 | Structured `is_hermitian` native path | CLOSED | `structured.py:1280-1291` → `handle.is_hermitian(tol)` (GIL-released) on all structured handles. |
| R3-3 | Pauli `__eq__`/`__hash__` materialize | CLOSED | `pauli.py:553-568` → `content_eq`/`content_hash`, no term materialization. |
| R3-4 | `_hybrid_arrays` terms-only rebuild | CLOSED | `structured.py:2063-2144` — unconditional handle materialize, terms-only path gone. |
| R3-5 | Dead `_native_data` + 7 helpers | CLOSED | `structured.py:952-977` — field, ctor param, and all 7 helpers removed; `__slots__` is `space/_terms/_native_handle/_locked`. |
| R3-6 | Commutator 4 crossings vs native | CLOSED | `structured.py:1158-1270`, `majorana.py:398-428` dispatch to `handle.commutator(...)`. `_native.pyi` declares for all 5 families. |
| R3-7 | SPPS adaptive exact-reference test | CLOSED | `test_spps.py:58-77` binds value/grad to `exact_value_and_gradient`. |
| R3-8 | MajoranaOperator.multiply test | CLOSED | `test_majorana_mapping.py:161-177` compares dense matrix product. |
| R3-9 | PropagationEngine.profile fields | CLOSED | `test_propagation.py:234-246` validates all 7 fields. |
| **R3-10** | U1 plan-level methods untested | **PARTIAL** | Public facade covers all four; plan-level API is now private `_U1CircuitPlan`. `probability`/`expectation` have exact-value tests; `state_full`/`probability_full` only shape-asserted (`test_u1_circuit.py:27`, `test_public_api.py:220`). Residual test gap re-filed as R4-TG5. |
| **R3-11** | Generic charge per-apply scratch | **OPEN (deferred)** | `charge.rs:1538-1543` — 5 buffers per `apply_into`. Phase-9.5 §8 benchmarked (~4.1% steady delta, below 10% threshold) and **recorded** the `defer_scratch_reuse` decision (`phase-9.5-closure-evidence` §R3). Documented deferral, not oversight. |
| **R3-12** | U1-lazy/structured MVP scratch | **OPEN (deferred)** | `sector.rs:639-642`, `structured.rs:2411-2412` — per-call allocs. Same recorded deferral. |
| R3-13 | reverse_frame hashmap rebuild | CLOSED | `propagation.rs:338-339,1443-1447` — `&mut FxHashMap` threaded, `clear()`+`reserve()` reuse. |
| **R3-14** | PackedKey clone per transition | **OPEN (narrow)** | `propagation.rs:1449` etc. — `term.key.clone()` per insert. `Inline` (≤128q) is a stack copy; only `Wide` (>128q) heap-allocates. Unchanged, narrow benefit as originally dispositioned. |
| R3-15 | apply_fermions O(k·M) parity | CLOSED | `charge.rs:353-433` — O(1) `count_ones` on packed `[u64;2]` for ≤128 modes; incremental maintenance verified correct. O(k) fallback only >128 modes. |
| **R3-16** | Non-conserved charge Vec clone per term | **OPEN** | `charge.rs:1602-1604` — `destinations.entry(destination.clone())` per term. No borrowed-key hashmap. Unchanged. |
| R3-17 | Pauli handle materialize nested → rebuild | CLOSED | `operator.rs:102-129`, `hamiltonian.rs:37-80` — `pauli_*_handle` variants; flat `NumpyPauliPackedOutput`. |
| R3-18 | pauli_canonicalize_array nested return | SUPERSEDED | Function removed; construction via `pauli_operator_native_array` / `pauli_canonicalize_batch_numpy` (flat). |
| R3-19 | Structured handles nested, no GIL | CLOSED | `structured.rs:325-341,1348-1370`, `majorana.rs:206-228` — flat `PyArray1` + `allow_threads`. |
| R3-20 | Charge csr/coo split_complex | CLOSED | `charge_sector.rs:623-653` — `PyArray1::from_vec(self.values.clone())` zero-copy complex. |
| R3-21 | 4 dead Pauli PyO3 FFI | CLOSED | `lib.rs:100-105`, `hamiltonian.rs:37-80` — only `_handle` variants registered; `test_pauli_operator.py:209` asserts absence. |
| R3-22 | Propagation/SPPS symbolic Jacobian | CLOSED | `Parameter`/`ParameterExpr`/`_evaluate_angle` removed (grep zero hits); native `GateTape` with slots. |
| **R3-23** | SPPS adaptive serializes across terms | **OPEN** | `spps.rs:478-520` — sequential `for term_index in 0..term_count` with shared `&mut scratch`; fixed path uses `par_iter_mut`. Adaptive-SPPS JAX explicitly deferred (Phase-9.5 §3), so this was not targeted. |
| R3-24 | SPPS adaptive divergent-budget test | CLOSED | `test_spps.py:80-111` — unequal coeffs, asserts non-uniform budget + exact ref. |
| R3-25 | Propagation double expectation compute | CLOSED | `propagation.rs:317-324` — fused single pass, iterator order preserved. |

**Net:** 21 closed, 5 still open (R3-11/12 documented-deferred, R3-14 narrow, R3-16 low, R3-23 medium), R3-10 partial. The 5 open items are all perf (4 scratch/clone + 1 SPPS parallelism) that Phase 9/9.5 did not target; they remain valid at the cited lines.

---

## 3. New Findings

> Categories: **correctness** · **test-gap** · **dead** (dead/superseded FFI) · **perf**. Severities are post-adversarial-verification.

### 3.1 Correctness

> **Classification key (real-user-scenario vs over-defense):** each finding below is tagged `[REAL]` (a failure a real user can hit on a representative workload), `[NARROW]` (real but only on an uncommon/edge workload), or `[OVER-DEFENSE]` (guards against magnitudes or hypothetical future changes no real workload produces — Phase 9 was right to remove these; they are recorded here so future rounds do not re-file them). Over-defense items are explicitly **not** recommended for re-implementation; if a fix would add layered/intermediate checks, do not do it.

#### R4-N1 — SPPS `run_samples` lost its *final* non-finite check; long circuits silently produce `NaN`/`inf` estimators — **correctness · low · [NARROW]**

**File:** `crates/tencir-pauli-core/src/spps.rs:896-936` (prefix/suffix/nonzero_product accumulation); `spps.rs:961-982` (`combine_fixed` — no finite check on output).

**Summary.** Phase-9.5's over-defense removal stripped every `NonFiniteCoefficient` guard from `run_samples`: the `!prefix[..].is_finite()`, `!suffix[..].is_finite()`, `!nonzero_product.is_finite()`, `!sample_value.is_finite()`, and `!stats.sum.is_finite() || !stats.sum_squared.is_finite()` checks are all gone. `combine_fixed` does not re-check. So an overflow in the path-product accumulation silently propagates into `stats.sum` as `inf`/`NaN`, and the estimator returns `Ok((NaN, NaN-gradient, NaN-SE))` instead of `Err`.

**Failure scenario.** The SPPS path-product is a product of L ratios, each bounded by `O(1/smoothing)`. Worst-case ratio ≈ `(2+2s)/s ≈ 2/s`. The product overflows f64 (max ≈ 1.8e308) when `L·log10(2/s) > 308`. For default `smoothing=0.01`: threshold `L ≈ 134` gates **in the worst case where every gate's ratio is near the `2/s` ceiling** (i.e. every angle near-degenerate). For `smoothing=0.001` (valid — `smoothing > 0` is the only guard at `spps.rs:185`): `L ≈ 93` worst-case. A long parameterized molecular-VQE SPPS circuit hitting a near-degenerate angle on one optimizer iteration silently returns `NaN` expectation. The old behavior raised `NonFiniteCoefficient`; the new behavior is silent and non-deterministic.

**Why this is `[NARROW]`, not `[REAL]`.** The threshold requires *every* gate ratio near its ceiling simultaneously — a realistic circuit has a mix of >1 and <1 ratios, so the product grows far slower than the worst case. It bites only on long circuits × near-degenerate angles × aggressive smoothing, which is an uncommon combination. It is filed because the failure is *silent* (NaN returned as `Ok`), not because it is frequent.

**Why this is NOT over-defense to fix.** Phase-9 §4.1 authorizes removing guards for "extreme repeated scaling," and the *per-branch intermediate* checks that were removed (prefix/suffix/nonzero_product `is_finite` at every step) **were** over-defense — removing them was correct. What remains missing is a **single exit check** in `combine_fixed`, which is a boundary guard, not layered intermediate scanning.

**Quantified benefit of the (removed) per-branch checks.** ~6 `is_finite()` branches (2 comparisons each) per sample path. For a 1000-sample/50-gate run: ~300k branch checks — negligible vs gate-application cost. The per-branch removal buys ~nothing and was correct.

**Dominant cost of the missing exit check.** Silent `NaN` estimator output for the narrow long-circuit case; non-deterministic, hard to diagnose.

**Fix direction (minimal, non-layered).** Add **one** `if !stats.sum.is_finite() || !stats.sum_squared.is_finite() { return Err(NonFiniteCoefficient) }` at the end of `run_samples` (or top of `combine_fixed`). Do **not** restore the per-branch prefix/suffix/nonzero_product checks — those were correctly removed as over-defense. ~3 lines, no complexity. If even this single check is judged not worth it for the narrow trigger, it is acceptable to leave it and document that SPPS path-product overflow is silent by Phase-9 §4.1 design.

**Verification.** Re-read `spps.rs:855-936` and `961-982`; confirmed no `is_finite` call remains in the path-product or accumulator. Confirmed via `git diff ab311ad..HEAD -- crates/tencir-pauli-core/src/spps.rs` that the guards were removed in the Phase-9.5 window. The `q <= 0.0` / `probability <= 0.0` removals (old lines ~871/882) ARE safe — unreachable given `smoothing > 0`. Not in any exclusion ledger.

#### R4-N2 — `operator.rs`/`majorana.rs` aggregation no longer rejects `inf` coefficients — **over-defense (correctly removed by Phase 9; NOT a finding)**

**File:** `crates/tencir-pauli-core/src/operator.rs:75-78,147-152,246-251,327-333`; `majorana.rs:177-180`; tests at `tests.rs:180-186,206-210` assert `is_infinite()`.

**Status: withdrawn as a finding.** This was originally filed as a low-severity correctness item. On re-review it is the textbook over-defense the owner wants removed: the only trigger is `scale(f64::MAX)` (or a Taylor step producing a coefficient near 1.8e308), which no real physics workload produces. Phase-9 §4.1 explicitly accepts "overflow after extreme repeated scaling," the tests were deliberately rewritten to assert `is_infinite()`, and public-input validation still rejects non-finite *inputs* at the boundary (`operator.rs:52,133,171,270`). The intermediate-aggregation finiteness scan that was removed added one `is_finite()` per term per op purely to catch a magnitude no representative workload reaches.

**Action: none. Do not re-implement.** Recorded here only so a future round does not re-file it as a regression. The propagation-side removal of `checked_scale`/`checked_add` is similarly correct (trig-bounded → finite×finite stays finite); the only path where the removal is questionable is SPPS (R4-N1), which has an *unbounded* product.

### 3.2 Dead / superseded FFI (Phase-9 slice-7 residue)

#### R4-D1 — Dead `NativeChargeSectorPlan.compile_mvp` PyO3 method violates §11 GIL contract and §10 — **dead · medium · [REAL: dead-code cleanup, zero complexity]**

**File:** `crates/tencirpauli-native/src/charge_sector.rs:252-302`; stub at `_native.pyi:249-270`.

**Summary.** The array-input `compile_mvp` method accepts nested `Vec<Vec<u32>>` / `Vec<Vec<(u32,u32,u32)>>` / `Vec<Vec<u8>>` structural arrays, takes **no** `py: Python<'_>` parameter, and does **not** wrap its work in `allow_threads`. It calls `charge_terms_from_inputs` (clones every nested `Vec` per term) then `build_native_charge_mvp_plan` (O(terms × basis-transitions) layout + fast-fermion work) — all under the GIL. The handle variants `compile_mvp_pauli_handle` (`:317-348`) and `compile_mvp_hybrid_handle` (`:364-396`) are the production path and DO release the GIL.

**Failure scenario.** Any caller invoking `sector._native_plan.compile_mvp(...)` with a non-trivial operator runs full compilation holding the GIL — exactly the §11 violation the closure docs claim fixed. No production caller exists (`grep` confirms only `_handle` variants are called from `python/` and `tests/`), so today it is dead; the risk is a future caller routing through the slower, GIL-holding variant believing it is intended.

**Benefit/cost.** Removes ~50 lines Rust + 1 stub signature. Zero callers. The charge analogue of R3-21 (which closed the Pauli side); this charge-side variant was not previously filed.

**Verification.** Re-read `charge_sector.rs:252-302` (no `py` param, no `allow_threads`); `grep -rn "\.compile_mvp(" python/ tests/` returns only `_handle` matches. Not in exclusion ledger.

#### R4-D2 — Dead non-handle `expectation`/`value_and_grad` on `NativeU1CircuitPlan`/`NativeU1FinalState` rebuild operators from nested arrays — **dead · medium · [REAL: dead-code cleanup, zero complexity]**

**File:** `crates/tencirpauli-native/src/u1_circuit.rs:152-210` (`NativeU1CircuitPlan`) and `:285-327` (`NativeU1FinalState`); stubs at `_native.pyi:504-531,538-549`.

**Summary.** Four `#[pymethods]` accept `structures: Vec<Vec<u8>>` + split `coefficients_re`/`coefficients_im`, then inside `allow_threads` call `build_canonical_operator(...)` to *reparse* a `PauliOperator` from nested Python lists, then run the plan. The `_handle` variants (`expectation_handle` `:212-232`, `value_and_grad_handle` `:234-254`, and the `NativeU1FinalState` versions `:329-358`) exist alongside and consume `&NativePauliOperatorHandle` directly. Per Phase-9 §6.3: "A non-handle array entry point may remain only for a genuine external array input boundary, not as a workaround for a native handle that lacks a direct terminal method." The handle method exists.

**Failure scenario.** A user unnecessarily calls `.terms`/`to_dict()`, packs into Python lists, passes to `plan.expectation(structures, re, im, params)` — the operator is reparsed in Rust from nested lists, exactly the §3.1-forbidden round trip. No production caller (`grep` shows U1 production uses only `_handle` variants).

**Benefit/cost.** Removes ~80 lines Rust + 4 stub signatures. Zero callers.

**Verification.** Re-read `u1_circuit.rs:152-358`; `grep` confirms no production non-handle caller. Not in exclusion ledger.

#### R4-D3 — `pauli_canonicalize_batch` / `pauli_canonicalize_batch_array` return nested `Vec<Vec<u8>>` + split re/im, superseded by the `_numpy` flat variant — **dead · low-medium · [REAL: consolidation, modest refactor]**

**File:** `crates/tencirpauli-native/src/operator.rs:259-344`; wrappers `pauli.py:410-471` (`canonicalize_batch`, `canonicalize_code_arrays`); stubs `_native.pyi:682-704`.

**Summary.** These two module-level functions return `(Vec<Vec<u8>>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<u8>)` — nested structures + split re/im. The Python wrapper immediately re-zips re/im into `complex` and walks the nested list. The sibling `pauli_canonicalize_batch_numpy` (`:346-400`) returns the same data as flat `PyArray1<u8>` codes + flat `PyArray1<Complex64>`. Phase-9 §3.1: "the implementation chooses one canonical numeric representation per read-back contract and deletes superseded parallel variants" and "A single complex value array is preferred over separate real and imaginary Python sequences."

**Failure scenario.** A 10000-term `canonicalize_batch` call: per-term heap allocation, PyO3 per-inner-list marshal, Python re-zip loop — all avoided by the flat variant.

**Benefit/cost.** Consolidating (route wrappers through the numpy variant, flattening only at the public-result boundary) eliminates per-term Python object creation. The public `CanonicalizationResult` API returns tuples-of-tuples, so the wrappers must convert at the terminal boundary — but that is one explicit materialization, not an internal round trip.

**Verification.** Re-read `operator.rs:259-400`, `pauli.py:410-508`. Not in exclusion ledger; this §3.1 cleanup was not flagged by the Phase-9 reviews.

#### R4-D4 — `pauli_qwc_group_handle` returns `masks` as a redundant parallel `Vec<Vec<Vec<u64>>>` duplicating the native handle's storage — **dead · low-medium · [REAL] · TO DO (API not yet frozen; remove the redundant field)**

**File:** `crates/tencirpauli-native/src/grouping.rs:12-18,96-151` (returns `groups, bases, masks, handle`); `grouping.py:211-238` (unpacks `masks_raw` → `reconstruction_masks`).

**Summary.** The handle internally stores `masks: Vec<Vec<Vec<u64>>>` (`:23`), cloned from the same source (`:146`). The function additionally marshals `masks` as a 3-deep nested Python list, which the wrapper iterates into `reconstruction_masks: Tuple[Tuple[int,...],...]`. `QWCGroupingResult.reconstruct` (`grouping.py:84-124`) uses **only** `self._native_handle.reconstruct(...)` — never `reconstruction_masks`. `tests/test_grouping.py:104-109` proves this by corrupting `reconstruction_masks` and confirming `reconstruct` still returns the correct result. Per Phase-9 §10: "Redundant … parallel Python arrays … are deleted once their native path is complete."

**Failure scenario.** A 1000-term/50-group QWC grouping clones `masks` into the handle AND marshals it as a nested Python list AND the wrapper builds a parallel tuple-of-tuples — three representations of the same bits, none consumed by execution.

**Benefit/cost.** Stop returning `masks` from `pauli_qwc_group_handle` and remove the `reconstruction_masks` field from the `QWCGroupingResult` dataclass. Eliminates one `Vec<Vec<Vec<u64>>>` clone, one PyO3 nested-list marshal, and one Python tuple-of-tuples-of-tuples build per grouping call — three representations of the same bits, none consumed by `reconstruct`. The `test_grouping.py:106` corruption test (which exists only to prove `reconstruction_masks` is unused) should be deleted alongside the field.

**Disposition: TO DO.** The public API is not yet frozen (pre-0.3.0; `BackendMVPPlan.to_scipy_linear_operator` was just removed in the working tree for the same reason — eliminating a redundant/unused surface while the API can still change). Removing a field that is provably unused by any execution path and exists only as a redundant parallel copy of handle-owned storage is exactly the kind of cleanup to do now rather than preserve as compatibility debt. Drop the field outright; do not add a lazy-getter shim (that would retain the field's complexity for no caller).

### 3.3 Test-coverage gaps

#### R4-TG1 — SciPy adapter: "compiles once" unverified; (no-materialization / single-adapter-delegation are over-defense, demoted) — **test-gap · low · [REAL: compile-count only; the other two are over-defense] · partially closed in working tree**

**Files:** `tests/test_scipy_linear_operator.py`; untested contracts at `python/tencirpauli/integrations/scipy.py:13-49` and the (now 4) `to_scipy_linear_operator` methods.

**Summary.** Of the three Phase-10 §6.3 gates originally grouped here, only one is a real-user-scenario test:
- **"Compiles once" [REAL, still open]** — `test_pauli_convenience_linear_operator_compiles_and_reuses_native_plan` (name claims it) never instruments `native_mvp_plan()` with a counter. Removing the compile cache would pass (matvec correct) while regressing perf by orders of magnitude. ~3 lines to add a counter. Still open.
- **"No materialization" [OVER-DEFENSE, demoted]** — monkeypatching `plan.dense()`/`coo()`/`csr()` to raise to prove the wrapper never calls them. This tests "something does not happen," guarding against a hypothetical future maintainer adding a small-plan fast path. Skip.
- **"Single-adapter delegation" [OVER-DEFENSE, partially moot]** — the working tree removed `BackendMVPPlan.to_scipy_linear_operator` (the one plan type that lacked a native-MVP `apply` contract and would have been the natural candidate for a divergent override) and added `test_backend_plan_does_not_advertise_native_scipy_interop` asserting its absence. The remaining plan types all delegate to the shared helper by construction. Skip.

**Action:** add only the compile-count instrumentation (~3 lines) if desired; it is the only `[REAL]` residual. Do not add the no-materialization or single-adapter monkeypatch tests — they are the "test that X never happens" over-defense pattern.

**Files:** `tests/test_scipy_linear_operator.py` (entire file, 96 lines); untested contracts at `python/tencirpauli/integrations/scipy.py:13-49` and the 5 `to_scipy_linear_operator` methods (`pauli.py:991-998`, `charge.py:856-862,1042-1048`, `hamiltonian.py:302-308,521-527`).

**Summary.** Phase-10 §6.3 requires three explicit gates the tests do not enforce:
1. **No materialization** — no test monkeypatches `plan.dense()`/`coo()`/`csr()` to raise, proving wrapper construction and `matvec` never touch them. A future "small-plan fast path" that pre-materializes CSR would pass every existing test (matvec still correct).
2. **Single-adapter delegation** — each plan type is tested independently, but no test asserts all 5 route through the one `integrations.scipy.to_scipy_linear_operator` helper. A parallel inline matvec on one plan type would pass.
3. **"Compiles once"** — `test_pauli_convenience_linear_operator_compiles_and_reuses_native_plan` (name claims it) never instruments `native_mvp_plan()` with a counter. Removing the compile cache would pass (matvec correct) while regressing perf by orders of magnitude.

**Failure scenarios.** (1) A maintainer adds a `LinearOperator` subclass that pre-materializes for "small" plans — matrix-free guarantee silently lost. (2) A `ChargeLazyMvpPlan` override builds a dense matrix internally — tests pass. (3) `native_mvp_plan()` caching removed — every `linear @ vector` recompiles.

**Benefit.** ~10 lines of monkeypatch test closes three explicit spec gates. Genuinely new.

**Verification.** Re-read `test_scipy_linear_operator.py:26-96` and `scipy.py:13-49`; confirmed no monkeypatch/instrumentation of materializers, the shared helper, or compile count. Not in exclusion ledger.

#### R4-TG2 — SciPy adapter: random `(dimension,1)` matvec and 2D invalid-shape branches untested — **test-gap · low · [REAL but trivial]**

**File:** `tests/test_scipy_linear_operator.py:42-43` (only `np.ones(3)` 1D-wrong-length tested); untested branches at `scipy.py:24-35`.

**Summary.** The `values.ndim == 2` + `shape != (dimension,1)` branch, the `ndim != 1` (3D+) branch, and the combination of *random* + `(dimension,1)` form are untested. Only a deterministic `(dimension,)` and a deterministic `(dimension,1)` are exercised.

**Failure scenario.** The 2D shape check accidentally inverted to `values.shape[0] != dimension`; `linear.matvec(np.ones((3,1)))` on a 4-dim plan silently slices `[:,0]` instead of raising. A `values[:,0]`-vs-`values[0,:]` bug passes deterministic tests but fails random `(d,1)` inputs.

**Benefit.** ~4 parametrized cases. Minor but closes spec-literal gaps (§6.3 "invalid vector dimensions" and "random complex vectors in both forms").

**Verification.** Re-read `scipy.py:24-35` and `test_scipy_linear_operator.py:26-43`. Not in exclusion ledger.

#### R4-TG3 — PySCF: no Hermiticity, particle-number, or independent dense-matrix reference test — **test-gap · medium · [REAL: catches silent ERI-axis regressions] · CLOSED in working tree**

**Status: CLOSED.** The working tree adds `_independent_integral_reference` (built directly from PySCF `ao2mo` output + raw fermion terms, independent of the adapter's native path) and now asserts: full dense matrix equality vs the independent reference (RHF + UHF), individual one-body/two-body coefficient values via `_term_coefficient`, and JW-mapped dense agreement. Additionally the *block-path* Hermitian check that this finding's "soft" sub-concern flagged is now enforced natively (`validate_hermitian_pair` in `crates/tencir-pauli-core/src/structured.rs`, applied to all four ERI blocks + one-body blocks). The only uncovered sub-item is an explicit `operator.is_hermitian()` assertion and a `[operator, N] == 0` particle-number-commutator check — but the full dense-matrix equality vs an independent reference is strictly stronger evidence than either of those, so the gap is effectively closed.

**File:** `tests/test_pyscf_integration.py:56-131` (all tests).

**Summary.** Phase-10 §5.4 requires "Hermiticity and particle-number conservation" and "direct fermionic dense matrices against a trusted small-system construction." Existing tests assert only: (a) nuclear-repulsion constant, (b) the single aggregate determinant-energy diagonal element `matrix[1100,1100] == e_tot`, (c) ordering permutation equivalence, (d) JW-vs-fermionic *self*-agreement (both from the same native implementation — not independent per §15). No test asserts `operator.is_hermitian()`, no particle-number-commutator check, no full dense matrix vs an independent from-terms oracle, and no individual one-body/two-body coefficient value assertions.

**Failure scenario.** A regression that transposes the AB/BA chemist axes but leaves the RHF determinant energy ~unchanged (H2's α=β makes the AA block dominant) passes the energy-only test while producing wrong open-shell UHF physics. The UHF test uses `abs=1e-7` on a single diagonal — too coarse to catch an AB/BA transposition if the UHF determinant is AA+BB-dominated. A sign error in one `g_pqrs` element that cancels in `<1100|H|1100>` passes.

**Benefit.** Catches ERI-axis / sign regressions that energy-only tests miss. For a 2-orbital system the AB/BA blocks are ~10-30% of correlation signal; a transposition shifts off-diagonal elements by O(1) while leaving the determinant diagonal within 1e-7. ~40 lines to add `is_hermitian()` + `[operator, N] == 0` + a from-terms oracle + per-coefficient assertions.

**Verification.** Re-read `test_pyscf_integration.py:56-131`; confirmed no `is_hermitian`, no commutator, no independent dense oracle, no per-coefficient `assert_allclose`. Not in exclusion ledger.

#### R4-TG4 — PySCF: UHF missing mapped-Pauli agreement, both orderings, and complex-MO-coefficient coverage — **test-gap · low-medium · [REAL] · CLOSED in working tree (UHF mapped-Pauli + dual ordering); complex-MO branch still untested**

**Status: PARTIALLY→CLOSED.** The working tree's UHF test now compares `operator.compile("dense")` against the independent `_independent_integral_reference` dense matrix, asserts individual UHF two-body coefficients across multiple `(creation, annihilation)` pairs (including mixed alpha/beta), and adds the interleaved-UHF + FSWAP-permutation comparison (`transform @ interleaved @ transform.conj().T`). The mapped-Pauli-agreement and both-orderings sub-items are closed. The only residual is the complex-MO-coefficient `ao2mo.incore.general` branch (`pyscf.py:105-117`), which remains unexercised by CI — low severity, leave for when a real complex-MO workload arrives.

**File:** `tests/test_pyscf_integration.py:112-131` (UHF test); untested paths at `pyscf.py:90-131` (complex-MO branch), `:54-78` (`_validate_orbitals`).

**Summary.** The RHF test compares `mapped.dense()` vs fermion dense; the UHF test does **not**. The UHF test uses only `alpha_then_beta` — no interleaved UHF, no FSWAP permutation comparison for UHF. The complex-MO-coefficient ERI path (`_has_nonzero_imaginary_part` → `ao2mo.incore.general` branch at `pyscf.py:105-117`) is never exercised by CI (all tests use real H2/sto-3g coefficients).

**Failure scenario.** A UHF-specific spin-block-ingestion bug (alpha/beta transposition) produces a fermion operator whose JW mapping differs from its dense form; determinant energy matches, mapped Pauli is wrong. A complex-MO molecule (spin-orbit, field-perturbed) hits the unexercised `ao2mo.incore.general` branch; a shape/order bug there is silent.

**Benefit.** ~4 lines for UHF mapped-Pauli + interleaved + permutation; one complex-phase test for the `incore.general` branch. Genuinely new.

**Verification.** Re-read `test_pyscf_integration.py:112-131` and `pyscf.py:90-131,54-78`. Not in exclusion ledger.

#### R4-TG5 — U1 `state_full`/`probability_full` lack exact-value tests (R3-10 residual) — **test-gap · low · [REAL but low-value: thin delegation wiring]**

**File:** `tests/test_u1_circuit.py:27`, `tests/test_public_api.py:220` (shape-only); entry points `python/tencirpauli/u1_circuit.py:248-279` (`state_full`/`to_dense`/`probability_full`).

**Summary.** R3-10 is partially closed: the public facade covers all four methods and `probability`/`expectation` have exact-value tests, but `state_full`/`probability_full` (delegating to native `to_dense`/`probability_full`) have only shape assertions. The plan-level API is now private (`_U1CircuitPlan`), so the public-API concern is closed; the residual is the missing exact-value binding on the two state/probability-full methods.

**Failure scenario.** A regression in `to_dense`/`probability_full` FFI delegation passes shape tests but returns wrong values.

**Benefit.** ~3 `assert_allclose` lines: `plan.state_full(...)[basis] == plan.run(...)`, `probability_full` sums to 1. Low cost, closes R3-10 fully.

**Verification.** Re-read cited test lines; confirmed shape-only. This is the explicit carry-forward of R3-10's partial verdict.

#### R4-TG6 — Phase-9 residency/ABI tests: propagation & compilation monkeypatch, and `isinstance(np.ndarray)` — **test-gap · low · [OVER-DEFENSE — skip unless trivial]**

**Files:** `tests/test_propagation.py`, `tests/test_structured_algebra.py`, `tests/test_lazy_operator.py`, `tests/test_native_mvp_resources.py`.

**Summary.** Phase-9 §15 nominally requires monkeypatch materializer-failure tests for propagation/compilation and explicit `isinstance(x, np.ndarray)` on terminals. On re-review these are the "test that X does not happen" over-defense pattern: monkeypatching `_materialized_terms` to raise proves a *hypothetical* future refactor does not reintroduce a round-trip; `isinstance(np.ndarray)` guards against a *hypothetical* PyO3 return-type change.

**Action: skip unless already trivial.** The existing `_terms is None` assertions (`test_lazy_operator.py:131-135`) and `writeable`/`c_contiguous` checks are sufficient for real regressions; the monkeypatch + isinstance layer is testing imagined futures, which is the pattern the owner wants avoided. Do not add complexity here.

**Files:** `tests/test_propagation.py`, `tests/test_structured_algebra.py` (no propagation/compile monkeypatch); `tests/test_lazy_operator.py`, `tests/test_native_mvp_resources.py` (no `isinstance(np.ndarray)` on terminals).

**Summary.** Phase-9 §15 requires handle-residency tests proving propagation and compilation "do not populate `.terms` or invoke array/materialization exports," and read-back ABI tests proving terminals "return flat NumPy arrays … with no nested Python sequence." Monkeypatch materializer-failure tests exist for charge restriction, embedding, U1 hermiticity, equality/hash — but **not** for propagation forward execution or `compile("dense"/"coo"/"csr"/"native_mvp")`. Terminal outputs are checked for `writeable`/`c_contiguous` but never explicitly `isinstance(x, np.ndarray)`.

**Failure scenario.** A `PropagationEngine.forward` refactor accidentally calls `operator._materialized_terms()`; the `_terms is None` assertion still passes (because `_materialized_terms` caches into `_terms`), but the no-materialization contract is violated. A PyO3 change returning `Vec<f64>` as a Python list for `csr.data` passes `writeable`/contiguity but degrades downstream numpy to object-dtype.

**Benefit.** ~10 lines monkeypatch + ~3 `isinstance` assertions. Low cost, closes spec-literal gates.

**Verification.** Re-read `test_lazy_operator.py:36-87`, `test_native_mvp_resources.py:43-46,161`; confirmed no `isinstance(np.ndarray)` and no propagation/compile monkeypatch. Not in exclusion ledger.

#### R4-TG7 — JAX PyTree-with-repeated-leaf and snapshot-immutability only tested for Propagation, not U1/SPPS — **test-gap · low · [NARROW: shared helper makes this structural-not-empirical]**

**File:** `tests/test_circuit_jax.py:18-63` (Propagation PyTree), `:100-112` (Propagation snapshot), `:120-135` (U1 — no repeated leaf), `:139-153` (SPPS — trivial single-scalar).

**Summary.** The Phase-9.5 R1 closure delivered the required *one* PyTree fixture (Propagation) and the *one* callback-count/snapshot test (Propagation). Spec §7.3 requires `expectation_jax()` to work "for all three circuit families." The U1 JAX test has arithmetic but no repeated leaf driving multiple gates; the SPPS JAX test is a trivial single-`ry` case. The one-callback/zero-backward contract is instrumented only on `PropagationEngine.value_and_grad`, not U1/SPPS.

**Why this is not a false-closure.** R1's *minimal* closure asked for "one PyTree fixture" (singular) — delivered. The shared `jax_support.py` helper means the VJP structure is identical across families, so the Propagation test exercises the shared mechanism. The closure text is careful to say "instruments `PropagationEngine.value_and_grad`" specifically.

**Why it still matters.** A wrong U1/SPPS cotangent-summation for a repeated outer leaf, or a missed U1/SPPS snapshot bug, could pass. The shared-helper argument is structural, not empirical.

**Benefit.** One U1 + one SPPS PyTree-with-repeated-leaf test; ~20 lines. Closes spec §7.3 bullets for all three families.

**Verification.** Re-read `test_circuit_jax.py:18-153` and `jax_support.py:68-80`. Not in exclusion ledger.

#### R4-N3 — No regression test pins the signed-zero hash contract — **test-gap · low · [REAL contract, trivial test]**

**File:** (none — grep for `content_hash`/`neg_zero`/`signed_zero` across `crates/**/tests*` and `crates/tencir-pauli-core/src/tests.rs` returns zero matches).

**Summary.** Phase-9 S3 correctly canonicalizes `+0.0`/`-0.0` in `hash_f64` (`scalar.rs:10-13`: `if value == 0.0 { 0 } else { value.to_bits() }`), applied pattern-wide via `hash_complex`. The contract `op == op.adjoint() ⇒ hash(op) == hash(op.adjoint())` now holds for real operators (verified by reading `operator.rs:369-382` adjoint + `:463-471` content_hash; `PauliWord::adjoint()` returns `self.clone()`, adjoint only conjugates the coefficient; `hash_complex` canonicalizes the `-0.0` imaginary part). But no Rust-side test pins this — a future revert to `Complex64`'s derived `Hash` (which uses `to_bits()`, distinguishing signed zeros) would silently re-break the contract.

**Note.** The Phase-9 closure evidence claims Python-side signed-zero tests exist across all 6 families (`test_value_semantics.py:14-29`), which the test-gaps finder confirmed. This finding is specifically about the *Rust core* having no `content_hash` regression test, so the contract is pinned at the layer that actually implements it, not only at the Python boundary.

**Benefit.** ~5 lines: construct a real Hermitian Pauli op, assert `op.content_hash() == op.adjoint().content_hash()`; plus a `Complex64::new(1.0,-0.0)` vs `new(1.0,0.0)` equal-hash assertion.

**Verification.** Re-read `scalar.rs:10-13`, `operator.rs:363-471`; grep confirms no `content_hash` test in `crates/`. Not in exclusion ledger.

### 3.4 Migration residue (informational)

#### R4-I1 — `evaluate_parameters` and `node_adjoint` names retained despite spec §6.2 ordering their deletion — **naming · low (informational)**

**Files:** `crates/tencir-pauli-core/src/circuit_ir.rs:109-126` (`evaluate_parameters`); `crates/tencir-pauli-core/src/u1_circuit.rs:813,836,885` (`node_adjoint` parameter of `accumulate_gate_derivative`).

**Summary.** The expression-DAG is genuinely gone (no `ParameterExprNode`, no `reverse_parameter_program`, no node-adjoint type — verified by grep). The surviving `evaluate_parameters` is now a pure numerical resolver (`AngleRef::Static(v) => v`, `Slot(i) => parameters[i]`), and `node_adjoint` is a runtime-slot adjoint. Only the *names* are misleading — the spec §6.2 literally says "Delete … `evaluate_parameters()` … expression-node adjoints."

**Scenario.** A future reviewer greps for `evaluate_parameters` (the spec's exact deletion target), finds it, and incorrectly concludes the expression evaluator survived. The Phase-9.5 closure-evidence search command matches this token and attributes it to "low-level numerical slot APIs" — functionally correct but hiding the naming collision.

**Severity.** Naming/maintenance only. No numerical or contract impact. Suggested renames: `resolve_angle_values`, `angle_adjoint` (the caller already uses the latter name). Brand-new residue the Phase-9.5 reviews (R5 covered prose/docstrings) did not audit.

---

## 4. Verified-correct items (no finding, recorded for confidence)

These were actively investigated and confirmed faithful to spec; they are **not** findings.

- **Phase-10 ERI convention.** PySCF `ao2mo.kernel`/`ao2mo.general`/`ao2mo.incore.general` return `(ij|kl)` indexed `out[i,j,k,l]` (verified against PySCF source einsum `pqrs,pi,qj,rk,sl->ijkl`). Rust `chemist_index(n,p,r,q,s)` reads `eri[p,r,q,s] = (p_σ r_σ | q_τ s_τ)`, exactly matching spec `g[p_σ,q_τ,r_σ,s_τ] = eri[p,r,q,s]` for all four spin blocks. RHF passing `(eri,eri,eri,eri)` is correct (α=β for RHF). The Hermitian pair `eri[p,r,q,s]` ↔ `eri[r,p,s,q]` is the correct conjugate pair.
- **Hermitian averaging / 0.5 prefactor.** `hermitian_average(g[p,q,r,s], conj(g[r,s,p,q])) * 0.5` does **not** double-count — verified numerically (`/tmp/check_herm.py`, n=2): Rust-implementation-sum / spec-sum = exactly 1.0. The 0.5 compensates for both members of each Hermitian pair appearing in the loop. One-body `hermitian_average` (no 0.5) verified separately, ratio 1.0.
- **Fermion word emission.** `[(p,0),(q,0),(s,1),(r,1)]` with 0=create, 1=annihilate spells `a†_p a†_q a_s a_r` (spec §4.1). Canonicalizer handles the word correctly.
- **Fused commutator signs.** `binary_fermion_terms(.., reverse_sign=-1)` = commutator `AB-BA`; `+1` = anticommutator `AB+BA`; `0` = product. `binary_hybrid_terms` and `binary_majorana_terms` (graded parity `(-1)^(kl-|A∩B|)`) verified correct for ordinary Lie bracket and anticommutator; the `factor == 0` skip (e.g. anticommutator when supports anticommute) is a sound optimization.
- **Signed-zero hash (Phase-9 S3).** `hash_f64` canonicalizes both signed zeros to bit `0`, pattern-wide. `op == op.adjoint() ⇒ hash` equality holds for real operators. (NaN payloads are not canonicalized — different NaN bit patterns hash differently — but `NaN != NaN` preserves the `Eq`/`Hash` contract technically.)
- **GIL release (Phase-9 S1/S2).** QWC `reconstruct` (`grouping.rs:87-91`) and charge `compile_mvp_pauli_handle`/`compile_mvp_hybrid_handle` (`charge_sector.rs:317-348,364-396`) genuinely wrap material work in `allow_threads`. Confirmed by re-reading.
- **JAX VJP contract (Phase-9.5).** `jax_support.py:68-80`: forward calls `call_native` once returning `(value, gradient)`; backward returns `(cotangent * gradient,)` with no callback. `expectation()` does not allocate a gradient (forward uses `nparameters==0` tapes / cached final state). `AngleRef` is a clean 2-variant numerical enum, no expression DAG.
- **R2-4.** Reproduced by two independent agents: 64q/64rot/`max_weight=1`/`max_bytes=None` now constructs (was `OverflowError`); `max_weight=None` still correctly raises on genuine unbounded growth. Correct projected-vs-unbounded semantics.

---

## 5. Dropped findings (audit trail)

For transparency, findings the adversarial pass rejected:

| Claim | Why dropped |
|---|---|
| `i128` charge-weight arithmetic is "over-defense" (Phase-9 §4.2) | Spec-ambiguous. §4.1 explicitly permits "the existing bounded native integer representation" for indices/small weights; `i128` is bounded (not arbitrary-precision). `analyze_charge` conservation verdict uses `f64`/`Complex64` (§9-compliant). The `i128` is confined to rank/unrank and target storage, not coefficient cancellation. Owner-interpretation-dependent; not filed. |
| TensorCircuit `id()`-based objective cache is not GC-safe | Pre-existing (Phase 8 facades), explicitly permitted by Phase-9.5 §4.4 ("observable identity or immutable handle"). Observable held alive by user's objective closure in practice. Latent, not a regression. Not filed. |
| `evaluate_parameters`/`node_adjoint` are expression-DAG survivors | Functionally false — the DAG is genuinely gone; only the *names* survive. Re-filed as low-severity naming residue (R4-I1), not a correctness/contract finding. |
| PySCF `_fermion_from_integral_blocks` skips Hermitian validation that `from_integrals` enforces | Soft contract gap, not a hard violation: spec §5.2 does not explicitly require the private block path to pre-check Hermiticity (the native side's pairwise averaging is the documented Hermitian-by-construction mechanism). Silently symmetrizing a non-Hermitian PySCF output would hide a PySCF-side bug, but produces a valid Hermitian operator. Too soft to file as a standalone finding; absorbed into R4-TG3 (add an `is_hermitian()` test, which would catch it). |

---

## 6. Cross-cutting assessment

**Where Phase 9 succeeded (and was verified, not assumed).** The "wrapper is not thin / boundary frequency is too high" debt that dominated Round 3 is substantially retired. Read-back is flat-NumPy across Pauli/structured/Majorana handles; terminals are handle-accepting; commutators are fused-native; QWC reconstruction and charge compilation release the GIL; `_native_data` and the dead Pauli FFI quartet are gone; the symbolic parameter system is removed end-to-end. The Phase-9 closure docs are accurate on every point I independently re-verified — there are **no false-closure claims** in this round.

**Where Phase 9/9.5 went slightly too far.** The over-defense removal (§4.2) is spec-authorized and safe on the propagation path (trig-bounded), but it was applied uniformly to the SPPS path-product accumulator, which is *unbounded*. The result (R4-N1) is the only medium-severity new correctness item: silent `NaN` estimators on realistic ~134-gate SPPS circuits. This is the kind of "AI-authored-code failure mode" the checklist warns about — a fix (remove over-defense) applied by pattern-matching across paths without checking whether each path's numerics are actually bounded.

**Where Phase 9's slice-7 cleanup is incomplete.** Four dead/superseded array-input FFI entry points (R4-D1..D4) survived slice 7 because their handle variants made them unreachable rather than broken. They are not called by production, but they remain registered and typed, carrying GIL-contract violations (R4-D1) and forbidden round-trip shapes (R4-D2/D3) that a future caller could route through.

**Phase 10 posture.** Numerically faithful — the ERI convention, Hermitian averaging, spin ordering, and SciPy matvec all match the frozen spec and were verified against PySCF source and numerically. After the working-tree re-verification, the chemistry test coverage is now strong: an independent `_independent_integral_reference` dense oracle (built from raw PySCF `ao2mo` + fermion terms) pins the full dense matrix, individual one-/two-body coefficients, JW-mapped Pauli agreement, and dual spin ordering for both RHF and UHF; the block-path Hermitian/finite contract is enforced natively in `validate_hermitian_pair`. Remaining Phase-10 residuals are narrow (complex-MO-coefficient `ao2mo.incore.general` branch unexercised) or over-defense (SciPy no-materialization/single-adapter monkeypatch — skip).

## 7. Carried-forward open ledger (for the next round)

The following remain open and should not be re-raised as new unless newly incomplete. Over-defense items are marked and explicitly **not** recommended for re-implementation.

1. **R3-11** — Generic charge per-apply scratch (`charge.rs:1538-1543`). **Deferred** per Phase-9.5 §8 with recorded `defer_scratch_reuse` decision (~4.1% steady delta < 10% threshold).
2. **R3-12** — U1-lazy/structured MVP per-apply scratch (`sector.rs:639-642`, `structured.rs:2411-2412`). Same recorded deferral.
3. **R3-14** — PackedKey clone per transition (`propagation.rs:1449` etc.). Narrow benefit — only `Wide` (>128q) heap-allocates.
4. **R3-16** — Non-conserved charge `Vec<u64>` clone per term (`charge.rs:1602-1604`). Low-priority perf.
5. **R3-23** — SPPS `value_and_grad_adaptive` serializes across terms (`spps.rs:478-520`). Adaptive-SPPS JAX explicitly deferred (Phase-9.5 §3); not targeted by Phase 9/9.5.
6. **R3-10 (residual)** — U1 `state_full`/`probability_full` exact-value tests. Re-filed as R4-TG5.
7. **R4-N1** — SPPS missing exit finiteness check (silent NaN on long circuits). New this round, `[NARROW]`; minimal ~3-line fix, do not restore layered per-branch checks.
8. **R4-D1/D2/D3/D4** — Dead/superseded FFI residue + redundant `masks` field. New this round, `[REAL]` zero-complexity cleanup. R4-D4 is TO DO (API not frozen; drop the unused `reconstruction_masks` field outright).
10. **R4-TG1 (compile-count only) / R4-TG5 / R4-TG7 / R4-N3** — Test-coverage gaps still open. (R4-TG3 and R4-TG4 closed in working tree; R4-TG1's no-materialization/single-adapter sub-items are over-defense, skip.)
11. **R4-I1** — `evaluate_parameters`/`node_adjoint` naming residue. New this round, informational.

**Closed since the initial report (working-tree changes):**
- **R4-TG3** — independent dense oracle + per-coefficient + Hermitian validation now present (native `validate_hermitian_pair` + `_independent_integral_reference`). CLOSED.
- **R4-TG4** — UHF mapped-Pauli agreement + dual ordering + per-coefficient now tested. CLOSED (complex-MO branch residual left for a real complex-MO workload).
- **R4-TG1 (single-adapter sub-item)** — `BackendMVPPlan.to_scipy_linear_operator` deleted; test asserts absence. The inconsistency is moot.

**Explicitly NOT open (over-defense, correctly removed by Phase 9 — do not re-implement):**
- **R4-N2** — operator/majorana intermediate-aggregation finiteness scan. Withdrawn. Only triggers near `f64::MAX`; Phase 9 §4.1 authorizes the removal. Do not re-add.
- **R4-TG1 (no-materialization + single-adapter monkeypatch)** — "test that X never happens" over-defense. Skip.
- **R4-TG6 (residency monkeypatch + `isinstance(np.ndarray)`)** — "test that X never happens" over-defense. Skip unless trivial.

**Recommended sequencing.** R4-N1 first (only correctness item; ~3 lines, no layered checks). Then the cheap real-coverage tests (R4-N3/R4-TG1-compile-count — R4-TG3/TG4 already closed in working tree). Then the zero-complexity dead-FFI / redundant-field cleanup (R4-D1/D2/D3/D4, including dropping the unused `reconstruction_masks` field while the API is still mutable). Skip all over-defense items (R4-N2, R4-TG6, R4-TG1 no-materialization/single-adapter). The carried-forward perf items (R3-11/12/14/16/23) remain profile-gated per their documented deferrals.

---

*Audit method: 6 parallel finder dimensions + independent firsthand re-verification of all load-bearing claims (R2-4 arithmetic trace, ERI index math, GIL releases, scratch-reuse status, fused-commutator signs, Hermitian-averaging numerics). No source code was modified. Line numbers re-resolved to the current tree at commit `32bc7d8`.*
