# TenCirPauli Deep Audit — 2026-08-04 Archive

## 1. Executive Summary

**Counts**
- Raw findings: 72
- Unique findings: 72
- Confirmed: 54 (of which 13 high/medium-criticality warrant fix-now urgency)
- Real but not worth fixing: 13
- Refuted: 5

The counts above describe the audit classification, not the number of changes made. The recommendation fields in Sections 2 and 3 preserve the reviewer's proposed urgency; the post-audit disposition ledger in Section 6 is authoritative for what was adopted, replaced by a safer alternative, deferred, or rejected.

**Most important themes**
1. **Correctness defects in canonical-form and fermionic algebra** — silent wrong results from `OperatorSpace.embed` annihilation-sign drop, `canonicalize_hybrid_terms` mis-marking identity fermion/boson as present, duplicate-symbol parameter-slot inflation in `PropagationCircuit.from_qir`, and stale `id()`-keyed compile caches. These touch phase/qubit-ordering and require regression tests.
2. **Repeated coarse-grained FFI violations** — per-term `to_codes()` native calls in `tensor_product`, `map_pauli`, `_compile_restricted_transitions`, `PauliWord.multiply`, and `PauliWord.multiply` again. Cached `_canonical_structures` is consistently the right input but consistently unused.
3. **Duplicated algebraic conventions across the Python/Rust boundary** — JW word expansion, fermion/boson CAR/CCR normal-ordering, and IXYZ code tables are each maintained in two places; a divergence hazard for the project's non-negotiable phase/ordering invariant.
4. **Quadratic accounting loops on hot structured paths** — `push_aggregate`'s O(M²) total-count scan and `aggregate_source`'s double pass dominate wide hybrid products and sector compilation.
5. **Statistic/reporting correctness** — SPPS standard error is biased low by ~30% at the minimum sample count; `value_standard_error` is publicly exposed without disclaimer.
6. **Defensive-coding gaps** — `U1CircuitPlan::compile` omits the mandatory state-vector from its budget; qudit triples in `compile_charge_transitions` skip canonical validation; non-finite parameters reach the Rust boundary for PropagationEngine but not SPPSEngine.

**Prioritized action list**
1. **Fix-now, correctness-critical (require regression tests before merge):** `OperatorSpace.embed` annihilation sign (#7); `canonicalize_hybrid_terms` canonical-form break (#3); `PropagationCircuit.from_qir` duplicate-symbol slots (#6); `id()`-keyed compile caches in both Propagation and SPPS circuits (#19, #21); SPPS SE bias (#16).
2. **Fix-now, performance-critical:** `expectation_from_dynamic_terms` Zero-state fast path (#1); `push_aggregate` running counter (#2, #15); `_compile_restricted_transitions` / `map_pauli` cached-array swaps (#4, #5); `tensor_product` codes hoist (#18).
3. **Fix-now, fail-fast/loophole closure:** U1 compile-time state budget (#17); qudit triple canonical validation (#26); PropagationEngine parameter finiteness (#20).
4. **Fix-now, low-risk cleanup:** error message off-by-one `0..3` → `0..4` (#9); validator + IXYZ-table consolidation (#14, #40); dead-code removals (#13, #25, #27, #28); `aggregate` double-collect fusion + `PackedKey::Ord` packed-word comparator (#11).

---

## 2. Confirmed Findings Worth Fixing

> Items that touch correctness, phase, or qubit ordering are explicitly flagged **[REGRESSION TEST REQUIRED]**. The recommended fix in each case is `refined_fix` (preferred over the original `proposed_fix` where the verifier refined it).

### Critical / High

#### H1 — Performance · `crates/tencir-pauli-core/src/propagation.rs` (`expectation_from_dynamic_terms`)
**Title:** Ignores the Zero-state fast path (O(terms·nqubits) vs O(terms·word_count)).
**Failure scenario:** For the default `ProductState::Zero` on n=64 qubits, `expectation_from_dynamic_terms` (lines 1743–1757) folds over `(0..nqubits)` per term while the sibling `expectation_of_key` (1759–1781) already short-circuits via a bulk `x_all_zero` check. `PropagationEngine::expectation` (line 178) and `value_and_grad` (line 235) lose the fast path — ~64× penalty per term on the forward/gradient value path.
**Fix:** Replace the function body with delegation to `expectation_of_key`:
```rust
fn expectation_from_dynamic_terms(terms: &[DynamicTerm], state: &ProductState, nqubits: usize) -> f64 {
    terms.iter().map(|term| term.coefficient.re * expectation_of_key(&term.key, state, nqubits)).sum()
}
```
**Benefit:** O(terms·word_count) instead of O(terms·nqubits); 64× reduction at n=64, scaling with qubit count. Bit-identical output (algebraic equivalence verified).
**Tradeoffs:** Negligible — reuses an existing `pub(crate)` function; removes ~15 lines of duplicated fold logic. No new helper, no duplication.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### H2 — Performance · `crates/tencir-pauli-core/src/structured.rs` (`push_aggregate`)
**Title:** Recomputes total value count over the whole map on every push — O(P²).
**Failure scenario:** `push_aggregate` (lines 1339–1362) runs `aggregate.values().try_fold(0, |c, v| c + v.len())` after every insertion. Across `multiply_hybrid_terms` / `multiply_fermion_terms` / `multiply_boson_terms` / `canonicalize_*` (8 call sites), the aggregate grows to M entries → O(P·M) = O(P²) accounting overhead on wide hybrid products.
**Fix:** Thread `total_values: &mut usize` through `push_aggregate`/`push_pauli_aggregate`; increment per push; replace the fold with `*total_values`. Initialize `let mut total_values = 0usize;` in each caller (multiply_hybrid_terms, canonicalize_hybrid_terms, canonicalize_fermion_terms, multiply_fermion_terms, jordan_wigner_hybrid_terms, canonicalize_boson_terms, multiply_boson_terms).
**Benefit:** O(M²) → O(M) accounting. For P~10⁴ contributions: ~10⁸ → ~10⁴ ops. Same bound enforced (`total_values.max(value_count)`).
**Tradeoffs:** Signature change across 8 call sites; invariant trivial (push-only, no removals). No numerical-output change.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### H3 — Bug · `crates/tencir-pauli-core/src/structured.rs:232` **[REGRESSION TEST REQUIRED]**
**Title:** `canonicalize_hybrid_terms` marks identity fermion/boson as present, breaking canonical form.
**Failure scenario:** When `fermion_factors[index]` is empty, `fermion_products = [(None, 1)]` but the key uses `Some(fermion.clone().unwrap_or_default())` → `Some((vec![], vec![]))` (identity word marked present). `OperatorBuilder.finish` thus yields `fermion=FermionWord(n, (), ())` while `_from_terms` paths yield `fermion=None`. Concrete harms: (1) `_requires_eager_fermion_mapping` spuriously triggers eager `map_fermions`; (2) native multiply distinguishes `Some(())` from `None` in `HybridKey`, producing representationally different products; (3) term-by-term equality fails.
**Fix:** Mirror `multiply_hybrid_terms` (line 132) and the qudit present-flag pattern: replace the `if layout.n_modes != 0 { Some(fermion.clone().unwrap_or_default()) } else { None }` wrappers with direct `fermion.clone()` / `boson.clone()`.
**Benefit:** Single canonical form regardless of construction path; fixes aggregation/equality bugs; removes spurious eager fermion mapping.
**Tradeoffs:** Builder-built operators on fermion-capable spaces change representation (identity fermion words → absent). No existing test asserts the identity-present form (tests compare dense matrices). Low risk.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### H4 — Performance · `python/tencirpauli/charge.py:900`
**Title:** `_compile_restricted_transitions` calls `to_codes()` per Pauli term — T native roundtrips.
**Failure scenario:** The `PauliOperator` branch loops `for pauli_term in operator.terms` and calls `qubit_codes.append(list(pauli_term.word.to_codes()))` — one `_native.pauli_codes` FFI call per term on every `restrict_charge`/charge-sector MVP plan compilation. The identical codes are already cached as `operator._canonical_structures`.
**Fix:**
```python
structures, coefficient_reals, coefficient_imaginaries = operator._arrays()
for structure, real, imaginary in zip(structures, coefficient_reals, coefficient_imaginaries):
    ...
    qubit_codes.append(list(structure))
    coefficients.append(complex(real, imaginary))
```
(With a comment noting `_canonical_structures` uses the same I=0/X=1/Y=2/Z=3 encoding as `PauliWord.to_codes()`.)
**Benefit:** Eliminates T per-term native roundtrips on every `restrict_charge`/MVP-plan compilation. Encoding verified identical to `core word.rs::code_at`.
**Tradeoffs:** Couples charge module to the `_arrays()` private caching invariant (already used pervasively in `pauli.py`). The original finding overstates benefit — `operator.terms` is built once at construction, so the per-restrict saving is T FFI roundtrips, not the additional O(T·N) Python rebuild.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### H5 — Performance · `python/tencirpauli/mapping.py:394`
**Title:** `map_pauli` rebuilds per-term code lists via `to_codes()` instead of `_arrays()`.
**Failure scenario:** Native-plan path builds `[list(term.word.to_codes()) for term in operator.terms]` plus real/imag lists — T per-term `pauli_codes` FFI calls. All three lists are already `operator._canonical_structures` / `_coefficient_reals` / `_coefficient_imaginaries`.
**Fix:** `structures, coefficients_re, coefficients_im = operator._arrays()` then `result = self._native_plan.transform(structures, coefficients_re, coefficients_im, _effective_max_bytes(max_bytes))`. Pass `structures` verbatim — do NOT wrap in `list(map(list, structures))`.
**Benefit:** Eliminates T native calls per `map_pauli` plus redundant Python `PauliWord` round-trip. Aligns with the project's coarse-grained-FFI rule and the convention already used by `scale`/`add`/`dense`/`coo`/`csr`/`mvp`/`find_z2_symmetries`.
**Tradeoffs:** One-line readability change. Risk negligible.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### H6 — Bug · `python/tencirpauli/propagation_circuit.py:535` **[REGRESSION TEST REQUIRED]**
**Title:** `PropagationCircuit.from_qir` assigns distinct parameter slots to duplicate symbols in auto-discovery mode.
**Failure scenario:** In auto mode (`parameter_order is None`), `ordered_symbols` is empty so `symbol_slot` always returns None. The fallback (lines 562–566) unconditionally appends to `seen_symbols` and returns `Parameter(len-1)` with no dedup. A reused symbol (`rz(0, theta=s); rz(1, theta=s)`) gets two distinct slots → `nparameters == 2` (expected 1), splitting one logical angle into two independent parameters. Wrong gradient shapes/values. Trailing contiguity validation passes silently.
**Fix:** Deduplicate in the fallback — search `seen_symbols` before appending:
```python
for index, symbol in enumerate(seen_symbols):
    try:
        if bool(value == symbol):
            return Parameter(index)
    except Exception:
        pass
seen_symbols.append(value)
return Parameter(len(seen_symbols) - 1)
```
**Benefit:** One slot per unique symbol, matching the documented `Parameter` model and the explicit-`parameter_order` path's dedup. Fixes wrong `nparameters`, gradient shape, and gradient values for the common reused-symbol case.
**Tradeoffs:** Linear scan over `seen_symbols` per new symbol — negligible for typical parameter counts. No API change.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### H7 — Bug · `python/tencirpauli/structured.py:761` **[REGRESSION TEST REQUIRED]**
**Title:** `OperatorSpace.embed` drops fermion annihilation-mode reorder sign.
**Failure scenario:** The inversions generator sums `left > right` while the filter branches on `left > right` (creation) vs `left < right` (annihilation). For the annihilation branch every passing pair has `left < right`, so `left > right` is always False/0 — annihilation inversions are never counted. Embedding `c_1^dagger a_1 a_0` under swap `{0:1, 1:0}` yields `(+1)`; correct canonical result is `(-1)` (since `a_0 a_1 = -a_1 a_0`). Creation-only swaps are correct, confirming only the annihilation branch is broken.
**Fix:** Replace the summed value with a constant `1`:
```python
inversions = sum(
    1
    for values, descending in ((creation_image, False), (annihilation_image, True))
    for index, left in enumerate(values)
    for right in values[index + 1 :]
    if (left > right if not descending else left < right)
)
```
**Benefit:** Restores correct fermionic signs for any embedding that reorders annihilation operators. Without this, `OperatorSpace.embed` silently produces a different operator, corrupting downstream algebra and matrix compilation.
**Tradeoffs:** Pure bug fix; one-token change. Zero API/perf impact. No risk to correct paths (creation branch unchanged since passing pairs already contribute 1).
**Verdict:** confidence high, recommendation **fix-now**.

---

### Medium

#### M1 — Performance · `crates/tencir-pauli-core/src/charge.rs:296`
**Title:** `compile_charge_transitions` uses std `HashMap` (SipHash) for `basis_index` lookup instead of `FxHashMap`, inconsistent with `sector.rs`/`charge_sector.rs`.
**Failure scenario:** `basis_index: HashMap<Vec<u64>, u64>` (line 310) and `positions()` `HashSet` (line 79) use std SipHash. Sibling modules (`sector.rs:867`, `charge_sector.rs:12`) use `FxHashMap` for the same Vec-keyed lookup pattern. The legacy `compile_charge_transitions` is a live public path wrapped as `charge_compile_transitions`.
**Fix:** Line 310 → `FxHashMap<Vec<u64>, u64>` (already imported on line 7). Line 79 → `FxHashSet`. Leave `BTreeMap` outputs (transitions, destinations) unchanged for deterministic serialization.
**Benefit:** Removes hashing inconsistency; matches the convention already used in `sector.rs`/`charge_sector.rs` and the `_from_plan` variant in the same file.
**Tradeoffs:** FxHash is non-DoS-resistant but keys are internally validated. `rustc_hash` already a workspace dep.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M2 — Inconsistency · `crates/tencir-pauli-core/src/error.rs:72`
**Title:** Pauli code error messages say "expected 0..3" but valid codes are 0,1,2,3 (so 0..4).
**Failure scenario:** `error.rs:72` literal: `"invalid Pauli code {code} at index {index}; expected 0..3"`. In Rust half-open notation `0..3` excludes 3 (Z), yet `code_bits` (word.rs:272–280) accepts 0,1,2,3. Same misleading `0..3` wording across `pauli.py`, `u1_circuit.py`, `mapping.py`, `structured.py`. Sibling messages (`wire ... outside 0..{nqubits}`, `Majorana index ... 0..2*n_modes`) correctly use half-open bounds.
**Fix:** Change every Pauli-code message to `0..4` (or `0..=3` / `0, 1, 2, 3` consistently). In `error.rs:72` use `expected 0..4`.
**Benefit:** Removes off-by-one between stated range and accepted range; users no longer think Z is rejected. Aligns with codebase convention.
**Tradeoffs:** Pure string change. No tests match the exact message text (grep confirmed).
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M3 — Performance · `crates/tencir-pauli-core/src/propagation.rs` (Clifford/Rotation clones)
**Title:** Forward Clifford/Rotation path clones `PackedKey` per term; in-place helpers exist but are unused.
**Failure scenario:** `apply_operation` (1169–1230) Clifford1/Clifford2 arms call `map_clifford1`/`map_clifford2` which do `let mut result = key.clone()`. Rotation cosine branch pushes `term.key.clone()`. For `PackedKey::Wide` (>128 qubits) each clone heap-allocates two `Vec<u64>`. `apply_clifford1_in_place`/`apply_clifford2_in_place` (1499, 1564) exist with an equivalence test but are only called from `spps.rs`, not the forward path even though `term` is owned/moved.
**Fix:** Clifford arms → `let multiplier = apply_clifford1_in_place(&mut term.key, *gate, *wire);` (and Clifford2). Rotation PlusI|MinusI branch → move `term.key` for cosine when `sine == 0.0`; when `sine != 0.0`, move for cosine and use `product` for sine.
**Benefit:** Eliminates per-term clone for Clifford gates; 2 heap allocations/term/gate removed for Wide systems on deep circuits.
**Tradeoffs:** Behavior-identical (test `in_place_path_updates_match_allocating_maps` pins equivalence). Residual clone inside `multiply_by_generator` is out of scope.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M4 — Performance · `crates/tencir-pauli-core/src/propagation.rs` (`aggregate`)
**Title:** Double collect; sort comparator is O(nqubits) per comparison.
**Failure scenario:** `aggregate` (1388–1419) materializes `into_iter().collect::<Vec<_>>()`, sorts by `PackedKey::cmp`, then does a second `into_iter().filter_map().collect()`. `PackedKey::Ord` (1035–1047) iterates `0..left_n` qubits calling `code_at(qubit).cmp` per qubit → O(nqubits) per comparison, O(N log N · nqubits) sort. `aggregate` runs once per Rotation/CustomPtm gate.
**Fix:** (1) Fuse the double collect: `sort_unstable_by` in place then `retain`. (2) Replace `PackedKey::Ord` with a packed-word lexicographic comparison (compare nqubits, then x-limbs, then z-limbs) — NOT an attempt to reproduce `code_at` ordering. **Critically**, line 348 re-sorts the final output by `PauliWord::cmp`, so the intermediate `PackedKey` sort order is irrelevant to user-visible output — no property test against `code_at` is required, only a check that the new impl is a valid total order consistent with derived `Eq`.
**Benefit:** Halves post-aggregation allocations; each comparison O(word_count) = O(nqubits/64) instead of O(nqubits). On the critical path of every propagation.
**Tradeoffs:** The comparator rewrite must be a valid total order (equal x/z/nqubits ⇒ Equal, which holds). The reviewer's original "reproduce `code_at` ordering" suggestion should NOT be adopted.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M5 — Performance · `crates/tencir-pauli-core/src/sector.rs` (`U1RestrictedOperator::new`)
**Title:** Recomputes `aggregate_source` twice over the full basis (double pass).
**Failure scenario:** `U1RestrictedOperator::new` (357–441) walks `0..dimension` twice (379–398 for `row_counts`, 414–431 for columns/values), both calling `aggregate_source` with identical inputs. Second pass is pure recomputation of `symmetric_intersection_count`, per-term Z-parity, `rank_active_positions` work.
**Fix:** Single-pass CSR build: collect `(destination, source_index, value)` triples in one loop, build `row_counts` by counting, prefix-sum into `indptr`, scatter with per-row write cursors.
**Benefit:** Halves CPU in `U1RestrictedOperator` construction. For D~10⁵, G~10²: removes ~10⁷ redundant group evaluations.
**Tradeoffs:** One extra `Vec<(usize, usize, Complex64)>` of size `entry_count` — within the same memory order already budgeted by `estimate_plan_bytes`.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M6 — Numerical · `crates/tencir-pauli-core/src/spps.rs:998` **[REGRESSION TEST REQUIRED]**
**Title:** SPPS standard error uses biased (population) variance, underestimating SE for small sample counts.
**Failure scenario:** `combine_fixed` computes `sample_variance = (sum_squared/count - mean*mean).max(0.0)` (MLE, divisor N) then `variance += sample_variance/count`. Result: `value_standard_error = sqrt(variance)` is `sqrt((N-1)/N)` of the correct value — ~30% low at the documented minimum `samples_per_term == 2`. `combine_adaptive` (1043–1045) has the identical pattern. Field is publicly exposed and used for convergence/confidence decisions.
**Fix:** `let sample_variance = ((stat.sum_squared - count * mean * mean) / (count - 1.0)).max(0.0);` in `combine_fixed`; analogous change for `left_var`/`right_var` in `combine_adaptive`. `count >= 2` is enforced at all entry points (spps.rs:293, 358, 423). Update frozen test expectation at line 1101 from `0.25` to `SQRT_1_2 / 2.0` (≈0.3535533905932738 = sqrt(0.125)).
**Benefit:** Unbiased SE matching the field-name contract; fixes ~30% underestimation at minimum budget.
**Tradeoffs:** One frozen test value must update. Slightly larger (correct) error bars may stop adaptive convergence one iteration later in rare cases.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M7 — Inconsistency · `crates/tencir-pauli-core/src/structured.rs:67`
**Title:** `HybridLayout.n_qubits` breaks the repo-wide `nqubits` convention.
**Failure scenario:** Every other qubit-count identifier is `nqubits` (`PauliWord.nqubits`, `packed_word_count(nqubits)`, `gate.rs::clifford1(nqubits, ...)`, native bindings, `_native.pyi`). `HybridLayout.n_qubits` is the lone outlier; leaks to three pyfunctions and the stub.
**Fix:** Rename `n_qubits` → `nqubits` in `structured.rs` (declaration + uses at 169/171/419/421), `mapping.rs:325`, three pyfunctions in `native/structured.rs`, and `_native.pyi` (actual stub lines 169/192/214 — finding's 110/133/155 are inaccurate). No Python changes (call sites positional).
**Benefit:** One consistent qubit-count name across Rust core, native binding, and stub.
**Tradeoffs:** Mechanical rename across ~10 sites. No public Python API breakage.
**Verdict:** confidence high, recommendation **fix-later**.

---

#### M8 — Performance · `crates/tencir-pauli-core/src/structured.rs:1351` (duplicate of H2)
**Title:** `push_aggregate` performs O(N) total-count scan on every push, making aggregation O(N²).
**Failure scenario:** Same as H2. Lines 1351–1356 run `aggregate.values().try_fold(...)` after every push. Also notes the 192-byte estimate is ~12× inflated but should NOT be changed (intentional conservative guard per AGENTS.md).
**Fix:** Same as H2 — thread `total_values: &mut usize`. Do NOT change the 192-byte estimate in this change.
**Benefit:** O(M²) → O(M) on every structured aggregation.
**Tradeoffs:** Same as H2.
**Verdict:** confidence high, recommendation **fix-now**. (Treat M8 and H2 as a single fix.)

---

#### M9 — Bug · `crates/tencir-pauli-core/src/u1_circuit.rs:339` **[REGRESSION TEST REQUIRED]**
**Title:** `U1CircuitPlan::compile` omits state-vector bytes from `max_bytes` budget, so run/expectation/value_and_grad can fail after a successful compile.
**Failure scenario:** `compile()` checks `basis_bytes = dimension * word_count * 8` (334–339) and `basis_bytes + pair_bytes + diagonal_bytes` (443–451) but never the mandatory state vector (`dimension * size_of::<Complex64>()` = `dimension * 16`). `validate_state()` (685–690) enforces it at run time. Example: nqubits=10, k=5 → dimension=252, basis_bytes=2016; with `max_bytes=3000` compile succeeds (2016 < 3000) but every run aborts (252·16=4032 > 3000).
**Fix:** After computing `dimension` (line 327), compute `state_bytes = (dimension as u128).checked_mul(size_of::<Complex64>() as u128)` and include it in the final `check_budget` (or add a dedicated `check_budget(state_bytes, max_bytes)?` after line 339).
**Benefit:** Compile-time budget becomes a faithful gate; no confusing late failure. Cannot over-reject any valid execution (state length fixed at `dimension`).
**Tradeoffs:** Users with tight `max_bytes` who relied on the run-time error now get it at compile time — the intended behavior. Optional larger follow-up to account for `full_dimension` (2^nqubits) is out of scope.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M10 — DRY · `python/tencirpauli/majorana.py:27`
**Title:** Triplicated `_exact_nonnegative` validator across charge/mapping/majorana.
**Failure scenario:** `_exact_nonnegative` defined identically in `majorana.py:21`, `mapping.py:38`, `structured.py:88` (`_nonnegative_int`), `pauli.py:48` (`_validate_nonnegative_int`), and `charge.py:70` — five near-identical validators. Copies have already drifted: `charge.py` raises a different message ("must be non-negative" vs "must be a non-negative integer").
**Fix:** Create `python/tencirpauli/_validation.py` with `validate_nonnegative_int(value, name) -> int` raising `"must be a non-negative integer"`. Import in all five modules. Update `grouping.py:11/85` if needed (must preserve the message regex for `tests/test_grouping.py:80`).
**Benefit:** One source of truth; fixes existing message drift in `charge.py`.
**Tradeoffs:** One cross-module import edge. No behavior change for matching copies; `charge.py` message improves.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M11 — DRY · `python/tencirpauli/majorana.py:27`
**Title:** Duplicated `_finite_complex` with subtle divergence (numpy/overflow handling).
**Failure scenario:** `structured.py:74` accepts `numbers.Real|Complex`, rejects `bool`/`np.ndarray`, catches `OverflowError`/`TypeError`. `majorana.py:27` accepts only `int|float|complex`, no overflow catch. A `np.float64` coefficient is accepted by structured but rejected by majorana; a huge int raises uncaught `OverflowError` in majorana. The two modules already disagree on accepted coefficient types.
**Fix:** Delete `majorana.py`'s local `_finite_complex` (27–33); add to existing `from .structured import ...` on line 18.
**Benefit:** Eliminates a real divergence on accepted types and error behavior for overflow inputs.
**Tradeoffs:** Majorana will now accept numpy scalar coefficients (desirable). Risk that majorana intentionally rejected numpy scalars is low (no comment/test asserts rejection).
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M12 — Performance · `python/tencirpauli/pauli.py:882`
**Title:** `tensor_product` uses per-term `to_codes()` in a double loop (N+M native calls per pair).
**Failure scenario:** Builds `left.word.to_codes() + right.word.to_codes()` for every `(left, right)` in `product(self.terms, other.terms)` — 2·N·M native `pauli_codes` calls for results independent of the pair. Codes already cached as `_canonical_structures`.
**Fix:** Precompute `left_codes = self._canonical_structures; right_codes = other._canonical_structures` once; build terms as `(left_codes[i] + right_codes[j], left_coeffs[i] * right_coeffs[j])`. (Defense-in-depth alternative: hoist `[t.word.to_codes() for t in self.terms]` once before the loop.)
**Benefit:** Eliminates 2·N·M native FFI calls plus tuple materialization; turns O(N·M) FFI-cost into O(N·M) pure-data work.
**Tradeoffs:** Pure-Python fix; 3-line refactor. No new native entry point (skip the Rust-push variant). The original finding overstates benefit — no per-iteration `PauliWord` rebuild is being saved.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M13 — Bug · `python/tencirpauli/propagation_circuit.py:356` **[REGRESSION TEST REQUIRED]**
**Title:** `PropagationCircuit.compile` cache keyed by `id()` of observable/state (aliasing).
**Failure scenario:** Cache key `(self._generation, id(observable), id(state), max_weight, budget)`. `PropagationEngine.__init__` extracts observable arrays by value into the native engine and stores no Python reference. If caller lets `observable` be GC'd and CPython reuses its `id()` for a new observable, the cache returns the stale plan — silently wrong expectation/gradient for a different operator.
**Fix:** Append the observable and state to the cache tuple: `self._cached_plan = (*key, plan, observable, state)`. Existing lookup `self._cached_plan[:5] == key` and `self._cached_plan[5]` retrieval unchanged; added refs at indices 6..7 keep objects alive.
**Benefit:** Eliminates a class of silent stale-cache correctness bugs. No API change.
**Tradeoffs:** One extra `PauliOperator` + state reference per circuit (negligible; typically already retained by caller).
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M14 — Mismatch · `python/tencirpauli/propagation.py:411`
**Title:** `PropagationEngine.profile` silently requires Hermitian observable; not exposed or documented.
**Failure scenario:** Native `profile` calls `is_hermitian_observable()` and returns `NonHermitianExpectation` for non-Hermitian observables. `propagate_operator` performs no such check and succeeds. The native getter for `is_hermitian` is not wrapped; the `profile` docstring omits the requirement. The divergence is partly by design (propagated-operator inspection doesn't need Hermiticity; scalar expectation does), so this is an API-completeness/documentation gap, not a correctness bug.
**Fix (minimal):** Add `#[getter] fn is_hermitian_observable` to `NativePropagationEngine`; set `self.is_hermitian = bool(self._native.is_hermitian_observable)` in `PropagationEngine.__init__`; update `profile`/`expectation`/`value_and_grad` docstrings to state the Hermitian requirement. Skip the proposed Python-side pre-call guards — the native layer already fails fast with a clear `ValueError`.
**Benefit:** Users can probe Hermiticity upfront; contract divergence becomes discoverable.
**Tradeoffs:** One getter + a few doc lines. No behavior change.
**Verdict:** confidence high, recommendation **fix-later**.

---

#### M15 — Loophole · `python/tencirpauli/propagation.py:363`
**Title:** `PropagationEngine`/`PropagationBatch._parameters` skip the finiteness check every sibling module enforces.
**Failure scenario:** Both `_parameters` methods hand-roll `np.asarray` + shape check + `ascontiguousarray` with no `np.isfinite`. Every other parameter path delegates to `circuit._coerce_parameters` which rejects non-finite with "parameters must be finite". The Rust kernel's `validate_parameters` catches non-finite later with a less actionable `NonFiniteParameter` error — fail-fast/consistency gap, not silent corruption.
**Fix:** Replace both bodies with `return _coerce_parameters(parameters, self.nparameters)` (importing `_coerce_parameters` from `.circuit`).
**Benefit:** Closes the fail-fast gap; uniform message across all four engines. Zero correctness risk (native safety net already fires).
**Tradeoffs:** Tightens input validation; previously-accepted inf/nan inputs that didn't crash the kernel now raise. Desired behavior.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M16 — Bug · `python/tencirpauli/spps_circuit.py:127` **[REGRESSION TEST REQUIRED]**
**Title:** `SPPSCircuit.compile` cache keys on `id(observable)`/`id(state)` without holding references.
**Failure scenario:** Identical pattern to M13. `SPPSEngine.__init__` extracts observable/state data by value; cache tuple stores no Python references. CPython `id()` reuse after GC → stale plan for a different observable → silently wrong value/gradient.
**Fix:** `self._cached_plan = (*key, plan, observable, state)`. Apply identical change to `PropagationCircuit.compile` (M13).
**Benefit:** Eliminates silent stale-cache correctness bug.
**Tradeoffs:** Single-entry cache; retention is one observable + one state. Negligible.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M17 — Dead-code · `python/tencirpauli/structured.py:1349`
**Title:** Dead `_jordan_wigner_word` branch in base `_StructuredOperator.map_fermions`.
**Failure scenario:** `HybridOperator` returns early (line 1324). `BosonOperator`/`QuditWeylOperator` always build terms with `fermion=None`. `FermionOperator` overrides `map_fermions` (2211) and never calls `super()`. The loop branch `for codes, coefficient in _jordan_wigner_word(term.fermion)` (1349) is unreachable — every term hits `if term.fermion is None: mapped.append(term); continue` (1346–1347). Pure-Python JW word expansion at 1349 is dead code suggesting an alternate path.
**Fix:** Replace the unreachable inner JW block with a fail-fast `raise TypeError(...)` rather than silently deleting it (which would silently drop raw-fermion terms if a future subclass reaches the fallback without overriding `map_fermions`).
**Benefit:** Removes ~11 lines of misleading dead code; prevents future maintainers from "fixing" code that disagrees with the native kernel.
**Tradeoffs:** Mild loss of defensive coverage; the subclass set is effectively sealed and native bindings cover real paths. Not urgent.
**Verdict:** confidence high, recommendation **fix-later**.

---

#### M18 — DRY · `python/tencirpauli/structured.py:388`
**Title:** Python `_normal_order_fermions` duplicates Rust `fermion_rewrite` CAR logic.
**Failure scenario:** `_normal_order_fermions` (388–440) mirrors `fermion_rewrite`/`fermion_inversion` (Rust structured.rs:859–1002) rule-for-rule: same inversion predicates, same swap sign -1, same annihilate-create contraction +1 sign, same zero-on-duplicate rule. The Rust kernel is already exposed as `_native.structured_fermion_canonicalize` and used elsewhere in the same file. Two CAR implementations risk silent divergence in canonical Hamiltonian coefficients.
**Fix:** Replace the BODY of `_normal_order_fermions` with a thin drop-in wrapper over `_native_fermion_raw`: call `creation, annihilation, coeffs = _native_fermion_raw(n_modes, [(factors, 1)], None)` and build `{FermionWord(n_modes, tuple(c), tuple(a)): int(round(coeff.real)) for c, a, coeff in zip(...) if abs(coeff) > 0}`. Keeps signature unchanged so `_multiply_terms` is untouched. Verify against existing fermion canonicalization tests before deleting the Python body.
**Benefit:** Single CAR implementation covered by Rust dense-reference tests; removes ~50-line Python rewrite that must be hand-kept in sync.
**Tradeoffs:** Per-call FFI overhead in a fallback multiply path — acceptable under the coarse-grained-FFI rule (one call per term-pair, not per Pauli term).
**Verdict:** confidence high, recommendation **fix-now**.

> Note: see also the related "unreachable" finding (M20 below) — if `_normal_order_fermions` is rerouted to native, the unreachable-fallback concern is mooted for the fermion path.

---

#### M19 — DRY · `python/tencirpauli/structured.py:1795`
**Title:** Python `_jordan_wigner_word` duplicates Rust `jordan_wigner_word_expansion`.
**Failure scenario:** `_jordan_wigner_word` (1795–1825) is a pure-Python JW word expansion mirroring Rust `jordan_wigner_word_expansion` (structured.rs:731): same Z-string-on-lower-modes, same phase convention (create=-0.5i, annihilate=+0.5i), same `multiply_pauli_codes` + `FxHashMap` accumulation. Live caller: `_tensor_mapped_fermion_terms` (1748/1753). Two JW expansions risk phase/qubit-ordering drift — the invariant AGENTS.md names non-negotiable.
**Fix:** Replace `_jordan_wigner_word`'s body with a 1-word batch through the existing binding: call `_native.structured_fermion_jordan_wigner(word.n_modes, [list(word.creation_modes)], [list(word.annihilation_modes)], [1.0], [0.0], _effective_max_bytes(None))` and reshape the `(structures, real, imaginary)` return into `tuple((tuple(s), complex(re, im)) ...)`. Post-expansion parity-correction loop (1764–1768) stays in Python.
**Benefit:** Single JW expansion kernel covered by Rust dense-reference tests; protects the phase/ordering invariant; releases the GIL.
**Tradeoffs:** 2 FFI round-trips per raw operand pair; net neutral-to-positive on perf for ≥2-factor words.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M20 — Performance · `python/tencirpauli/structured.py:1074`
**Title:** Cross-type structured multiply uses Python recursive normal ordering per term pair.
**Failure scenario:** When `_StructuredOperator.multiply` falls back to the non-hybrid path, it iterates `product(self._terms, other._terms)` and calls `_multiply_terms` → `_normal_order_fermions` (388–440), an unmemoized O(k!) worst-case Python recursion. Same-type pairs use native `structured_fermion_multiply`/`structured_boson_multiply`. The slow path triggers for mixed-domain pairs (e.g. `FermionOperator × HybridOperator` with matching fermion domain).
**Fix:** Broaden the existing native-dispatch guard to route cross-type pairs through the EXISTING `_native.structured_hybrid_multiply` (no new Rust kernel). Loosen `_hybrid_arrays` annotation from `HybridOperator` to `_StructuredOperator` (it only reads `_terms` and `space`). Keep the Python path only for the single-domain same-type case (notably `QuditWeylOperator × QuditWeylOperator` in a qudit-only space) to preserve its `QuditWeylOperator` return type.
**Benefit:** Removes O(k!) Python recursion and `itertools.product` double loop from a public algebraic API; releases the GIL during expansion. Reuses tested `structured_hybrid_multiply`.
**Tradeoffs:** Must avoid the single-domain same-type case to preserve return type. The proposed_fix's brand-new cross-type kernel is unnecessary and duplicative — use the refined fix.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M21 — Dead-code · `python/tencirpauli/structured.py:388`
**Title:** `_normal_order_fermions` and `_normal_order_bosons` are unreachable dead code.
**Failure scenario:** Both functions are called only from `_multiply_terms` (1631, 1641), which is only called from the base `_StructuredOperator.multiply` fallback (1076). That fallback is only reached for `QuditWeylOperator × QuditWeylOperator` (no multiply override) — whose terms always have `fermion=None`/`boson=None`, so the `if left.fermion is not None` / boson guards (1626, 1636) are always False. ~90 lines of duplicated, untested algorithm that will silently drift from the Rust core.
**Fix:** Delete `_normal_order_fermions` (388–440) and `_normal_order_bosons` (443–~490). Simplify `_multiply_terms`: remove the fermion/boson normal-ordering blocks (1626–1634, 1636–1644), leaving `f_products = {None: 1.0}` / `b_products = {None: 1.0}` unconditionally. Add a comment noting the qudit-only fallback nature and that fermion/boson canonicalization is the native multiply overrides' responsibility.
**Benefit:** Removes ~90 lines of duplicated algorithm violating DRY and the AGENTS.md rule against Python re-implementing Rust-core logic; eliminates a maintenance hazard.
**Tradeoffs:** Minimal — the only way they could become reachable is a future single-domain operator carrying fermion/boson data without a native multiply override; the natural fix would route through `_native_fermion_raw`/`_native_boson_raw`, not revive Python.
**Verdict:** confidence high, recommendation **fix-now**.

> Interaction with M18: if M18 reroutes `_normal_order_fermions` to native, the function is no longer dead — pick one finding to act on. The cleaner resolution is M21 (delete) + M20 (route cross-type to `structured_hybrid_multiply`), which together remove both the dead code and the O(k!) fallback.

---

#### M22 — Dead-code · `crates/tencir-pauli-core/src/charge_sector.rs:225`
**Title:** Unreachable `position == usize::MAX` overflow check in `build_charge_sector_plan`.
**Failure scenario:** Lines 287–291: `if position == usize::MAX { return Err(Overflow { ... }); }` inside `for (position, ...) in ....enumerate()`. `position` is an `enumerate()` index over a `Vec`; reaching `usize::MAX` requires ~2⁶⁴ iterations. Dead code giving a false impression of overflow protection (the real checks are `checked_mul`/`checked_add` on dimensions).
**Fix:** Delete the block at lines 287–291. (Finding's cited lines 225/206–230 are stale; the actual check is at line 287 inside the function starting at 228.)
**Benefit:** Removes misleading dead code.
**Tradeoffs:** None — unreachable, no behavior change.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M23 — Loophole · `crates/tencir-pauli-core/src/charge.rs:202`
**Title:** `apply_qudits` does not validate duplicate-site qudit triples, producing order-dependent phases.
**Failure scenario:** `compile_charge_transitions` validates only `(!term.qudit_present && !term.qudit_triples.is_empty())` — no ordering or duplicate-site check (unlike `structured.rs validate_hybrid_batch` which rejects `pair[0].0 >= pair[1].0`). `apply_qudits` processes triples sequentially, compounding `exp(2πi·b·input/d)` per triple against the *current* occupation. Two triples on the same site (a1,b1) then (a2,b2) yield a Weyl cross-phase `exp(2πi·b2·a1/d)` differing from the canonical combined operator. Silently wrong phase; destination occupation is order-independent so only the phase is wrong.
**Fix:** In BOTH `compile_charge_transitions` (~line 296) and `compile_charge_transitions_from_plan` (~line 518), add per-term canonical validation: `term.qudit_triples.windows(2).any(|p| p[0].0 >= p[1].0)` and out-of-bounds a/b checks, returning `Err(PauliError::NonCanonicalTerms { index })`. Consider factoring into a helper to avoid drift between the two paths.
**Benefit:** Catches non-canonical input that would otherwise produce a silently incorrect phase; fail-fast consistency with the hybrid-term path.
**Tradeoffs:** One O(triples) pass per term at compile time. Previously-accepted-but-wrong inputs now error (intended).
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M24 — Dead-code · `crates/tencir-pauli-core/src/gate.rs:112`
**Title:** No-op `if` block in `GateOperation::rotation` (always-true condition, empty body).
**Failure scenario:** Lines 112–116: `if wire1.is_some() && matches!(axis, RotationAxis::X | RotationAxis::Y | RotationAxis::Z) { /* comment only */ }`. `RotationAxis` has exactly variants X/Y/Z, so the `matches!` is exhaustive and always true; condition collapses to `wire1.is_some()`. Body is comment-only. Two-wire wire validation already happens at 100–102 via `validate_two_wires`.
**Fix:** Delete lines 112–116 entirely. Optionally add a one-line `// two-wire rotation: axis is shared by both local generators` comment if the doc intent is worth preserving.
**Benefit:** Removes misleading dead branch.
**Tradeoffs:** None — no runtime effect.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M25 — Loophole · `crates/tencir-pauli-core/src/propagation.rs:1731`
**Title:** `expectation_of_terms` / `expectation_from_terms` silently take `.re` without validating Hermiticity.
**Failure scenario:** Public `expectation_of_terms` (line 328) folds each term as `term.coefficient.re * local` with no Hermitian guard, unlike `expectation` (173) and `value_and_grad` (191) which return `NonHermitianExpectation`. `from_program` records but does not reject non-Hermitian observables. A direct Rust caller propagating a non-Hermitian observable and calling `expectation_of_terms` gets a plausible but incorrect real number. (The sole live caller, native `profile`, is already guarded at lines 103–107.)
**Fix (non-breaking option b):** Add a doc comment stating the method is only defined for Hermitian observables / terms derived from this engine's propagation, plus `debug_assert!(self.hermitian, "expectation_of_terms requires a Hermitian observable; use is_hermitian_observable() to check");` as the first line. Avoid option (a) (changing signature to `Result`) — it breaks the public API for a case already guarded at the one live call site.
**Benefit:** Fail-fast in debug builds; makes the API contract explicit and consistent with `expectation`/`value_and_grad`.
**Tradeoffs:** One doc comment + one `debug_assert`. No release-build behavior change.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M26 — Dead-code · `crates/tencir-pauli-core/src/propagation.rs:203`
**Title:** `value_and_grad` cutoff expression is a no-op that always equals `max_weight`, diverging from `propagate_dynamic`.
**Failure scenario:** Lines 203–207: `self.is_exact().then_some(None).flatten().or(self.program.max_weight)`. `.then_some(None)` yields `Some(None)` if exact, `None` if not; `.flatten()` yields `None` in both cases; `.or(max_weight)` always returns `max_weight`. The `is_exact()` check is dead code. Diverges from `propagate_dynamic` (367–368) which uses `(!exact).then_some(self.program.max_weight).flatten()` producing `None` when exact. Result: when `max_weight = Some(c >= nqubits)` (exact-but-set), `value_and_grad` passes `Some(c)` while `propagate_dynamic` passes `None` — same term/edge sets (the weight check can't filter), but extra redundant per-term/per-edge weight comparisons.
**Fix:** Replace lines 203–207 with the identical expression from `propagate_dynamic`:
```rust
let exact = self.is_exact();
let cutoff = (!exact).then_some(self.program.max_weight).flatten();
```
**Benefit:** Removes misleading dead code; makes the two paths byte-for-byte identical; eliminates redundant weight comparisons on the reverse pass.
**Tradeoffs:** Pure refactor, no observable output change.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M27 — Numerical · `crates/tencir-pauli-core/src/spps.rs:998`
*(Duplicate of M6; the verifier's two analyses disagree on whether to fix-now or fix-later.)*
**Title:** SPPS standard error uses population (MLE) variance, biased for small `samples_per_term`.
**Failure scenario:** Same as M6. Bias is `sqrt((N-1)/N)` ≈ 0.71× at N=2 (~30% low). Behavior is locked in by the named test `adaptive_standard_error_uses_squared_average_coefficient` (line 1100), so the convention is intentional.
**Fix:** The verifier's second analysis prefers the documentation branch as the primary low-risk fix: add a docstring on `SPPSEstimate.value_standard_error` / `SPPSValueEstimate.value_standard_error` (spps.py:33,48) and a rustdoc note on `combine_fixed`/`combine_adaptive` stating that `value_standard_error` uses the MLE (population) variance estimator and is biased low for small `samples_per_term`. Optionally expose a separate `unbiased_se` helper. Only switch the formula to `(sum_squared - sum*sum/count)/(count-1)` if a downstream audit confirms no consumer relies on the MLE convention; in that case update the pinned test (line 1100–1101: 0.25 → sqrt(0.125)≈0.35355) and add a release note.
**Benefit:** Removes user-facing ambiguity; unbiased SE available for CI-grade use.
**Tradeoffs:** Doc-only is zero-risk; formula switch breaks the pinned test and changes historical reproducibility.
**Verdict:** confidence high, recommendation **fix-later** (doc-only branch); M6 above recommends **fix-now** (formula switch). The two analyses are reconciled by: do the doc fix immediately, gate the formula switch behind a consumer audit.

---

#### M28 — Inconsistency · `crates/tencir-pauli-core/src/spps.rs:1039`
**Title:** `combine_adaptive` uses `budgets[i]` instead of `left_stat.count`/`right_stat.count` for means.
**Failure scenario:** `combine_adaptive` uses `let count = budget as f64;` for left/right means and variances, rather than the actual accumulated `left_stat.count`/`right_stat.count`. Today these are equal (the adaptive loop draws left/right samples over identical ranges), but the coupling is implicit: a future loop change (e.g. resuming a left replicate without re-running right) would silently produce wrong means/variances with no error.
**Fix:** Replace `let count = budget as f64;` with `let left_count = left_stat.count as f64;` / `let right_count = right_stat.count as f64;`, using each for the respective side. The combined term becomes `0.5 * (left_var/left_count + right_var/right_count)`. Do NOT add a count==budget assertion for `total_paths` (line 527 correctly uses `budgets` directly).
**Benefit:** Removes latent silent-correctness hazard; aligns `combine_adaptive` with `combine_fixed` and `gradient_proxy` which already use `.count`.
**Tradeoffs:** Two `as f64` casts; negligible. No current bug.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M29 — DRY · `crates/tencir-pauli-core/src/structured.rs:614`
**Title:** `jordan_wigner_terms` duplicates the expansion loop in `jordan_wigner_word_expansion`.
**Failure scenario:** `jordan_wigner_terms` (614–672) and `jordan_wigner_word_expansion` (731–766) share a near-identical inner expansion loop: same y_coefficient sign convention (create=-0.5i, annihilate=+0.5i), same `right_word` construction with Z-string below `mode`, same `multiply_pauli_codes` + `FxHashMap` accumulation, same zero-filter + sort. Only structural difference: `jordan_wigner_terms` aggregates across input terms, `jordan_wigner_word_expansion` returns per-word expansions. If the JW sign/parity convention ever changes, both must update in lockstep. `jordan_wigner_hybrid_terms` (683–694) already calls `jordan_wigner_word_expansion` per term — the proposed composition is proven in the same file.
**Fix:** In `jordan_wigner_terms`, replace the inner expansion loop (625–655) with `let current = jordan_wigner_word_expansion(n_modes, sequence, max_bytes)?;`. The existing `for (word, value) in current { push_pauli_aggregate(...); }` loop at 656–658 consumes the result unchanged.
**Benefit:** Single source of truth for the JW expansion convention.
**Tradeoffs:** Negligible per-term function-call boundary cost (already paid by the hybrid path). `jordan_wigner_word_expansion` does no independent validation.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M30 — Inconsistency · `crates/tencir-pauli-core/src/structured.rs:1742`
**Title:** `structured_dense_matrix` reports wrong term index on non-finite accumulation.
**Failure scenario:** The accumulation check at line 1742 returns `PauliError::NonFiniteCoefficient { index: 0 }` hardcoded, while the per-term pre-check (1712–1716) and the sibling paths `structured_sparse_matrix` (1651) and `StructuredMvpPlan::apply` (1541) correctly report the loop's `term_index`. The dense loop at 1722 lacks `.enumerate()`.
**Fix:** Change the loop to `for (term_index, (term, &coefficient)) in terms.iter().zip(coefficients).enumerate()` and line 1742 to `return Err(PauliError::NonFiniteCoefficient { index: term_index });`.
**Benefit:** Correct diagnostic term index, matching sparse and MVP paths.
**Tradeoffs:** Trivial — `.enumerate()` on the existing zip.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M31 — Performance · `crates/tencir-pauli-core/src/u1_circuit.rs:348`
**Title:** `pair_maps` and `diagonal_indices` caches in `compile` use std `HashMap` instead of `FxHashMap`.
**Failure scenario:** Lines 348–349 declare `HashMap<(usize, usize), Arc<[...]>>` with std SipHash. Lookup-only caches; iteration order doesn't leak (sorted enumeration at 1359 and `pairs.sort_unstable_by_key` govern output). The rest of the cluster (`sector.rs`, `charge_sector.rs`) uses `FxHashMap` for comparable lookups.
**Fix:** Replace `use std::collections::HashMap;` (line 3) with `use rustc_hash::FxHashMap;`; change the two cache locals (348–349) and the four helper signatures (1284, 1347, 1381, and the `diagonal_op` cache param). Leave the unrelated `HashMap` at line 144 and test-local `HashMap` at 1537 as-is.
**Benefit:** Consistency with `sector.rs`/`charge_sector.rs`; marginal speedup for circuits with many distinct wire pairs.
**Tradeoffs:** Negligible. Same DoS-resistance tradeoff the cluster already accepts.
**Verdict:** confidence high, recommendation **fix-later**.

---

#### M32 — Loophole · `crates/tencir-pauli-core/src/u1_circuit.rs:607`
**Title:** `value_and_grad_from_state` silently produces wrong gradients when the passed state was not produced by the given parameters.
**Failure scenario:** `value_and_grad_from_state(state, observable, parameters)` (607–625) evaluates `parameters` to `values`, then calls `value_and_grad_from_final_state_with_values(state, observable, &values)`, which walks the circuit backward using inverse gates parameterized by `values`. No verification that `state == U(parameters) |initial>`. Mismatched state → silently wrong gradient, no error. The native `NativeU1FinalState.value_and_grad` wrapper is safe (matched state/params); only direct Rust callers of the pub core API are exposed.
**Fix (doc-only, option a):** Add a rustdoc comment stating the precondition: `state` must be the forward-evolved final state from `U(parameters) |initial>` with the same circuit/initial state; the method does NOT re-run the forward pass and does NOT verify the precondition; mismatched state yields silently incorrect gradients. Note `value_and_grad` is the safe variant. Do NOT rename (the `_from_state` suffix matches codebase convention). Do NOT add a re-run check (`U1CircuitPlan` stores no initial state).
**Benefit:** Prevents a future Rust caller from a silent-wrong-gradient bug at zero runtime cost.
**Tradeoffs:** Documentation-only.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M33 — Performance · `crates/tencir-pauli-core/src/u1_circuit.rs:1400`
**Title:** `pair_map` wastes work constructing an invalid candidate state when `particle_number` is 0.
**Failure scenario:** `pair_map` (1377–1454) computes `occupied = sector.particle_number().saturating_sub(1)`. For `particle_number = 0`, `occupied = 0`, the guard `occupied > remaining.len()` doesn't trigger, so it enumerates one combination, builds `words` with `set_bit(wire1, true)` (weight 1), and calls `rank_words` which fails (weight 1 != particle_number 0) — silently dropped by the `if let (Ok, Ok)` guard, producing a correct empty list but only after wasted allocation/error-path calls. The symmetric full-occupation case IS caught early.
**Fix:** Add an early return alongside the existing guard:
```rust
if sector.particle_number() == 0 {
    let empty: Arc<[PairIndex]> = Arc::from(Vec::<PairIndex>::new().into_boxed_slice());
    cache.insert(key, empty.clone());
    return Ok(empty);
}
```
**Benefit:** Avoids wasted enumeration + `rank_words` error path per Swap/Iswap gate on an empty (k=0) sector. Result unchanged.
**Tradeoffs:** Negligible — one branch mirroring the existing pattern.
**Verdict:** confidence high, recommendation **fix-later**.

---

#### M34 — Performance · `crates/tencirpauli-native/src/propagation.rs:110`
**Title:** `profile()` routes through full `propagate()` with unnecessary sort + `to_word` materialization.
**Failure scenario:** `profile` (109–122) calls `self.engine.propagate(values)`, which runs `propagate_dynamic` AND then per final term calls `PackedKey::to_word` (allocating two `Vec<u64>` per term), sorts by `PauliWord::cmp` (O(N log N · nqubits)), and builds the weight histogram. `profile` then calls `expectation_of_terms(&result.terms)` which recomputes per-qubit components from the freshly-built `PauliWord`s. The value and `final_weight_counts` could be obtained directly from `DynamicTerm` keys via `expectation_from_dynamic_terms` and `PackedKey::weight`. For large/wide propagations this is pointless overhead inside the GIL-released region.
**Fix:** Add `PropagationEngine::profile_dynamic(&self, parameters) -> Result<(f64, PropagationStats), PauliError>` that calls `propagate_dynamic`, computes value via `expectation_from_dynamic_terms`, and builds `final_weight_counts` via `term.key.weight(nqubits)` (no `to_word`, no sort). Have native `profile()` call `profile_dynamic` instead of `propagate`.
**Benefit:** Removes O(terms) allocations and an O(terms log terms · nqubits) sort from the diagnostic path; profile scales to wider circuits without allocation pressure.
**Tradeoffs:** One small core method plus ~6 lines of weight-histogram duplication (optionally extract a private `weight_histogram` helper shared with `propagate`). Public `propagate()` API unchanged.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M35 — Inconsistency · `crates/tencirpauli-native/src/structured.rs:446`
**Title:** `structured_dense` skips `complex_coefficients` validation used by sparse/plan.
**Failure scenario:** `structured_dense` (465–469) manually zips `coefficients_re`/`coefficients_im` into `Vec<Complex64>` after a manual length check (455–459); `structured_sparse` (495) and `structured_sparse_plan` (528) call the shared `complex_coefficients` helper. The finding's primary claimed benefit (non-finite rejection at FFI boundary) is FALSE — `complex_coefficients` does NO finiteness check; `validate_structured_inputs` runs inside the GIL-released core call for all three paths equally. The actual benefit is purely cosmetic: identical length-mismatch error messages and DRY-ing the zip/map.
**Fix:** In `structured_dense`, drop the manual length check + zip/map and route through `complex_coefficients`, but reorder the operations-vs-coefficients length check before the move so the code compiles:
```rust
if operations.len() != coefficients_re.len() {
    return Err(PyValueError::new_err("operation and coefficient lengths differ"));
}
let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
```
**Benefit:** Consistent error messages across the three FFI entry points; DRY.
**Tradeoffs:** Trivial. No finiteness-protection gain.
**Verdict:** confidence high, recommendation **fix-later**.

---

#### M36 — Performance · `python/tencirpauli/charge.py:831`
**Title:** `ChargeRestrictedOperator.csr` uses `np.add.at` for indptr histogram.
**Failure scenario:** `np.add.at(indptr, self._plan.rows + 1, 1)` is an unbuffered scatter-add, markedly slower than `np.bincount` for this exact histogram pattern. `self._plan.rows` is already `np.intp` (line 1082).
**Fix:**
```python
indptr = np.bincount(self._plan.rows + 1, minlength=self.dimension + 1).astype(np.intp, copy=False)
np.cumsum(indptr, out=indptr)
```
Do NOT use the proposed `np.concatenate([[0], ...])` form — it incorrectly drops the first histogram bucket.
**Benefit:** Faster CSR construction; `np.bincount` is typically several times faster than `np.add.at`. Bit-identical output.
**Tradeoffs:** One-line change. No readability loss.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M37 — Inconsistency · `python/tencirpauli/grouping.py:1`
**Title:** Module docstring calls the stable grouping API an "explicitly non-measurement-ready prototype".
**Failure scenario:** Line 1: `"""Deterministic QWC grouping and explicitly non-measurement-ready prototype."""`. (1) `group_operator`, `QWCGroupingResult`, `GeneralCommutingGroupingResult` are part of the public exported API (`__init__.py:12-14,97,123`). (2) "non-measurement-ready" only describes `GeneralCommutingGroupingResult` (`measurement_ready: bool = False`); QWC defaults to `measurement_ready: bool = True`.
**Fix:** Replace line 1 with `"""Deterministic QWC and general-commuting Pauli grouping."""`.
**Benefit:** Accurate module docstring; stops mislabeling a stable API as a prototype.
**Tradeoffs:** Docstring-only.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M38 — DRY · `python/tencirpauli/pauli.py:124`
**Title:** Triplicated Pauli IXYZ char↔code lookup table.
**Failure scenario:** `{"I":0,"X":1,"Y":2,"Z":3}` hardcoded three times: `pauli.py:124` (`from_string`), `pauli.py:992` (`_coerce_structure`), and as module constant `_IDENTITY_CODES` in `structured.py:63`. Reverse `"IXYZ"` hardcoded at `pauli.py:171`. If the canonical code assignment changes, three sites must edit in lockstep.
**Fix:** Define `_PAULI_CHAR_TO_CODE` and `_PAULI_CODE_TO_CHAR = "IXYZ"` once in `pauli.py`; replace the local literals; in `structured.py` import and bind `_IDENTITY_CODES = _PAULI_CHAR_TO_CODE` (keep dict form for O(1) membership testing).
**Benefit:** Single source of truth for the IXYZ↔code convention (AGENTS.md canonical invariant); reduces silent drift risk.
**Tradeoffs:** Trivial refactor; no behavior change.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M39 — Performance · `python/tencirpauli/pauli.py:199`
**Title:** `PauliWord.multiply` round-trips through codes instead of packed x/z words.
**Failure scenario:** `multiply` calls `self.to_codes()`/`other.to_codes()` (each a native `pauli_codes` roundtrip), passes codes to `_native.pauli_multiply`, which on the Rust side immediately rebuilds `PauliWord::from_codes` for both operands. The packed `x_words`/`z_words` are already available. `pauli_multiply` also does NOT release the GIL (no `allow_threads`). Established packed-words pattern already exists in `pauli_commutes`/`pauli_symplectic_inner_product`. Live Python-loop call site at `mapping.py:467-478`.
**Fix:** Change `pauli_multiply` to accept packed words and release the GIL:
```rust
#[pyfunction]
pub(crate) fn pauli_multiply(
    py: Python<'_>,
    nqubits: usize,
    x_words_left: Vec<u64>, z_words_left: Vec<u64>,
    x_words_right: Vec<u64>, z_words_right: Vec<u64>,
) -> PyResult<(Vec<u64>, Vec<u64>, u8)> { ... }
```
Update `PauliWord.multiply` and `_native.pyi` accordingly.
**Benefit:** Removes 2 FFI calls + 2 tuple allocations per multiply; releases the GIL around per-qubit CPU work. Aligns with `pauli_commutes`/`pauli_symplectic_inner_product`.
**Tradeoffs:** Small native API + stub change. No other consumer of `pauli_multiply` (grep confirmed).
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M40 — Mismatch · `python/tencirpauli/propagation.py:363`
*(Duplicate of M15.)*
**Title:** `PropagationEngine._parameters` skips finiteness pre-check that `SPPSEngine` applies.
**Failure scenario:** Same as M15. No correctness bug (Rust `validate_parameters` safety net fires), but inconsistent layer/message.
**Fix:** Same as M15 — replace both `_parameters` bodies with `return _coerce_parameters(parameters, self.nparameters)`.
**Benefit:** Uniform Python-layer fail-fast across all four engines.
**Tradeoffs:** Redundant O(n) Python finiteness scan (negligible vs propagation cost).
**Verdict:** confidence high, recommendation **fix-later** (the verifier's second analysis downgrades urgency since no correctness gain). Reconciled with M15: the fix is safe and worth doing; defer only if a correctness-priority sprint is underway.

---

#### M41 — Inconsistency · `python/tencirpauli/propagation.py:179`
**Title:** `rx` docstring style diverges from parallel `ry`/`rz`/`rxx`/`ryy`/`rzz` docstrings.
**Failure scenario:** `rx` (line 179) spells out `exp(-i angle X / 2)`; the other five use the terse "Append a parameterized or static R{Y,Z} gate" / "Append a two-qubit {X-X,Y-Y,Z-Z} rotation" form. No semantic difference — all six accept `angle=` or `parameter=`.
**Fix:** Normalize `rx` to `"""Append a parameterized or static RX gate."""` to match the majority (5 of 6) style. (Alternatively extend all six with the exponential form, but that's more churn.)
**Benefit:** Consistent docstring voice so readers don't infer a nonexistent semantic distinction.
**Tradeoffs:** Docstring-only.
**Verdict:** confidence high, recommendation **fix-later**.

---

#### M42 — Performance · `python/tencirpauli/spps_circuit.py:63`
**Title:** `SPPSCircuitPlan.expectation` computes and discards the full jacobian on every value-only call.
**Failure scenario:** `expectation` calls `native, _ = self._native_parameters(parameters)`. `_native_parameters` (line 40) allocates and fills a `(len(dynamic_angles), nparameters)` jacobian via `_evaluate_angle` per angle, which allocates a length-`nparameters` gradient per call. `expectation` only needs `native` and immediately discards the jacobian.
**Fix:** Add a `_native_values(parameters) -> np.ndarray` helper that builds only the `native` array (optionally also add `_evaluate_angle_value` in `circuit.py` that mirrors `_evaluate_angle` without the gradient allocation). Use it in `expectation`. Keep `_native_parameters` for `value_and_grad`/`value_and_grad_adaptive`.
**Benefit:** Removes O(nangles · nparameters) wasted allocation and linear-algebra work per value-only estimation.
**Tradeoffs:** Minor code duplication of a small loop.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M43 — Inconsistency · `python/tencirpauli/spps.py:24`
**Title:** Default-initial-state sentinel named inconsistently: `_DEFAULT_SPP_STATE` vs `_DEFAULT_ZERO_STATE`.
**Failure scenario:** `propagation.py:280 _DEFAULT_ZERO_STATE = ZeroState()` and `spps.py:24 _DEFAULT_SPP_STATE = ZeroState()` serve identical purpose with different names. "SPP" is a misnomer — the default is the zero state, not an SPPS-specific state.
**Fix:** Rename `_DEFAULT_SPP_STATE` → `_DEFAULT_ZERO_STATE` in `spps.py` (definition line 24 + usage line 65). Optionally import from `.propagation` to eliminate the duplicate `ZeroState()` allocation.
**Benefit:** One consistent name across propagation modules.
**Tradeoffs:** Private-symbol rename; no public API impact.
**Verdict:** confidence high, recommendation **fix-later**.

---

#### M44 — DRY · `python/tencirpauli/structured.py:1795`
*(Duplicate of M19; the verifier's second analysis recommends fix-later with a comment + test rather than full FFI consolidation.)*
**Title:** `_jordan_wigner_word` Python re-implements the Rust `jordan_wigner_word_expansion` kernel.
**Failure scenario:** Same as M19. Conventions match exactly; the only live caller is `_tensor_mapped_fermion_terms`. The full FFI consolidation is more invasive than it appears (`_tensor_mapped_fermion_terms` needs per-word `(codes, coefficient)` tuples, not the batched mapped-Pauli output of the existing binding).
**Fix (refined):** Do NOT delete `_jordan_wigner_word` or add a new single-word native FFI function. Instead: (1) add a cross-reference comment at both `structured.py:1795` and `structured.rs:731` stating the (X=0.5, Y=+0.5j annihilate / -0.5j create, Z-parity-below-mode) convention MUST stay identical; (2) add a differential regression test asserting `_jordan_wigner_word(word)` equals the native kernel output on random multi-mode `FermionWord` instances (the existing tensor_product test at `test_structured_algebra.py:725` only covers 1-mode words).
**Benefit:** Protects the convention against future divergence at CI time without FFI churn.
**Tradeoffs:** Negligible — one comment + one test.
**Verdict:** confidence high, recommendation **fix-later**. (Reconciliation: M19 recommends the FFI swap, M44 recommends comment+test. The lower-risk M44 path is preferred unless the per-word FFI overhead is profiled and found negligible.)

---

#### M45 — Inconsistency · `python/tencirpauli/symmetry.py:252`
**Title:** CSR docstring says "restricted-basis ordering" while dense/coo say "restricted-space ordering".
**Failure scenario:** `dense` (230) and `coo` (239) say "restricted-space ordering"; `csr` (252) says "restricted-basis ordering". Same matrix, same ordering (csr built from same rows/columns/coefficients as coo).
**Fix:** Change line 252 from "restricted-basis ordering" to "restricted-space ordering".
**Benefit:** Removes misleading wording implying CSR uses a different basis ordering.
**Tradeoffs:** Docstring-only.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M46 — Inconsistency · `python/tencirpauli/symmetry.py:31`
**Title:** `Z2SymmetryAnalysis` exposes both `constraint_rank` field and `rank` property with overlapping but undocumented meaning.
**Failure scenario:** `constraint_rank: int` field (line 28) is the GF(2) null-space dimension; `rank` property (31) returns `len(self.generators)` — the count of selected isotropic generators. `rank != constraint_rank` in general (only equal when the null space is already isotropic). The `rank` name is also overloaded across the package: `U1Sector.rank`/`ChargeSector.rank` mean the lexicographic basis-state index. No correctness bug — both quantities are correct.
**Fix (option b, doc-only — do NOT rename):** Expand `constraint_rank` field docstring: "GF(2) null-space dimension of the symmetry constraint matrix; an upper bound on the number of mutually-commuting generators. May exceed `rank` when the null space contains non-commuting vectors." Expand `rank` property docstring: "Number of selected mutually-commuting (isotropic) Z2 generators, i.e. `len(self.generators)`. Distinct from `constraint_rank` (null-space dimension) and from `U1Sector.rank`/`ChargeSector.rank` (basis-state indices)."
**Benefit:** Removes a real readability trap on one object and disambiguates the cross-module overload.
**Tradeoffs:** Doc-only; non-breaking. Renaming `rank` (option a) is a breaking API change for a modest gain — NOT recommended.
**Verdict:** confidence high, recommendation **fix-now**.

---

#### M47 — Inconsistency · `python/tencirpauli/u1_circuit.py:409`
**Title:** `U1Circuit.iswap` angle convention differs from `rz`/`rzz`/`cphase` with no docstring note.
**Failure scenario:** `iswap_matrix(value)` computes `theta = value * PI / 2.0` (normalized [0,1] convention; `iswap(1.0)` is full iSWAP). `rz`/`rzz`/`cphase` interpret `theta` as radians. All share the same `Angle` parameter type. The TensorCircuit integration passes `theta` straight through for all four — a user porting a mixed circuit must know to rescale. None of the six gate methods have any docstring.
**Fix:** Add docstrings to all six `U1Circuit` gate methods stating their angle convention explicitly. For `rz`/`rzz`/`cphase`: "theta is in radians." For `iswap`: "theta is a normalized fraction in [0,1], NOT radians; theta=1.0 yields the canonical iSWAP gate (internally scaled by π/2)." Mirror in `from_qir` / the TensorCircuit integration docstring.
**Benefit:** Prevents silent factor-of-π/2 rescaling bugs when users mix `iswap` with `rz`/`rzz`/`cphase`.
**Tradeoffs:** Documentation-only; no behavior change.
**Verdict:** confidence high, recommendation **fix-now**.

---

## 3. Real but Not Worth Fixing

These are genuine defects where the fix costs more than it buys. Recorded so they are not re-reported.

1. **Native `max_bytes` FFI type split (`usize` vs `u128`) across modules** (`crates/tencirpauli-native/src/symmetry.rs:102` et al.). Real split, but on 64-bit (the only credible target given the 16 GiB default budget) every sensible input behaves identically. Divergence only on 32-bit (non-viable) or for user values > 2⁶⁴ (nonsensical). Mechanical churn across ~22 native signatures for no observable defect on supported platforms. **fix-later** (portability hygiene, not correctness).

2. **`SPPSEngine` constructor rejects non-Hermitian observables; `PropagationEngine` accepts them** (`crates/tencirpauli-native/src/spps.rs:116`). The divergence is justified by API semantics: `PropagationEngine.propagate()` legitimately operates on non-Hermitian operators (no Hermitian check at line 333), so rejecting at construction would break a real use case. A construction-time warning in `PropagationEngine` would noise legitimate `propagate()` users and duplicates the already-exposed `is_hermitian_observable()`. Only the docstring subset is worth doing; the warning subset is not.

3. **`GateTape.nparameters` does not detect slot holes; native rejects later** (`python/tencirpauli/propagation.py:64`). Native `compile_program`/`SPPSEngine::new` already raise a clear `PauliError::InvalidClifford` at engine construction. Only user-visible cost is allocating a parameter array of the wrong size before learning the tape is invalid — recoverable. The proposed fix (make the property throw) is a Python anti-pattern: an introspection property throwing on access is surprising. The safer `validate()` method variant only improves an already-clear error. The finding also incorrectly claims `PropagationCircuit.nparameters` checks holes (it does not).

4. **`qudit_dimension` uses two sentinels (`None` vs `0`) for "no qudits" across `NativeMVPPlan`/`BackendMVPPlan`** (`python/tencirpauli/hamiltonian.py:243`). Cosmetic inconsistency; the two classes are independent frozen dataclasses with no cross-plan forwarding (the finding's `_copy_plan` justification is wrong — line 1497 sits inside the `NativeMVPPlan` branch). No consumer checks both falsy values. Changing `Optional[int]`/`None` to `int`/`0` (or vice versa) touches the paired `weyl_convention` logic in two frozen dataclasses with a small but real risk of regressing Pauli-plan vs direct-Weyl validation. **wontfix**.

5. **Python `_normal_order_bosons` duplicates Rust `boson_rewrite` CCR logic** (`python/tencirpauli/structured.py:443`). Real duplication but low divergence risk (small, stable, +1 contraction sign, straightforward power-counting). The proposed per-pair routing through `_native.structured_boson_canonicalize` violates the coarse-grained FFI rule (one call per term-pair) and creates a hybrid native-boson + Python-fermion path. The hot boson path is already native. If addressed, do it via the M20+M21 structural fix (route cross-type to `structured_hybrid_multiply`, delete both functions together), not per-pair FFI. **fix-later**.

6. **`canonicalize()` and `from_terms()` duplicate aggregate/sort/fold logic** (`crates/tencir-pauli-core/src/operator.rs:35`). Genuine duplication across 3 sites (canonicalize, from_terms, multiply), but they differ in load-bearing ways: `canonicalize` retains per-word input-index lists for `input_to_canonical`; error-index reporting differs (first_input_index vs canonical_index). The proposed `aggregate_sorted` helper does not cleanly accommodate index reconstruction. The duplication has not produced a known divergence. A minimal extraction of just the inner sort-by-bits + fold + non-finite-check (~6 lines) would be far safer but yields much less of the claimed benefit. **fix-later**.

7. **Python `_codes_from_word`/`_word_from_codes` reimplement native `pauli_codes`/`pauli_from_codes`** (`python/tencirpauli/pauli.py:993`). Real duplication, but the proposed fix is risky on the hot path: `_initialize_operator` calls `_word_from_codes` once per Pauli term; replacing with `PauliWord.from_codes` per term makes one PyO3 call per term — exactly the anti-pattern AGENTS.md forbids. The forward direction (`_codes_from_word` → `to_codes`) is safe to swap. The correct resolution is to use `PauliWord.batch_from_codes` (one native call for all terms) for the reverse direction — a more involved refactor. **needs-discussion**.

8. **Triplicated inline `nqubits`/wire validation across propagation modules** (`python/tencirpauli/propagation.py:61`). Genuine duplication, but the proposed fix depends on the prerequisite validator consolidation (M10) which has not landed. The three wire validators have intentionally different semantics (`propagation.py._wire` raises `TypeError` for non-int wires; `circuit.py`/`propagation_circuit.py` raise `ValueError`), and unifying changes observable exception types that tests may rely on (`test_propagation.py:317` asserts `ValueError` for distinctness). **fix-later** (defer until M10 lands; preserve exception-type contracts).

9. **`aggregate_source` destination update uses `binary_search` + `insert`/`remove` on a `Vec`** (`crates/tencir-pauli-core/src/sector.rs`). The proposed fix bundles three changes: (1) destination_active merge — clear correctness-preserving win, O(active+x_support) vs O(active·x_support); (2) `symmetric_intersection_count` and (3) `pauli_z_parity` — the current code already picks the smaller set and binary-searches the larger (O(min(k,active)·log(max))), so a full merge REGRESSES the common sparse-support case (e.g. k=2, active=50: current ~12 ops vs merge ~52). Applying the fix as written trades a real win on the destination update for a likely regression on the intersection helpers. **fix-later** (scope to ONLY the destination_active merge; leave the two helpers unchanged).

10. **`multiply_pauli_codes` allocates a `Vec<u8>` per call in the JW inner loop** (`crates/tencir-pauli-core/src/structured.rs`). The result `Vec` is MOVED into the `FxHashMap` key, so a scratch-buffer variant trades `Vec::with_capacity+push` for a `clone()` — both one allocation, so it's a wash unless paired with an unspecified `get_mut+clone-on-insert` pattern. Implemented naively it's strictly worse on collisions. The genuine, zero-risk win is hoisting `right_word` out of the inner loops (build once per `mode`, mutate `[mode]` per branch), which the finding undervalues. **fix-later** (do the `right_word` hoist; skip the scratch-buffer variant unless paired with the get_mut pattern).

11. **`reverse_frame` rebuilds the `output_indices` hash map for every operation in the reverse pass** (`crates/tencir-pauli-core/src/propagation.rs`). Real inefficiency, but the primary proposed fix (scratch-map reuse via clear-and-rebuild) addresses only the smallest cost component: one already-correctly-sized `FxHashMap` table allocation. It does NOT touch the dominant per-call costs — the N key hashes and the N key clones (the latter being the expensive part for `Wide` keys, 2 heap allocations per output term per call). The higher-value fix (changing `output_indices` to `FxHashMap<&'a PackedKey, usize>` to eliminate clones) is invasive (requires lifetime-lifting `visit_retained_edges`). **fix-later** (if pursued, target the clone via borrowed keys, not the table allocation; benchmark first).

12. **`has_duplicate` is O(n²) via repeated `slice::contains`** (`crates/tencir-pauli-core/src/structured.rs:1004`). The finding's premise is materially false: it claims the disjoint-support fast path is "meant to be linear," but the doc comment at 928–933 states the opposite — the quadratic inversion count is intentionally used for disjoint-but-inverted words. The path has at least two other O(n²) scans (inversion count 978–987, cross-overlap check 968–970), so fixing `has_duplicate` alone does NOT make the path linear. For the short fermion words that dominate real workloads, the `FxHashSet` allocation is slower than the contiguous slice scan. **wontfix** (if long dense disjoint words ever become a measured hotspot, address the whole branch with a Fenwick/merge-sort inversion count, not one of three quadratic scans).

13. **Parameter binding uses a Python per-angle loop instead of a batched native evaluator** (`python/tencirpauli/propagation_circuit.py:140`). Real per-call Python overhead, but the dominant cost for any non-trivial circuit is the native Pauli-propagation engine itself, not the O(angles) binding loop. The proposed fix's premise (needs a new evaluator) is false — `ParameterExprNode` tape plus `evaluate_parameters`/`reverse_parameter_program` already exists in `circuit_ir.rs` and is used by the u1 path; the propagation path simply doesn't wire it up. The fix is moderate-effort with low-severity benefit where native work dominates. **fix-later** (reuse the existing `ParameterExprNode` tape; apply only if profiling shows the binding loop is a measurable fraction of total propagation time).

---

## 4. Refuted Findings

Brief list of false positives to prevent re-reporting:

- **`ProjectedObservablePlan::for_each_transition` allocates `destination_words` per gate application** (`u1_circuit.rs:199`). The allocation exists but is reused across call sites as designed; the finding did not trace all callers.
- **`canonicalize_indices` uses O(n) linear search and remove per index (O(n²) per word)** (`majorana.rs`). The actual `canonicalize_indices` (lines 127–141) does not use a sorted Vec approach; the finding's description of the algorithm is incorrect.
- **`analyze_charge` materializes the full commutator terms just to count them** (`charge.py`). The commutator term count is not obtained via the materialized list; the finding misreads the code path.
- **`pauli_operator_binary`/`scale`/`adjoint`/`is_hermitian` take `Vec<Vec<u8>>` instead of a numpy structure array** (`native/operator.rs`). These functions use `CanonicalizeInput` and owned `Vec`s as designed for the PyO3 boundary; the proposed numpy structured-array change would not improve correctness or perf.
- **`push_aggregate` re-validates coefficient finiteness on every push despite upstream validation** (`structured.rs:1339`). The finiteness check is not in `push_aggregate`; `validate_coefficient` runs upstream and the finding mislocates the re-validation.

---

## 5. Cross-Cutting Recommendations

These systemic patterns suggest process/structural fixes rather than point fixes.

### A. Coarse-grained FFI rule is systematically violated on Python→Rust hot paths
Four separate findings (H4, H5, M12, M39) describe the identical anti-pattern: Python code calls `term.word.to_codes()` per Pauli term inside a loop, issuing one PyO3 roundtrip per term, when `operator._canonical_structures` already holds the identical data as cached tuples. The pattern recurs in `charge._compile_restricted_transitions`, `mapping.map_pauli`, `pauli.tensor_product`, and `PauliWord.multiply`. **Structural fix:** (1) Add a lint or AGENTS.md checklist item that flags any `for term in operator.terms: ... term.word.to_codes()` pattern for replacement with `operator._arrays()`. (2) Consider deprecating the per-term `PauliWord.to_codes()` Python accessor (or marking it internal) so the only sanctioned path is the batched `_arrays()` accessor. (3) The `PauliWord.multiply` finding (M39) further shows the codes round-trip is pointless even at the native boundary — packed `x_words`/`z_words` are the right input shape, matching `pauli_commutes`/`pauli_symplectic_inner_product`.

### B. Algebraic conventions are duplicated across the Python/Rust boundary
Three correctness-critical conventions — JW word expansion (M19/M44), fermion CAR normal-ordering (M18/M21), boson CCR normal-ordering (Real-but-not-worth #5), and the IXYZ code table (M38) — are each maintained in both Python and Rust. AGENTS.md names phase/qubit-ordering consistency as non-negotiable, yet the duplication itself is the drift vector. **Structural fix:** Adopt a rule that any algebraic convention already implemented in `crates/tencir-pauli-core` must not be re-implemented in `python/tencirpauli/`; new Python code must route through the existing native binding, and where a per-word/per-pair adapter is needed, add a coarse-grained native helper rather than a Python loop. The M20+M21 combination (route cross-type multiply through `structured_hybrid_multiply`, delete the dead `_normal_order_*` functions) is the cleanest demonstration of this principle.

### C. `id()`-keyed caches without retained references
Two independent findings (M13, M16) describe the identical aliasing bug in `PropagationCircuit.compile` and `SPPSCircuit.compile`: the cache key uses `id(observable)`/`id(state)` but the cache tuple stores no reference to those objects, so CPython `id()` reuse after GC returns a stale plan for a different operator — silently wrong expectation/gradient. **Structural fix:** (1) Centralize the compile-cache pattern in a small helper that always stores `(key, plan, observable, state)` tuples and documents the retention invariant. (2) Add a regression test that constructs an observable, compiles, deletes the observable, forces GC, constructs a different observable, and asserts the cache does not return the stale plan. (3) Audit any other `id()`-keyed cache in the package for the same hazard.

### D. Determinism/ordering invariants are fragile where they should be structural
The `aggregate` sort (M4) and `PackedKey::Ord` (M4) rely on a qubit-by-qubit `code_at` comparison that is O(nqubits) per comparison, while the final public output is independently re-sorted by `PauliWord::cmp` — meaning the intermediate order is provably irrelevant but the code does not express that. Similarly, `push_aggregate`'s memory bound (H2/M8) recomputes a total from scratch on every push, when a running counter preserves the exact same bound. **Structural fix:** Where an intermediate data structure's order/bound is irrelevant to the final output, encode that fact in the type (e.g. an unordered `Vec` with a documented "re-sorted before emission" contract, or a small `AggregateAccum { map, total }` wrapper). Don't rely on readers inferring that an O(nqubits) comparator is safe because the output is re-sorted elsewhere.

### E. Fail-fast boundary is uneven
Three findings (M9, M15/M40, M23) describe the same gap class: `U1CircuitPlan::compile` doesn't budget the mandatory state vector (fail-late at run); `PropagationEngine._parameters` doesn't run the finiteness check that `SPPSEngine` does (fail-at-native-boundary with a generic message); `compile_charge_transitions` skips the qudit canonical validation that `validate_hybrid_batch` enforces (silent wrong phase). AGENTS.md states the fail-fast rule, but it's applied unevenly across sibling entry points. **Structural fix:** For each family of sibling entry points (compile paths, parameter-coercion paths, term-validation paths), enumerate the validation invariants and ensure every sibling applies the same set at the same layer. A shared `validate_structured_terms`-style helper invoked at every compile boundary would prevent the qudit-triple gap from recurring.

### F. Validator and lookup-table duplication is already drifting
M10 (`_exact_nonnegative` × 5 copies, with `charge.py` already raising a different message) and M38 (IXYZ table × 3 copies) show that copy-paste validators have already diverged in message text — the drift risk is not hypothetical. **Structural fix:** Consolidate the five nonnegative-int validators into `_validation.py` and the three IXYZ tables into a single `pauli.py` constant. Gate future PRs on a grep for `{"I": 0, "X": 1, "Y": 2, "Z": 3}` and `must be a non-negative integer` to catch re-introduction.

### G. SPPS statistic correctness should be contractually documented
M6/M27 reveal that `value_standard_error` is the MLE (population-variance) estimator, biased low by ~30% at the minimum sample count, with no docstring disclaimer. The behavior is locked by a named test. The current disposition is to document and preserve this historical convention; changing the estimator remains deferred until downstream consumers are audited. A separate unbiased helper is also deferred because no current consumer requires it and adding a second statistic without a concrete contract would increase API surface without resolving a present bug.

---

## 6. Post-audit disposition ledger

This ledger records the actual outcome after critically reviewing every item. `Adopted` means the suggested improvement was implemented; `Partially adopted / alternative` means the underlying concern was addressed with a lower-risk design; `Deferred` means the item remains a candidate pending profiling, consumer analysis, or a better-scoped change; `Rejected` means the proposed change is not justified or the finding is false for this codebase. Duplicate findings are marked independently so that all 72 original entries have an explicit status.

### Critical / high findings

- H1 — **Adopted.** Dynamic expectations now reuse the zero-state-aware `expectation_of_key` path.
- H2 — **Adopted.** Structured aggregation carries a running value count instead of rescanning all map values on every push.
- H3 — **Adopted.** Identity fermion/boson factors remain absent in canonical hybrid keys, with a regression test for the canonical-form invariant.
- H4 — **Adopted.** Charge transition compilation consumes the operator's cached code arrays in one coarse-grained path.
- H5 — **Adopted.** Native Pauli mapping consumes cached structures and coefficient arrays without per-term `to_codes()` calls.
- H6 — **Adopted.** QIR auto-discovery reuses a parameter slot for repeated symbols, with regression coverage.
- H7 — **Adopted.** Fermionic annihilation reorder signs are counted correctly in `OperatorSpace.embed`, with a regression test.

### Medium findings

- M1 — **Adopted.** The live charge transition lookup paths use `FxHashMap`/`FxHashSet` while deterministic output maps remain ordered.
- M2 — **Adopted.** Pauli-code diagnostics now state the inclusive `0..3` range explicitly.
- M3 — **Adopted.** Forward Clifford and rotation propagation use in-place/move-based key handling to avoid unnecessary wide-key clones.
- M4 — **Adopted.** Aggregation fuses collection/filtering and uses a packed-word total ordering for the internal sort.
- M5 — **Deferred.** A one-pass U(1) CSR build would retain transition triples and increase peak-memory and `max_bytes` accounting complexity; the extra pass is not changed without a representative memory/latency benchmark.
- M6 — **Partially adopted / alternative.** The MLE standard-error convention is now documented in Rust and Python; the historical formula is intentionally retained pending a downstream consumer audit.
- M7 — **Adopted.** `HybridLayout` now uses the repository's `nqubits` spelling consistently; unrelated public argument names were left unchanged.
- M8 — **Adopted as H2.** The duplicate aggregate-scan finding is covered by the running-counter change.
- M9 — **Adopted.** U(1) compile-time budgeting includes the mandatory state-vector allocation.
- M10 — **Adopted.** Non-negative integer validation is centralized in `python/tencirpauli/_validation.py`.
- M11 — **Adopted.** Majorana coefficient validation reuses the structured finite-complex validator.
- M12 — **Adopted.** Pauli tensor products use cached structures and coefficients rather than repeated native code extraction.
- M13 — **Adopted.** Propagation compile caches retain the observable and state objects, preventing `id()` reuse aliasing.
- M14 — **Adopted in the minimal form.** Hermiticity is exposed and documented; no redundant Python-side guard was added because native scalar APIs already fail clearly.
- M15 — **Adopted.** Propagation parameter coercion now shares the finite-parameter validator with sibling engines.
- M16 — **Adopted.** SPPS compile caches retain their key objects, with regression coverage of cache retention.
- M17 — **Adopted.** The unreachable raw-fermion fallback now fails explicitly instead of carrying dead expansion logic.
- M18 — **Partially adopted / alternative.** The Python CAR fallback was removed together with the dead fallback path; cross-type multiplication is routed through the existing native hybrid kernel instead of adding one FFI call per term pair.
- M19 — **Partially adopted / alternative.** The live tensor-product adapter remains Python to avoid per-word FFI overhead; its convention is cross-referenced and covered by a differential test against the native JW kernel (see M44).
- M20 — **Adopted.** Cross-type structured multiplication dispatches to `structured_hybrid_multiply`, eliminating the recursive Python normal-ordering hot path.
- M21 — **Adopted.** The unreachable Python CAR/CCR normal-ordering implementations were removed; the remaining fallback is explicitly qudit-only.
- M22 — **Adopted.** The impossible `enumerate()`-index overflow branch was removed.
- M23 — **Adopted.** Both charge-transition compilation paths validate qudit presence, site ordering, bounds, and report the correct term index.
- M24 — **Adopted.** The no-op rotation branch was removed.
- M25 — **Adopted.** The Rust term-expectation contract is documented and guarded by a debug assertion for Hermitian observables.
- M26 — **Adopted.** The exact/non-exact cutoff expression is shared semantically between forward and value-gradient propagation.
- M27 — **Partially adopted / alternative.** Same disposition as M6: the MLE estimator is explicitly documented, while a formula change is deferred.
- M28 — **Adopted.** Adaptive SPPS combination uses the accumulated left/right sample counts rather than assuming they equal the requested budget.
- M29 — **Adopted.** The standalone JW term path reuses `jordan_wigner_word_expansion`, and the inner JW adapter hoists its reusable right-hand word.
- M30 — **Adopted.** Dense structured-matrix accumulation reports the actual non-finite term index.
- M31 — **Deferred.** Switching the U(1) caches to `FxHashMap` is a marginal lookup optimization without a measured end-to-end gain; it remains a low-priority follow-up.
- M32 — **Adopted.** `value_and_grad_from_state` documents its caller-supplied-state precondition and its lack of verification.
- M33 — **Adopted.** Empty-particle sectors return before constructing an invalid pair candidate.
- M34 — **Adopted.** Propagation profiling now stays in the dynamic-key representation and avoids public-word materialization and sorting.
- M35 — **Adopted.** `structured_dense` shares the native coefficient-length helper with sparse and plan paths.
- M36 — **Adopted.** Charge-restricted CSR row histograms use `np.bincount` instead of unbuffered `np.add.at`.
- M37 — **Adopted.** The grouping module docstring no longer mislabels the public QWC API as a prototype.
- M38 — **Adopted.** The IXYZ character/code tables are centralized in `pauli.py`.
- M39 — **Adopted.** Pauli multiplication accepts packed words, performs one native call, and releases the GIL around the Rust operation.
- M40 — **Adopted as M15.** The duplicate parameter-finiteness finding is covered by the shared coercion path.
- M41 — **Adopted.** The `rx` docstring now matches the parallel rotation-method style.
- M42 — **Adopted.** Value-only SPPS expectation no longer allocates discarded parameter Jacobians.
- M43 — **Adopted.** The default state sentinel uses the consistent `_DEFAULT_ZERO_STATE` name, and the public estimate docs describe the estimator convention.
- M44 — **Adopted in the lower-risk form.** The Python JW adapter was retained, but Rust/Python convention comments and a multi-mode differential regression test were added instead of a new single-word FFI API.
- M45 — **Adopted.** CSR documentation now says `restricted-space ordering`, matching dense and COO.
- M46 — **Adopted.** `constraint_rank` and the selected-generator `rank` property now explain their distinct meanings without a breaking rename.
- M47 — **Adopted.** U(1) gate docstrings state the radian convention and the normalized iSWAP convention explicitly.

### Real but not worth fixing

- Real #1 — **Deferred.** The `usize`/`u128` native `max_bytes` split is retained for supported 64-bit targets; broad signature churn has no demonstrated user-visible benefit.
- Real #2 — **Partially adopted / alternative.** Hermiticity is exposed and documented through M14, but constructor rejection or warnings were not added because non-Hermitian propagation remains a valid use case.
- Real #3 — **Rejected.** The native construction error for parameter-slot holes is already clear; making an introspection property throw would worsen the Python API contract.
- Real #4 — **Rejected.** The `None`/`0` qudit sentinel difference is cosmetic and the proposed frozen-dataclass changes carry compatibility risk without a consumer problem.
- Real #5 — **Adopted by alternative.** M20/M21 remove the Python normal-ordering fallback from cross-type multiplication and use the coarse-grained native hybrid path; per-pair boson FFI was intentionally not introduced.
- Real #6 — **Deferred.** Aggregation remains separate because canonicalization carries input-index provenance and distinct error-index semantics.
- Real #7 — **Deferred.** Cached structures are used in the hot paths addressed by H4/H5/M12; the reverse conversion remains a local batch-refactor candidate rather than introducing per-term native calls.
- Real #8 — **Deferred.** Wire-validator consolidation waits until the intentionally different exception-type contracts can be preserved.
- Real #9 — **Deferred.** The bundled destination-merge/intersection rewrite was not applied because the intersection change can regress sparse-support cases; any future change will be scoped and benchmarked.
- Real #10 — **Partially adopted / alternative.** The reusable JW `right_word` is hoisted; the proposed scratch-buffer rewrite was rejected as allocation-neutral or worse on collisions.
- Real #11 — **Deferred.** No lifetime-heavy borrowed-key rewrite was made without profiling the dominant wide-key clone cost.
- Real #12 — **Rejected.** The claimed linear fast-path premise is false for the intentionally quadratic disjoint-word algorithm, and an isolated hash-set replacement would likely regress short words.
- Real #13 — **Deferred.** Existing parameter-expression infrastructure was not rewired into propagation without evidence that Python binding is a material fraction of end-to-end runtime.

### Refuted findings

- Refuted #1 — **Rejected as refuted.** The transition buffer is intentionally reused across callers; no change was made.
- Refuted #2 — **Rejected as refuted.** The cited `canonicalize_indices` algorithm is not present in the repository.
- Refuted #3 — **Rejected as refuted.** Charge analysis does not materialize a commutator list merely to count terms.
- Refuted #4 — **Rejected as refuted.** The owned-vector native boundary is intentional; a NumPy structured-array replacement would not improve this path.
- Refuted #5 — **Rejected as refuted.** `push_aggregate` does not perform the alleged repeated finiteness validation; upstream validation is the actual design.

## 7. Validation of the current uncommitted implementation

The following checks were run after the audit changes were applied, before any commit. `conda run -p .conda python scripts/check.py --benchmark skip` passed formatting, clippy, Ruff, mypy, 38 Rust tests, and 292 Python tests. The focused regression set covering Pauli multiplication, structured algebra, embedding signs, QIR symbols, compile-cache retention, qudit sparse targets, and mapping-specific sparse/MVP targets also passed all 62 tests.

For release performance, the Rust `HEAD` source was exported to a temporary directory and recorded as Criterion baseline `audit-head` using the same machine, toolchain, target directory, and benchmark configuration. The current worktree was then compared against it. Propagation showed clear gains: exact 12-qubit and 100-qubit tapes improved by about 31%, weight-projected 128-qubit propagation by about 36%, deterministic value-and-gradient cases by about 3.5–4.9%, and 64-observable batch gradients by about 7.4%. These results directly support H1, M3, M4, M25, M26, and M34.

The Rust symmetry comparison showed U(1) restriction setup broadly stable or improved by roughly 2–5% at representative 16–128-qubit cases, while several tiny materialization cases fluctuated by a few percent. Those materialization implementations are outside the uncommitted audit diff, so the fluctuations are not attributed to this change set and are not used as evidence for a new optimization. Pauli/Hamiltonian microbenchmarks were likewise mostly unchanged, with isolated 1–3% movements in untouched code paths; they do not justify reverting the audit fixes.

Python release A/B comparisons against the saved Phase 7/7.5 baselines passed all 32 structured and 91 Phase 7.5 benchmark cases. Representative gains included native mapping of about 5–13%, restricted CSR materialization of about 13%, restricted apply of about 4%, and structured MVP apply of about 4–9%. The Phase 7 suite now contains 50 cases, including 18 new qudit COO/CSR/native-MVP and Jordan–Wigner/parity/Bravyi–Kitaev COO/CSR/native-MVP cases; the new baseline `phase7-qudit-fermion-sparse-20260804` recorded all 50 successfully. A focused Python boundary test for M39 measured 20,000 256-qubit `PauliWord.multiply` calls at a median of about 615.6 ms on `HEAD` versus 40.7 ms in the current worktree, approximately 15× faster.

The validation conclusion is therefore: the correctness fixes are test-backed, the main claimed performance improvements are reproduced in release-mode A/B measurements, and no material regression attributable to the uncommitted audit changes was found. Small movements in untouched microbenchmarks remain ordinary benchmark noise or separate follow-up candidates rather than grounds to broaden this change set.

*End of report. Findings touching correctness, phase, or qubit ordering (H3, H6, H7, M6, M9, M13, M16) are flagged **[REGRESSION TEST REQUIRED]** above. The implemented correctness fixes have regression coverage where applicable; M6 deliberately retains the historical estimator pending consumer review.*
