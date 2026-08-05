# TenCirPauli Deep Audit — Round 2 (2026-08-04)

> **Scope.** An independent second-pass scan of the **current working tree** (Rust core + PyO3 native + Python), run after the Round‑1 archive (`docs/vibe/audit-report-2026-08-04.md`) was applied. Round‑1 recorded 72 findings; this pass deliberately excludes every item Round‑1 marked **Deferred**, **Rejected**, **Refuted**, or **Real‑but‑not‑worth‑fixing**, and does not re‑raise the **Adopted** fixes (they are already in the tree — a finding only appears here if it is genuinely new, or shows an adopted fix is incomplete/buggy).
>
> **No code was changed.** This is a report only.

## 1. Executive Summary

**Method.** A 20‑agent ultracode workflow: 10 dimension finders (Rust core algebra, propagation, U1/charge/sector, native binding, Python algebra, Python engines, API/stubs, test gaps, performance, cross‑language mismatch) fanned out, each followed by an adversarial verifier that re‑read the cited code in the current tree, checked the exclusion list, and judged whether the fix's benefit clearly exceeds its risk. The workflow returned **17 survivors / 11 dropped**. I then independently re‑anchored every high/medium claim against the source before writing this report (verification notes are inline per finding).

**Counts.**
- Raw round‑2 findings produced by finders: 28
- Survived adversarial verification: 17
- Dropped (false‑positive / in‑exclusion‑list / benefit‑not‑greater‑than‑risk / stale‑lines): 11
- Of the 17 survivors, all 17 passed my final gate (benefit clearly > risk, not in the Round‑1 deferred/rejected ledger).

**Themes new in this pass.**
1. **Phase‑8.5 native MVP surface left loose ends** — dead/orphaned charge FFI entry points, `apply_into` ignoring `max_bytes`, untested lazy‑plan preflight, and an untested parallel‑CSR branch that only a shape‑only benchmark exercises.
2. **Canonical‑form / numerical‑exactness gaps in the qudit paths** — `hybrid_qudit_product` emits `qudit_present=true` with empty triples (breaks aggregation), and the direct‑Weyl MVP apply does not reduce the phase exponent mod `d` before the `f64` cast (wrong amplitudes for `d > 2^26.5`).
3. **Sibling‑parity debt left by Round‑1** — Round‑1 fixed the SPPS plan's discarded Jacobian (M42) but the **Propagation** plan still computes and discards the full Jacobian on every value‑only call; and Round‑1 added qudit canonical validation to `compile_charge_transitions` (M23) but the `map_hybrid_terms` entry point still skips it.
4. **Private‑stub accuracy** — `_native.pyi` annotates ~25 structured/majorana/mapping parameters as bare `object` where the native side expects concrete tuple sequences, and one `apply_lazy` default is missing; strict mypy cannot catch shape/arity mistakes that surface only as opaque FFI `TypeError`s.

**Prioritized action list**
1. **Fix‑now, correctness (regression test required):** qudit canonical‑form (`R2‑2`); Weyl phase mod‑reduction (`R2‑3`).
2. **Fix‑now, test‑miss / fail‑fast:** parallel‑CSR branch correctness test (`R2‑1`); Propagation‑plan Jacobian discard (`R2‑5`); U1 lazy `mvp_plan(max_bytes)` preflight test (`R2‑15`); stub `object`→typed (`R2‑16`); `apply_lazy` stub default (`R2‑17`).
3. **Fix‑later, medium:** batch‑worker overflow blocks tractable projected batches (`R2‑4`); SPPS proxy/converged native‑space mismatch (`R2‑6`); dead charge FFI surface (`R2‑7`).
4. **Fix‑later, low‑risk cleanup / hardening:** `map_hybrid_terms` validation parity (`R2‑8`); `pair_map` symmetry invariant (`R2‑9`); U1 compile budget for Static payloads (`R2‑10`); `map_clifford1` DRY (`R2‑11`); `add_product` lowercase Pauli (`R2‑12`); `_state_payload` ndarray error (`R2‑13`); eager `apply_into` `max_bytes` guard (`R2‑14`).

---

## 2. Confirmed Findings Worth Fixing

> Findings touching correctness, phase, or qubit/canonical ordering are flagged **[REGRESSION TEST REQUIRED]**. The `Verdict` line records my independent re‑check against the current tree.

### Critical / High

#### R2-1 — Test‑miss · `crates/tencir-pauli-core/src/charge.rs:13,56-67`
**Title:** The parallel branch of `apply_charge_csr_into` is never correctness‑checked; the only exerciser is a shape‑only benchmark.
**Failure scenario:** `apply_charge_csr_into` takes the Rayon parallel branch when `parallel && values.len() >= CSR_PARALLEL_TRANSITION_THRESHOLD` (`1 << 19`, line 13/56). The default `tests/` suite never reaches that threshold, so the serial branch runs. The sole place the parallel branch executes is `benchmarks/python/test_native_mvp_resources_benchmark.py::test_generic_charge_large_csr_gather_ab`, which is `performance_large`, lives outside `tests/` (`testpaths=['tests']`), needs `pytest‑benchmark` (not in CI), and only asserts `result.shape == state.shape` and `transition_count == 823680` — never comparing apply output to any reference. Every eager charge `apply()`/`apply_into()` hardcodes `parallel=true`, so a parallel‑only race/wrong‑write/reduction‑order bug ships undetected for all large eager plans.
**Fix:** Add a real correctness test to `tests/`. Reuse the benchmark's `_all_to_all_charge_restricted(16)` (transition_count 823680 > `1<<19`). Build `restricted.mvp_plan(storage='eager')`, then call `apply_with_parallelism(state, 2**63-1, parallel=False)` and `parallel=True` and assert the two arrays are bit‑identical (`np.testing.assert_array_equal`). Both branches share the same per‑row sequential reduction with no cross‑row accumulation, so they must agree exactly; any divergence is a parallel‑only bug. No production change required.
**Benefit:** Closes the only untested correctness branch of the shared CSR execution kernel backing every eager charge apply at scale.
**Tradeoffs:** The n=16 all‑to‑all eager plan is ~12K dimension / ~823K transitions; two applies cost on the order of seconds and ~tens of MB. Mark `@pytest.mark.performance_large` only if the team wants a lean default suite — prefer running by default since the kernel is correctness‑critical.
**Verdict:** confirmed (`CSR_PARALLEL_TRANSITION_THRESHOLD=1<<19` at `charge.rs:13`; branch gate at `:56`; benchmark asserts shape only). Not in Round‑1 ledger. **fix‑now.**

---

#### R2-2 — Correctness · `crates/tencir-pauli-core/src/structured.rs:375-380` **[REGRESSION TEST REQUIRED]**
**Title:** `hybrid_qudit_product` returns `Some([])` when every site cancels to identity, breaking the `qudit_present` canonical‑form invariant.
**Failure scenario:** When the per‑site Weyl product reduces every site to identity (`aa==0 && bb==0`), the `if aa != 0 || bb != 0` guard (line 371) drops all sites from `by_site`, so `triples` is empty — yet line 380 still returns `(Some(triples), phase)`. `finish_hybrid_aggregate` then maps `Some([])` to `qudit_present=true, qudit_triples=[]`, violating the Python `QuditWeylWord` invariant (identity ⟺ empty triples ⟹ absent). Re‑feeding this non‑canonical output into `canonicalize_hybrid_terms` produces `Some([])` vs `None` keys that do **not** merge, so two numerically‑equivalent identity‑qudit terms survive as separate entries and the non‑canonical representation propagates across repeated native calls. Reachable by multiplying two hybrid operators where one term carries `X(a)Z(b)` and the other `X(-a)Z(-b)` on the same site.
**Fix:** In `hybrid_qudit_product`, after computing `triples`/`phase`, return `(None, phase)` when `triples.is_empty()` and `(Some(triples), phase)` otherwise. One‑line change; the identity‑site drop already happens at line 371.
**Benefit:** Restores the canonical‑form invariant (`qudit_present` ⟹ non‑empty non‑trivial Weyl factor) matching the Python representation; eliminates duplicate‑term output when identity‑qudit products mix with qudit‑absent terms.
**Tradeoffs:** One `is_empty` check per qudit product pair, negligible vs the existing cos/sin. No behavior change for canonical input.
**Verdict:** confirmed (line 380 returns `(Some(triples), ...)` unconditionally; `finish_hybrid_aggregate` at ~503‑509 maps `Some(triples)` → `qudit_present=true`). Distinct from Round‑1 M23 (which validated *input* triples; this is *output*). **fix‑now.**

---

#### R2-3 — Correctness · `crates/tencir-pauli-core/src/structured.rs:1925-1927` **[REGRESSION TEST REQUIRED]**
**Title:** Direct‑Weyl MVP/dense/sparse apply does not reduce the phase exponent mod `d` before the `f64` cast — wrong amplitudes for large qudit dimension.
**Failure scenario:** `apply_structured_operation` for the direct‑Weyl kind computes `b_digit = u128::from(operation.q) * digit as u128` (line 1925) then `angle = 2π * b_digit as f64 / local_dimension as f64` (line 1927) **without** first reducing `b_digit` modulo `local_dimension`. The advertised qudit range is `3 <= d <= 2^32-1`. For `d > 2^26.5`, `q*digit` can exceed `2^53` and the `as f64` cast rounds the integer, so the angle (and hence cos/sin) is wrong. Example: `d = 2^28`, a pure‑Z Weyl op with `q = d-1` applied to `digit = d-1` gives `b_digit = (d-1)^2 ≈ 7.2e16 > 2^53`; the rounded angle differs from the true `ω^((d-1)^2 mod d)` by O(1). The symbolic `hybrid_qudit_product` path (line 379) already reduces `phase_exponent` mod `d` before the cast, so the two paths disagree on large‑d operators. Affects `structured_dense_matrix`, `structured_sparse_matrix`, and the MVP builder.
**Fix:** Reduce before the cast: `let exponent = (u128::from(operation.q) * digit as u128) % u128::from(local_dimension);` then `let angle = 2.0 * PI * exponent as f64 / local_dimension as f64;`. After reduction `exponent < d <= 2^32`, exactly representable in f64.
**Benefit:** Makes the direct‑Weyl kernels numerically exact for the full advertised qudit range and removes the inconsistency with the symbolic path.
**Tradeoffs:** One extra `u128 % u128` per Weyl op, negligible vs the transcendental cos/sin. No change for `d <= ~2^26` where the current computation is already exact.
**Verdict:** confirmed (lines 1925‑1927 lack the mod; symbolic path at 379 reduces first). **fix‑now.**

---

### Medium

#### R2-4 — Loophole · `crates/tencir-pauli-core/src/propagation.rs:850,572`
**Title:** `estimate_batch_worker_bytes` overflow rejects tractable projected batches for `nqubits >= 64` with `>= 64` rotations.
**Failure scenario:** `growth_exponent = min(sum_branch_exponents, nqubits)` (line 849). For nqubits=100, 400 rotations, `max_weight=2`: `growth_exponent = min(400, 100) = 100`; `checked_pow_two(100)` (line 930) sees `100 >= usize::BITS (64)` and returns `Err(Overflow)`; `PropagationBatch::new` propagates it via `?` at line 572, so construction fails — even though the projected path bounds actual term growth to a polynomial in nqubits. The single‑observable `PropagationEngine` path never calls `estimate_batch_worker_bytes` and works fine, so batched multi‑observable evaluation is silently unavailable for wide rotation‑heavy projected circuits the single‑engine path handles.
**Fix:** Align construction‑time budget handling with the execution‑time path (`map_observables`, ~692‑734), which already tolerates estimate overflow via `.unwrap_or(usize::MAX)` / `.unwrap_or(1)`. Switch the three call sites in `PropagationBatch::new` (lines 572, ~579, ~586‑598) from `?`/checked to the same saturating pattern; keep the final `check_budget(shared_bytes, program.max_bytes, …)` (line 599) as the single fail‑fast gate. **Do not** cap `growth_exponent` at `max_weight` — `2^max_weight` drastically underestimates the true polynomial term count (max_weight=2 on n=100 bounds ~5000 terms, not 4) and would let through allocations `max_bytes` was meant to guard.
**Benefit:** Restores the batched multi‑observable API for wide projected circuits (a documented weight‑projected Heisenberg use case); removes the construction‑vs‑execution inconsistency while keeping `max_bytes` as a best‑effort guard.
**Tradeoffs:** Three call sites changed in lockstep; reviewer must confirm no intermediate `checked_add` masks a genuinely‑too‑large `shared_bytes` when `max_bytes` is set (it cannot — the final `check_budget` still fires on `usize::MAX > limit`). Single‑engine path unchanged.
**Verdict:** confirmed (construction `?` at 572 vs execution `unwrap_or` at ~692‑734; `checked_pow_two` at 930). Not in Round‑1 ledger. **fix‑later.**

---

#### R2-5 — Performance · `python/tencirpauli/propagation_circuit.py:153-215`
**Title:** `PropagationCircuitPlan` computes and discards the full Jacobian on every value‑only call (`expectation`/`propagate_operator`/`profile`).
**Failure scenario:** `_native_parameters` (line 161) always allocates a `(len(dynamic_angles), nparameters)` float64 Jacobian plus per‑angle length‑`nparameters` gradient vectors via `_evaluate_angle`. `expectation` (177), `propagate_operator` (203), and `profile` (214) all do `native, _ = self._native_parameters(parameters)` and discard the Jacobian. For an outer optimizer calling `expectation` thousands of times on a 50‑gate / 100‑parameter circuit this is pure allocation/fill overhead. The SPPS sibling already avoids it via `_native_values` (added by Round‑1 M42).
**Fix:** Add a `_native_values(parameters)` method mirroring `SPPSCircuitPlan._native_values` (`spps_circuit.py:75‑84`): iterate `self._dynamic_angles` calling the existing `_evaluate_angle_value(angle, values, self.nparameters)` (`circuit.py:206`) into a preallocated float64 array, no Jacobian. Replace the three `native, _ = self._native_parameters(parameters)` sites with `native = self._native_values(parameters)`. Leave `value_and_grad` (191) on `_native_parameters` (it needs the Jacobian).
**Benefit:** Eliminates per‑call Jacobian + per‑angle gradient allocations on the three value‑only terminals; brings the sibling plan to parity with the SPPS fix. The Jacobian was already discarded — no correctness risk.
**Tradeoffs:** Minor duplication of the `_native_values` helper across the two plan classes (or a small shared module function). Pure‑Python prelude before the single batched native call; no FFI/coarse‑graining concern.
**Verdict:** confirmed (`expectation`/`propagate_operator`/`profile` all discard the Jacobian; `_evaluate_angle_value` exists at `circuit.py:206`). Distinct from the deferred Round‑1 "parameter binding batched native evaluator" item (that is about native work dominating; this is about discarded Python‑side allocations). **fix‑now.**

---

#### R2-6 — Mismatch · `python/tencirpauli/spps_circuit.py:103-147`
**Title:** `SPPSCircuitPlan.value_and_grad[_adaptive]` returns `gradient_error_proxy` / `converged` / `term_gradient_error_proxies` in **native**‑parameter space while the returned `gradient` is in **public**‑parameter space.
**Failure scenario:** `circuit.rz(0, 2*Parameter(0)); plan = circuit.compile(obs); est = plan.value_and_grad(params, samples_per_term=64, seed=0)`. The returned `est.gradient[0] = 2 × (native gradient)` via `jacobian.T @ result.gradient`, but `est.gradient_error_proxy` / `est.term_gradient_error_proxies` / `est.converged` are left unchanged from the native `SPPSEstimate` (`replace(result, gradient=gradient)` at 119‑121 and 145‑147 only swaps `gradient`). For nonlinear `ParameterExpr` angles the Jacobian is non‑identity, so a user relying on `gradient_error_proxy` to judge public‑space convergence — or `value_and_grad_adaptive` reporting `converged=True` — gets a numerically wrong threshold check when the native gradient met tolerance but the amplified public gradient does not.
**Fix (doc‑only, option a):** Add a clear note to `value_and_grad` / `value_and_grad_adaptive` docstrings: `gradient_error_proxy`, `term_gradient_error_proxies`, and `converged` refer to the native (pre‑Jacobian‑chain) parameter space and must not be compared directly to the returned (public‑space) `gradient` when angles are nonlinear `ParameterExpr` values; for direct `Parameter` angles (identity Jacobian) the two spaces coincide. **Do not** attempt to transform per‑term proxies (option b) — the per‑term Jacobian structure is not well‑defined in general and risks a silent correctness bug.
**Benefit:** Surfaces a silent mismatch at the API contract; prevents wrong convergence decisions for nonlinear‑expression SPPS users. Most SPPS users use direct `Parameter` angles (identity Jacobian) where there is no issue, so impact is bounded.
**Tradeoffs:** Doc‑only fix informs but does not enforce. Zero code risk.
**Verdict:** confirmed (`replace(result, gradient=gradient)` at 119‑121/145‑147 leaves proxy/converged native). Not in Round‑1 ledger. **fix‑later.**

---

#### R2-7 — Dead‑code · `crates/tencirpauli-native/src/charge.rs:17,107,171` + `charge_sector.rs:103,179-305`
**Title:** Five charge‑transition native entry points have zero Python callers (dead FFI surface) and inline a 4‑copy `ChargeTransitionTerm` construction.
**Failure scenario:** Not a runtime bug today. The live phase‑8.5 path is `compile_mvp` (`charge.py:990`) → `NativeChargeMvpPlan::{apply, apply_into, compile_eager}`. The five exported‑but‑uncalled entry points — `charge_compile_transitions`, `charge_mvp_apply`, `charge_mvp_apply_into` (`charge.rs:17/107/171`), `NativeChargeSectorPlan::compile_transitions` (`charge_sector.rs:103`), and `NativeChargeSectorPlan::apply_lazy` (`charge_sector.rs:202`) — each inline the identical ~30‑line `ChargeTransitionTerm` construction (4 copies) instead of reusing the helper at `charge_sector.rs:47‑61`. A maintainer adding a field to `ChargeTransitionTerm` must update four copies; missing one silently produces defaulted fields on whichever dead path is later revived.
**Fix:** Remove the three standalone `#[pyfunction]`s in `charge.rs`, their `use` import and `wrap_pyfunction!` registrations in `lib.rs`, the two `NativeChargeSectorPlan` methods, and the corresponding `_native.pyi` declarations. Keep `NativeChargeSectorPlan::{rank, unrank, basis_states, compile_mvp}` and `NativeChargeMvpPlan::{apply, apply_into, compile_eager}` (the live path). Leave the core `compile_charge_transitions` (`crates/tencir-pauli-core/src/charge.rs:893`) untouched — it is simply orphaned with no remaining caller. **Note:** if this is adopted, `R2-17` (the `apply_lazy` stub default) is mooted — pick one.
**Benefit:** Removes ~150 lines of unused private FFI plus registrations/declarations; eliminates the 4‑copy `ChargeTransitionTerm` drift hazard; shrinks the surface a future field change must touch from 4 sites to 1.
**Tradeoffs:** `apply_lazy` is a distinct one‑shot lazy path; if a future feature wants it, re‑add as a thin wrapper over the existing core helper. Phase 8.5 is in active flux (`charge.rs`/`charge_sector.rs` already modified in the working tree) — verify no in‑progress branch wires these before deleting. `_native` is private by contract, so out‑of‑tree callers are unsupported.
**Verdict:** confirmed (zero callers across `python/`, `tests/`, `benchmarks/`; only `_native.pyi` declarations match). The Round‑1 M1 note ("live public path wrapped as `charge_compile_transitions`") is **stale** — phase‑8.5 rerouted the live path to `compile_mvp`; this is new evidence, not a re‑raise. **fix‑later.**

---

### Low

#### R2-8 — Correctness · `crates/tencir-pauli-core/src/mapping.rs:343-384`
**Title:** `map_hybrid_terms` skips the boson‑block and qudit‑triple canonical validation that `validate_hybrid_batch` enforces.
**Failure scenario:** `MappingPlan::map_hybrid_terms` validates field lengths, `n_modes` compatibility, fermion‑absence, qubit/mapped code lengths, and coefficient finiteness — but does **not** call `validate_boson_blocks` nor run the qudit‑triple canonical check (site ordering / bounds) that `validate_hybrid_batch` enforces (`structured.rs:414, 433-443`). Non‑canonical boson blocks or qudit triples are accepted, cloned into the `HybridMappingKey`, and emitted verbatim; downstream native multiply on that result applies `hybrid_boson_products` / `hybrid_qudit_product`, which assume canonical input (`boson_block_product` silently overwrites duplicates via `BTreeMap::insert`) → silently wrong coefficients/phases rather than a fail‑fast `NonCanonicalTerms` error.
**Fix:** In the per‑term loop of `map_hybrid_terms` (after line 364), add two inline checks mirroring `validate_hybrid_batch`: `validate_boson_blocks(layout.n_bosons, &batch.boson_blocks[index])?;` and the same qudit‑triple `windows(2)`/bounds check used at `structured.rs:433-443`, returning `PauliError::NonCanonicalTerms { index }`. Keep it inline — do **not** factor a shared helper (speculative abstraction against the minimal‑change guidance).
**Benefit:** Closes the fail‑fast gap between the two hybrid native entry points; satisfies the AGENTS.md fail‑fast non‑negotiable for the native boundary.
**Tradeoffs:** Adds O(triples + blocks) validation per term, trivial vs the transform already done. Practical reachability is narrow (Python validates friendly input first), so this is consistency hardening rather than a hot‑path fix.
**Verdict:** confirmed (no `validate_boson_blocks` call and no qudit‑triple check in `map_hybrid_terms`). Distinct from Round‑1 M23 (different entry point). **fix‑later.**

---

#### R2-9 — Loophole · `crates/tencir-pauli-core/src/u1_circuit.rs:1389-1402`
**Title:** `pair_map`'s sorted cache key silently swaps `PairIndex` semantics across wire orderings — correct only because every pair matrix is symmetric.
**Failure scenario:** `pair_map` caches under the sorted key `(min(wire0,wire1), max(wire0,wire1))`. A block whose first gate is `Iswap(3,5)` builds pairs with `zero_one`/`one_zero` labeled for that order; a later block `Iswap(5,3)` hits the cache and gets pairs whose labeling is swapped relative to the caller's `wire0=5, wire1=3`. `apply_pair_matrix` (1101) writes `state[zero_one] = m00*left + m01*right` etc. This is correct **only** because every `PairMicroOp` (Swap, iSWAP) yields a symmetric matrix (`m01 == m10`); a future non‑symmetric pair op would silently produce wrong amplitudes/gradients with no diagnostic. No existing test exercises mixed wire orderings for the same physical pair.
**Fix:** Add a `debug_assert!` at the top of `apply_pair_matrix` documenting the symmetry invariant the sorted‑key cache relies on: `debug_assert!(matrix.values[0][1] == matrix.values[1][0] && matrix.values[0][0] == matrix.values[1][1])`. Zero‑cost in release. (Dropping the sort / keying by exact `(wire0, wire1)` is a larger change because the compile loop's within‑block grouping would also need normalization; defer until a non‑symmetric op is actually introduced.)
**Benefit:** Documents the invariant the cache silently relies on; fails fast in debug if a future non‑symmetric `PairMicroOp` is added without updating the cache strategy.
**Tradeoffs:** Debug‑only (catches debug builds, not release). Zero runtime cost. Does not fix the hypothetical asymmetric case (which does not exist today).
**Verdict:** confirmed (sorted key at `~1396`; cache.insert at 1464; `apply_pair_matrix` at 1101). Not the deferred Round‑1 M31 (FxHashMap type, not key sorting). Preventive; near‑zero risk. **fix‑later.**

---

#### R2-10 — Performance · `crates/tencir-pauli-core/src/u1_circuit.rs:448-457`
**Title:** `U1CircuitPlan::compile` budget omits the `Static`/`Diagonal` payload allocations cloned into the gates vector.
**Failure scenario:** A `CircuitProgram` with a `Diagonal` gate on 20 wires (payload `2^20 Complex64` ≈ 16 MiB, valid — `validate_gate` imposes no arity cap) compiled with `max_bytes = 32 MiB`: the final `check_budget` (448‑457) sums `basis_bytes + pair_bytes + diagonal_bytes + state_bytes` but never the `Static` payload cloned into the gates vector at ~1344‑1346 and ~1024‑1027. The plan then holds an extra ~16 MiB, pushing real memory over the caller's limit with no error; the overflow surfaces only as RSS/OOM at runtime.
**Fix:** After the gates vector is fully built and before the final `check_budget`, accumulate `static_bytes`: iterate gates, and for each `CompiledU1Gate::DiagonalBlock { operations }` → each `DiagonalOp::Static { wires, payload }` add `payload.len() * size_of::<Complex64>()` (using `checked_mul`/`checked_add`, consistent with the existing pattern at 423‑447). Include `static_bytes` in the final `check_budget` sum. (Arc‑sharing the payload instead of cloning is a `CircuitProgram` API/serialization change — defer.)
**Benefit:** Makes `max_bytes` a faithful guard for the plan's own major persistent compiled allocations, consistent with AGENTS.md's fail‑fast‑for‑excessive‑allocations principle. Currently the Cz/Cphase index cache is counted but the comparable (often larger) `Static` payload is not.
**Tradeoffs:** One extra pass over the gates vector during compile (negligible). This is simple accumulation of a major persistent allocation that is currently entirely absent — **not** the complex/exact‑peak‑RSS accounting AGENTS.md forbids; it matches the existing best‑effort `pair_bytes`/`diagonal_bytes` pattern.
**Verdict:** confirmed (final `check_budget` sum at 448‑457 has no `static_bytes` term). Distinct from Round‑1 M9 (state‑vector bytes, adopted) and deferred M31 (FxHashMap). **fix‑later.**

---

#### R2-11 — DRY · `crates/tencir-pauli-core/src/propagation.rs:1504-1553`
**Title:** `map_clifford1` duplicates the full Clifford1 match table of `apply_clifford1_in_place`.
**Failure scenario:** `map_clifford1` (1504‑1553) inlines a ~45‑line Clifford1 sign/code table that is byte‑for‑byte identical to `apply_clifford1_in_place` (1499). A future one‑side change (e.g. an S/Sdg convention fix) would make the reverse pass (`visit_retained_edges`, uses `map_clifford1`) disagree with the forward pass (`apply_operation`) and SPPS (`run_samples`) — silently wrong adjoints/gradients. The existing `in_place_path_updates_match_allocating_maps` test catches this only at test time, not by construction. The 2‑qubit pair (`map_clifford2` / `apply_clifford2_in_place`) already delegates via `clifford2_local_map`, so the 1‑qubit pair is the lone outlier.
**Fix:** Replace the body of `map_clifford1` with `let mut result = key.clone(); let sign = apply_clifford1_in_place(&mut result, gate, wire); (result, sign)`. `map_clifford1` already does `let mut result = key.clone()` on line 1550, so the refactor adds zero clones/allocations.
**Benefit:** Removes ~45 lines of duplicated table; makes forward/reverse/SPPS Clifford conventions provably identical by construction; brings the 1‑qubit pair in line with the 2‑qubit pair.
**Tradeoffs:** Negligible — `map_clifford1` is reverse‑pass only (not the hottest path), no new allocations, pure refactor, no behavior change.
**Verdict:** confirmed (duplicate table at 1504‑1553; delegation pattern already used by the 2‑qubit pair). Not the deferred Round‑1 `canonicalize/from_terms` duplication (different module). **fix‑later.**

---

#### R2-12 — Inconsistency · `python/tencirpauli/structured.py:2546`
**Title:** `OperatorBuilder.add_product` rejects lowercase Pauli string codes while `from_string`/`_coerce_structure` accept them.
**Failure scenario:** `space.builder().add_product(qubits=[(0,'x'),(1,'z')])` raises `ValueError('Pauli code must be one of I, X, Y, Z')` because `_IDENTITY_CODES` (`= _PAULI_CHAR_TO_CODE`, uppercase‑only) is the membership table and `finish()` does not uppercase the input — while `PauliWord.from_string('xz')` and `_coerce_structure` (`pauli.py:1169`) both call `value.upper()` first and accept the same logical input.
**Fix:** In the `isinstance(code, str)` branch of `OperatorBuilder.finish` (2546‑2549), insert `code = code.upper()` immediately after the `isinstance` check, before the membership test. One line; preserves the strict `ValueError` for genuinely invalid characters.
**Benefit:** Uniform Pauli‑letter acceptance across the three public entry points; removes a surprising fail‑fast boundary that rejects input the sibling paths accept.
**Tradeoffs:** Marginal — accepts lowercase where it previously errored. No correctness impact (the uppercase mapping is the canonical table already used elsewhere). Does not touch numeric‑code or non‑str inputs.
**Verdict:** confirmed (`_IDENTITY_CODES = _PAULI_CHAR_TO_CODE` at `structured.py:67`; `from_string`/`_coerce_structure` upper‑case first). Not in Round‑1 ledger. **fix‑later.**

---

#### R2-13 — Loophole · `python/tencirpauli/propagation.py:606-619`
**Title:** `_state_payload` raises an ambiguous‑truth `ValueError` instead of the documented `TypeError` for a raw ndarray `initial_state`.
**Failure scenario:** `PropagationCircuit(2, initial_state=np.array([1.0,0.0,0.0,0.0]))` constructs without validation. `compile(obs)` later calls `_state_payload(array, 2)`; `if state == "zero" or isinstance(state, ZeroState):` (line 606) — for a multi‑element ndarray `state == "zero"` returns a bool array and `array or …` raises `ValueError: truth value of an array with more than one element is ambiguous` instead of the documented `TypeError` at line 619. The user sees an opaque numpy error rather than `initial_state must be 'zero' or a typed state descriptor`.
**Fix:** Reorder the opening guard to test type before equality: `if isinstance(state, ZeroState) or (isinstance(state, str) and state == "zero"):`. A raw ndarray then falls through to the `TypeError` at 619 with the documented message. (Optionally also validate `initial_state` type in `_CircuitBuilder.__init__` at construction time — better fail‑fast, but the one‑line reorder is the minimal fix.)
**Benefit:** Users who pass a raw state vector get the documented `TypeError` at the boundary; aligns actual behavior with the documented contract and AGENTS.md fail‑fast.
**Tradeoffs:** Trivial reordering; no behavioral change for valid inputs (`ZeroState`, `'zero'`, `ComputationalBasisState`, `ProductBlochState`). A raw array is already outside the documented type contract, so this only improves the error message.
**Verdict:** confirmed (line 606 `if state == "zero" or isinstance(state, ZeroState):` → ndarray truth‑value error; `TypeError` at 619 is the intended contract). **fix‑later.**

---

#### R2-14 — Test‑miss · `crates/tencirpauli-native/src/charge_sector.rs:712-725`
**Title:** Eager charge `apply_into` silently ignores `max_bytes`, diverging from eager `apply()`.
**Failure scenario:** `NativeChargeEagerMvpPlan.apply_into` takes `_max_bytes: u128` and never reads it (712). The eager `apply()` (677) and `apply_with_parallelism` (702) both guard `if output_bytes > max_bytes { return MemoryError }` before computing; `apply_into` skips the guard entirely. A caller passing `max_bytes` below the output size to `apply_into(state, output, max_bytes=small)` expecting a preflight `MemoryError` (matching `apply()`) instead gets a silent success — the output buffer is written regardless. (The strict‑buffer contract — overlap/contiguity/writeable — is already enforced in Python and covered for all plans; the gap is specifically the `max_bytes` guard.) The U1 eager `apply_into` (`symmetry.rs:174`) has the same `let _ = max_bytes;` pattern.
**Fix:** Mirror the eager `apply()` guard in `apply_into`: compute `output_bytes = (self.dimension as u128).checked_mul(size_of::<Complex64>() as u128)` and return `PyMemoryError` if `output_bytes > _max_bytes`, before the `apply_values` call. Add a test: `restricted.mvp_plan(storage='eager').apply_into(state, output, max_bytes=0)` raises `MemoryError`. Apply the same one‑line guard to the U1 eager `apply_into` for cross‑plan consistency.
**Benefit:** Restores `max_bytes` contract parity between eager `apply()` and `apply_into()` (AGENTS.md: `max_bytes` is a best‑effort guard for major outputs). Numerical apply path is already shared with `apply()` and covered; gain is contract coverage, not arithmetic coverage.
**Tradeoffs:** One‑line native change + one‑line test; zero runtime cost. No false‑positive risk — the guard adds the same preflight `apply()` already performs, on the caller‑allocated output size only (not complex transient accounting).
**Verdict:** confirmed (eager `apply_into` ignores `_max_bytes` at 712‑725; eager `apply` guards at 677‑681; U1 sibling has the same pattern at `symmetry.rs:174`). Genuinely new in phase 8.5 (commit `a534e0d`). **fix‑later.**

---

#### R2-15 — Test‑miss · `python/tencirpauli/symmetry.py:350-355`
**Title:** U1 lazy `mvp_plan(max_bytes)` preflight `MemoryError` path is untested.
**Failure scenario:** `U1RestrictedOperator.mvp_plan` (350) calls `_check_allocation(int(self._native_lazy_plan.estimated_bytes), max_bytes, 'U1 MVP plan')` for the lazy branch, raising `MemoryError` when `max_bytes` is below the lazy plan estimate. No test asserts this. `test_cached_eager_plan_still_obeys_call_budget_and_exposes_csr_storage` covers the charge eager `mvp_plan(max_bytes=0)` `MemoryError`, and `test_u1_lazy_apply_into_accounts_for_native_scratch` covers `apply_into(max_bytes=0)`, but the U1 lazy `mvp_plan(max_bytes=0)` preflight is covered by neither. A refactor removing the `_check_allocation` call (or changing its argument) would pass all existing tests while breaking the preflight — callers passing `max_bytes` below the true lazy plan size would get a plan handle instead of `MemoryError`, failing later inside `apply`.
**Fix:** Add one assertion mirroring the charge test: `with pytest.raises(MemoryError): operator.restrict_charge(tcp.U1Sector(8,1)).mvp_plan(max_bytes=0)`. Place it adjacent to `test_u1_lazy_apply_into_accounts_for_native_scratch` in `tests/test_native_mvp_resources.py`.
**Benefit:** Locks in the new U1 lazy plan preflight so the `max_bytes` guard stays consistent with the charge plan's and survives future refactors.
**Tradeoffs:** Negligible — one short test, no production change, reuses an adjacent fixture.
**Verdict:** confirmed (preflight at `symmetry.py:350‑355`; grep of `tests/` shows no such assertion). Genuinely new in phase 8.5 (commits `2ca1d72`/`a534e0d`). **fix‑now.**

---

#### R2-16 — Inconsistency · `python/tencirpauli/_native.pyi` (multiple lines)
**Title:** `_native.pyi` annotates ~25 structured/majorana/mapping operation, factor, index, and structure parameters as bare `object` where the native side expects concrete tuple/int sequences.
**Failure scenario:** The stub annotates `operations`/`factors`/`indices`/`input`/`left`/`right`/`structures`/`creation`/`annihilation` of `structured_dense` (180), `structured_sparse` (187), `structured_sparse_plan` (194), `structured_fermion_canonicalize` (201), `structured_boson_canonicalize` (226), `structured_fermion_jordan_wigner` (218‑219), `majorana_canonicalize` (343), `majorana_multiply` (350/353), `majorana_to_fermion` (360), `fermion_to_majorana` (372‑373), and `NativeMappingPlan.transform` (26) as bare `object`, even though the native pyfunctions accept concrete `Vec<Vec<(usize, u8, u32, u32)>>`, `Vec<Vec<(usize, u8)>>`, `Vec<Vec<u64>>`, `Vec<Vec<u32>>`, `Vec<Vec<u8>>`. Other structurally analogous stubs in the same file **do** use precise tuple annotations (`pauli_propagation_engine` `operations`, `clifford_operations`, `pauli_incompatibility_edges`), so the loose `object` is an inconsistency, not a convention. A caller passing a wrong‑shape tuple (e.g. a 3‑tuple instead of the required 4‑tuple `(axis, kind, p, q)`) passes mypy silently under `object`; the error surfaces only as an opaque PyO3 arity `TypeError` inside the FFI boundary.
**Fix:** Tighten the affected stub parameters to mirror the native shapes (pure stub change, zero runtime effect):
- `structured_dense`/`sparse`/`sparse_plan` `operations` → `Sequence[Sequence[tuple[int, int, int, int]]]`.
- `structured_fermion_canonicalize`/`structured_boson_canonicalize` `factors` → `Sequence[Sequence[tuple[int, int]]]`.
- `majorana_*` `indices`/`left_indices`/`right_indices` → `Sequence[Sequence[int]]`.
- `structured_fermion_jordan_wigner`/`fermion_to_majorana` `creation`/`annihilation` → `Sequence[Sequence[int]]`.
- `NativeMappingPlan.transform` `structures` → `Sequence[Sequence[int]]`.

Leave `transform_hybrid input` (12‑element heterogeneous `HybridInput`) and the `*_array` functions that accept raw NumPy buffers as `object`.
**Benefit:** Lets strict mypy catch arity/shape mismatches in the Python wrappers; turns runtime FFI `TypeError`s into compile‑time type errors. The in‑repo typed helpers (`_native_term_arrays`/`_native_factor_arrays`) already produce precisely‑typed lists, so they pass unchanged; only genuinely wrong‑shaped direct calls are newly rejected. Also makes the private stub self‑documenting for the operation schema `(axis, kind, p, q)` and factor schema `(index, code)`.
**Tradeoffs:** Pure stub change, zero runtime effect. `_native` is private (AGENTS.md), so in‑repo callers go through the typed helpers (grep‑confirmed). Any downstream code passing a type‑checker‑incompatible‑but‑runtime‑valid sequence would newly fail mypy and need a cast, but no such caller exists in‑tree.
**Verdict:** confirmed (~25 `object` params across the listed lines; sibling stubs use precise tuples). Not in Round‑1 ledger. **fix‑now.**

---

#### R2-17 — Inconsistency · `python/tencirpauli/_native.pyi:112`
**Title:** `apply_lazy` stub omits the default for `fast_fermion_particles` that the native signature provides.
**Failure scenario:** A type‑checked caller invoking `NativeChargeSectorPlan.apply_lazy` while omitting `fast_fermion_particles` (which the native PyO3 signature permits, defaulting to `None` per `charge_sector.rs:205`) is flagged by mypy as a missing required argument, even though the call is valid at runtime. The stub also disagrees with its sibling `compile_mvp` at `_native.pyi:134`, which correctly writes `fast_fermion_particles: int | None = ...`.
**Fix:** Change line 112 from `fast_fermion_particles: int | None,` to `fast_fermion_particles: int | None = ...,`. One‑token, mechanical edit.
**Benefit:** Stub faithfully reflects the runtime API; mypy no longer falsely rejects valid calls; restores intra‑class consistency with `compile_mvp`.
**Tradeoffs:** No runtime impact today (the only live caller, `charge.py:1009`, passes the argument positionally). Pure type‑checker/stub consistency fix; zero behavioral risk.
**Verdict:** confirmed (native `fast_fermion_particles=None` at `charge_sector.rs:205`; stub at `_native.pyi:112` omits the default; `compile_mvp` stub at 134 has `= ...`). **Note:** if `R2-7` is adopted (delete the dead `apply_lazy`), this finding is mooted — pick one. **fix‑now** (standalone) / mooted‑by‑`R2-7`.

---

## 3. Dropped Findings (not worth re‑reporting)

Recorded so they are not re‑raised. Reasons: `false-positive`, `in-exclusion-list`, `already-fixed`, `benefit-not-greater-than-risk`, `stale-lines`.

1. **Lazy U1 operator CSR/COO/dense materialization redundantly re‑canonicalizes and re‑compiles terms** (`python/tencirpauli/symmetry.py:380`) — *benefit‑not‑greater‑than‑risk* (the lazy→eager transition is by design; re‑canonicalization is the cost of deferred materialization, not a removable duplication).
2. **`U1RestrictedOperator.estimated_bytes` undercounts for lazy‑built operators by omitting `transition_count`** — *benefit‑not‑greater‑than‑risk* (best‑effort guard; tightening risks the exact‑accounting AGENTS.md forbids).
3. **Eager U1 restriction skips the charge‑conservation pre‑check that lazy runs, diverging failure modes** — *benefit‑not‑greater‑than‑risk* (the two paths have intentionally different validation timing; eager input is already canonicalized upstream).
4. **`_multiply_mapped` Python re‑implements Rust `multiply_pauli_codes` (IXYZ product table duplication)** (`structured.py:1632`) — *benefit‑not‑greater‑than‑risk* (small, stable table; routing per‑pair through native violates the coarse‑grained‑FFI rule).
5. **Majorana `_canonicalize_indices`/`_multiply_canonical` duplicate Rust `canonicalize_indices`/`multiply_words`** — *benefit‑not‑greater‑than‑risk* (small, stable; per‑pair FFI not justified).
6. **`PropagationCircuit.from_circuit` skips parameter‑slot contiguity validation that `from_qir` enforces** — *in‑exclusion‑list* (contiguity/param‑slot validation is covered by Round‑1 H6 and the deferred `GateTape.nparameters` item; `from_circuit` is the documented fast path).
7. **`NativeChargeEagerMvpPlan.apply_with_parallelism` is dead public surface with no caller or test** — *in‑exclusion‑list* (subsumed by `R2-1`: that method is the cleanest way to exercise the serial branch above threshold for the parity test, so keep it and add the test rather than removing it).
8. **`MajoranaOperator.__init__`/`_from_native` overwrite duplicate canonical words instead of summing, unlike `add()`** — *benefit‑not‑greater‑than‑risk* (construction‑time overwrite is the documented fast path; `add()` is the accumulation API).
9. **`PauliOperator.from_code_arrays` rejects an empty Python list of structures with a confusing 'not rectangular' error** — *benefit‑not‑greater‑than‑risk* (edge‑case message quality; empty‑operator construction has a dedicated factory).
10. **U1 lazy `apply_into` scratch budget only tested at `max_bytes=0` failure boundary, never at a tight success boundary** — *benefit‑not‑greater‑than‑risk* (failure boundary is the contract that matters; a tight‑success test adds brittleness without catching a distinct bug class).
11. **Charge transition term‑building + length validation inlined in three sites instead of reusing `charge_terms_from_inputs` helper** — *benefit‑not‑greater‑than‑risk* (the three sites differ in input shape; folding them risks the speculative‑abstraction AGENTS.md rule; the drift hazard is addressed by `R2-7`'s deletion of the dead copies).

---

## 4. Cross‑Cutting Observations

### A. Phase‑8.5 left dead/orphaned FFI and untested guard paths in its wake.
`R2-7` (dead charge FFI), `R2-14` (`apply_into` ignoring `max_bytes`), `R2-15` (untested U1 lazy preflight), and `R2-1` (untested parallel‑CSR branch) are all phase‑8.5 artifacts (commits `a534e0d`/`2ca1d72`). They share a shape: a new MVP execution path was wired as the live route, leaving the *previous* native entry points orphaned and the *new* path's guard/parallel branches under‑tested. **Structural suggestion:** when a phase reroutes the live path (here `compile_transitions` → `compile_mvp`), add a checklist step to (1) delete the orphaned native surface in the same change or mark it explicitly retained, and (2) add a parity test for every new budget/parallel branch the new path introduces.

### B. Round‑1 left sibling‑parity debt.
Round‑1 fixed the SPPS plan's discarded Jacobian (M42) and added qudit canonical validation to `compile_charge_transitions` (M23). The **siblings** — `PropagationCircuitPlan` (`R2-5`) and `map_hybrid_terms` (`R2-8`) — were missed. The same pattern explains `R2-6` (SPPS proxy in native space after Round‑1's gradient‑chain work). **Structural suggestion:** when adopting a fix to one entry point in a family (compile paths, parameter‑coercion paths, term‑validation paths, value‑only plan methods), enumerate the siblings in the same change and either apply the same fix or explicitly defer with a tracked item — otherwise the debt resurfaces as a "new" finding next pass.

### C. The private stub has drifted from the native signatures.
`R2-16` (~25 `object` params) and `R2-17` (missing default) show `_native.pyi` no longer faithfully reflects the PyO3 signatures for the structured/majorana/mapping/charge surface. Because `_native` is private and the in‑repo callers go through typed Python helpers, the drift is invisible until a caller reaches into `_native` directly. **Structural suggestion:** add a small `mypy --strict`‑against‑`_native.pyi` step that type‑checks a representative direct call for each binding (or a stubgen diff check) so signature drift fails CI rather than accumulating.

---

## 5. Verification & Final Gate

**Workflow.** 10 finders × 10 adversarial verifiers (20 agents, ~1.63M tokens, 733 tool calls). Verifiers re‑read cited code in the current tree, checked the Round‑1 exclusion ledger, and judged benefit vs risk; they dropped 11 of 28 raw findings.

**My independent final gate.** After the workflow returned 17 survivors, I re‑anchored every **high/medium** claim (`R2-1` through `R2-7`) and a sample of the low claims against the source before writing this report — verification notes are in each finding's `Verdict` line. Specifically I confirmed in the current tree: `structured.rs:380` returns `Some(triples)` when empty (`R2-2`); `structured.rs:1925-1927` lacks the mod‑reduction (`R2-3`); `propagation.rs:572` uses `?` where the execution path uses `unwrap_or` (`R2-4`); `propagation_circuit.py` discards the Jacobian on the three value‑only methods (`R2-5`); `spps_circuit.py:119-121,145-147` leaves `gradient_error_proxy`/`converged` native (`R2-6`); `charge.rs:13,56` parallel branch + shape‑only benchmark (`R2-1`); `propagation.py:606` ndarray truth‑value error (`R2-13`); `_native.pyi` `object` params and missing `apply_lazy` default (`R2-16`/`R2-17`).

**Final gate conclusion.** All 17 survivors clear the bar the task set — the proposed fix's benefit is **clearly greater than its risk/cost** in each case — and none duplicate an item the Round‑1 archive marked **Deferred**, **Rejected**, **Refuted**, or **Real‑but‑not‑worth‑fixing**. The two findings that touch phase/canonical ordering (`R2-2`, `R2-3`) are flagged **[REGRESSION TEST REQUIRED]**; `R2-1` and `R2-15` are test‑only and carry zero production risk; the low‑severity cleanups (`R2-8`…`R2-14`) are each one‑line or one‑test changes with no behavior change on valid input.

**Interaction note.** `R2-7` (delete dead `apply_lazy`) and `R2-17` (fix the `apply_lazy` stub default) address the same surface — adopt `R2-7` and `R2-17` is mooted; otherwise adopt `R2-17` standalone. Do not act on both.

*No source code was modified. This document is the deliverable.*

---

## 6. Structural Improvements Worth Discussing (Not Action Items)

> **Status: discussion only.** The following are **not** findings from this audit and are **not** pending fixes. They are cross‑cutting observations about how the same *classes* of issue keep recurring across audit rounds — recorded here so an owner can decide whether and when to invest. None of them conflict with frozen owner decisions (`architecture.md` §5 non‑goals, S1–S4, the `max_bytes` best‑effort contract, the no‑wall‑time‑CI policy); they are about *process and guardrails*, not about changing shipped semantics.

Two audit passes (Round‑1 archive + this round) have now surfaced the same patterns repeatedly: Python↔Rust convention duplication, sibling‑entry‑point drift, dead surface left after a reroute, and untested fail‑fast/parallel branches. Each individual finding was real and worth fixing; the *recurrence* is what justifies a structural look. The six suggestions below are ordered by rough benefit‑to‑cost ratio; the first three are the highest‑leverage.

### S‑A. Cross‑implementation differential tests for algebraic conventions (highest leverage)
Round‑1 (cross‑cutting B, M19/M44) and Round‑2 (`R2-2`, `R2-3`, plus the dropped `_multiply_mapped`/Majorana DRY items) all share one root: JW expansion, CAR/CCR normal ordering, the IXYZ code table, and Weyl phase conventions are each implemented in *both* Python and Rust, and the only thing keeping them in sync is human review. AGENTS.md names phase/qubit‑ordering consistency non‑negotiable, yet nothing *machine‑enforces* it. The cheapest structural protection is a cross‑implementation differential test per convention — assert, on random multi‑mode/word inputs, that the Python reference equals the native kernel bit‑for‑bit (Round‑1's M44 already proved this pattern for JW). This is strictly cheaper and safer than routing per‑pair through native (which violates the coarse‑grained‑FFI rule). **Discussion point:** which conventions warrant the test (JW, CAR, CCR, IXYZ table, Weyl phase, `canonicalize`), and whether to gate the suite in CI.

### S‑B. A three‑line PR/acceptance checklist to stop sibling drift (highest leverage, zero tooling)
A clear recurring shape is "fix one entry point, miss its sibling": Round‑1 fixed the SPPS plan's discarded Jacobian (M42) but missed the Propagation plan (`R2-5`); Round‑1 added qudit canonical validation to `compile_charge_transitions` (M23) but missed `map_hybrid_terms` (`R2-8`); Round‑1 touched the SPPS gradient chain but left `gradient_error_proxy`/`converged` in native space (`R2-6`). Separately, Phase 8.5 rerouted the live path to `compile_mvp` and left five dead native entry points (`R2-7`), and new budget/parallel branches went untested (`R2-1`, `R2-14`, `R2-15`). Three one‑sentence checks per PR would catch the whole class:
1. **Sibling enumeration** — when touching one entry point in a family (compile paths, value‑only plan methods, term‑validation boundaries), list the siblings; fix together or record a tracked deferral.
2. **Orphan removal on reroute** — when rerouting the live path, delete the orphaned path in the same PR or annotate why it is retained.
3. **Guard/branch coverage** — every fail‑fast guard and each branch of `if/else` on a hot path must have a test that triggers it (including the non‑taken side).
This is the single change most likely to *prevent* the next audit round from finding the same kinds of issues. **Discussion point:** whether to encode this in `AGENTS.md` or a `docs/vibe/review-checklist.md`.

### S‑C. Protect the propagation advantage with a same‑machine regression baseline
Propagation is the project's largest measured advantage over TensorCircuit/JAX (SPPS 6.5–7.6×, deterministic 18–44× at 20–32 qubits). Round‑1's performance fixes (H1, M3, M4, M34) are backed by release A/B, but several related optimizations are *deferred, not resolved* (`reverse_frame` borrowed‑key rewrite, `aggregate_source` single‑pass, U1 cache→`FxHashMap`) and will stay latent until someone profiles. AGENTS.md rightly forbids wall‑time CI gates, but an *informational* same‑machine Criterion/`benchmarks/run.py record` baseline, run periodically against a stored reference, would surface a silent regression (e.g. a refactor re‑introducing a per‑term clone) before it reaches users. **Discussion point:** cadence, which workloads, and who owns re‑recording the reference.

### S‑D. Tighten fail‑fast and parallel‑path test coverage
Round‑2's "evident test miss" findings cluster in two areas: budget/parallel guards (`R2-1`, `R2‑14`, `R2‑15`) and wrong error *types* on invalid input (`R2‑13`). The suite is large (326 Python / 39 Rust) but skews toward happy‑path correctness. Two cheap improvements: (1) a `fail‑fast` matrix that, for each AGENTS.md‑named fail‑fast case (unsupported gate, invalid dim, incompatible word length, excessive allocation, missing dependency), asserts both that it errors *and* with the documented exception type; (2) a serial‑vs‑parallel parity test as a standard for every Rayon path (CSR apply, batch propagation, Hamiltonian row build) — `R2-1`'s `assert_array_equal(parallel, serial)` is the template. **Discussion point:** whether these belong in the default suite or a marked `performance_large`/`slow` subset.

### S‑E. Differential coverage at the *advertised* range bounds
`R2‑3` (Weyl phase wrong for `d > 2^26.5`) is a correctness bug that survived because the reference didn't sample near the top of the *advertised* qudit range (`3 <= d <= 2^32‑1`). The same gap class exists wherever a public quantity declares a range (`nqubits < usize::BITS`, the 63/64/65 and 127/128/129 boundaries already tested for U1, etc.). A standing rule — "for every advertised range, the differential test samples at and near both bounds" — would have caught `R2‑3` at Phase 7. **Discussion point:** whether to formalize this as a test‑authoring convention in `AGENTS.md`.

### S‑F. Classify the structured‑algebra domains by support tier
Phase 7 expanded scope substantially (fermion + boson + hybrid + direct‑Weyl + Majorana + JW/parity/BK mappings). Two of this round's correctness findings (`R2‑2`, `R2‑3`) and several Round‑1 items (H3, M18/M21) sit in this domain, suggesting its test/reference density is not keeping pace with its breadth. An owner decision that explicitly tiers the domains — first‑class contract (likely fermion + JW, given Hubbard/chemistry workloads) vs. research stub with documented range limits — would let the test investment follow the workload rather than spreading thin across every surface. This is consistent with `architecture.md` §5 (the non‑goals are already clear); it only asks that *implemented‑but‑research* sub‑domains be labelled, not that anything be removed. **Discussion point:** which domains are first‑class, and what "research stub" labeling looks like at the `__init__`/docstring level.

### Priority if only a subset is pursued
1. **S‑B** (PR checklist) — cheapest, most direct structural prevention.

2. **S‑A** (convention differential tests) — directly guards the project's named non‑negotiable invariant.
3. **S‑C** (propagation regression baseline) — protects the largest competitive moat.

`S‑D`/`S‑E` are medium‑term test‑density investments; `S‑F` is an owner‑decision‑level scoping question. None of the six are requests for action in this report; they are recorded so a future owner can choose to invest, and so the next audit can reference them rather than re‑deriving the pattern from scratch.

## Remediation closure (2026-08-05)

All 17 surviving Round‑2 findings are closed in the current implementation. The high-severity canonical and numerical findings have regression coverage for hybrid qudit identity aggregation and large-dimension Weyl phase reduction; the parallel CSR branch, lazy U1 budget preflight, and eager `apply_into` budget parity are covered by focused tests. The remaining production hardening covers propagation value-only parameter evaluation, batch estimate overflow handling, hybrid mapping validation parity, U1 static-payload accounting and pair-map invariants, Clifford-map deduplication, lowercase Pauli input, invalid ndarray state diagnostics, charge/U1 output guards, and precise private native stubs. The dead charge FFI entry points were removed, so R2-17 is intentionally moot rather than independently changed.

The Phase 8.5 second-round acceptance items SR1–SR3 are also closed: generic eager transition compilation releases the GIL, dense/COO/CSR materialization performs target-budget preflight before uncached construction, and the generic aggregation benchmark uses a mixed-domain fixture with an explicit `term_direct` assertion. `conda run -p .conda python scripts/check.py --benchmark skip` passed Black, Ruff, strict mypy, Clippy, release maturin build, 41 Rust tests, 331 Python tests, and 10 doctests; the affected focused suite passed 153 tests. Benchmark results remain informational and machine-specific as required by the project policy.
