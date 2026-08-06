# Phase 9 Implementation and Performance Review, 2026-08-06

Status: open remediation report. Phase 9 is not acceptance-closed at commit `2937537`.

## 1. Review scope and verdict

This review covers commits `b7f51a0` (`feat: close native lazy operator remediation`) and `2937537` (`feat: implement phase 9 native data plane`) against their common pre-change baseline `3b9c58b`, with `docs/vibe/phase-9-spec.md` as the frozen acceptance contract. The review inspected the public Python facades, private PyO3 ABI, Rust core ownership of scalable work, ordinary production call paths, tests, and release-mode benchmarks.

The two commits make substantial and useful progress: ordinary Pauli and structured algebra is now predominantly lazy and native-backed; the main mapping, conversion, terminal-compilation, Z2, Pauli U1, and propagation-observable paths use handles; flat NumPy read-back exists for the principal operator and matrix/vector terminals; `Fraction` and `_native_data` are gone; and the full correctness/quality gate passes. The current lazy BCH paths are also materially faster than deliberately eager term materialization.

Phase 9 nevertheless cannot be closed. Native embedding is absent; structured/generic charge work still serializes operator terms in Python; QWC sample reconstruction remains Python; propagation and SPPS do not cache a native gate-tape handle; canonical equality/hash and some scalar queries still materialize; mapping plans eagerly retain a parallel Python representation; completed native capabilities still carry unreachable array/Python fallbacks; and post-operation non-finite-result defenses prohibited by the specification remain widespread. Two representative performance regressions are independently reproducible: generic fermion charge restriction and deterministic propagation value-and-gradient.

The benchmark answer is therefore: there is no credible repository-wide slowdown, and several important paths improved by roughly one order of magnitude, but the benchmark set is not regression-free. The two localized regressions in Sections 5.1 and 5.2 must be fixed before Phase 9 acceptance.

## 2. Verification performed

- Compared release-mode Python benchmarks on the same machine against `3b9c58b`, then repeated the focused generic-charge cases in reverse order. There were 373 directly comparable completed cases. Four baseline TensorCircuit cases whose external example-result directories were unavailable were excluded rather than treated as product regressions.

- Rebuilt the Rust baseline and current tree into independent target directories and compared Criterion median point estimates for `pauli_word`, `symmetry`, and `propagation`. This avoids reusing a stale executable from the baseline target.

- Ran the 23 Phase 9 lazy-algebra and handle-boundary benchmarks separately on the current release build; all passed.

- Ran `python scripts/check.py --benchmark skip` after reinstalling the current release extension. `cargo fmt`, Black, Clippy with warnings denied, Ruff, mypy, and `git diff --check` passed; Rust tests were 41/41, Python tests were 362/362, and doctests were 10/10.

## 3. What is already complete and should be preserved

- Pauli, Fermion, Boson, Hybrid/Qudit, and Majorana constructors normally produce immutable native-backed facades, and ordinary algebra returns handles without eagerly populating typed Python terms.

- Same-family native algebra and structured-family promotion are substantially implemented; Fermion, Boson, Hybrid, and Majorana commutator paths have native fused kernels.

- Public parity/BK mapping, Fermion/Majorana conversions, Hybrid-to-Pauli projection, Z2 tapering, Pauli U1 restriction, and the principal dense/COO/CSR/native-MVP terminals have handle-consuming paths.

- Propagation accepts native observable handles and returns a Pauli handle for operator results. U1Circuit observable terminals also consume Pauli handles.

- The principal numeric materializers return flat NumPy buffers with explicit shape/count metadata rather than per-term Python objects.

- The charge-analysis public metadata now reports `native_float_selection_rules`, and production code no longer imports `fractions.Fraction` or retains `_native_data`.

Current release-build lazy/eager checks demonstrate that this direction is valuable. In the latest standalone run, Pauli BCH was 4.68x faster at 8 qubits/16 input terms and 3.19x faster at 16 qubits/32 terms; Fermion and Boson BCH were 4.33x and 5.30x faster. These compare the same current native algebra with and without explicit typed-term materialization, so they are evidence for laziness rather than commit-to-commit A/B evidence.

## 4. Open implementation findings

### R1 — Native embedding is not implemented

Priority: P0 Phase 9 blocker.

Evidence: `OperatorSpace.embed()` validates compact maps in Python, then traverses every term through `_materialized_terms()`, rebuilds every factor, computes fermionic inversion signs, and reconstructs a new operator in `python/tencirpauli/structured.py:570-742`. There is no embedding implementation in either Rust crate.

Impact: embedding breaks the required `native handle -> native handle` chain, creates typed Python objects proportional to term count, retains a second correctness implementation for fermionic signs, and holds the GIL during all scalable work.

Resolution route:

1. Keep the current friendly map normalization in Python, but emit only compact source-to-target integer arrays and the target layout descriptor.

2. Add one core embedding operation per native structured storage shape, or one layout-aware Hybrid embedding kernel reused by the specialized handles. It must remap factors, apply the fermionic permutation sign, aggregate collisions deterministically, and return canonical native storage.

3. Add thin PyO3 handle methods that call the core kernel inside `allow_threads`, then construct the existing public facade from the returned handle.

4. Delete the production term loop after independent differential tests pass.

Closure evidence: existing permutation/sign tests must remain, add mixed-domain collision differentials, add a residency test that makes `_materialized_terms()` fail if called, and add a release benchmark with input/output term counts and a nontrivial fermion-axis permutation.

### R2 — Generic and structured charge analysis/restriction still use a Python data plane

Priority: P0 Phase 9 blocker.

Evidence: `ChargeRestrictedOperator.__init__()` calls `len(operator.terms)` and builds Python transition descriptors in `python/tencirpauli/charge.py:1133-1170`; `_termwise_charge_conserved`, `_fast_fermion_particles`, and `_restricted_transition_inputs` traverse `_arrays()` or `_materialized_terms()` in `python/tencirpauli/charge.py:1396-1565`; structured charge analysis uses the Python `_exact_charge_commutator` in `python/tencirpauli/charge.py:1607-1689`; and the structured all-qubit U1 shortcut materializes structured terms to build a new Pauli operator in `python/tencirpauli/structured.py:1414-1428`. The Pauli-only native analysis in `crates/tencirpauli-native/src/charge_analysis.rs` performs the selection-rule algebra in the binding crate rather than delegating to the pure Rust core.

Impact: this is the largest remaining ordinary operator-sized Python round trip. It also leaves two implementations of charge semantics and prevents the structured restriction path from satisfying the GIL and residency contracts.

Resolution route:

1. Move Pauli charge aggregation from the PyO3 crate into `tencir-pauli-core`, then extend the core operation to native Fermion/Boson/Hybrid batches with compact charge/layout descriptors.

2. Move termwise-conservation and fast-fermion eligibility detection onto native canonical storage and return only scalar metadata.

3. Make generic transition-plan construction accept Pauli or structured handles directly. Sector constraints, axis positions, cutoffs, storage choice, and `max_bytes` remain compact inputs; term descriptors and coefficients do not cross Python.

4. Add a direct native conversion/restriction route for an all-qubit structured handle instead of rebuilding a Python `PauliOperator`.

5. Delete `_exact_charge_commutator`, `_restricted_transition_inputs`, and their term-serialization helpers after the native differentials pass. Update stale “exact fraction” wording at the same time; semantics are deterministic binary64 exact-zero aggregation, not arbitrary precision.

Closure evidence: independently compare particle number, spin, excitation number, mixed qubit/Boson spectators, and cancellation-after-aggregation cases; monkeypatch all term/array materializers to fail during analysis and restriction; benchmark analysis, lazy setup, first apply, steady apply, and dense/COO/CSR materialization.

### R3 — Grouping is handle-native, but reconstruction is not

Priority: P0 Phase 9 blocker.

Evidence: `pauli_group_handle` computes groups and bases in Rust, but returns nested support lists. Python converts those supports into integer masks in `python/tencirpauli/grouping.py:199-220`, stores all reconstruction masks in the public dataclass, and `QWCGroupingResult.reconstruct()` loops over masks and samples in Python/NumPy at `python/tencirpauli/grouping.py:79-116`. There is no native grouping/reconstruction handle.

Impact: grouping itself no longer exports Pauli terms, but repeated measurement reconstruction still performs work proportional to shots times group terms in Python and cannot release the GIL as required by Section 7.2 of the specification.

Resolution route:

1. Return a `NativeQWCGroupingHandle` that owns deterministic groups, bases, packed reconstruction masks, and any coefficient data needed by current reconstruction terminals.

2. Materialize `groups`, `bases`, `term_to_group`, and other documented public metadata once when constructing the result facade; keep the native handle privately for execution.

3. Implement `reconstruct(group_index, bitstrings)` as one native call over a contiguous binary sample array, with parity reduction inside `allow_threads`, returning the existing flat `int8` result shape.

4. Delete the Python support/mask conversion and reconstruction loop.

Closure evidence: retain the known eigenvalue tests, add random differential batches across shot/group sizes, reject invalid sample shapes at the boundary, prove reconstruction does not traverse Python masks, and benchmark both grouping construction and repeated reconstruction.

### R4 — Propagation/SPPS still rebuild tapes; batch preparation violates the complete GIL contract

Priority: P0 Phase 9 blocker.

Evidence: the low-level `GateTape` stores only Python operation tuples and has no versioned native cache in `python/tencirpauli/propagation.py:58-301`. Circuit facades rebuild a new Python tape on each uncached plan construction in `python/tencirpauli/propagation_circuit.py:403-433`; their single cached plan is keyed to one observable/state configuration at `python/tencirpauli/propagation_circuit.py:682-721` and `python/tencirpauli/spps_circuit.py:173-217`, so changing the observable recompiles the same structural tape. PyO3 compiles every operation again in `crates/tencirpauli-native/src/propagation.rs:254-283` and the corresponding SPPS constructor. `pauli_propagation_batch_handles` additionally deep-clones all observable operators before entering `allow_threads` at `crates/tencirpauli-native/src/propagation.rs:359-374`.

Impact: the public plan cache hides repeated compilation only for one exact observable, not for tape reuse across native consumers. Large tapes are repeatedly serialized and lowered, and batched observable preparation performs O(total term count) work while holding the GIL.

Resolution route:

1. Introduce the specified immutable native `GateTape` handle containing validated/lowered operations plus parameter-layout metadata.

2. Give the low-level `GateTape` and the circuit builder a monotonic structural version and a cached `(version, native_tape, dynamic_parameter_descriptor)`. Every append/replace/remove/QIR restore invalidates it; parameter values do not.

3. Change PropagationEngine, PropagationBatch, SPPSEngine, and the propagation/SPPS circuit plans to consume the cached tape handle plus observable/state handles. Keep U1Circuit’s existing native-plan cache, but align invalidation tests and terminology rather than creating a second public concept.

4. Reshape the batch binding so Python reference extraction is the only GIL-held work and all term-sized cloning/preparation occurs inside the detached Rust section. Do not introduce `Arc` or copy-on-write; those remain explicitly deferred.

Closure evidence: test first compile, same-tape/different-observable reuse, parameter-only reuse, mutation invalidation, independent circuit instances, and QIR reconstruction; add cold compile versus cached-tape benchmarks and a GIL-concurrency probe without a wall-time CI threshold.

### R5 — Canonical storage and the private ABI still have dead dual tracks

Priority: P0 Phase 9 blocker and maintainability issue.

Evidence: every public `PauliOperator` constructor installs a native handle, yet the class retains optional `_canonical_structures`, split real/imaginary caches, `_from_native`, `_as_native_handle`, and native-versus-array branches throughout `python/tencirpauli/pauli.py:306-356`, `525-594`, and `642-735`. Majorana public construction is likewise always native, but `_initialize`, `_from_canonical`, and all no-handle algebra branches remain in `python/tencirpauli/majorana.py:191-240` and `283-495`. Mapping plans are factory-only and always native, but retain `_native_plan is None` reference branches in `python/tencirpauli/mapping.py:196-293` and `422-625`. U1 plans still contain complete Python execution fallbacks at `python/tencirpauli/symmetry.py:491-616` and `671-742`, although `_restrict_u1` now requires a native handle. Superseded nested private outputs remain as `CanonicalizeOutput`, `MappingOutput`, `JordanWignerOutput`, and array-only propagation/symmetry functions in the PyO3 crate.

Impact: ordinary public execution usually takes the right native branch, but the codebase still reasons about impossible storage states and preserves multiple ABIs for the same capability. This is the main source of interface irregularity and makes future changes likely to repair only one sibling path.

Resolution route:

1. Prove the storage invariant at each public/private constructor, then make the native handle mandatory for Pauli, Majorana, mapping plans, and completed structured families. Keep lazy public term caches only where an explicit `.terms` request needs them.

2. Delete unreachable no-handle algebra, Python U1 execution, compatibility probes, `_from_native` reparse paths, and their private array FFI entry points in one pattern-wide change.

3. Keep one flat numeric read-back per family for explicit exports. The deferred Python tensor-product helpers may use that read-back, but each remaining call must be documented as that deliberate exception rather than relying on a general fallback substrate.

4. Remove split real/imaginary Python caches; use one complex array at an explicit materialization boundary.

Closure evidence: repository search should leave no native-capability fallback dispatch, no completed-path nested numeric output, and no operator-sized `_arrays()`/`_materialized_terms()` call except explicit public export or the documented deferred tensor-product helper. Run the full public compatibility suite after deletion.

### R6 — Equality, hashing, and canonical scalar queries are not uniformly native

Priority: P1 Phase 9 blocker.

Evidence: Pauli equality and hashing materialize `.terms` at `python/tencirpauli/pauli.py:610-616`. Structured and Majorana operators do not define content equality/hash and therefore retain object-identity behavior. Native Hermiticity methods now exist for all main handles, but Structured still uses `hasattr` fallback dispatch at `python/tencirpauli/structured.py:1360-1378`, and Pauli’s `_exact_hermitian_value` retains an array fallback at `python/tencirpauli/pauli.py:756-770`.

Impact: a scalar query can unexpectedly allocate all typed terms, and mathematically equivalent operator families do not expose one coherent immutable-value contract.

Resolution route:

1. Add core canonical-storage equality and content hashing for each handle family and expose scalar handle methods that release the GIL for scalable comparisons.

2. Define equality conservatively for the same public family and compatible layout; do not introduce cross-family mathematical equivalence rules as part of this cleanup.

3. Delegate Python `__eq__`, `__hash__`, and Hermiticity directly to the handle after trivial type/layout/identity checks, then delete `hasattr` and materializing fallbacks.

Closure evidence: compare independently constructed equal and unequal large operators, verify equal objects have equal hashes, and monkeypatch all materializers to fail during equality/hash/Hermiticity. Add a benchmark because these operations are explicitly required by the Phase 9 performance gate.

### R7 — Mapping-plan facades eagerly duplicate native plan data

Priority: P1 architecture and interface issue.

Evidence: `_build()` creates a native mapping plan, immediately exports its encoding matrix, then `FermionQubitMapping.__init__()` exports the inverse and CNOT list and stores `_encoding`, `_inverse_encoding`, `_cnot_operations`, and `_clifford_operations` alongside `_native_plan` in `python/tencirpauli/mapping.py:196-254` and `262-293`. `encode_occupation()` performs an O(n²) Python matrix loop at `python/tencirpauli/mapping.py:367-383`.

Impact: mapping operator transforms are now handle-native, but plan construction still creates parallel scalable state and the facade presents a different storage model from the other native plans.

Resolution route:

1. Retain the native mapping handle plus scalar metadata at construction.

2. Materialize encoding, inverse, and CNOT provenance lazily only when their documented public properties are requested; return flat NumPy arrays for matrices and construct tuples only for the explicitly textual/structured provenance property.

3. Delegate `encode_occupation()` to the native plan and delete the non-native plan-construction branch.

Closure evidence: a plan-construction residency test must show that no matrix/provenance getter ran; retain public property-value tests; benchmark 128/512-mode construction separately from explicit read-back and occupation encoding.

### R8 — Prohibited post-operation non-finite checks remain in hot algebra

Priority: P0 Phase 9 semantic blocker.

Evidence: public input validation is correctly retained, but trusted-handle arithmetic still rejects internally produced non-finite results. Examples include canonical aggregation, handle addition, scaling, multiplication, and final aggregation in `crates/tencir-pauli-core/src/operator.rs:78-81`, `155-158`, `259-260`, `292-293`, `335-338`, and `352-355`; the fused Majorana pair loop checks both already-canonical operands and every product in `crates/tencir-pauli-core/src/majorana.rs:497-525`; mapping, structured finite-basis kernels, propagation, SPPS, sector, and U1Circuit contain the same pattern. Tests in `crates/tencir-pauli-core/src/tests.rs` still assert internal-overflow exceptions.

Impact: this contradicts the frozen ordinary-IEEE semantics and leaves branches/scans in scalable loops. It is an acceptance issue even where the measured cost is small; this report does not claim a material speedup without a profile.

Resolution route:

1. Classify every `is_finite`/`NonFiniteCoefficient` site. Retain one-time checks for direct public coefficients, angles, parameters, states, and matrices, plus all dimension/index/allocation overflow checks.

2. Remove checks whose operands come from trusted canonical handles or whose sole purpose is rejecting a later arithmetic overflow/underflow/NaN. Aggregate with ordinary `Complex64` arithmetic and remove only exact final zero.

3. Delete tests whose sole contract is an exception for an implausible internal numeric overflow; keep representative finite-value numerical differentials.

4. Maintain a short allowlist comment only where a surviving check is a true public boundary, not as a way to preserve ambiguous internal checks.

Closure evidence: a repository audit should account for every surviving finite check, ordinary small-coefficient differentials must remain unchanged, and all memory-safety checked arithmetic must still pass its failure tests.

## 5. Confirmed performance regressions

### 5.1 Generic 8-mode fermion charge restriction

Priority: P0 performance regression.

The focused release benchmark was repeated in both run orders. Median results from the reverse-order run are:

| Python-visible operation | `3b9c58b` | `2937537` | Change |
| --- | ---: | ---: | ---: |
| Steady restricted apply | 28.17 µs | 35.96 µs | +27.7% |
| First apply | 61.54 µs | 66.21 µs | +7.6% |
| Dense materialization | 72.79 µs | 81.83 µs | +12.4% |
| COO materialization | 73.79 µs | 82.04 µs | +11.2% |
| CSR materialization | 73.71 µs | 81.54 µs | +10.6% |

The code change provides a direct explanation: `apply_fermions()` now scans every fermion mode to build a two-word occupation mask on every term application for all layouts up to 128 modes (`crates/tencir-pauli-core/src/charge.rs:351-366`). The benchmark has 8 modes and 14 canonical hopping terms, so this adds repeated O(mode count) work before a small number of parity queries; the old prefix scan was cheaper for this workload.

Resolution route:

1. Build the packed source occupation once per basis column in the outer apply/materialization loop, not once per term. Pass a two-word copy into each term application and update that local copy as creation/annihilation acts.

2. Preserve the existing non-packed path above 128 modes. If hoisting alone does not recover the 8/16-mode cases, restore the old prefix-parity calculation for short sparse words based on measured operation counts; do not add an arbitrary size threshold without the A/B matrix.

3. Benchmark 8, 16, 65, and 128 modes with one-body and longer words, covering lazy apply, eager construction, first/steady apply, and sparse materialization. The fix is accepted only if the small representative case returns to baseline without surrendering the intended wide-word gain.

### 5.2 Deterministic propagation value-and-gradient

Priority: P0 performance regression.

Independent-target Criterion results are:

| Rust kernel | `3b9c58b` | `2937537` | Change |
| --- | ---: | ---: | ---: |
| Deterministic value+gradient, checkpoint interval 1 | 297.15 µs | 358.85 µs | +20.8% |
| Deterministic value+gradient, checkpoint interval 4 | 467.20 µs | 534.37 µs | +14.4% |
| Deterministic value+gradient, checkpoint interval 16 | 483.06 µs | 543.01 µs | +12.4% |

This is localized rather than a general propagation slowdown: the same run improved SPPS value+gradient by 10.7%, 64-observable batch expectation by 13.6%, exact tape propagation by about 3%, and weight-projected tape propagation by 3.8%. Commit `2937537` changes only two relevant deterministic-VJP details: it fuses final expectation/lambda construction and reuses one cleared `FxHashMap` across reverse frames in `crates/tencir-pauli-core/src/propagation.rs:236-257` and `1374-1400`.

Resolution route:

1. A/B the two hunks independently with the existing interval 1/4/16 Criterion cases. First restore per-frame map construction while keeping the fused expectation loop; if that removes the regression, revert the scratch-map reuse rather than layering on a second optimization.

2. If the regression remains, restore the previous expectation/lambda construction and repeat. Use a profiler only after the two-hunk isolation if neither explains the result.

3. Retain whichever variant meets or beats the baseline across all three checkpoint intervals and does not regress SPPS/batch propagation. Add no new data structure or threshold without measured dominant-cost evidence.

## 6. Benchmark summary versus the pre-two-commit baseline

The strongest Python-visible improvements are real and large:

| Operation | `3b9c58b` | `2937537` | Result |
| --- | ---: | ---: | ---: |
| U1 restriction setup | 268.33 µs | 24.63 µs | 10.9x faster |
| 26-qubit U1 restriction setup | 622.88 µs | 45.04 µs | 13.8x faster |
| Z2 taper setup and transform | 35.08 µs | 3.54 µs | 9.9x faster |

Rust QWC grouping was effectively flat in the independent Criterion comparison: 128 terms changed by -1.1% and 1024 terms by +0.5%. A separate Python run in which both the native grouping and pure-Python control slowed substantially was therefore treated as machine noise rather than a product regression.

No other observed movement is strong enough to report as a blocker without a dedicated rerun. The correct overall disposition is “major wins with two confirmed localized regressions,” not either “all benchmarks improved” or “the rewrite regressed globally.”

## 7. Missing acceptance evidence

Priority: P1 closure blocker.

The new benchmark files cover construction, lazy BCH, mapping, Majorana conversion, grouping construction, Pauli U1 restriction, and terminal compilation, but the Phase 9 minimum matrix remains incomplete. There is no representative release benchmark for native embedding, QWC sample reconstruction, operator equality/hash, native gate-tape first compile versus reuse/invalidation, or peak/retained memory removed by the operator-sized Python round trips. Several new lazy BCH cases also omit the required input/output term-count and materialization metadata.

The residency tests are similarly narrower than the completion contract. Existing tests prove native handles exist and validate several handle materializers, but they do not comprehensively fail all materialization hooks during embedding, grouping reconstruction, structured charge analysis/restriction, equality/hash, and tape reuse. A passing result from the same native implementation is not an independent numerical oracle.

Resolution route:

1. Add the missing benchmarks only as the corresponding native paths are implemented, recording input/output terms, qubits/modes, groups/shots, tape length, materialization inclusion, and retained/peak memory where the old path built Python intermediates.

2. Add pattern-wide residency tests that monkeypatch explicit Python term/array exports to fail during every ordinary handle-native operation.

3. Keep independent dense/dictionary references for small embedding, charge, grouping reconstruction, equality/Hermiticity, and tape results under `tests/`; do not retain production fallbacks as oracles.

4. Record a final same-machine release A/B only after R1-R8 and both performance regressions are closed.

## 8. Recommended remediation order

1. Isolate and fix the two measured regressions first, retaining the existing benchmark cases as guards.

2. Implement the three missing native data planes: embedding, generic/structured charge, and grouping reconstruction.

3. Introduce the cached native gate tape and move all batch term-sized preparation under GIL release.

4. Complete native equality/hash and thin the mapping-plan facade.

5. Delete the impossible storage branches, Python execution fallbacks, compatibility probes, and superseded nested/split private ABIs across the whole pattern.

6. Apply the ordinary-IEEE finite-check audit without weakening dimension/index/allocation safety.

7. Fill the residency, independent differential, GIL, memory, and benchmark matrix; run `python scripts/check.py` and the complete release benchmark suite.

This order keeps each deletion behind a working native replacement, avoids speculative abstractions, and uses the existing representative workloads instead of adding defenses for unrealistic numeric extremes.

## 9. Acceptance decision

Decision: do not mark Phase 9 complete at `2937537`.

The current tree is a strong partial implementation and is suitable as the remediation base. Acceptance should follow only when R1-R8 are closed, both confirmed regressions are removed or justified by a larger representative end-to-end win, repository search shows only explicit/deferred materialization boundaries, and the expanded correctness/residency/performance gates pass on a release build.
