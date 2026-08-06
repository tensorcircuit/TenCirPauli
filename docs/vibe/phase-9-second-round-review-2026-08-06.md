# Phase 9 Second-Round Remediation Review, 2026-08-06

Status: open remediation report. The current worktree is not acceptance-closed.

## 1. Scope and verdict

This second-round review audits the uncommitted remediation work layered on commit `2937537` against `docs/vibe/phase-9-review-2026-08-06.md`, `docs/vibe/phase-9-spec.md`, and the repository rules in `AGENTS.md`. It focuses only on findings that remain open or were introduced by the remediation. Findings from the first review that are now closed are intentionally not restated.

The remediation implements most of the intended native data plane, but seven closure items remain. Two scalable native paths still execute while holding the GIL, equality and hashing violate Python's equal-values/equal-hash contract for signed zero, native embedding changes specialized public facade types, the required residency/GIL/performance evidence is incomplete, numbered development-stage labels remain outside `docs/vibe/`, and the generic-charge steady-apply performance result has not yet been closed with the required controlled A/B.

Acceptance decision: do not close Phase 9 until S1-S7 below are resolved and their stated closure evidence passes.

## 2. Verification performed

- Inspected the current Python, PyO3, and Rust core paths for embedding, generic charge, QWC reconstruction, native gate-tape reuse, canonical equality/hash, mapping-plan residency, and finite-result semantics.

- Ran `.conda/bin/python scripts/check.py --benchmark skip`. Cargo formatting, Black, Clippy with warnings denied, Ruff, mypy, `git diff --check`, 41 Rust tests, 362 Python tests, and 10 doctests passed.

- Reproduced the equality/hash defect on the release extension: a real-coefficient `PauliOperator` and its adjoint compare equal but have different hashes; the same failure occurs for a real-coefficient `MajoranaOperator` and its adjoint.

- Ran the focused deterministic propagation Criterion group. Current medians were approximately 277, 434, and 457 microseconds for checkpoint intervals 1, 4, and 16, respectively, which are better than the first review's pre-change baseline values of 297, 467, and 483 microseconds.

- Ran the focused 8-mode generic-charge Python benchmarks. Current medians were 29.71 microseconds for steady apply, 42.79 microseconds for first apply, and approximately 57.0-57.3 microseconds for dense/COO/CSR materialization. First apply and materialization recovered beyond the first-review baseline, while steady apply remained about 5.5% above its recorded 28.17-microsecond baseline.

- Scanned the worktree outside `docs/vibe/` for numbered development-stage labels. No remaining source filename contains such a label after the two pending renames, but 102 content occurrences remain across 19 files.

## 3. Open findings

### S1 — QWC reconstruction remains GIL-bound

Priority: P0 acceptance blocker. First-review relation: R3 remains only partially closed.

Evidence: `NativeQwcGroupingHandle.reconstruct()` validates the NumPy view, allocates the complete output, and performs the full shots-by-group-terms parity loop directly inside the PyO3 method at `crates/tencirpauli-native/src/grouping.rs:28-70`. The method receives `py: Python`, but never calls `py.allow_threads`. Python also performs an O(sample count) binary-value scan with `np.any` before the native call in `python/tencirpauli/grouping.py:99-109`.

Impact: reconstruction is native in language only; it still blocks all Python threads for work proportional to sample count times group support. This violates the complete GIL-release contract and leaves the first-review reconstruction finding open.

Required fix:

1. Keep only trivial type, rank, contiguity, and shape normalization at the Python boundary.

2. Move binary-value validation, output allocation, and the complete parity loop into one Rust helper that takes borrowed packed masks and a contiguous `i8` sample slice.

3. Call that helper inside `py.allow_threads`, then construct the flat NumPy output after the detached computation returns.

4. Do not restore a Python or NumPy reconstruction fallback. The retained public `reconstruction_masks` metadata must not participate in execution.

Closure evidence:

- Add randomized differential tests over multiple qubit, shot, group-size, and support-density combinations against a small independent NumPy oracle.

- Add a residency test that replaces or corrupts the public `reconstruction_masks` after result construction and proves native reconstruction remains correct.

- Add a concurrent-observer GIL probe for a sufficiently large reconstruction workload, without a wall-time CI threshold.

- Add a release benchmark that separately records grouping construction and repeated reconstruction, including qubits, shots, group size, and support density.

### S2 — Handle-based charge compilation still performs operator-sized work under the GIL

Priority: P0 acceptance blocker. First-review relation: R2 remains only partially closed.

Evidence: `charge_terms_from_pauli_handle()` rebuilds every Pauli transition descriptor at `crates/tencirpauli-native/src/charge_sector.rs:92-109`, and `charge_terms_from_hybrid_handle()` clones all structured word arrays at `crates/tencirpauli-native/src/charge_sector.rs:111-128`. `compile_mvp_pauli_handle()` and `compile_mvp_hybrid_handle()` invoke those conversions and `build_native_charge_mvp_plan()` directly at `crates/tencirpauli-native/src/charge_sector.rs:330-388`; neither method accepts `Python` nor enters `allow_threads`.

Impact: Python no longer serializes terms, but the PyO3 call still holds the GIL while performing O(term count) cloning, layout preparation, and optional fast-plan construction. This violates the repository rule that complete scalable conversion and preparation must occur inside the GIL-released Rust section.

Required fix:

1. Change both handle-consuming compile methods to accept `py: Python` and retain only Python reference extraction and scalar argument conversion before detaching.

2. Move handle-to-transition conversion, layout preparation, fast-path detection/building, and construction of `NativeChargeMvpPlan` into one `py.allow_threads` closure.

3. Prefer a core constructor that borrows canonical operator storage directly while preparing the retained native plan. If the plan must own transition descriptors, clone them inside the detached section exactly once.

4. Return ordinary Rust errors from the detached helper and map them to `PyErr` after the closure. Do not add a second Python descriptor path.

Closure evidence:

- Add materializer-failure tests for Pauli, Fermion, Boson, and mixed Hybrid restriction setup.

- Add a concurrent-observer GIL probe for large handle-based plan construction.

- Retain independent particle-number, spin, excitation-number, spectator, and cancellation-after-aggregation differentials.

- Benchmark analysis, lazy setup, eager setup, first apply, steady apply, and dense/COO/CSR materialization with input term count and sector dimension recorded.

### S3 — Equality and hashing disagree for signed-zero coefficients

Priority: P0 correctness blocker. First-review relation: R6 is not closed.

Evidence: Python equality delegates to native numeric coefficient equality in `python/tencirpauli/pauli.py:553-568` and `python/tencirpauli/majorana.py:233-246`. Pauli adjoint preserves the numerically equal sign change from `+0.0` to `-0.0` in the imaginary component at `crates/tencir-pauli-core/src/operator.rs:369-381`, while `PauliOperator::content_hash()` hashes raw IEEE bit patterns at `crates/tencir-pauli-core/src/operator.rs:463-470`. Majorana uses the same mismatch between numeric equality at `crates/tencirpauli-native/src/majorana.rs:195-200` and bitwise hashing at `crates/tencirpauli-native/src/majorana.rs:39-46`.

Reproduction: for a real Pauli X operator, `operator == operator.adjoint()` is `True`, but `hash(operator) == hash(operator.adjoint())` is `False`; the same result occurs for a real single-generator Majorana operator.

Impact: equal immutable values can occupy different dictionary/set buckets, violating Python's hash contract and making cache behavior incorrect. Existing tests exercise equality but do not bind equal-value hashing to signed-zero cases.

Required fix:

1. Introduce one shared coefficient-hash rule that canonicalizes every numeric zero to the same hash representation before hashing; for example, hash zero as `0_u64` when `value == 0.0`, otherwise hash `value.to_bits()`.

2. Apply the rule pattern-wide to Pauli, Fermion, Boson, Hybrid/Qudit, and Majorana content hashes. Do not fix only the two reproduced families.

3. Preserve current same-family/layout equality semantics. Do not introduce cross-family mathematical equality as part of this repair.

Closure evidence:

- Add direct `+0.0/-0.0`, real-operator adjoint, independently constructed equal operator, unequal operator, infinity, and ordinary complex-coefficient cases for every hashable family.

- Assert both `left == right` and `hash(left) == hash(right)` for every equal pair.

- Monkeypatch all materializers to fail during equality and hashing.

- Add representative large-operator equality/hash benchmarks required by the specification.

### S4 — Native embedding loses specialized public facade types

Priority: P1 public-API blocker. First-review relation: R1 is functionally incomplete despite the new native kernel.

Evidence: `OperatorSpace.embed()` converts `NativeFermionOperatorHandle` and `NativeBosonOperatorHandle` to `NativeHybridOperatorHandle` at `python/tencirpauli/structured.py:656-665`, calls the native Hybrid embedding kernel, and unconditionally returns `_hybrid_from_native(self, result)` at `python/tencirpauli/structured.py:666-682`. A `FermionOperator`, `BosonOperator`, or `QuditWeylOperator` embedded into a compatible pure target space therefore returns `HybridOperator`. Before this remediation, `_make_operator()` selected the domain-specific facade for a pure target layout.

Impact: embedding changes the public type and its documented domain-specific methods even when no domain promotion is required. The current embedding tests check numerical matrices but not result family, so the regression passes the full suite.

Required fix:

1. Route the native result according to the target `OperatorSpace`, not unconditionally through the Hybrid facade.

2. For a pure fermion target, return `FermionOperator` backed by `NativeFermionOperatorHandle`; for a pure boson target, return `BosonOperator` backed by `NativeBosonOperatorHandle`; for a pure uniform-qudit target, return `QuditWeylOperator`; return `HybridOperator` only for genuinely mixed layouts.

3. Implement native extraction/conversion from the canonical Hybrid result when necessary. Validate the pure-layout invariant once and do not export terms through Python to reconstruct a specialized handle.

4. Replace the remaining user-facing development-stage wording in the `embed()` type error with capability language.

Closure evidence:

- Add type-preservation tests for Fermion, Boson, QuditWeyl, and genuinely mixed Hybrid embeddings.

- Add mixed-domain collision, nontrivial fermion-permutation sign, deterministic ordering, and materializer-failure tests.

- Add a release embedding benchmark with source/output term counts, layout widths, and a nontrivial fermion-axis permutation.

### S5 — Required acceptance evidence remains incomplete

Priority: P1 closure blocker. First-review relation: the implementation portions of R1, R3, R4, R6, and R7 advanced, but their required evidence was not completed.

Evidence: `benchmarks/python/test_handle_boundaries_benchmark.py` covers flat Pauli construction, mapping, Majorana conversion, grouping construction, U1 restriction, and terminal compilation, but contains no embedding, QWC reconstruction, equality/hash, native gate-tape cold-versus-reuse, invalidation, GIL, or retained/peak-memory cases. Existing tests contain one fixed QWC reconstruction example, no signed-zero hash contract, no embedding result-family assertion, and no comprehensive gate-tape cache lifecycle test.

Required fix:

1. Add embedding residency and benchmark coverage as specified in S4.

2. Add QWC reconstruction differential, residency, GIL, and benchmark coverage as specified in S1.

3. Add equality/hash correctness, residency, and benchmark coverage as specified in S3.

4. Add gate-tape tests for first compile, same tape with a different observable, parameter-only reuse, structural mutation invalidation, independent circuit instances, and QIR reconstruction. Instrument or monkeypatch the native tape constructor so each test asserts the exact compile count.

5. Add cold native-tape compile versus cached reuse benchmarks for PropagationEngine, PropagationBatch, SPPSEngine, PropagationCircuit, and SPPSCircuit. Record gate count, parameter count, observable term count, and whether structural conversion is included.

6. Add retained/peak-memory evidence for paths that previously constructed complete Python intermediates. Keep results local and informational; do not add wall-time CI gates.

Closure evidence: every benchmark must report the workload metadata required by the specification, every residency test must make the prohibited materializer fail, and numerical tests must compare against independent small-system references rather than replaying the same native implementation.

### S6 — Numbered development-stage labels remain outside `docs/vibe/`

Priority: P1 repository-rule blocker.

Evidence: the pending filename changes remove the two non-vibe filenames containing development-stage labels, but a repository scan still finds 102 content occurrences across 19 files outside `docs/vibe/`. They include root and benchmark documentation, production docstrings and error text, Rust comments, test names/docstrings, benchmark test names and metadata keys, benchmark suite selectors, examples, and a reference note. Representative locations include `README.md:65-228`, `python/tencirpauli/structured.py:578`, `crates/tencir-pauli-core/src/mapping.rs:533`, `benchmarks/run.py:23-274`, `benchmarks/python/test_majorana_mapping_charge_benchmark.py:238-996`, and `tests/test_u1.py:88-173`.

Impact: the current tree violates the non-negotiable rule now recorded in `AGENTS.md`: development-stage labels may appear only under `docs/vibe/`, while formal artifacts must use capability or behavior names.

Required fix:

1. Rename test and benchmark functions by capability or workload, preserving their numerical parameters and semantics.

2. Rename benchmark suite selectors such as the structured and Majorana/charge suites to capability names. Update the runner help, README, and any internal references in the same change.

3. Rewrite root documentation, examples, production docstrings/errors, Rust comments, and test docstrings to describe behavior rather than rollout chronology. Root documentation that needs historical context should link to the generic `docs/vibe/README.md` index instead of embedding a development-stage label in non-vibe content.

4. Move genuinely experimental reference notes into `docs/vibe/` or remove their numbered-stage wording if they remain formal references.

5. Add a deterministic repository check for numbered development-stage labels in both paths and file contents outside `docs/vibe/`. Use a narrow pattern that does not reject legitimate scientific uses of the word “phase,” such as Pauli phase or complex phase.

Closure evidence: the repository check returns zero path and content matches outside `docs/vibe/`, and the full test/benchmark collection still discovers the renamed cases.

### S7 — Generic-charge steady-apply performance is not formally closed

Priority: P1 performance-closure blocker. First-review relation: the deterministic propagation regression is recovered; the generic-charge regression needs one final controlled comparison.

Evidence: the packed source occupation is now hoisted outside per-term application in `crates/tencir-pauli-core/src/charge.rs`, and the focused rerun shows substantial recovery. Current first apply and dense/COO/CSR materialization beat the first-review pre-change baselines, but steady apply measured 29.71 microseconds versus the recorded 28.17-microsecond baseline, approximately 5.5% slower. This single run was not the independent-target, reverse-order A/B required by the first review.

Required fix:

1. Do not add another optimization before confirming the remaining delta. Rebuild baseline and remediation into independent target directories and repeat the 8-mode focused case in both execution orders.

2. Run the specified 8, 16, 65, and 128-mode matrix with one-body and longer fermion words, covering lazy apply, eager setup, first/steady apply, and sparse materialization.

3. If the 8-mode steady regression remains outside run-to-run noise, profile the term application path and quantify the share attributable to packed-source construction/copying before changing thresholds or data structures.

4. Retain the hoisted packed-source implementation only if it recovers the small representative case without surrendering the intended wide-word gain. Any short-word specialization must be justified by the measured operation matrix rather than an arbitrary cutoff.

Closure evidence: same-machine independent-target results show no meaningful regression in the representative 8-mode steady case, the wide-word cases do not regress materially, and the benchmark record includes both run orders and numerical equivalence metadata.

## 4. Required remediation order

1. Fix S3 first because it is a deterministic public correctness violation with a small, pattern-wide repair surface.

2. Fix S1 and S2 next so all scalable native execution and preparation satisfies the complete GIL contract.

3. Fix S4 before adding embedding evidence, because current numerical-only tests would otherwise bless the wrong public result family.

4. Complete the evidence matrix in S5 after the corresponding implementations stabilize.

5. Perform the mechanical repository-wide label cleanup in S6 and add the recurrence check.

6. Run the controlled performance comparison in S7 last, after implementation and test instrumentation no longer change the measured paths.

## 5. Final acceptance gate

Phase 9 may be acceptance-closed only when S1-S7 are resolved, the full repository quality gate passes after a release rebuild, the non-vibe label scan is empty, equality/hash tests cover signed zero pattern-wide, embedding preserves specialized public facades, QWC and charge scalable work demonstrably release the GIL, the missing residency/performance evidence exists, and the controlled generic-charge comparison is regression-free or has an owner-approved representative tradeoff backed by measured end-to-end evidence.
