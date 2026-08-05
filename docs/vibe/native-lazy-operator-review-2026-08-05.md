# Native-backed lazy operator implementation review

Review date: 2026-08-05

Reviewed baseline: `3b9c58b26fb6b0a278849a7ee4bdd65276a931e2` plus the uncommitted native-backed lazy operator, mapping, benchmark, example, test, and documentation changes in the working tree.

Scope: review the attempt to keep public Python operator facades backed by Rust-owned handles so symbolic algebra can remain native across intermediate results. The review prioritizes numerical and algebraic correctness, coarse FFI, high availability, end-to-end hot-path performance, and same-layout API compatibility. Cosmetic cleanup, exact transient allocator accounting, and speculative public abstractions are excluded.

## Verdict

The central design is sound and already produces a material benefit. Pauli, Fermion, Boson, Hybrid/Qudit, and Majorana constructors and most primitive algebra results retain private Rust handles; `.terms` is generally an explicit typed-word materialization boundary. On the review machine, the included Pauli BCH benchmark measured the native-backed path at approximately 7.58 ms versus 25.19 ms for the eager-materializing comparison at 8 qubits/16 input terms, and 146.87 ms versus 371.19 ms at 16 qubits/32 input terms, corresponding to approximately 3.3x and 2.5x speedups.

The implementation is not yet complete enough to describe mapping or terminal compilation as fully handle-native. Pauli array construction performs a native-to-Python-to-native canonicalization round trip, reusable Pauli mapping forces typed-term and array materialization, pure Hybrid-to-Pauli mapping exports and rebuilds the operator, and finite compilation still serializes complete operators through Python. Structured and Majorana commutators retain native primitive operations but are not fused: they execute two multiplications plus scale and addition through multiple PyO3 calls and allocate all intermediate operators.

Recommended status: **accept the private-handle architecture and measured algebra speedup; keep remediation open for coefficient invariants, compatible mixed-family addition, duplicate Pauli construction, fully handle-native mapping, fused commutators, GIL release, and direct handle-native compilation/conversion.**

## Validation performed

- Rebuilt the current PyO3 extension with `maturin develop --release`.
- Ran the complete Python suite: 358 tests passed.
- Ran the complete Rust workspace suite: 41 tests passed.
- Ran `cargo fmt --check`, Clippy for all workspace targets and features with warnings denied, Black, Ruff, and strict mypy; all passed.
- Ran the 12 new lazy algebra benchmark cases in release mode. The Pauli results above demonstrate a real materialization reduction; the structured cases demonstrate retained native results but do not include an old-implementation or independent eager algebra baseline.
- Instrumented code-array Pauli construction and observed one `pauli_canonicalize_array` call followed by one `pauli_operator_native` call for a single construction.
- Verified that `FermionQubitMapping.map_pauli` changes a previously untouched input from `_terms is None` and `_canonical_structures is None` to both caches populated.
- Reproduced compatible-space `FermionOperator.add(HybridOperator)` failing with `TypeError: native structured handles have incompatible families`.
- Reproduced native Fermion, Boson, Hybrid, and Majorana scaling retaining a `0j` term after underflow and retaining `inf` after overflow.

## Existing decisions and implementation consequences

| Topic | Decision status | Required implementation consequence |
| --- | --- | --- |
| Finite coefficients and exact-zero removal | Already frozen by `semantics.md` and the Phase 7/7.5 contracts | Native coefficient transforms and aggregation must reject non-finite outputs and omit exact zeros. Fuse these checks into the existing coefficient/aggregation loop; do not add a second pass. |
| Compatible same-layout addition | Already frozen by the Phase 7 contract | Addition across compatible specialized and Hybrid facades must remain valid and promote to `HybridOperator` when the result is not representable by one specialized family. |
| Private handles and public facades | Established by the current lazy-result design | Add handle-to-handle native methods rather than exposing a new public lazy type or returning raw native handles to users. |
| Long-running native work and the GIL | Repository-wide rule | Release the GIL around O(term count), O(pair count), canonicalization, mapping, conversion, and compilation work. Python object creation remains outside the detached section where required. |
| `max_bytes` | Existing best-effort contract | Preserve cheap major-output/workspace checks without attempting exact allocator accounting. Passing handles must not silently bypass the existing guards. |

No additional owner decision is required for the items above unless the project intentionally reopens and changes a frozen public coefficient or compatibility contract.

## Findings

### NL1 — MAJOR: native coefficient transforms violate frozen finite and zero-free operator invariants

Locations: `crates/tencirpauli-native/src/structured.rs:122-141`, `crates/tencirpauli-native/src/structured.rs:406-432`, `crates/tencirpauli-native/src/structured.rs:542-581`, and `crates/tencirpauli-native/src/majorana.rs:45-85`.

The new handle-local scale implementations multiply coefficients and retain the structural arrays without validating the products or removing products that compare exactly equal to complex zero. Some new addition paths similarly accumulate directly into a `BTreeMap` without validating the aggregated result. Consequently an operator may expose a non-finite coefficient or report a nonzero `term_count` for a stored `0j` term.

This is not an optional defensive check under the current contract: `docs/vibe/semantics.md` rejects NaN and infinity and requires exact-zero removal, while `phase-7-spec.md` explicitly states that coefficient overflow is an error. The efficient repair is to use one shared checked coefficient helper while already iterating over results. It should return an error for non-finite output and skip exact zero without an additional scan. Regression tests must cover every handle family because the duplicated implementations have already drifted.

### NL2 — MAJOR: compatible mixed-family addition regressed

Locations: `python/tencirpauli/structured.py:1072-1093` and `python/tencirpauli/structured.py:2365-2393`.

`_StructuredOperator.add` dispatches to `_add_native_handles` whenever both operands have handles, but `_add_native_handles` rejects different handle classes even when the two public operators have the same `OperatorSpace`. This breaks previously supported addition such as a specialized `FermionOperator` plus a factory-produced `HybridOperator` on the same fermion layout.

Required resolution: preserve the specialized same-handle fast paths, but route compatible mixed handle families into a native Hybrid addition/promotion path. Do not materialize typed Python terms merely to recover compatibility. Add operand-order regressions for Fermion/Hybrid, Boson/Hybrid, and Qudit/Hybrid where the spaces are equal, plus incompatible-space failures.

### NL3 — MAJOR PERFORMANCE: code-array Pauli construction canonicalizes and crosses the boundary twice

Locations: `python/tencirpauli/pauli.py:336-355`, `python/tencirpauli/pauli.py:395-412`, and `python/tencirpauli/pauli.py:1336-1349`.

Code-array input calls `pauli_canonicalize_array`, exports the canonical operator as Python-visible arrays, and immediately passes those arrays to `pauli_operator_native`, which rebuilds and canonicalizes the operator again. `_from_native` applies the same rebuild pattern to legacy native functions that return arrays.

Required resolution: add a contiguous array constructor that returns `NativePauliOperatorHandle` directly. Native producers of Pauli results should return the handle whenever the Python contract does not require arrays. Retain array-returning APIs only for public canonicalization results and backend data exchange whose contract explicitly requests arrays.

### NL4 — MAJOR PERFORMANCE: reusable mapping is not fully handle-native

Locations: `python/tencirpauli/mapping.py:422-462`, `python/tencirpauli/mapping.py:464-488`, `python/tencirpauli/mapping.py:553-609`, and `python/tencirpauli/structured.py:1472-1523`.

`map_pauli` calls `len(operator.terms)`, which constructs typed Pauli terms, then calls `_arrays()`, which separately exports canonical arrays. The native mapping plan accepts those arrays and returns arrays that `_from_native` sends back into Rust. Named parity and Bravyi-Kitaev Fermion mapping inherit this route after the already-native Jordan-Wigner step. Majorana mapping already has the desired `transform_majorana_handle` path, and mixed Hybrid mapping already has `transform_hybrid_handle`, showing that the handle-native design is technically viable.

Pure fermion/qubit Hybrid mapping has another round trip: the native Hybrid handle performs Jordan-Wigner mapping, then Python materializes every mapped term, reorders codes by `OperatorSpace`, and constructs a new Pauli handle. This affects formula-style `OperatorSpace` factories even though the separate `FermionOperator` path is direct for Jordan-Wigner.

Required resolution:

1. Expose the underlying core Pauli operator internally and add `NativeMappingPlan.transform_pauli_handle`, returning `NativePauliOperatorHandle`.
2. Chain Fermion handle Jordan-Wigner output directly into the mapping-plan handle transform for parity and Bravyi-Kitaev.
3. Add a native Hybrid-to-Pauli projection that accepts the compact axis-order descriptor and returns `NativePauliOperatorHandle` without term export.
4. Replace term-based size estimates with `term_count` or a native estimate; an estimate must never force materialization.

This can be fully Rust-native for operator-sized data. Passing an O(number of axes) layout descriptor across PyO3 once is still a coarse-grained boundary and does not undermine the design.

### NL5 — MAJOR PERFORMANCE: Structured and Majorana commutators are native-resident but not fused

Locations: `python/tencirpauli/structured.py:1203-1240` and `python/tencirpauli/majorana.py:427-448`.

Pauli has direct native `commutator` and `anticommutator` handle methods. Fermion and Boson do not: although each individual `multiply`, `scale`, and `add` is native, they inherit `_StructuredOperator.commutator`, which performs `A*B`, `B*A`, scaling, and addition as separate PyO3 calls with complete intermediate handles. Hybrid and Qudit use the same base implementation. Majorana follows the same composed pattern.

There is no algebraic reason these families cannot support direct kernels. A fused implementation can feed contributions from both operand orders into one deterministic aggregate, using signs `+1/-1` for the commutator and `+1/+1` for the anticommutator. Fermion and Boson still require their CAR/CCR normal-order expansions, but they need not store two complete canonical products before the final addition. Hybrid can reuse the domain-product machinery with an operation mode. Majorana can additionally use canonical-word commutation parity to skip pairs whose commutator is identically zero.

“Fused” here changes only the internal execution, not the public operation or mathematical result. The current route materializes four native intermediates conceptually equivalent to `ab = multiply(A, B)`, `ba = multiply(B, A)`, `minus_ba = scale(ba, -1)`, and `result = add(ab, minus_ba)`. The fused route allocates one aggregate, emits every canonical contribution of `A_i B_j` with factor `+1`, emits every canonical contribution of `B_j A_i` with factor `-1`, and performs one deterministic finish step. Anticommutator uses `+1` for both orders. The same `max_bytes` guard covers the combined contribution workspace, and exact-zero cancellation happens only after contributions sharing a canonical key have been aggregated.

Concrete family plans:

- **Pauli:** retain the existing direct core and handle methods. A pair either commutes and contributes zero to the commutator or produces the ordinary Pauli product with the exact discrete phase and the commutator factor. No change is required beyond keeping this as the reference native interface shape.
- **Fermion:** add a core binary-operation mode around the existing CAR product expansion. For every canonical input pair, normal-order `A_i B_j` into the shared aggregate with `+c_i c_j` and normal-order `B_j A_i` into the same aggregate with `-c_j c_i`. Do not first finish either directional product. Expose `NativeFermionOperatorHandle.commutator` and `anticommutator` and dispatch directly from Python when both handles are Fermion handles.
- **Boson:** mirror the Fermion plan using the existing CCR contraction expansion and one shared canonical Boson aggregate. Expose the corresponding Boson handle methods.
- **Hybrid/Qudit:** generalize `multiply_hybrid_terms` with an internal binary-operation mode that emits both operand orders into one shared complete Hybrid-key aggregate. Reuse the existing Fermion, Boson, Pauli, mapped-fermion, and Weyl factor-product logic. Preserve the existing raw/mapped-fermion ambiguity checks before starting the fused kernel. Pure Qudit operators use the same Hybrid handle implementation.
- **Majorana:** add a direct core operation over canonical Majorana supports. Before multiplying a word pair, compute the exact graded commutation sign; skip a commutator pair when its two ordered products cancel identically and skip an anticommutator pair when they cancel there. For surviving pairs, compute the canonical XOR support and exact sign once and emit the resulting coefficient into the shared aggregate. Expose direct Majorana handle methods.

Required resolution: implement the family plans above, then make the Python base dispatch to them when compatible native handles are available. Keep the existing composed path only as a materialized/reference fallback. Differentially compare every fused result with both the existing composed recurrence and an independent small dense reference, and benchmark complete BCH recurrences rather than isolated multiplication.

### NL6 — MAJOR AVAILABILITY/PERFORMANCE: several new O(n) handle operations retain the GIL and use avoidable deep-copy aggregation

Locations: `crates/tencirpauli-native/src/structured.rs:106-119`, `crates/tencirpauli-native/src/structured.rs:392-403`, `crates/tencirpauli-native/src/structured.rs:542-621`, and `crates/tencirpauli-native/src/majorana.rs:45-85`.

Hybrid and Boson add/scale/materialize do complete nested-vector cloning and aggregation without releasing the GIL. Fermion and Majorana addition clone or rebuild the complete inputs before entering `allow_threads`. These operations can stall unrelated Python work for large valid symbolic operators.

The add implementations also rebuild a `BTreeMap` and deep-clone every structural key even though each input is already canonical and sorted. A two-way ordered merge is O(n), preserves deterministic order, aggregates only equal adjacent keys, and avoids tree-node allocation. Coefficient scaling continues to produce an ordinary independently owned handle in this remediation. Structural sharing through `Arc` or copy-on-write storage is explicitly deferred and must not be introduced as part of this work.

Required resolution: move all pure-Rust O(n) preparation and aggregation under `allow_threads`, replace canonical add reconstruction with linear merge, and add a concurrent observer regression or diagnostic without introducing a wall-time CI gate.

### NL7 — MEDIUM PERFORMANCE: terminal compilation still serializes complete handles through Python

Locations: `python/tencirpauli/pauli.py:924-1088` and `python/tencirpauli/structured.py:3356-3602`.

Pauli dense/COO/CSR/MVP targets call `_arrays()` and then pass the same operator back into native kernels. Structured compilation calls `_hybrid_arrays`, constructs per-term operation lists in Python according to the axis order, and then passes those lists into Rust. This avoids typed public term objects in some paths but does not achieve the intended handle-to-handle or handle-to-terminal boundary.

Required resolution: add terminal methods or native overloads that consume `NativePauliOperatorHandle` directly. For Structured handles, pass the compact ordered axis descriptors and boson cutoffs once and perform term-to-operation lowering in Rust. Dense and sparse results may still cross into Python as their public arrays; reusable native MVP plans should remain native handles.

Pauli `native_mvp_plan` also calls `handle.materialize()` solely to count distinct X masks for a Python-side byte estimate. The native plan should expose the required estimate or X-mask count instead of exporting all structures, X words, Z words, and coefficients.

### NL8 — MEDIUM PERFORMANCE: remaining conversions and tensor products are not handle-native

Locations: `python/tencirpauli/majorana.py:595-612`, `python/tencirpauli/pauli.py:1178-1219`, and `python/tencirpauli/structured.py:1368-1430`.

Fermion-to-Majorana exports the complete Fermion handle to Python arrays even though Majorana-to-Fermion already has a direct handle method. Add the symmetric `NativeFermionOperatorHandle.to_majorana` conversion.

Pauli and Structured tensor products materialize operands and construct the Cartesian product in Python. Native tensor product is technically feasible, but the owner has deferred it because no current representative workload establishes it as a dominant hotspot. It must not delay construction, mapping, fused commutators, GIL release, or compilation. Embedding and charge analysis remain explicit structure-level fallbacks and are outside this remediation scope; charge analysis intentionally needs semantic word inspection.

### NL9 — MEDIUM EVIDENCE: new tests and benchmarks do not cover the reproduced regressions or all claimed native paths

Location: `tests/test_lazy_operator.py:15-29`, `benchmarks/python/test_pauli_lazy_algebra_benchmark.py`, and `benchmarks/python/test_structured_lazy_algebra_benchmark.py`.

Two lazy Pauli tests compare an expression with the same expression recomputed by the same implementation, so they are not independent correctness evidence. Existing random dense Pauli tests provide broader protection, but the lazy-specific suite must assert storage behavior while comparing against an independent dense or plain-dictionary reference.

The structured benchmark has native BCH and BCH-plus-materialization cases but no old/eager algebra baseline. Neither benchmark covers array construction, parity/BK mapping, Hybrid-to-Pauli projection, conversion, terminal compilation, GIL behavior, or peak memory. Add focused release benchmarks for the repaired boundaries and record term counts so accidental canonical collapse cannot create misleading speedups.

## Implementation order

1. Restore coefficient finite/zero invariants and compatible mixed-family addition, with durable regressions.
2. Eliminate the double Pauli array construction and make Pauli, Fermion, Majorana, and Hybrid mapping handle-native end to end.
3. Release the GIL and replace canonical handle addition with deterministic linear merge.
4. Add fused native commutator and anticommutator kernels for Fermion, Boson, Hybrid/Qudit, and Majorana; rerun BCH benchmarks and correctness differentials.
5. Make Pauli and Structured dense/sparse/MVP compilation consume handles directly and add direct Fermion-to-Majorana conversion.
6. Keep native tensor products deferred. Reopen them only after the higher-frequency boundaries are closed and representative profiling identifies tensor composition as material.

## Owner decisions recorded

The owner recorded the following scope decisions on 2026-08-05:

1. **No `Arc`/copy-on-write structural sharing now:** keep ordinary owned handle storage. `Arc` here would mean sharing immutable word/key arrays between scaled operator results while allocating only new coefficients. The added ownership complexity is not justified without a profile showing that structural copying materially dominates BCH runtime or peak memory. Reopen only with such evidence.
2. **Defer native tensor products:** retain the current Python/materialized tensor-product implementation for now. Reopen a Rust-native implementation only for an identified representative tensor-composition workload.
3. **Implement fused native commutators:** this is a production-core optimization, not an example-only change and not a new public API. Keep the public `operator.commutator(other)` contract unchanged, add the internal Rust core functions and private handle methods described in NL5, and route compatible handle-backed operators through one native call and one final aggregate.

No owner decision remains open for this review. The finite-coefficient invariant, exact-zero removal, compatible-space promotion, fully private native handles, GIL release, handle-native mapping and compilation, and fused native commutator direction are implementation requirements under the current repository contracts and reviewed performance goal.

## Frozen remediation scope

| Finding | Disposition | Frozen delivery requirement |
| --- | --- | --- |
| NL1 coefficient invariants | Required | Fuse finite-result validation and exact-zero removal into every affected native coefficient/aggregation loop and add family-wide regressions. |
| NL2 compatible addition | Required | Preserve same-layout mixed-family addition and promote the result to a native-backed `HybridOperator` without typed-term materialization. |
| NL3 duplicate Pauli construction | Required | Add a direct contiguous-array-to-handle constructor and remove the canonical-array round trip from ordinary operator construction/results. |
| NL4 mapping | Required | Keep operator-sized Pauli, Fermion, Hybrid, and Majorana mapping data in Rust handles end to end; only compact plan/layout descriptors may cross as control input. |
| NL5 fused commutators | Required and owner-approved | Add internal fused Rust core functions and private handle methods for Fermion, Boson, Hybrid/Qudit, and Majorana; public APIs remain unchanged. |
| NL6 GIL and canonical addition | Required | Release the GIL around complete O(n)/O(pair count) native work and replace sorted-handle `BTreeMap` addition with deterministic linear merge. Do not add `Arc`/copy-on-write storage. |
| NL7 terminal compilation | Required | Consume native handles directly for Pauli and Structured dense/sparse/native-MVP compilation; public terminal arrays cross the boundary only as final outputs. |
| NL8 conversions and tensor products | Split | Direct Fermion-handle-to-Majorana-handle conversion is required. Native tensor products are deferred. Embedding and charge-analysis fallback are outside scope. |
| NL9 evidence | Required | Replace self-comparisons, add independent family differentials and handle-residency assertions, and benchmark the repaired construction/mapping/commutator/compilation boundaries in release mode. |

## Acceptance recommendation

After every item marked Required in the frozen remediation scope is repaired, the implementation can accurately claim that ordinary symbolic algebra, reusable fermion mappings, and native terminal compilation keep operator-sized data in Rust until the user explicitly requests terms or public output arrays. Closure requires the full local quality gate, independent dense/dictionary differentials for every family, storage assertions proving intermediate `.terms` remain unmaterialized, release-mode end-to-end benchmarks including conversion and PyO3 costs, and a concurrent GIL diagnostic for large valid operations. Deferred and out-of-scope items are not acceptance blockers.

This review added only this report and its `docs/vibe/README.md` index entry. It did not modify production source, tests, benchmark sources, examples, specifications, or existing user changes.
