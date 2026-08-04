# Phase 7 implementation review report

Review date: 2026-08-03

Scope: commits `56300eb`, `56407e3`, and `2a32c34` against the frozen contract in `docs/vibe/phase-7-spec.md`, including correctness, target availability, performance architecture, memory behavior, tests, benchmarks, and P0–P5 delivery.

## Verdict

Phase 7 is not complete and should not be marked release-ready. The current code is a useful vertical slice and all existing quality gates pass, but there are two silent algebra-corruption paths, an inconsistent finite-boson numeric path, an incomplete uniform-qudit backend/TensorCircuit target, and an incorrectly implemented structured `native_mvp` target that materializes and retains a complete COO matrix instead of providing matrix-free application. The repository's own status file accurately says “implemented vertical slice; under acceptance review” and “full P0–P5 handoff remain acceptance work” (`docs/vibe/implementation-status.md:7,16`).

No implementation, test, benchmark, or specification source files were changed during this review. This report is the only added file.

## Compliance checklist

| Requirement | Result | Evidence |
| --- | --- | --- |
| Rust core remains independent of Python/TensorCircuit | PASS | Structured algorithms are in `crates/tencir-pauli-core/src/structured.rs`; PyO3 conversion remains in the native crate. |
| Existing format, lint, type, Rust, Python, and benchmark-smoke gates pass | PASS | `python scripts/check.py --benchmark smoke` passed: 31 Rust tests, 201 Python tests, and 156 selected Python benchmark-smoke cases. |
| Small-system CAR/CCR/Weyl and finite-target basics | PASS | The focused Phase 7 file has 13 passing tests and current dense/COO/CSR/native-MVP cases agree within that coverage. |
| Public algebra remains correct after partial mapping and embedding | FAIL | CRITICAL C1 and C2 below reproduce silent wrong operators. |
| Full target matrix, including uniform-qudit backend MVP on TensorCircuit NumPy/JAX backends | FAIL | MAJOR M1; the adapter assumes binary local dimensions and cannot execute the returned qudit plan. |
| Native structured MVP is reusable, compact, accurately described, and matrix-free | FAIL | MAJOR M2; the plan stores a complete COO triplet and reports a much smaller byte estimate. |
| Finite targets have consistent complex128 semantics | FAIL | MAJOR M3; a valid boson monomial yields non-finite dense output but a sparse-path exception. |
| Expansion limits guard work before major recursive allocation | FAIL | MAJOR M4; CAR/CCR rewrites recurse and clone without receiving `max_bytes`. |
| Builder product semantics do not silently discard factors | FAIL | MAJOR M5; repeated Pauli factors on one site are overwritten. |
| Hot paths use coarse FFI and avoid duplicate semantic implementations | FAIL | MAJOR M6; adjoint crosses PyO3 per term and finite transitions are duplicated in Python and Rust. |
| P0–P5 acceptance evidence, public documentation, examples, and release records are complete | FAIL | MAJOR M7 and the delivery matrix below. |

## CRITICAL

### C1. Multiplying a partially mapped hybrid operator by a raw fermion operator silently drops the already mapped Pauli factor

Locations: `crates/tencir-pauli-core/src/structured.rs:102-138`, `crates/tencir-pauli-core/src/structured.rs:650-688`, and `python/tencirpauli/structured.py:891-911`.

Hybrid multiplication can legitimately produce a term containing both `fermion_present` and `mapped_present`. During the next `map_fermions()` call, `jordan_wigner_hybrid_terms` expands the raw fermion word but does not multiply each expansion by `batch.mapped_codes[index]`; it merely replaces the mapped component with the new expansion. A public workflow such as `(create * bdag).map_fermions() * annihilate` therefore compiles to a different matrix from `(create * bdag) * annihilate`.

Reproduction on the reviewed build gave a maximum dense-matrix difference of `1.0`; the expected result had one unit transition, while the partially mapped workflow produced a different complex half-amplitude transition. This is silent numerical corruption in an ordinary immutable-algebra workflow.

Resolution: combine the pre-existing mapped word with every new Jordan–Wigner expansion using the Pauli product table and its phase before aggregation. Add a regression covering mapped-plus-raw multiplication and tensor product, or explicitly prohibit further fermion algebra after partial mapping at the public boundary rather than accepting it and returning a wrong result.

### C2. `OperatorSpace.embed()` can ignore an explicit map or silently change the operator algebra

Locations: `python/tencirpauli/structured.py:584-686`, especially the unconditional equal-space return at `589-590`, permissive integer coercion at `609-621`, and target-dimension reconstruction at `666-675`.

There are several manifestations of one missing embedding invariant layer:

- An explicit same-space permutation is ignored because `operator.space == self` returns before maps are inspected.
- Target values are not required to be injective, so multiple source modes/sites can collapse onto one target.
- Unexpected map keywords are ignored, and keys/values are coerced with `int()`, accepting values the rest of the public API rejects.
- Source and target qudit dimensions are not compared. A reviewed probe embedded a nontrivial dimension-4 `X**3` word into a dimension-3 space and silently obtained the identity matrix.
- A non-monotone fermion embedding is rebuilt without canonical reordering and its fermionic sign, so it is either rejected by a constructor or would have the wrong graded semantics.

Resolution: validate map names, exact source coverage, integer types, injectivity, target ranges, and domain dimensions before the equal-space fast path. Canonicalize permuted fermion factors with the required sign and sort boson/qudit blocks after mapping. Add dense differential tests for same-space permutations, expanded target spaces, collisions, and cross-dimension rejection.

## MAJOR

### M1. Uniform-qudit `backend_mvp` cannot execute nonbinary local dimensions, while one unsupported mixed layout is silently accepted

Locations: `python/tencirpauli/structured.py:1114-1146`, `python/tencirpauli/structured.py:2227-2237`, and `python/tencirpauli/integrations/tensorcircuit.py:231-311`; contract: `docs/vibe/phase-7-spec.md:403-424,517-521`.

`QuditWeylOperator.compile("backend_mvp")` returns a `BackendMVPPlan` with `nqubits=0`, empty Pauli structural arrays and coefficients, and a private generic COO payload. `BackendMVPPlan.apply()` happens to execute that payload with NumPy, but the public TensorCircuit backend adapter derives a binary dimension from `plan.nqubits`, iterates the empty coefficient array, and rejects a length-`d` qudit state as incompatible. The missing capability is direct execution of the Weyl Hamiltonian through TensorCircuit's NumPy/JAX backend operations; converting the Hamiltonian to `QuditCircuit` is not needed and, by owner decision during this review, should be removed from the Phase 7 acceptance contract.

The availability check only rejects `backend_mvp` when bosons are present. A qubit-plus-qudit mixed layout therefore returns the same generic plan even though the frozen target matrix says mixed-dimension hybrid backend execution is deferred and must raise `NotImplementedError` without silent fallback.

This is a moderate, localized fix rather than a difficult new algorithm: every direct-Weyl term is a product of a diagonal phase and a cyclic permutation. The basic repair route is:

1. Add a versioned `direct_weyl` plan variant carrying `qudit_dimension`, `n_sites`, `local_dimensions`, canonical coefficients, and compact `a`/`b` exponent arrays of shape `(term_count, n_sites)`; do not encode the plan as `nqubits=0` or hide it in `_generic_entries`.
2. Make `BackendMVPPlan.dimension` and validation use the checked product of `local_dimensions`, not `2**nqubits`, and dispatch the executor by plan variant.
3. In `backend_mvp`, reshape a flat state to `(d,) * n_sites`. For each active site of each term, multiply by the broadcast local phase vector `omega ** (b * arange(d))`, then implement `X**a` as a static cyclic slice-and-`concat` shift along that axis. TensorCircuit backends already expose `reshape`, `concat`, `cast`, and ordinary static slicing, so this path can remain compatible with both NumPy and JAX/JIT without constructing a circuit or a COO matrix.
4. Preserve the existing optional coefficient override in canonical term order and validate its length against `term_count`.
5. Continue to reject finite-boson and genuinely mixed local-dimension backend plans with a direct `NotImplementedError`.
6. Add backend differential tests: compare `backend_mvp(plan)(state)` with `operator.compile("dense") @ state` for `d=3,4,5,6`, flat and rank-`n_sites` states, default and overridden coefficients, and TensorCircuit NumPy/JAX backends. Here “differential” means comparing two independent executions of the same Hamiltonian, not converting to `QuditCircuit`.

The frozen spec should be amended at `docs/vibe/phase-7-spec.md:420-422,519-521` to remove `QuditCircuit` conversion as an acceptance requirement while retaining uniform-qudit backend-MVP execution and differential tests.

### M2. Structured `native_mvp` is implemented as a retained COO matrix rather than the required matrix-free MVP

Locations: `crates/tencirpauli-native/src/structured.rs:57-101,526-553`, `crates/tencir-pauli-core/src/structured.rs:1122-1202`, and `python/tencirpauli/structured.py:2180-2191`.

`structured_sparse_plan` calls `structured_sparse_matrix`, stores `rows`, `columns`, and `values` for every compiled nonzero, and `apply()` walks those arrays in a single serial loop. Construction therefore costs `O(D * term_count)` transition work plus a hash table and sort, while plan memory is `O(nnz)`. This is a reusable sparse matrix disguised behind the `NativeMVPPlan` interface, not the matrix-free structured plan promised by the Phase 7 purpose and performance contract. It defeats the principal reason to request `native_mvp` instead of COO/CSR and must be treated as an incorrect target implementation, not merely a future optimization opportunity.

The public `estimated_bytes` is set to `dimension * 16`, which accounts only for one complex state-sized buffer and not the stored plan. In a reviewed 4096-dimensional, 8-term boson workload, the plan reported 65,536 bytes while the corresponding 32,004 COO triplets alone occupied 1,024,128 bytes, a 15.6x underestimate before native vector capacity or object overhead.

Resolution: store canonical local operations and coefficients once, apply them directly with reusable per-worker mixed-radix scratch, and parallelize only after release profiling identifies the crossover. If a sparse-materialized strategy is retained for selected workloads, name it honestly, expose compiled transition count, calculate actual array bytes, and select between matrix-free and sparse strategies using measured end-to-end evidence.

### M3. A valid finite boson operator can produce non-finite dense output even though its true complex128 matrix element is finite

Locations: `crates/tencir-pauli-core/src/structured.rs:1206-1261,1387-1411`; compare the sparse finite check at `1164-1166`.

The native boson transition kernel multiplies all ladder factors first and takes one square root afterward. The intermediate factorial can overflow even when the final square-root amplitude is representable. For `(b†)^171` with inclusive cutoff 171, the expected matrix element is approximately `3.522808638313566e154`, but `compile("dense")` returned `inf+nanj`. The same operator's COO path raised `ValueError` because sparse compilation checks the non-finite amplitude, so target semantics disagree.

Resolution: accumulate the amplitude as products of square roots or use another overflow-safe recurrence, then reject only when the final complex128 value is non-finite. Validate accumulated dense entries as well as sparse entries, and add target-equivalence regressions around the overflow boundary.

### M4. CAR/CCR expansion protection is applied after unbounded recursive rewriting, and the boson algorithm is unsuitable for the promised structured representation

Locations: `crates/tencir-pauli-core/src/structured.rs:499-579,731-811,814-946,994-1029`.

`fermion_rewrite` and `boson_rewrite` recursively clone factor sequences and build `BTreeMap`s but do not accept `max_bytes` and cannot stop on a running budget. The public callers call `push_aggregate()` only after a complete rewrite has returned. The multiplication preflight counts input term pairs, not contraction branches. In addition, `boson_sequence` expands compact power blocks back into repeated factors, discarding the main benefit of the public block representation. Integer contraction coefficients are accumulated in unchecked `i64`, so sufficiently high but structurally valid powers can wrap in release mode before conversion to complex128.

The running aggregate estimate is also incomplete: `push_aggregate()` checks `aggregate.len() + value_count` for only the current bucket rather than total stored contributions across all buckets (`1024-1028`).

Resolution: implement block-level CCR multiplication from the frozen closed form with checked combinatorial arithmetic and a checked product of per-mode expansion counts. Return `Result` from expansion routines and charge generated keys/contributions before allocation. For fermions, use canonical mode sets/bitsets and contraction-aware dynamic programming rather than recursive adjacent swaps.

### M5. `OperatorBuilder.add_product()` silently overwrites repeated Pauli factors on a site

Location: `python/tencirpauli/structured.py:2039-2124`, especially `2075-2087`.

The builder describes each entry as a product, and it correctly multiplies repeated qudit factors with phases, but qubit factors are treated as assignments. `qubits=((0, "X"), (0, "Y"))` returns `Y`; the correct ordered product is `iZ`. Even if repeated-site factors were intended to be unsupported, silent last-write behavior is unsafe.

Resolution: multiply qubit factors in input order using the Pauli product table and absorb the phase into the coefficient, or reject duplicate sites explicitly. Add a builder differential against readable operator expressions for all noncommuting local Pauli pairs.

### M6. Several hot paths violate the coarse-FFI/one-authoritative-kernel design and add maintenance complexity for negligible small-case gains

Locations: `python/tencirpauli/structured.py:337-437,939-992,1206-1253,2163-2179,2336-2500`.

Structured adjoint loops over terms and calls `FermionWord.adjoint()` or `BosonWord.adjoint()`, each of which constructs an operator through PyO3, creating one or two FFI crossings per term. Tensor product sends disjoint-domain factors through recursive Python normal ordering even though contractions are impossible and the graded sign can be computed directly. Meanwhile the finite transition semantics are duplicated in a long Python fallback and in Rust, selected by a hard-coded `dimension * term_count >= 64` threshold and guarded with `hasattr()` checks despite the native extension being mandatory.

This is both a performance problem and an over-defensive design: a required native-symbol mismatch should fail fast, while maintaining two numerical kernels creates target- and size-dependent behavior. The status document's smallest-case result was approximately 5.8 microseconds for both paths, which is not enough benefit to justify a second production implementation.

Resolution: make adjoint and tensor product direct canonical transformations with one batched aggregation, remove per-term FFI, and use one authoritative native finite kernel. Keep an independent Python implementation only under tests as a reference, not as a silent runtime fallback.

### M7. The P0–P5 handoff is materially incomplete despite the smoke gate passing

Locations: `docs/vibe/phase-7-spec.md:440-472,491-531`, `tests/test_structured_algebra.py:1-287`, and `benchmarks/python/test_structured_algebra_benchmark.py:1-288`.

The current 13 focused tests do not cover direct-Weyl multiplication/adjoint/commutation/Hermiticity, explicit embedding, partial-mapping algebra, uniform-qudit backend MVP, TensorCircuit NumPy/JAX execution, deterministic thread-count outputs, expansion limits, or property-based identities. The benchmark source has no uniform-Weyl chain benchmark, Holstein/spin-boson scaling, aggregation-heavy duplicate workload, guarded expansion case, or required memory/transition/thread/error metadata. `benchmarks/run.py list` contains no Phase 7/HEAD release record.

P5 public delivery is also absent: searches found no Phase 7 structured API material in `README.md`, `CHANGELOG.md`, non-vibe docs, or `examples/`. `python/tencirpauli/structured.py` has only seven docstrings across roughly 150 definitions/methods, and several public returns are typed as `Any`, so a passing mypy run does not establish the frozen strict public contract.

Resolution: do not change the status to complete until the missing correctness matrix, representative release records, public target-support table, examples, compatibility notes, release notes, and executable documentation examples are present.

## MINOR

### N1. `OperatorSpace` does not enforce the frozen upper bound on qudit dimensions

Locations: `python/tencirpauli/structured.py:480-503`; compare `QuditWeylWord` validation at `244-247`.

`OperatorSpace(qudits=...)` checks uniformity and `d >= 3` but not `d <= 2**32 - 1`. The invalid space can exist until a later word or FFI conversion fails. Validate the bound at space construction so all public qudit sites obey the same contract.

### N2. Domain-specific compile signatures still accept irrelevant mapping keywords

Locations: `python/tencirpauli/structured.py:1107-1146` and contract `docs/vibe/phase-7-spec.md:368-383`.

Boson and qudit operators inherit the generic `**kwargs` compile path and accept an explicit `fermion_mapping="jordan_wigner"` even though the contract requires domain-specific typed signatures that reject unrelated keywords. Replace the generic public signature with explicit overrides while retaining a shared private implementation.

## OBSERVATIONS

- The code has several sound foundations: Rust/Python separation is clean, long native calls release the GIL, public outputs are sorted deterministically in the covered paths, coefficient pruning remains exact-zero only, and existing small finite-target differentials pass.
- The full smoke workflow passing is useful regression evidence but is not Phase 7 acceptance evidence. The implementation-status warning at `docs/vibe/implementation-status.md:18` correctly distinguishes smoke execution from comparable release benchmarking.
- The adaptive Python/Rust finite-target split and `hasattr()` compatibility checks are the clearest over-engineering candidates. Removing them would reduce code size, semantic drift, and test burden without sacrificing a measured common-path advantage.

## P0–P5 delivery matrix

| Slice | Result | Main gap |
| --- | --- | --- |
| P0 references/API alignment | FAIL | Basic references and callable Pauli plans exist, but the frozen convention vectors and complete plan metadata/target tests are not delivered. |
| P1 fermion-to-Pauli | FAIL | Core construction/mapping exists, but property coverage, expansion-guard behavior, and post-mapping algebra correctness are incomplete. |
| P2 boson/native finite | FAIL | Basic targets exist, but overflow-safe numeric equivalence, guarded block-level expansion, and a genuinely compact native MVP are missing. |
| P3 hybrid native | FAIL | Core factories/algebra exist, but embedding is unsafe, partial mapping corrupts products, builder qubit products are wrong, and required hybrid differentials/scaling are absent. |
| P4 direct Weyl | FAIL | Basic Python algebra and finite targets exist, but the required nonbinary backend plan/executor, NumPy/JAX backend differentials, property suite, and release Weyl benchmarks are missing; `QuditCircuit` conversion is explicitly out of scope. |
| P5 delivery/handoff | FAIL | Quality smoke passes, but public docs/examples/release notes and a representative Phase 7 release benchmark record are absent. |

## RECOMMENDED IMPROVEMENTS

1. Fix C1 and C2 first and add dense, COO, CSR, and MVP regressions; both can silently return a different operator than requested.
2. Fix M3 and M5 next so all accepted inputs have consistent target semantics and builder construction cannot discard factors.
3. Replace the unguarded recursive expansion in M4 with checked block/set algorithms before increasing workload sizes or publishing performance claims.
4. Implement the factorized direct-Weyl backend path in M1 without a `QuditCircuit` conversion, and implement a matrix-free structured native plan with accurate metadata for M2.
5. Simplify M6 by removing per-term FFI and production Python fallback dispatch; retain Python logic only as an independent test oracle.
6. Complete the P0–P5 evidence in M7, rerun `python scripts/check.py --benchmark smoke`, then record a clean release benchmark run with the required workload and metadata matrix before changing Phase 7 status to complete.

## Validation performed

- `conda run -p .conda pytest -q tests/test_structured_algebra.py` — 13 passed.
- `conda run -p .conda python scripts/check.py --benchmark smoke` — passed format, Clippy, Ruff, mypy, Rust tests, release build, 201 Python tests, Rust benchmark smoke, and 156 selected Python benchmark-smoke cases.
- Targeted read-only probes reproduced C1, C2, M1, M3, M5, and the M2 metadata underestimate on the release-built extension.
