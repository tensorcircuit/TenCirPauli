# Phase 7.5 implementation and performance review

Review date: 2026-08-03

Reviewed commit: `3a18072` (`docs: record ChargeSector performance evidence`), covering the Phase 7.5 implementation range after the accepted Phase 7 baseline.

Scope: adversarial review of Majorana algebra and conversion, JW/parity/BK mapping plans, exact additive-charge analysis, charge-sector construction, restricted native execution, memory behavior, and the Phase 7.5 correctness and benchmark evidence. The review prioritizes correctness, availability, end-to-end performance, and alignment with `phase-7.5-spec.md`; it deliberately excludes cosmetic cleanup and speculative abstractions.

This report records findings and remediation requirements only. No Rust, Python, test, benchmark, specification, or status implementation was changed during the review.

## Verdict

Phase 7.5 works correctly on the committed small-system fixtures, and the native migrations materially improve Majorana-to-fermion expansion, mapping-plan setup, Pauli batch transforms, and ChargeSector DP/rank/unrank/basis generation. However, the phase should not remain marked fully accepted in its current form. One exact-integer correctness defect can report a broken charge as conserved, the public Majorana mapping path has exponential intermediate complexity even when the exact mapped output contains one Pauli word, and two core scalability/resource contracts are not met. Reopen the P3 correctness gate and the P1/P2/P4 performance and availability gates until the CRITICAL and MAJOR findings below are closed.

## Executive remediation matrix

| ID | Severity | Area | Required outcome | Suggested owner task |
| --- | --- | --- | --- | --- |
| C1 | CRITICAL | Additive-charge analysis | Conservation decisions remain exact for arbitrary accepted integer weights and never pass through a lossy complex128 charge generator. | Implement exact transition/selection-rule analysis and large-integer regressions. |
| C2 | CRITICAL | Majorana mapping | Map a canonical Majorana word directly to one Pauli word without `Majorana -> fermion -> JW` expansion. | Add a direct Rust Majorana-to-Pauli batch kernel and route public mapping/compile through it. |
| M1 | MAJOR | Majorana representation/conversion | Use packed `u64` support internally and move fermion-to-Majorana expansion out of Python. | Replace index-vector hot-path keys and add a native inverse-conversion batch. |
| M2 | MAJOR | Resource guards | Reject oversized charge-sector requests before constructing per-level Python contribution tables; honor `max_bytes` on every analysis path. | Move compact contribution construction into Rust and fix Pauli analysis budget propagation. |
| M3 | MAJOR | Parity/BK execution and evidence | Separate public canonical CNOT provenance from the internal transform and benchmark real unique-term scaling. | Add a direct symplectic transform and repair the mapping workload generator. |
| M4 | MAJOR | Restricted compiler | Remove per-state heap keys and per-source-per-term occupation cloning from the generic restricted compiler. | Compile against `ChargeSectorPlan` with reusable scratch/rank primitives and representative scale benchmarks. |
| M5 | MAJOR | Acceptance matrix | Cover the representative workloads and mapping/restriction ordering required by the frozen specification. | Expand correctness tests and produce a new clean release benchmark record after C1–M4. |

## Compliance checklist

| Requirement | Result | Evidence |
| --- | --- | --- |
| Small-system Majorana signs, adjoints, Fock matrices, and round trips | PASS | The focused Phase 7.5 Python suite and Rust tests pass. |
| JW/parity/BK small encoded-basis differentials | PASS | Frozen matrix/provenance tests and small dense differentials pass. |
| Exact integer additive-charge semantics | FAIL | C1 reproduces a false conserved result for adjacent integers above the exact f64 range. |
| Direct scalable Majorana mapping | FAIL | C2 shows exponential intermediate expansion and approximately 749 ms for a degree-10 one-output word. |
| Packed Rust Majorana representation | FAIL | M1: the core stores sorted index vectors and performs quadratic cross scans. |
| Fail-fast major-allocation policy | FAIL | M2: a one-byte budget request allocates tens of MiB before rejection. |
| Representative mapping performance evidence | FAIL | M3: the nominal 16/64/128-term workload canonicalizes to four terms at every scale. |
| Scalable restricted native setup | PARTIAL | Numerical results are correct, but M4 retains allocation-heavy basis maps and is benchmarked only at dimension 70 in the official Phase 7.5 suite. |
| Existing optimized U1 implementation remains intact | PASS | The generic ChargeSector implementation does not replace the U1 engine. |
| Full frozen correctness/performance matrix | FAIL | M5 lists missing exact-integer, mapping/restriction, simultaneous-sector, boson, and mixed-workload gates. |

## CRITICAL

### C1. Exact integer charge analysis becomes lossy complex128 arithmetic and can return a false symmetry

Locations: `python/tencirpauli/charge.py:178-252,985-1015`; contract: `docs/vibe/phase-7.5-spec.md:35-44,245-270,471-475`.

#### Current problem

`AdditiveCharge` stores Python integers, but `as_operator()` converts fermion and boson weights with `complex(weight)`. `analyze_charge()` then constructs the charge generator and computes a complex128 canonical commutator. Distinct accepted integer weights above `2**53` can therefore become the same binary64 value before the conservation decision is made.

The reproduced case is:

```python
space = tcp.OperatorSpace(fermions=2)
h = tcp.FermionOperator.from_terms(
    2,
    [(((0, "create"), (1, "annihilate")), 1.0)],
)
charge = tcp.AdditiveCharge(
    space,
    fermions={0: 2**53, 1: 2**53 + 1},
)
analysis = h.analyze_charge(charge)
```

Mathematically, the hopping changes the charge by one and `[H,Q] != 0`. The reviewed implementation returns `is_conserved=True` and `commutator_term_count=0` because `complex(2**53) == complex(2**53 + 1)`.

#### Impact

The public `conserves()` and `analyze_charge()` APIs can make a false correctness claim for inputs explicitly accepted by the exact-integer contract. `restrict_charge()` performs an additional leakage check during transition construction and often still rejects the inconsistent sector, but that later check does not repair the incorrect public symmetry result and is not a substitute for exact P3 analysis.

#### Required resolution

1. Make the conservation authority independent of a materialized complex128 charge generator. Analyze canonical physical transitions using exact integer charge labels or exact charge deltas.
2. For raw fermion and boson monomials, compute charge changes from creation/annihilation multiplicities and exact integer weights. For qubit X/Y contributions, expand or group the finite local transitions so cancellation cases such as `XX + YY` are aggregated before the conservation decision.
3. Avoid multiplying arbitrary-size charge integers into complex128 merely to decide whether a commutator is zero. A robust approach groups equal physical transitions first, removes exact-zero complex amplitudes under the repository's existing canonical coefficient semantics, and then compares exact source/destination charges.
4. Preserve the documented `commutator_term_count`. If the optimized transition analyzer cannot derive the canonical count directly, retain a bounded exact symbolic path for the diagnostic count or document and obtain an owner decision for a revised result field. Do not silently return a transition count under the existing name.
5. Keep `AdditiveCharge.as_operator()` as a diagnostic/materialization API only. It must either prove every generated coefficient is complex128-compatible without changing the charge distinctions relevant to the result, or fail explicitly with a precise representability error. It must not be used as the exact symmetry oracle.

#### Required regressions

- Fermion hopping with weights `(2**53, 2**53 + 1)` is reported nonconserving.
- The same test is repeated near `2**63`, `2**127`, and with negative weights.
- Equal large weights remain conserved.
- Large offsets do not affect conservation.
- `XX + YY` cancellation remains conserved, while `XX` alone remains broken.
- Mixed fermion-boson and qubit fixtures compare with an independent exact-integer transition reference.
- `as_operator()` either produces a verified representation or raises an explicit representability error for integers outside its numeric contract.

#### Closure gate

C1 closes only when the large-integer false-positive reproducer fails before the fix and passes afterward, the cancellation regressions remain correct, and the focused plus full quality gates pass. A fix that merely rejects all values above `2**53` requires an explicit change to the frozen exact-integer contract and is not an implementation-only closure.

### C2. Majorana mapping uses an exponential fermion intermediate even when the exact output has one Pauli word

Locations: `python/tencirpauli/majorana.py:385-458`, `python/tencirpauli/mapping.py:411-423`, and `crates/tencir-pauli-core/src/majorana.rs:171-250`; contract: `docs/vibe/phase-7.5-spec.md:84-107,109-198,459-469`.

#### Current problem

`MajoranaOperator.map_fermions()` calls `self.to_fermion()`, and `FermionQubitMapping.map_fermion_operator()` maps the resulting FermionOperator through Jordan-Wigner before applying parity/BK Clifford conjugation. A canonical degree-`k` Majorana word maps directly to one Pauli word, but the current path can create `2**k` fermion branches followed by up to another `2**k` Jordan-Wigner branch expansion before cancellation.

The reviewed release build produced the following end-to-end timings for one Majorana word containing one even Majorana generator on each distinct mode. Every case returned exactly one mapped Pauli term:

| Majorana degree | Current JW mapping time |
| ---: | ---: |
| 4 | approximately 0.27 ms |
| 6 | approximately 1.16 ms |
| 8 | approximately 20.45 ms |
| 10 | approximately 748.94 ms |

The degree-8 to degree-10 increase is approximately 36x for an output whose structural size remains one term. This is an algorithmic mismatch, not ordinary Python overhead.

#### Impact

Moderate-degree Majorana chains and observables become unusable despite having compact exact Pauli representations. `MajoranaOperator.compile()` inherits the same path, so native/backend MVP compilation is also blocked by the symbolic intermediate. The existing benchmark only exercises degree-two Majorana terms and therefore does not detect this failure mode.

#### Required resolution

1. Implement a pure Rust batch kernel that maps canonical Majorana words directly to JW Pauli words with exact discrete phases. Each input Majorana term should produce one pre-aggregation Pauli word.
2. Apply parity or BK through the reusable mapping plan after the direct JW word has been formed. Do not construct a FermionOperator in this workflow.
3. Aggregate equal Pauli words once in Rust, remove exact zeros, sort deterministically, and cross PyO3 once for the complete operator.
4. Route both `MajoranaOperator.map_fermions()` and `MajoranaOperator.compile()` through the direct kernel. Keep `to_fermion()` as an explicit user-requested algebraic conversion, not a hidden compilation dependency.
5. Reuse the packed representation requested by M1 so generator products and phase accumulation do not allocate one Python or Rust vector per local factor.

#### Required regressions and benchmarks

- Dense differentials for all Majorana words through a bounded degree on two to four modes under JW, parity, and BK.
- Random bounded Majorana operators compared with the independent encoded-basis permutation reference.
- Degree 4/8/16/32 single-word mapping, recording input degree, output terms, mapped Pauli weight, runtime, throughput, and bytes.
- Multi-term Majorana-chain workloads with duplicate-output aggregation.
- A regression asserting that a single canonical Majorana word creates one pre-aggregation mapped word rather than `2**degree` fermion terms.
- First and reused mapping-plan timing, including Python/Rust boundary costs.

#### Closure gate

C2 closes when direct mapping no longer calls `to_fermion()`, degree scaling is consistent with word width and mapping-transform cost rather than exponential branch count, and all three mapping dense differentials pass.

## MAJOR

### M1. The Rust Majorana core does not implement the frozen packed-support representation, and inverse conversion remains a Python branch loop

Locations: `crates/tencir-pauli-core/src/majorana.rs:8-13,53-123,125-163,253-294` and `python/tencirpauli/majorana.py:36-70,487-525`; contract: `docs/vibe/phase-7.5-spec.md:103-107,459-463`.

#### Current problem

The Rust batch types and canonical aggregate use `Vec<Vec<u64>>` sorted index lists. Word multiplication counts cross inversions through a nested iterator and builds another index vector; operator aggregation uses those heap-owned vectors as `BTreeMap` keys. The public Python `MajoranaWord.multiply()` separately implements the same quadratic index-pair scan. The representation therefore does not match the frozen packed `u64` limb requirement.

The reverse `FermionOperator.to_majorana()` path remains entirely in Python. It expands factors with nested dictionaries, canonicalizes each branch using Python lists, and aggregates Python `MajoranaWord` objects. The unavoidable output branching is therefore combined with avoidable interpreter and allocation overhead.

#### Impact

Large mode counts, dense Majorana supports, and operator multiplication pay unnecessary heap traffic, comparisons, and quadratic scans. C2 is the dominant practical failure, but leaving the underlying representation unchanged would preserve avoidable costs after direct mapping is added and would continue to contradict the P1 delivery contract.

#### Required resolution

1. Introduce a private packed support key using `Vec<u64>` limbs or a small-inline/owned-wide representation consistent with the existing Pauli packed design. The public tuple of indices remains unchanged.
2. Implement support XOR for multiplication and exact cross-inversion parity using a limb scan. Start with a simple correct indexed-prefix/popcount implementation; add architecture-specific SIMD only if a release profile justifies it.
3. Use packed keys in native aggregation. Convert to sorted public indices only once when returning the final deterministic result.
4. Move fermion-to-Majorana conversion into one Rust batch call. Preserve exact `1/2` and `±i/2` local factors as exact discrete/power-of-two data until absorption into complex coefficients.
5. Keep the Python code as validation and object shaping only; do not retain a second production algebra implementation after the native path is established.

#### Required regressions and benchmarks

- Packed/index round trips across generator indices 63/64/65, 127/128/129, and multi-limb sparse/dense supports.
- Exhaustive small-word multiplication signs against explicit Fock matrices.
- Deterministic operator multiplication across repeated runs and supported thread counts if parallelism is introduced.
- Fermion-to-Majorana degree and term-count scaling at small/medium/large sizes, compared with the current Python path before removal.
- End-to-end Majorana construction, multiplication, both conversion directions, and mapping in the same release manifest.

#### Closure gate

M1 closes when production multiplication and conversion use packed native keys, the public API remains unchanged, and release A/B results demonstrate no regression for tiny words and a material improvement for representative wide/dense supports.

### M2. Charge-sector memory guards run after the largest Python allocation, and the Pauli analysis path does not honor the caller budget

Locations: `python/tencirpauli/charge.py:441-518,603-625,985-1015`, `crates/tencir-pauli-core/src/charge_sector.rs:189-315`, and `python/tencirpauli/pauli.py:486-504,841-846`; contract: `docs/vibe/phase-7.5-spec.md:35-44,401-418,433`.

#### Current problem

`ChargeSector.__init__()` constructs a full Python contribution table with one tuple per axis level before calling `_native.charge_sector_plan()`. The native memory guard therefore cannot prevent the Python list/tuple graph or PyO3 input conversion from consuming memory. The stored Rust `estimated_bytes` also does not include all contribution-table storage and cross-language temporaries.

A read-only probe using one boson mode, explicit inclusive cutoff 200,000, and `max_bytes=1` still spent approximately 466 ms and reached approximately 33.6 MB of Python-traced peak memory before raising `MemoryError`. The correct behavior for this request is an immediate cheap rejection.

Separately, the Pauli branch of `analyze_charge()` calls `operator.commutator(pauli_generator)` without forwarding `max_bytes`. The current Pauli `_binary()` native call does not consume the validated budget either. A direct `PauliOperator.analyze_charge(..., max_bytes=1)` can therefore return a result instead of enforcing the requested bound.

#### Impact

Malformed or accidentally huge boson cutoffs can cause avoidable latency and memory pressure before the documented guard is reached. On a service or notebook process, this is an availability defect even though the final call raises an exception. The ignored Pauli analysis budget makes public behavior domain-dependent.

#### Required resolution

1. Pass compact axis dimensions, exact weights/levels, offsets, and targets into Rust. Construct repeated level contributions inside the native plan after checked dimension and byte preflight.
2. Before any per-level Python or Rust allocation, compute a cheap upper bound from `sum(local_dimensions) * constraint_count`, checked container overhead, and planned DP storage. This remains best-effort; exact allocator accounting is not required.
3. Include persistent contribution storage, target vectors, suffix maps, and major temporary buffers in `estimated_bytes` with one documented convention.
4. Avoid serializing nested Python lists of every contribution when the contribution is a simple affine function of the local occupation.
5. Thread the caller's `max_bytes` through Pauli charge analysis and into any native commutator or replacement exact analyzer. Add the necessary native parameter rather than validating and then discarding the value.
6. Preserve `max_bytes=None` as an explicit unbounded setting and retain checked arithmetic/overflow even when unbounded.

#### Required regressions

- Large explicit boson cutoff plus `max_bytes=1` fails before constructing a per-level Python table.
- Low-budget tests for pure fermion, boson, qubit, simultaneous-charge, and qudit-spectator sectors.
- `PauliOperator.analyze_charge(..., max_bytes=1)` either raises `MemoryError` before a major output/workspace or completes only when the operation demonstrably fits the same documented budget policy.
- `max_bytes=None` succeeds for a bounded fixture.
- Plan `estimated_bytes` is monotonic with axis levels and constraint count and covers every retained native buffer category.

#### Closure gate

M2 closes when low-budget rejection occurs before material Python expansion, the Pauli and structured paths follow the same public policy, and a focused memory probe shows bounded behavior without introducing exact-RSS accounting or allocator instrumentation into production.

### M3. Parity execution replays an O(n²) provenance circuit, while the mapping benchmark silently collapses every term-count scale to four canonical terms

Locations: `crates/tencir-pauli-core/src/mapping.rs:44-98,102-177,222-250`, `python/tencirpauli/mapping.py:336-472`, and `benchmarks/python/test_phase75_benchmark.py:52-75,315-383`; contract: `docs/vibe/phase-7.5-spec.md:123-194,435-449,465-469`.

#### Current problem

The mapping plan correctly exposes the frozen canonical CNOT provenance, but `map_pauli_terms()` uses that provenance as the execution algorithm for every word. The prefix-parity plan contains `n(n-1)/2` canonical row-reduction CNOTs even though the same supported transform can be applied with a direct packed symplectic formula or a linear-size internal network. Public provenance and internal execution do not need to be identical under the frozen contract.

Using genuinely unique canonical Pauli terms, the reviewed release build produced:

| Modes | Unique terms | Mapping | CNOT provenance count | Median mapping time |
| ---: | ---: | --- | ---: | ---: |
| 32 | 64 | JW | 0 | approximately 0.29 ms |
| 32 | 64 | parity | 496 | approximately 0.36 ms |
| 64 | 128 | JW | 0 | approximately 0.94 ms |
| 64 | 128 | parity | 2,016 | approximately 1.42 ms |
| 128 | 256 | JW | 0 | approximately 3.42 ms |
| 128 | 256 | parity | 8,128 | approximately 7.05 ms |
| 128 | 256 | BK | 448 | approximately 3.63 ms |

The official mapping A/B workload is more problematic. Its generated Pauli codes are periodic modulo four, so the nominal 16-, 64-, and 128-term inputs all canonicalize to exactly four operator terms before the timed call. The benchmark records the requested `term_count` as `input_terms`, not the actual canonical input size, and therefore cannot support the claimed small/medium/large batch-scaling evidence.

The compatible-hybrid parity/BK path also transforms surviving mapped fermion codes in a Python per-term loop after the native JW step. This does not cross PyO3 per term, but it leaves one agreed-scope hot path outside the Rust batch transform.

#### Required resolution

1. Retain the canonical CNOT sequence as immutable public provenance and test metadata.
2. Add a separate internal packed symplectic transform for each supported mapping. For parity, use a direct prefix relation; for BK, use Fenwick update/parity sets or an equivalent packed GF(2) transform. The exact public encoded operator must remain unchanged.
3. Batch hybrid mapped-code transformation and aggregation in Rust. Preserve boson, physical-qubit, and qudit factors and their axis metadata.
4. Replace `_mapping_ab_workload()` with a deterministic generator that guarantees the requested number of unique canonical Pauli words. Assert `len(operator.terms) == requested_term_count` before timing.
5. Record both raw requested terms and actual canonical input/output terms, Pauli weight distribution, CNOT provenance count, plan bytes, throughput, and numerical error.
6. Include mapping-sensitive long parity strings rather than only dense periodic code patterns.

#### Required regressions and benchmarks

- Exact equality between the optimized transform and canonical CNOT conjugation for random bounded Pauli words under all three mappings.
- Dense encoded-basis differentials remain the correctness authority.
- Unique term-count scaling at 1/16/64/256/1024 terms and mode-count scaling at 8/32/128/512 where memory-safe.
- Pure fermion, compatible hybrid, and direct Majorana mapping cases.
- First-plan and reused-plan timings, including public Python conversion costs.

#### Closure gate

M3 closes when the timed implementation no longer loops over the full parity provenance per word, the hybrid transform is batched, and the repaired benchmark verifies its actual canonical term count before recording a clean release comparison.

### M4. The generic restricted compiler materializes allocation-heavy basis maps and clones one occupation vector per source-term pair

Locations: `python/tencirpauli/charge.py:720-982`, `crates/tencir-pauli-core/src/charge.rs:232-403`, and `crates/tencir-pauli-core/src/charge_sector.rs:17-185`; contract: `docs/vibe/phase-7.5-spec.md:301-399,435-449,477-481`.

#### Current problem

Restricted construction first materializes the complete selected basis into a NumPy array. Rust then constructs `HashMap<Vec<u64>, u64>` by allocating one owned vector per basis row. For every source state and every operator term, it clones the complete occupation vector and aggregates candidate destinations in a `BTreeMap<Vec<u64>, Complex64>`. Final transitions are accumulated in another global `BTreeMap<(u64,u64), Complex64>`.

This is numerically correct and preserves the essential rule that equal destinations are aggregated before leakage is rejected. The problem is the amount of allocation, hashing, ordered-tree work, and duplicated basis storage on the primary scalable native setup path.

An equivalent fixed-particle-number nearest-neighbor hopping workload produced:

| Modes | Sector dimension | Generic ChargeSector setup | Existing U1 setup | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 70 | approximately 0.43 ms | approximately 0.047 ms | 9.1x |
| 12 | 924 | approximately 2.39 ms | approximately 0.67 ms | 3.6x |
| 16 | 12,870 | approximately 45.7 ms | approximately 12.6 ms | 3.6x |
| 18 | 48,620 | approximately 209.9 ms | approximately 56.8 ms | 3.7x |
| 20 | 184,756 | approximately 973.4 ms | approximately 248.5 ms | 3.9x |

The generic path is not required to equal the specialized U1 engine, and preserving U1 separately is correct. The issue is that the official Phase 7.5 restricted benchmark exercises only 8 modes and dimension 70, so it neither exposes this scaling nor demonstrates acceptable behavior on the representative simultaneous, bosonic, or mixed sectors required by the specification.

#### Required resolution

1. Let restricted compilation borrow or own an `Arc<ChargeSectorPlan>` rather than receiving a fully materialized basis plus rebuilding a reverse lookup.
2. Iterate source states through native `unrank_into()` with caller-owned scratch. Add a private allocation-free `rank_from_validated_into()` or equivalent destination lookup using the existing suffix DP.
3. Reuse destination occupation scratch across terms. Avoid one `Vec<u64>` allocation per source-term pair.
4. Preserve cancellation-before-leakage exactly. For each source, aggregate equal destination occupations or compact destination keys before checking sector membership. Leaking destinations whose aggregate coefficient becomes exact zero must still be discarded without error.
5. Use an unordered private aggregate on the hot path and impose deterministic order once when final arrays are emitted. Hash iteration must not become public output order.
6. Avoid a second global ordered transition map when source traversal and destination ordering can produce deterministic arrays directly or with one final sort.
7. Precompute fermion parity support/prefix data and term-local active positions where profiling shows `apply_fermions()` prefix summation is material.
8. Keep the current sparse transition plan if representative memory/steady-apply benchmarks justify it; a new universal operator abstraction is not required.

#### Required regressions and benchmarks

- Existing exact-zero leakage cancellation regression remains mandatory.
- Independent `P† H P` comparisons for fermion, boson, qubit, hybrid, simultaneous charge, negative weights, and qudit spectators.
- Setup scaling for fixed-number sectors through at least dimensions 70/924/12,870/184,756 where the local memory policy permits.
- Simultaneous particle number plus `2Sz` on a spinful Hubbard fixture.
- Bose-Hubbard fixed total occupation and mixed fermion-boson excitation fixtures.
- Construction peak workspace, retained plan bytes, transition count, first apply, steady apply, COO/CSR materialization, and numerical error.
- Same-machine comparison with U1 remains informational and must not lead to replacing the specialized U1 engine.

#### Closure gate

M4 closes when restricted construction no longer materializes owned basis-row keys or clones a fresh occupation vector for every source-term pair, all exact leakage semantics pass, and the new release record covers representative sector dimensions and domains.

### M5. The current acceptance tests and benchmark manifest do not cover several frozen Phase 7.5 gates

Locations: `tests/test_phase75_charge.py`, `tests/test_phase75_majorana_mapping.py`, `benchmarks/python/test_phase75_benchmark.py`, and `docs/vibe/implementation-status.md:81-108`; contract: `docs/vibe/phase-7.5-spec.md:420-487`.

#### Current problem

The committed focused tests cover important small examples, but the following required gates are absent or too weak to catch C1–M4:

- No adjacent large-integer charge-weight differential.
- No direct high-degree Majorana mapping scale test; the mapping fixture uses degree-two terms.
- No packed Majorana boundary test despite the explicit packed-index requirement.
- No mapping/restriction ordering differential under all three mappings.
- No performance benchmark for charge analysis setup.
- No representative restricted setup benchmark beyond the 70-state sector.
- No simultaneous-charge, Bose-Hubbard, or mixed excitation-conserving restricted performance workload.
- The mapping term-count workload collapses to four canonical terms.
- The U1 comparison is only an 8-qubit, 70-state point.

The existing tests passing therefore establishes small-fixture regression safety, not completion of the frozen correctness and performance matrix.

#### Required resolution

1. Convert every C1–M4 reproducer into a deterministic committed test or benchmark before changing the implementation.
2. Add an independent mapping/restriction ordering differential. Construct the selected physical basis independently, compare restriction-first results with the appropriately encoded mapped full operator on small systems, and do not use implementation-generated basis permutations as the only oracle.
3. Add charge-analysis setup benchmarks separated from restricted-plan construction.
4. Add the representative workloads listed in the specification with safe small correctness sizes and larger release performance sizes.
5. Record a new clean Phase 7.5 benchmark manifest only after the workload generator assertions and implementation remediations pass. The manifest must include the commit, clean-tree state, input/canonical/output term counts, dimensions, transitions, plan/workspace/output bytes, thread environment, runtime/throughput, and numerical error.
6. Update `implementation-status.md` from accepted to remediation/under-review while the CRITICAL findings remain open, then record closure evidence only after a clean committed run.

#### Closure gate

M5 closes when the complete focused quality gate and the repaired release benchmark manifest pass on a clean committed tree and the status document records C1–M4 disposition with reproducible labels rather than relying on the superseded Phase 7.5 acceptance record.

## Recommended implementation order

1. **Correctness first:** fix C1 and add exact large-integer/cancellation regressions. Do not optimize the current floating commutator path before removing it as the conservation authority.
2. **Eliminate the exponential path:** implement C2 together with the minimum packed primitives from M1 needed for direct Majorana mapping.
3. **Close representation debt:** finish packed Majorana multiplication and native fermion-to-Majorana conversion, then run both-direction A/B benchmarks.
4. **Restore availability guarantees:** implement M2 before expanding bosonic performance tests so benchmark inputs cannot bypass preflight guards.
5. **Optimize mapping execution and evidence:** implement M3 and repair the unique-term workloads.
6. **Optimize restricted setup:** implement M4 against `ChargeSectorPlan`, retaining all cancellation/leakage semantics.
7. **Acceptance handoff:** complete M5, run the full quality gate, record the clean release manifest, and only then restore Phase 7.5 accepted status.

## Validation performed during this review

- `conda run -p .conda cargo test --workspace` — 36 Rust tests passed.
- `conda run -p .conda pytest -q tests/test_phase75_charge.py tests/test_phase75_majorana_mapping.py` — 23 focused Python tests passed.
- Exact-integer adversarial probe — reproduced false conservation for weights `2**53` and `2**53 + 1`.
- Majorana mapping probe — reproduced approximately 0.27/1.16/20.45/748.94 ms for degree 4/6/8/10 one-output words.
- Mapping workload audit — confirmed the nominal 16/64/128-term benchmark inputs each contain only four canonical terms.
- Unique mapping probe — measured JW/parity/BK scaling through 128 modes and 256 unique terms.
- Charge-sector allocation probe — reproduced approximately 466 ms and 33.6 MB Python-traced peak before a one-byte-budget rejection at boson cutoff 200,000.
- Restricted setup probe — measured generic ChargeSector and specialized U1 setup through 20 modes and sector dimension 184,756.

All probes were read-only with respect to production source. Benchmark numbers are informational same-machine release measurements, not CI wall-time gates.
