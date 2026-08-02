# Phase 5 implementation review — 2026-08-02

## Review scope

This review covers the Phase 5 arbitrary-width packed U1 specification, its Rust core implementation, the PyO3/Python boundary, correctness tests, and performance evidence in commits `cae1d06..15d1af0`. The review intentionally focuses on correctness, specification compliance, and performance issues that are material for the stated low-particle/low-hole workloads; it does not propose speculative abstractions or exact memory accounting.

No source files were changed. This review report is the only added file.

## Overall assessment

The arbitrary-width semantic work is sound and substantially complete. The implementation removes the 64/128-qubit occupation-word ceiling, preserves TensorCircuit basis ordering, groups contributions by X mask before exact-zero removal and leakage validation, retains deterministic destination-major storage, shares plan arrays through `Arc`, and keeps long native operations outside the GIL. The focused Rust and Python test suites pass, and this review found no clear algebraic, phase, ordering, or sparse-output correctness defect.

Phase 5 should nevertheless not be accepted as fully complete on performance grounds. The setup path still scales with every restricted source, every X group, and every packed limb in two passes. That cost is much larger than the canonical transition count for the representative low-k local Hamiltonians and is already visible between 128 and 512 qubits. In addition, the committed benchmark/profile package omits one required workload and several required metadata fields, so the completion claim is stronger than the evidence.

## Compliance checklist

| Area | Status | Evidence |
|---|---|---|
| Arbitrary-width packed occupation representation | PASS | `crates/tencir-pauli-core/src/sector.rs:112-137`, `821-895`; wide boundary tests in `tests/test_phase5_u1.py:84-153` |
| TensorCircuit ordering and particle-hole rank/unrank | PASS | `crates/tencir-pauli-core/src/sector.rs:140-227`, `300-330`; Rust exhaustive small-system checks at `crates/tencir-pauli-core/src/tests.rs:413-443` |
| Stable X-group aggregation, exact-zero removal, then leakage validation | PASS | `crates/tencir-pauli-core/src/sector.rs:919-975`; independent Python oracle at `tests/test_phase5_u1.py:49-81` |
| Deterministic flat destination-major plan shared by operator/MVP | PASS | `crates/tencir-pauli-core/src/sector.rs:334-418`, `437-439` |
| Public Python compatibility and batched wide basis materialization | PASS | `crates/tencirpauli-native/src/symmetry.rs:251-303`, `python/tencirpauli/symmetry.py:95-181` |
| Focused correctness verification | PASS | `cargo test --workspace`: 22 passed; `pytest -q tests/test_phase5_u1.py tests/test_symmetry.py`: 31 passed |
| Low-k setup complexity expected by the specification | FAIL | Specification requires rank near `O(min(k,n-k))` at `docs/vibe/phase-5-spec.md:212-215`; implementation repeatedly scans all limbs at `crates/tencir-pauli-core/src/sector.rs:300-324`, `930-945` |
| Required P5 benchmark/profile evidence | FAIL | Requirements at `docs/vibe/phase-5-spec.md:362-388`, `435-442`; gaps described below |

## CRITICAL

None.

## MAJOR

### 1. Restriction setup has a source × group × limb bottleneck that dominates the intended low-k workloads

The two construction passes at `crates/tencir-pauli-core/src/sector.rs:352-409` call `aggregate_source` for every restricted source. Each call visits every X group and scans every packed limb to form/count the destination at `crates/tencir-pauli-core/src/sector.rs:928-936`; it then scans every limb again for each term's Z parity at `crates/tencir-pauli-core/src/sector.rs:938-945`. Every retained destination is ranked through `rank_active_words`, which also walks all limbs before iterating active bits at `crates/tencir-pauli-core/src/sector.rs:300-324`. Thus the incremental low-particle/low-hole source iterator improves source generation, but the dominant work remains approximately `O(dimension × groups × word_count)` per pass rather than tracking actual candidate transitions or `O(min(k,n-k))` rank work as required by `docs/vibe/phase-5-spec.md:212-215`.

This is material rather than theoretical. A release-path nearest-neighbor `XX+YY`, k=2 check on this machine measured setup at approximately 11 ms / 124 ms / 1.24 s for 128 / 256 / 512 qubits, while canonical nnz was 32,004 / 129,540 / 521,220. Increasing n by four therefore increased nnz about 16.3× but setup about 113×. The corresponding steady MVP times were approximately 69 / 133 / 324 µs, confirming that the existing destination-major gather is not the problem. The repository's own 129q-to-256q measurements at `docs/vibe/implementation-status.md:138` already show the same setup scaling trend.

Resolution: retain the public API and destination-major steady plan, but change setup work so local/sparse masks do not scan all limbs for every source/group pair. A focused first step is to compile nonzero limb/support metadata for each X group and Z term, compute destination weight from the touched support, and rank low-k/low-hole destinations from occupied positions rather than a full limb scan. After that, use the existing candidate-count formula at `docs/vibe/phase-5-spec.md:233-244` to evaluate a group-oriented candidate enumeration or another internal setup strategy on the 128/256/512q chain. Parallel setup alone may lower constants but does not fix this scaling, so it should not be the first acceptance remedy.

### 2. P5 is marked complete without the benchmark/profile package required by its own frozen specification

The completion statement at `docs/vibe/implementation-status.md:130-142` says P0-P5 is implemented, but the committed benchmarks only cover nearest-neighbor hopping over the width/particle-number grid (`crates/tencir-pauli-core/benches/symmetry.rs:67-145` and `benchmarks/python/test_symmetry_benchmark.py:148-211`). The required long-range/duplicate-X aggregation workload at `docs/vibe/phase-5-spec.md:366-375` is absent. The wide Python records include n, k, word count, dimension, term count, and nnz, but omit distinct X-group count, plan/output bytes for most cases, thread count, and numerical error (`benchmarks/python/test_symmetry_benchmark.py:171-179`, `203-210`). Finally, `docs/vibe/implementation-status.md:142` calls Criterion timing plus code inspection a “profile” while also recording that the sampler attempt failed; this does not meet the explicit representative-profile acceptance condition at `docs/vibe/phase-5-spec.md:379-388`.

This evidence gap matters because the missing duplicate-X/long-range case is the workload that would distinguish group aggregation costs from simple chain scaling, and a scaling-oriented profile would have exposed the bottleneck above more clearly.

Resolution: add one stable cross-limb long-range/duplicate-X benchmark at both Rust and Python/FFI levels, record the missing metadata, and capture one actual release profile using an available profiler or deterministic instrumentation counters if OS sampling remains unavailable. Add a 512q/k=2 setup point or an equivalent scaling point because it remains modest in plan memory while making the asymptotic behavior visible. Update the completion checkpoint only after this evidence is recorded and the setup issue is either fixed or explicitly accepted as a measured limitation.

## MINOR

None that warrant blocking Phase 5. The remaining omissions seen in individual boundary-test permutations are covered sufficiently by the combined Rust property tests, small dense differential tests, and wide Python sparse oracle; expanding them mechanically would add little confidence.

## OBSERVATIONS

The wide `basis_words_packed` implementation at `crates/tencir-pauli-core/src/sector.rs:230-253` performs a full `unrank_into` scan for each output row rather than reusing the incremental iterator. This is not the main plan-setup bottleneck and the packed output itself is large, so it does not need to block acceptance. It is a reasonable follow-up only if a public `basis_words()` benchmark shows meaningful cost.

The `max_bytes` estimate is checked after `U1Sector` combinatorics and compiled terms have already been allocated (`crates/tencir-pauli-core/src/sector.rs:346-350`). For the stated 64-256q workloads these allocations are small relative to the plan, so this review does not recommend redesigning allocation accounting. It is worth preserving as a documented best-effort limitation rather than claiming that every setup allocation is guarded before it occurs.

## RECOMMENDED IMPROVEMENTS

1. Address Major 1 with support-aware/sparse-limb aggregation and low-k/low-hole rank data, then compare 128/256/512q setup, steady MVP, nnz, and memory against the current release results.
2. Address Major 2 by adding the long-range/duplicate-X workload, complete metadata, and one reproducible release profile before declaring P5 accepted.
3. Keep the existing destination-major MVP representation, Python API, exact-zero semantics, deterministic ordering, and post-aggregation leakage rules unchanged; the review found no evidence that these areas need redesign.

## Verification performed

- `conda run -p .conda cargo test --workspace` — 22 passed.
- `.conda/bin/python -m pytest -q tests/test_phase5_u1.py tests/test_symmetry.py` — 31 passed.
- Read-only release-path scaling check for nearest-neighbor `XX+YY`, k=2 at 64/128/192/256/384/512 qubits; no benchmark artifacts were written.
- Source review of the Phase 5 commits, frozen specification, implementation status, Rust/Python APIs, tests, and benchmark definitions.

## Remediation outcome — 2026-08-02

Major 1 is resolved in the current implementation without changing the public API or destination-major storage. Compiled X-group and Z-term support positions let setup compute active-weight intersections and parity without scanning every packed limb for every source/group pair; sector-preserving destinations are ranked from low-particle or low-hole active positions, and leakage remains checked only after stable aggregation and exact-zero removal.

Major 2 is resolved by adding the required cross-limb long-range/duplicate-X workload to both Rust Criterion and Python/FFI benchmarks, adding the 512q/k2 scaling point, and recording source dimension, term count, distinct X-group count, nnz, word count, plan/output bytes, thread count, and numerical error in the Python benchmark metadata. Release benchmark runs completed for the new cases; the local macOS sampler remains unavailable, so the repository records deterministic release-boundary workload metadata and timings instead of sampler percentages.

Remediation verification: `cargo test --workspace` passed 22 tests, `pytest -q tests/test_phase5_u1.py tests/test_symmetry.py` passed 31 tests, workspace Clippy passed with `-D warnings`, `cargo bench --locked -p tencir-pauli-core --bench symmetry -- --test` completed all registered cases, and the focused Python long-range benchmark passed. Existing exact-zero, Y-phase, wide-boundary, ordering, sparse-output, and post-aggregation leakage tests remain green.
