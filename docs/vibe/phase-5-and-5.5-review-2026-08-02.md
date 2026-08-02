# Phase 5 remediation and Phase 5.5 review — 2026-08-02

## Review scope

This review covers the Phase 5 remediation commit `2b162d5`, the Phase 5.5 implementation commit `a062789`, and the opt-in pre-commit benchmark change `49ee0b6`. The review focuses on numerical correctness, hot-path implementation, end-to-end performance behavior, parallel memory behavior, and the acceptance evidence required by the frozen Phase 5 and Phase 5.5 specifications. The removal of full benchmark recording from the default commit hook is treated as an intentional owner decision, not as a defect.

At the time of the initial review no source files had been changed. The remediation and evidence recorded below are the follow-up implementation for this report.

## Overall assessment

The Phase 5 remediation is technically sound and can be accepted at the implementation level. It removes repeated packed-limb scans from the source/group hot path, uses low-particle or low-hole active positions, preserves deterministic stable aggregation and post-aggregation leakage validation, and passes the complete current correctness and quality gates. The remaining source-by-X-group scaling is visible at 512 qubits, but it is consistent with the source-ordered baseline permitted by the frozen specification and is not by itself a correctness blocker.

Phase 5.5 is numerically sound and its main architecture is appropriate: one immutable program is shared through `Arc`, the scalar forward/reverse kernels remain authoritative, the FFI is coarse-grained, and row order is deterministic. It should not yet be called fully complete, however. The current parallel heuristic creates a severe latency cliff for light Clifford workloads, and the batch memory guard does not account for the known concurrent checkpoint/branching workspace of active workers. The committed acceptance benchmark also omits the workload and thread-scaling cases that would have exposed both issues.

The pre-commit change is correct for the stated workflow. The hook defaults to benchmark smoke checks, retains all formatting/lint/type/correctness/release-build gates, and leaves full release recording explicitly opt-in. Full performance records should remain manual and should not be restored to every commit.

## Compliance checklist

| Area | Status | Evidence |
|---|---|---|
| Phase 5 active-position setup semantics | PASS | `crates/tencir-pauli-core/src/sector.rs:991-1084`; focused and full suites pass |
| Phase 5 low-particle/low-hole rank path | PASS | `crates/tencir-pauli-core/src/sector.rs:333-352`, `1065-1077` |
| Phase 5 deterministic aggregation and leakage ordering | PASS | `crates/tencir-pauli-core/src/sector.rs:1017-1054` |
| Shared immutable Phase 5.5 program and scalar-kernel reuse | PASS | `crates/tencir-pauli-core/src/propagation.rs:65-89`, `477-568` |
| Coarse flattened Python/PyO3 batch boundary | PASS | `python/tencirpauli/propagation.py:460-482`; `crates/tencirpauli-native/src/propagation.rs:230-294` |
| Row-wise value/gradient determinism | PASS | `crates/tencir-pauli-core/src/propagation.rs:591-628`; `tests/test_propagation_batch.py:36-70`, `136-156` |
| Small-workload serial behavior | FIXED | Branch-aware work units, a conservative threshold, and threshold-adjacent Rust regressions keep light Clifford rows serial |
| Batch-level active-worker memory guard | FIXED | Worker estimates include branch growth, current/candidate terms, checkpoint/replay terms, adjoint/gradient storage, and budget-bounded Rayon chunks |
| Phase 5.5 P0/P4 performance acceptance package | FIXED | Release benchmark coverage now includes construction controls, one-thread/default-thread runs, light Clifford, 100-qubit near-Clifford, multi-term rows, term statistics, and estimated peak bytes |
| Full benchmark recording removed from default commit hook | PASS | `.githooks/pre-commit:15-24`; `scripts/check.py:76-107` |

## CRITICAL

None.

## MAJOR

### 1. The private parallel threshold produces a large performance cliff for light Clifford batches

`PropagationBatch::map_observables` estimates work as `observable_count * operation_count * maximum_initial_term_count` and switches to Rayon at `work >= 64` (`crates/tencir-pauli-core/src/propagation.rs:636-650`). This treats a cheap non-branching Clifford gate as equivalent to a rotation or a branching custom PTM and does not estimate the actual propagated term count.

On the current release build and default 14-thread pool, a batch of two single-term observables and 31 Clifford gates stays serial and takes approximately 2.44 microseconds for expectations. Adding one Clifford gate crosses the threshold and raises latency to approximately 24.48 microseconds, while two scalar engines take approximately 2.50 microseconds. Equivalent threshold cases measured approximately 29.03 versus 3.51 microseconds for four observables and 16 Clifford gates, and 31.66 versus 5.39 microseconds for eight observables and eight Clifford gates. The gradient path shows a smaller but still material regression in the same cases.

This contradicts the Phase 5.5 requirement that light workloads remain serial and is an implementation issue rather than merely missing benchmark documentation.

Resolution: recalibrate the private heuristic using gate-kind/branching cost and a minimum per-row cost, or conservatively raise the threshold while retaining the existing B=4/16/64 rotation-heavy wins. Add permanent release cases immediately below and above the threshold for cheap Clifford-only rows, rotation-heavy rows, and a branching PTM. No public strategy or thread-count option is needed.

Resolution (2026-08-02): the scheduler now sums each operation's branch factor, requires a minimum row work estimate, and uses a threshold of 128 work units. The 2x31/32 and 4x15/16 Clifford threshold-adjacent cases remain serial, while rotation-heavy batches retain observable-level parallelism. Execution uses bounded parallel chunks so the selected worker count is also the maximum number of simultaneously active row tasks.

### 2. `max_bytes` does not constrain the aggregate concurrent workspace of a batch

The construction estimate computes each worker from only twice the maximum initial observable term storage plus twice the gradient storage (`crates/tencir-pauli-core/src/propagation.rs:519-551`). It does not include known gate branching, forward/checkpoint state growth, reverse checkpoint storage, replay states, adjoints, or the requested checkpoint interval. Execution then gives every row engine the full batch `max_bytes` limit and may run up to `rayon::current_num_threads()` rows concurrently (`crates/tencir-pauli-core/src/propagation.rs:528-559`, `631-650`). Each worker can therefore remain below the individual limit while the batch exceeds it by approximately the active-worker count.

A focused release check demonstrated the gap: a 16-row, 12-qubit exact rotation batch with one expanding all-Z term per row completes with `max_bytes=1_000_000`; the scalar runtime checks report that one worker's known workspace reaches at least 571,392 bytes, so several concurrent workers already exceed the nominal one-megabyte batch budget. This is not a request for exact RSS accounting. The frozen contract only requires a cheap best-effort estimate of major workspaces, but the current estimate omits major workspace components that are already known to the engine.

Resolution: compute a conservative per-row major-workspace estimate from branch factors, cutoff, gate count, parameter count, and checkpoint schedule; derive the allowed active-worker count from the remaining batch budget; and use serial or bounded chunks when the budget cannot support the normal Rayon width. Retain the existing per-engine runtime checks as a second guard.

Resolution (2026-08-02): `estimate_batch_worker_bytes` accounts for branch-derived propagated term growth, candidate/current storage, gradient storage, and checkpoint/replay state slots. `allowed_batch_workers` limits active chunks by the remaining `max_bytes` budget, with a minimum of one row worker so existing per-engine runtime checks remain authoritative. The focused 16-row, 12-qubit expanding case now completes with `max_bytes=1_000_000` without launching the normal full Rayon width, and the new regression test covers this contract.

### 3. Phase 5.5 is marked complete without its specified performance acceptance coverage

The frozen specification requires one-thread and fixed/default multithread comparisons, 12-qubit rotation-heavy and 100-qubit near-Clifford workloads, light single-string and heavier multi-term observables, construction comparison against B scalar engines, crossover evidence, active-worker memory scaling, and term-count/estimated-memory metadata (`docs/vibe/phase-5.5-spec.md:183-224`, `238-270`). The committed Python benchmark uses only one 12-qubit, single-term observable family (`benchmarks/python/test_propagation_batch_benchmark.py:14-27`), does not time scalar-engine construction, and records no one-thread batch run, term statistics, or estimated peak bytes (`benchmarks/python/test_propagation_batch_benchmark.py:30-158`). The Rust benchmark similarly measures batch calls without a scalar control (`crates/tencir-pauli-core/benches/propagation.rs:213-265`).

The local dirty release record does demonstrate genuine benefit on its chosen rotation-heavy case: B=16 expectations/gradients are about 0.53/1.18 ms versus 2.53/6.43 ms for serial engines, and B=64 is about 1.33/2.95 ms versus 10.63/26.72 ms. A focused one-thread rerun is essentially equal to the scalar loop, which supports the architecture. Those good results do not cover the required workload space and did not expose Major 1 or Major 2.

Resolution: add only the missing high-information cases, run them manually through the opt-in release benchmark workflow at one thread and the normal fixed pool size, and record a clean post-fix label. The commit hook should remain in smoke mode.

Resolution (2026-08-02): the benchmark file now includes light Clifford crossover, 100-qubit near-Clifford, and 16-row eight-term observables, alongside the existing 12-qubit B=1/4/16/64 construction, expectation, gradient, and scalar controls. The default 14-thread release run and `RAYON_NUM_THREADS=1` run both passed all 21 cases; benchmark metadata records gate count, parameter count, initial/peak/final term maxima, estimated peak bytes, output bytes, thread count, checkpoint interval, and numerical error.

## MINOR

None that justify expanding the implementation scope.

## OBSERVATIONS

The Phase 5 remediation removes the extra limb factor but retains the permitted two-pass source-by-X-group construction. The recorded 128/256/512-qubit k=2 setup times of roughly 14.7/109/841 ms scale more steeply than canonical nnz, while steady MVP remains about 79/173/370 microseconds. Candidate-oriented setup or deterministic setup parallelism may be worthwhile later, but it should be driven by a representative user workload and is not required to correct the current patch.

The Phase 5 frozen specification asks for a representative profile. The remediation records release timings and deterministic support/group metadata but no successful sampler trace because local macOS process inspection was denied. This is a strict evidence gap, although it does not undermine the source-level diagnosis or the verified correctness of the fix. If strict P5 closure is important, record deterministic per-stage timing/counters or use an available profiler in a manual run.

The pre-commit default still compiles Rust benchmark targets and executes non-`performance_large` Python benchmark bodies once with timing disabled. This is a reasonable smoke gate and is materially different from a full statistical release record. If commit latency remains unacceptable, the existing `TENCIRPAULI_PRE_COMMIT_BENCHMARK=skip` path is documented; no change is required for this review.

## RECOMMENDED IMPROVEMENTS

1. Fix the light-workload parallel threshold and retain threshold-adjacent regression benchmarks.
2. Make active batch concurrency respect the cheap major-workspace budget, falling back to fewer workers or serial execution when appropriate.
3. Complete the missing Phase 5.5 benchmark matrix manually after those fixes; do not put full performance recording back into the commit hook.
4. Keep the Phase 5 active-position implementation and public semantics unchanged unless a new profile shows a different dominant bottleneck.

## Verification performed

- `python scripts/check.py --benchmark skip`: passed formatting, Black, Clippy with `-D warnings`, Ruff, mypy, Rust tests, release extension build, and the full Python suite.
- Rust workspace tests: 22 passed.
- Python suite: 140 passed, 4 optional TensorCircuit tests skipped.
- Focused Phase 5/propagation suite: 65 passed.
- Read-only release timing of batch versus scalar execution at default Rayon width and `RAYON_NUM_THREADS=1` for B=1/4/16/64.
- Read-only threshold-adjacent Clifford timing for 2x31/32, 4x15/16, 8x7/8, and 16x3/4 observable/gate cases.
- Read-only expanding-batch memory-guard checks at 8/10/12 qubits.
- Inspection of the retained local benchmark record associated with the Phase 5.5 dirty implementation state.

## Post-review remediation verification

The remediation passes `cargo fmt --check`, core Rust tests (25 passed), core Clippy with `-D warnings`, the release extension build, the focused propagation-batch suite (10 passed), benchmark smoke coverage (19 passed and 2 explicitly performance-large deselected), and the two added performance-large correctness cases (2 passed).

The release benchmark uses the installed optimized extension and 14 Rayon threads unless stated otherwise. In the default-thread run, batch versus scalar expectation medians were approximately 1.57 ms versus 10.53 ms at B=64, and batch versus scalar value/gradient medians were approximately 4.16 ms versus 26.35 ms; at B=16 the corresponding expectation medians were approximately 0.61 ms versus 2.54 ms and gradient medians approximately 1.26 ms versus 6.38 ms. The one-thread run is intentionally near the scalar control: B=16 expectation was approximately 2.54 ms versus 2.51 ms, and gradient was approximately 6.42 ms versus 6.32 ms. The light Clifford crossover remained approximately 3.6–3.7 microseconds in both configurations. These measurements are local informational evidence, not a wall-time gate.

The review blockers are therefore resolved at the implementation level. Future tuning can improve the conservative worker estimate or benchmark breadth, but no known 5.5 correctness, bounded-concurrency, or acceptance-evidence blocker remains.
