# Phase 9 Remediation Closure Evidence, 2026-08-06

Status: S1–S7 closure evidence complete on the local macOS arm64 environment. This record summarizes the final implementation, correctness, residency, GIL, repository, and performance gates after the second-round review.

S1 QWC reconstruction accepts only a C-contiguous `numpy.int8` sample array at the Python boundary, so rank/shape/dtype/contiguity checks are O(1) and no sample-sized Python conversion precedes the native call. Rust receives the borrowed contiguous `i8` slice, validates binary values, allocates the flat output, and runs the packed-mask parity loop inside one helper called through `allow_threads`; the packed masks remain exclusively in the native handle. Randomized tests compare every group against an independent NumPy oracle across single- and multiword qubit widths, and a large concurrent-observer test sees progress from another Python thread during reconstruction. The grouping benchmark records qubits, shots, group size, support density, group count, and output bytes separately from construction timing. A same-machine release A/B on 64 qubits, 96 terms, 2,000 shots, and group size 43 measured 826.8 microseconds median for the `i8` path versus 865.1 microseconds for the pre-remediation `i64` path, a 0.956 ratio; the benchmark artifacts remain local and untracked.

S2 charge restriction compiles Pauli, Fermion, Boson, and mixed Hybrid handles inside a detached Rust section. Materializer-failure tests cover all four domains, while independent particle-number, spin, excitation-number, spectator, and cancellation-after-aggregation differentials remain passing. The large handle-plan concurrent-observer test sees progress during setup. Charge benchmarks record analysis, lazy setup, eager setup, first and steady apply, dense/COO/CSR materialization, input terms, sector dimension, plan bytes, and output sparsity.

S3 signed-zero hashing is canonicalized pattern-wide through the shared scalar hash helpers. Pauli, Majorana, Fermion, Boson, Hybrid, and Qudit values now pass equal-value, real-adjoint, independently constructed, unequal, signed-zero, ordinary-complex, nonfinite-rejection, and no-materialization checks. Large native equality/hash cases are recorded for every hashable family.

S4 embedding keeps specialized Fermion, Boson, and uniform-Qudit facades for pure target layouts and returns Hybrid only for genuinely mixed targets. Tests cover type preservation, native residency, mixed-domain target-index collisions, nontrivial fermion permutation signs, deterministic ordering, and materializer failure. The release embedding benchmark records source/output term counts, layout widths, and the fermion permutation.

S5 GateTape lifecycle tests count the native tape constructor across first compile, different observables, two distinct parameter values on one compiled plan, mutation invalidation, a separately built isomorphic circuit, and an independently reconstructed QIR circuit. Cold-versus-cached benchmarks cover PropagationEngine, PropagationBatch, SPPSEngine, PropagationCircuit, and SPPSCircuit with gate count, parameter count, observable term count, and structural-conversion metadata. The full benchmark suite also includes QWC reconstruction, embedding, native value semantics for Pauli, Majorana, Fermion, Boson, Qudit, and Hybrid values, and charge eager setup.

S6 `scripts/check_stage_labels.py` scans paths and contents outside `docs/vibe/` with a narrow numbered-development-label pattern. The final scan is empty, including after excluding only generated caches and local environment files; legitimate scientific parameter names such as `p0` and `p1` are not treated as labels.

S7 used independent Cargo target directories and both execution orders. Each case used numerical checks before timing; the table reports the remediation median divided by the baseline median after combining A→B and B→A observations.

| modes | workload | lazy setup | eager setup | first apply | steady apply | dense | COO | CSR |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 8 | one-body | 0.564 | 0.507 | 0.482 | 0.819 | 0.597 | 0.547 | 0.555 |
| 8 | longer word | 0.518 | 0.564 | 0.550 | 0.754 | 0.542 | 0.587 | 0.581 |
| 16 | one-body | 0.401 | 0.458 | 0.459 | 0.653 | 0.430 | 0.482 | 0.485 |
| 16 | longer word | 0.414 | 0.494 | 0.450 | 0.548 | 0.513 | 0.530 | 0.527 |
| 65 | one-body | 0.255 | 0.358 | 0.355 | 0.426 | 0.365 | 0.380 | 0.388 |
| 65 | longer word | 0.490 | 0.608 | 0.617 | 0.631 | 0.877 | 0.612 | 0.615 |
| 128 | one-body | 0.194 | 0.326 | 0.325 | 0.362 | 0.328 | 0.330 | 0.332 |
| 128 | longer word | 0.506 | 0.653 | 0.640 | 0.649 | n/a | 0.646 | 0.662 |

The A/B runs preserved input term count, sector dimension, COO/CSR nonzero count, and dense Frobenius norm exactly. The 128-mode longer-word dense case was intentionally omitted because its 8128-dimensional dense output is outside the controlled representative matrix budget; COO and CSR remained covered.

Final local gates passed after a release rebuild: Cargo fmt check, Clippy with warnings denied, 41 Rust tests, 377 Python tests, 10 doctests, Black, Ruff, mypy, diff check, and the repository label check. The complete local benchmark record finished successfully with all three Rust Criterion groups and 419 Python benchmark cases; one optional TensorCircuit comparison case was skipped because its external dependency was unavailable. Benchmark outputs remain ignored local artifacts.
