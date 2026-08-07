# Phase 10 Chemistry and Scientific-Interop Implementation Review, 2026-08-06

Status: open remediation report. Phase 10 is not acceptance-closed at commit `366e46e`; the implementation direction is substantially correct, but two production hot-path issues and two required acceptance-evidence gaps remain.

## 1. Review scope and verdict

This review covers commits `32bc7d8` (`Implement Phase 10 chemistry and SciPy interop`) and `366e46e` (`Fix PySCF example with JAX Adam VQE`) against `docs/vibe/phase-10-spec.md`. The audit focuses on numerical correctness, native/Python ownership, GIL and allocation behavior, PySCF RHF/UHF conversion, SciPy iterative-solver interoperability, representative performance, and closure evidence. It intentionally excludes hypothetical extreme inputs, exact allocator accounting, broad defensive validation, and speculative optimizations that are not tied to a measured hot path.

The primary API direction should be preserved. `FermionOperator.from_integrals()` and the PySCF compact-block route share one coarse native ingestion implementation; RHF and UHF avoid a zero-padded complete spin-orbital tensor; PySCF remains lazily loaded; native, charge-restricted, and U1 plans delegate to one SciPy helper; the two ED research examples use the public wrapper; and the chemistry example reaches a mapped Pauli operator and TensorCircuit backend-MVP/JAX energy evaluation. The complete formatting, static-analysis, release-build, Rust, Python, and doctest gates pass.

Phase 10 nevertheless cannot be closed yet. The public integral constructor performs complete `O(n^4)` dtype conversion and Hermiticity validation in Python with several full-tensor temporaries before entering the GIL-released native call. The SciPy method is also exposed on `BackendMVPPlan`, which is outside the frozen native-plan scope and is a measured performance trap for iterative solvers. Separately, the PySCF tests do not independently bind the full RHF/UHF coefficient and spin-block contract, and the benchmark package contains only the minimal H2 workload without a valid medium-workload release record.

No production source was modified during this review.

## 2. Verification performed

- Inspected the complete `HEAD~2..HEAD` diff and the frozen Phase 10 specification.

- Built the PyO3 extension in release mode and ran the focused integral-ingestion, PySCF, SciPy, and chemistry-example suite: 18 tests passed.

- Ran `python scripts/check.py --benchmark skip`: Rust formatting, Black, stage-label validation, Clippy with warnings denied, Ruff, strict mypy, and `git diff --check` passed; Rust tests were 41/41, Python tests were 380/380, and doctests were 8/8.

- Verified a complex-orbital phase rotation on RHF H2. The imported dense Hamiltonian remained Hermitian and its spectrum agreed with the real-orbital result to a maximum error of approximately `1.67e-15`.

- Measured compact native spin-block ingestion with deterministic dense blocks. At 20 spatial orbitals, 640,000 raw two-body contributions produced 233,000 canonical terms in approximately 0.184 seconds with an approximately 180 MiB peak-RSS increase. This does not establish a general performance guarantee, but it indicates that the compact native ingestion kernel itself is not the principal defect found in this review.

- Measured the public `from_integrals()` validation path on a 48-mode all-zero tensor. An 81 MiB `complex128` input caused an approximately 335 MiB additional peak-RSS increase; a 40.5 MiB `float64` input caused an approximately 381 MiB increase. Because the operator contains zero output terms, the excess is attributable to conversion and validation work rather than unavoidable canonical output storage.

- Compared SciPy wrappers over matched 12-qubit, 64-term, dimension-4096 plans. The native plan took approximately 0.114 ms per measured `matvec`, while the backend plan took approximately 4.54 ms, a slowdown of about 40 times, with a maximum numerical difference of approximately `1.5e-14`.

- Inspected the local benchmark records. No completed record for commits `32bc7d8` or `366e46e` contains the Phase 10 small/medium acceptance package.

## 3. Completed implementation that should be preserved

- The public spin-orbital constructor and the PySCF compact spin-block adapter converge on one framework-neutral Rust ingestion function and one native handle result.

- RHF reuses one spatial ERI block and UHF retains only the four nonzero spin-pair blocks; neither path constructs a Python term dictionary or a complete zero-padded spin-orbital two-body tensor.

- Native construction applies the one-half two-body factor once, performs exact-zero removal, canonical CAR aggregation, deterministic output ordering, checked estimates, and releases the GIL for the native construction call.

- PySCF import is lazy, rejects unsupported SCF families, verifies convergence and orbital orthonormality before two-electron transformation, includes nuclear repulsion by default, and supports both specified spin-orbital orderings.

- `NativeMVPPlan`, `ChargeMvpPlan`, `ChargeLazyMvpPlan`, and `U1MvpPlan` delegate to one SciPy adapter and support the required vector, column-vector, `matmat`, memory-error, and `eigsh` paths without materializing a matrix.

- The migrated Fermi-Hubbard and SYK examples use the reusable public SciPy wrapper, and the optional chemistry example checks backend-MVP energy against a dense reference.

## 4. Open findings

### R1 — Public integral validation creates multiple full-tensor Python temporaries

Priority: P1 performance and availability blocker.

Evidence: `python/tencirpauli/structured.py:128-154` converts every supported `float64` integral array into a full `complex128` copy before FFI. `python/tencirpauli/structured.py:2581-2594` then validates the one- and two-body arrays with NumPy `allclose`; the two-body expression `two.transpose(2, 3, 0, 1).conj()` and `isclose` internals allocate complete `O(n^4)` temporaries. These scans and conversions occur before `crates/tencirpauli-native/src/structured.rs:1743-1785` enters `py.allow_threads()`.

Impact: the most common real-valued public input pays a full dtype-expansion copy, and both supported dtypes pay several additional complete-tensor workspaces. The 48-mode zero-output probe showed approximately 335-381 MiB additional peak RSS for only 40.5-81 MiB of input. This is a predictable major workspace, not exact-RSS bookkeeping or an extreme-coefficient recovery case. It weakens `max_bytes`, delays failure under memory pressure, and leaves complete scalable conversion preparation on the Python side.

Minimal closure: let the PyO3 boundary accept contiguous `float64` and `complex128` inputs without Python dtype expansion, then perform finite-value, Hermitian-pair tolerance checking, averaging, exact-zero filtering, and construction in the GIL-released Rust path. The scan can reuse the existing count/construction traversal; no fallback representation, allocator instrumentation, or repeated handle validation is needed.

### R2 — `BackendMVPPlan.to_scipy_linear_operator()` is outside scope and is a measured performance trap

Priority: P1 public hot-path blocker.

Evidence: `python/tencirpauli/hamiltonian.py:518-527` adds the SciPy method to `BackendMVPPlan`, although the frozen specification limits the adapter to native MVP plans and requires the actual iterative-solver MVP to remain in the native plan. `tests/test_scipy_linear_operator.py` tests ordinary native, charge lazy/eager, U1, and Pauli convenience paths but does not test or benchmark the backend method. A matched dimension-4096 probe measured the backend wrapper about 40 times slower than the native wrapper.

Impact: a public method with the same name advertises backend execution as interchangeable with the intended native iterative-solver path. Repeated SciPy callbacks then run the Python/backend executor and conversion path rather than the optimized native plan. The result is numerically correct in the measured case but materially violates the phase's performance purpose.

Minimal closure: remove the method from `BackendMVPPlan` and avoid making SciPy conversion a requirement of the protocol shared with backend plans. Retain the methods on `NativeMVPPlan`, charge lazy/eager plans, `U1MvpPlan`, and the `PauliOperator` convenience path. Do not add a second optimized backend-SciPy implementation unless a separately approved workload establishes a need.

### R3 — RHF/UHF tests do not independently bind the complete coefficient and spin-block contract

Priority: P1 correctness-evidence blocker. No numerical failure on the tested inputs was reproduced.

Evidence: `tests/test_pyscf_integration.py:56-70` checks the RHF nuclear constant, one occupied determinant energy, and equality between a mapped Pauli matrix and the dense matrix produced from the same imported fermion operator. `tests/test_pyscf_integration.py:112-130` checks that UHF alpha and beta orbitals differ and that one occupied determinant diagonal agrees with `mf.e_tot`, using only `alpha_then_beta`. The module does not directly assert independent one-body and two-body coefficients before and after spin expansion, all four UHF spin blocks, an independent full fermionic dense oracle, or both UHF orderings.

Impact: a wrong non-diagonal ERI axis permutation, cross-spin block placement, or UHF mode permutation can leave the selected determinant diagonal unchanged and therefore pass the existing tests. Mapping the imported operator and comparing it with its own dense form validates mapping consistency but is not an independent chemistry-ingestion oracle.

Minimal closure: construct small RHF and genuinely distinct-alpha/beta UHF references directly from PySCF AO/MO values and an independent `from_terms()` or dense fermionic oracle. Assert selected one- and two-body coefficients, the complete small dense matrix, mapped Pauli equality to that independent matrix, determinant energy, and the FSWAP-correct relation between both orderings. Keep the fixtures small and deterministic rather than adding a broad molecule matrix.

### R4 — The benchmark package does not satisfy the representative recorded acceptance gate

Priority: P1 performance-evidence blocker. This is not a demonstrated regression in the compact native kernel.

Evidence: every test in `benchmarks/python/test_chemistry_interop_benchmark.py` uses the same two-spatial-orbital H2 fixture. `test_native_compact_integral_ingestion()` compares only dense result shapes, and `test_complete_pyscf_conversion()` stores `max_abs_error: 0.0` without computing an independent error. The package does not measure a medium chemistry workload, plan construction, or dense/sparse materialization baselines as required by `docs/vibe/phase-10-spec.md:270-276`; no completed Phase 10 record exists for the reviewed commits.

Impact: the current benchmark source cannot establish realistic AO-to-MO versus ingestion dominance, conversion scaling, medium-workload memory, actual numerical accuracy, or the SciPy callback overhead relative to plan construction and materialization. A hard-coded zero error is metadata, not accuracy evidence.

Minimal closure: retain H2 as the small case and add one representative medium RHF or UHF workload with recorded orbital/basis size. Separately record PySCF transformation, native ingestion, complete conversion, plan construction, first and steady native/SciPy MVP, peak RSS, and a computed numerical error against an independent target; include dense/sparse materialization only at a size where it is meaningful and safe. Produce one synchronized release-mode record tied to the closure commit. No wall-time CI gate or large benchmark grid is required.

## 5. Acceptance recommendation

Do not redesign the chemistry representation or add density fitting, factorized tensors, custom eigensolvers, exact allocator accounting, or defensive recovery branches for this closure. Preserve the compact PySCF blocks, shared native canonicalizer, lazy optional dependency, native plan reuse, and single SciPy helper.

Phase 10 may be acceptance-closed after R1-R4 are resolved: eliminate the full-tensor Python validation/copy path, remove the backend-plan SciPy performance trap, add independent RHF/UHF coefficient and ordering evidence, and record the required small/medium release benchmark package with real accuracy values. The existing passing quality gate and the measured compact-ingestion behavior can then be reused as supporting evidence rather than rerunning unrelated optimization experiments.
