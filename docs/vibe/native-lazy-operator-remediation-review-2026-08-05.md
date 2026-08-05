# Native-backed lazy operator open remediation review

Review date: 2026-08-05

Status: open remediation handoff.

Reviewed target: the current uncommitted working tree after the remediation attempt for [`native-lazy-operator-review-2026-08-05.md`](native-lazy-operator-review-2026-08-05.md).

Scope: record only the issues that remain open after the remediation attempt and provide a concrete handoff for the next fixer. NL1 is excluded from the acceptance gate by the owner's explicit decision to use ordinary IEEE `f64`/`complex128` behavior without adding recurring internal overflow, underflow, or non-finite-result defenses. Items already repaired are intentionally omitted.

## Verdict

**Do not treat the current working tree as a completed interface-cohesion checkpoint.** Qudit multiplication and commutator dispatch have a confirmed public regression, same-family Qudit addition loses the specialized facade type, the abandoned `_native_data` representation remains throughout the Python wrapper, native read-back still exposes several nested-list/split-real-imaginary ABIs, and the GIL/evidence remediations are incomplete.

The ordinary quality gate passes because the affected Qudit operations and the required architecture assertions are not covered. A passing gate therefore does not override the confirmed reproducer below.

Recommended status: **keep remediation open; fix R1 before any code checkpoint, then remove R2/R3 before claiming that the private operator interface is cohesive.** R4 and R5 remain acceptance requirements. The report itself may be committed independently as the handoff record.

## Validation performed

- Rebuilt the extension with `conda run -p .conda maturin develop --release`.
- Ran `conda run -p .conda python scripts/check.py --benchmark skip`: Rustfmt, Black, Clippy with warnings denied, Ruff, strict mypy, `git diff --check`, 41 Rust tests, 360 Python tests, and 10 doctests all passed.
- Ran the focused lazy/operator suites separately: 105 tests passed.
- Reproduced `QuditWeylOperator * QuditWeylOperator` raising `AssertionError` against the rebuilt extension.
- Reproduced `QuditWeylOperator + QuditWeylOperator` returning `HybridOperator` while scale and adjoint preserve `QuditWeylOperator`.
- Audited every remaining `_native_data` reference and found no production construction call that supplies `native_data=`; the representation is retained only by dead or compatibility branches.
- Audited handle read-back signatures in `convert.rs`, `operator.rs`, `structured.rs`, `majorana.rs`, and `_native.pyi` and confirmed multiple nested sequence and split real/imaginary variants remain.

No production source file was modified by this review. This report and its `docs/vibe/README.md` index entry are the only review-authored changes.

## Confirmed findings

### R1 — BLOCKER: Qudit native-handle dispatch is broken and same-family addition loses its facade

Locations: `python/tencirpauli/structured.py:1123-1128`, `python/tencirpauli/structured.py:1232-1245`, `python/tencirpauli/structured.py:1288-1301`, and `python/tencirpauli/structured.py:2466-2470`.

`QuditWeylOperator` is backed by `NativeHybridOperatorHandle`. The first native-Hybrid branch in `_StructuredOperator.multiply` dispatches on the handle type and then asserts that both Python facades are `HybridOperator`. A Qudit operand satisfies the handle check but fails the facade assertion. Qudit commutator and anticommutator do not enter the fused native branch because that branch additionally requires `isinstance(..., HybridOperator)`; they fall back to `multiply` and hit the same assertion.

The same-handle addition branch always calls `_hybrid_from_native(left.space, result)` without passing `type(left)`. Consequently `q + q` returns `HybridOperator`, whereas the pre-remediation `_make_operator` path retained `QuditWeylOperator` for a pure-qudit space. This is inconsistent with scale and adjoint, which preserve the specialized facade.

Reproducer:

```bash
conda run -p .conda python -c 'import tencirpauli as tcp; q=tcp.QuditWeylOperator.from_terms(3,[(((0,1,1),),1.0)],n_sites=1); print(type(q+q).__name__); print(q*q)'
```

Observed result:

```text
HybridOperator
AssertionError at structured.py:1126
```

Required resolution:

1. Dispatch native Hybrid-handle algebra independently of whether the public facade is `HybridOperator` or `QuditWeylOperator`.
2. Preserve `QuditWeylOperator` for same-family pure-qudit add, multiply, commutator, and anticommutator results; continue promoting genuinely mixed Qudit/Hybrid addition to `HybridOperator`.
3. Add numeric dense-reference tests for Qudit multiply, commutator, and anticommutator, plus explicit return-type tests for same-family and mixed-family operations.

### R2 — MAJOR: the abandoned `_native_data` representation remains as a parallel interface

Locations: `python/tencirpauli/structured.py:900-930`, `python/tencirpauli/structured.py:960-1008`, `python/tencirpauli/structured.py:1087-1104`, `python/tencirpauli/structured.py:2140-2288`, and `python/tencirpauli/structured.py:2430-2704`.

`_StructuredOperator` still accepts three storage forms: typed terms, `_native_data`, and `_native_handle`. The working tree contains no production caller that supplies `native_data=`, while the constructors and native-result factories now install handles. Nevertheless, term counts, materialization, addition, scaling, adjoint, array conversion, and raw reconstruction all retain `_native_data` branches and helpers.

This is a dead parallel representation, not a required fallback. It directly conflicts with the repository rule to delete an abandoned path and with the Phase 9 requirement that native-backed operators retain one canonical private storage form. It also makes every future operation reason about a state that normal construction cannot produce.

Required resolution: remove the `_native_data` field, constructor argument, branches, casts, and helpers as one change. Retain `_terms` only where an explicitly deferred word-level fallback such as tensor product still needs it, and retain `_native_handle` for ordinary production paths.

### R3 — MAJOR: read-back and producer interfaces have not converged on one handle/flat-array ABI

Locations: `crates/tencirpauli-native/src/convert.rs:6-20`, `crates/tencirpauli-native/src/operator.rs:102-111`, `crates/tencirpauli-native/src/structured.rs:19-21`, `crates/tencirpauli-native/src/structured.rs:242-251`, `crates/tencirpauli-native/src/structured.rs:948-965`, `crates/tencirpauli-native/src/majorana.rs:10`, `python/tencirpauli/pauli.py:529-543`, `python/tencirpauli/propagation.py:441-451`, and `python/tencirpauli/symmetry.py:93-100`.

The private ABI still contains nested `Vec<Vec<...>>` materializers, multiple packed/code variants, and split real/imaginary Python sequences. `PauliOperator._from_native` accepts array results and immediately invokes `pauli_operator_canonical` to rebuild a handle. Propagation and Z2 tapering still use this native-to-Python-to-native route; symmetry additionally exports the input handle through `_arrays()` before calling Rust.

This violates the repository's one-read-back-ABI rule, so the relevant NL3 interface work remains open. The issue is interface cohesion, not a demand for false zero-copy ownership.

Required resolution:

1. Make propagation and symmetry producer/consumer methods accept and return `NativePauliOperatorHandle` whenever the public contract does not request arrays.
2. Define one flat NumPy schema per handle family: fixed-width Pauli arrays and offset/indptr-based variable-width Structured/Majorana arrays, with one complex coefficient array.
3. Delete superseded nested/split read-back variants after all callers use the handle or flat-array route.
4. Keep textual `to_dict()`/string exports and explicit `.terms` materialization as public boundaries, not as inputs to another native call.

### R4 — MAJOR: GIL release is incomplete for new O(n) handle operations

Locations: `crates/tencirpauli-native/src/structured.rs:298-299`, `crates/tencirpauli-native/src/structured.rs:805-819`, `crates/tencirpauli-native/src/structured.rs:898-945`, `crates/tencirpauli-native/src/structured.rs:1065-1089`, and `crates/tencirpauli-native/src/majorana.rs:137-154`.

Specialized-to-Hybrid promotion clones complete structures in `to_hybrid()` without `allow_threads`; Hybrid, Boson, and Majorana adjoints clone or traverse complete operators while holding the GIL; and Hybrid role scans are also linear handle traversals executed under the GIL.

No concurrent observer regression or diagnostic was added, so the original availability requirement has no durable evidence.

Required resolution: put complete O(n) promotion, adjoint, and role-scan work inside GIL-released sections, then add a deterministic concurrent observer diagnostic without a wall-time pass/fail threshold.

### R5 — MAJOR: the acceptance evidence remains incomplete

Locations: `tests/test_lazy_operator.py:27-30`, `tests/test_structured_algebra.py:80-110`, `benchmarks/python/test_structured_lazy_algebra_benchmark.py:52-69`, and the absence of family-wide fused-operation and GIL tests.

`test_lazy_scale_and_dense_result_match` still compares an expression with the same expression recomputed through the same implementation. The Structured fused test compares fused output only with the composed native recurrence; it does not bind Boson or Hybrid values to an independent dense reference. There is no operator-level independent fused test for Majorana or Qudit, no anticommutator coverage for the new family kernels, and no test that would have caught R1.

The Structured lazy benchmark still contains only native BCH and term-materialization cases, with no eager/reference algebra baseline. Focused release benchmarks for direct construction, parity/BK mapping, Hybrid-to-Pauli projection, Fermion-to-Majorana conversion, terminal compilation, GIL availability, and peak memory are absent.

Required resolution:

1. Replace the remaining self-comparison with a plain dictionary or dense oracle.
2. Add independent small-system numeric references for multiply, commutator, and anticommutator across Fermion, Boson, Qudit, Hybrid, and Majorana, while retaining handle-residency assertions.
3. Add the concurrent GIL diagnostic required by NL6.
4. Add the missing eager/reference Structured BCH baseline and focused repaired-boundary release benchmarks with stable term-count metadata.

### R6 — MEDIUM: the implementation-status document overclaims Qudit and tensor-product coverage

Location: `docs/vibe/operator-lazy-results.md:12-13`.

The status table claims native-default Qudit multiply and commutator even though those operations currently fail as described in R1. It also lists Qudit and Hybrid tensor products among native-default operations even though `python/tencirpauli/structured.py:1440-1502` explicitly materializes terms and native tensor products are owner-deferred.

Required resolution: correct the table when fixing R1. Tensor products must be listed as an intentional materialized fallback until profiling reopens the deferred native implementation.

## Minimum handoff checklist

A future fixer may declare the interface-cohesion checkpoint ready only after all of the following are true:

- [ ] Qudit add/multiply/commutator/anticommutator preserve correct facade types and match independent numeric references.
- [ ] `_native_data` and all unreachable branches/helpers are deleted.
- [ ] Propagation and symmetry no longer perform Pauli native-array-native round trips.
- [ ] Handle read-back uses one documented flat NumPy schema per family; nested/split variants are removed unless the public result explicitly requires text or typed objects.
- [ ] All new material O(n) native work releases the GIL and a concurrent observer diagnostic exists.
- [ ] Every fused family operation has independent numeric evidence, including anticommutators.
- [ ] Structured BCH has an eager/reference baseline and the reviewed construction/mapping/conversion/compilation boundaries have release-mode benchmarks.
- [ ] `operator-lazy-results.md` matches the implemented native and fallback paths.
- [ ] `python scripts/check.py --benchmark smoke` passes after a release rebuild; representative release measurements are recorded locally before any performance claim is published.

Until this checklist is complete, the working tree may be described as a substantial remediation in progress, but not as a closed native-lazy-operator or cohesive-private-interface checkpoint.
