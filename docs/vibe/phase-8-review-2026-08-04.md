# Phase 8 API coherence implementation review

Review date: 2026-08-04

Reviewed commit: `fe8a2831a86dbcde50d5c35317949c8abc0e361a` (`refactor: align public API with phase 8`).

Scope: independent contract review of the Phase 8 Python API cleanup against `phase-8-api-coherence-spec.md`, including circuit capability separation, Hermitian execution contracts, Pauli input validation, canonical count metadata, MVP plan behavior, U1 state and basis APIs, mapping construction, ordinary/advanced export tiers, documentation, typing, and regression coverage. This review did not modify production code.

## Verdict

Phase 8 is substantially implemented and the complete correctness and quality gate passes, but it should be recorded as `implemented; remediation open` rather than accepted. No numerical-algebra regression was found in the reviewed test matrix. The remaining defects are concentrated at the public API boundary: SPPS capability isolation can be bypassed through class-level method lookup, empty non-integer Pauli arrays escape the unified validator, U1 gradient calls repeat a prohibited full Hermiticity scan, a raw mapping constructor remains publicly usable, deterministic propagation omits required term metadata, and the advanced documentation/contract tests do not yet match the specification.

The recommended outcome is a short API-focused remediation followed by a second review. A redesign of the Rust kernels or executor architecture is not required.

## Closure note (2026-08-04)

R1–R7 were remediated in commit `35f9adc`; the full local quality gate, strict documentation build, and benchmark smoke passed. This report is now an archived historical review, and no machine-specific benchmark record is required for the 0.2 release.

## Review and validation performed

- Inspected the complete `origin/main..fe8a283` diff and the current public Python modules against every normative section of `phase-8-api-coherence-spec.md`.
- Ran `conda run --no-capture-output -p .conda python scripts/check.py --benchmark skip`: Rust formatting and Clippy, Black, Ruff, strict mypy, `git diff --check`, 39 Rust tests, release `maturin develop --release --locked`, 305 Python tests, and 10 doctests passed.
- Ran 142 focused Phase 8, propagation, SPPS, U1, symmetry, and structured-algebra tests; all passed.
- Ran `conda run --no-capture-output -p .conda mkdocs build --strict --site-dir /tmp/tencirpauli-phase8-site`; the strict documentation build passed.
- Ran targeted read-only runtime probes for class-level SPPS capability leakage, constructor exposure, MVP output layout, and empty-array validation. The probes reproduced R1, R2, and R4 below.
- No performance benchmark record was created because Phase 8 is an API-coherence milestone and this review did not make a performance claim.

## Acceptance matrix

| Area | Result | Assessment |
| --- | --- | --- |
| SPPS versus deterministic capability separation | FAIL | Instance-level `hasattr()` checks pass, but inherited class methods remain callable and static typing still advertises the deterministic surface. |
| Exact Hermitian scalar contracts | PARTIAL | Error behavior is correct in covered calls, but U1 repeats a full operator scan and PyO3 round trip on every gradient execution. |
| Unified Pauli code validation | PARTIAL | Scalar and nonempty array paths obey the new contract; empty bool, float, and object arrays are accepted. |
| Canonical count vocabulary | PARTIAL | Operators, grouping results, profiles, SPPS, and MVP plans are largely migrated, but `PropagationEngine.term_count` is missing. |
| Grouping result contract | PASS | Canonical defaults, immutable mappings, counts, mode, and measurement-ready invariants are implemented. |
| MVP protocol, budgets, and immutability | PASS WITH EVIDENCE GAP | The four implementations follow the flat-vector execution contract in production paths, but the explicit Phase 8 contract suite does not exercise the complete cross-plan matrix. |
| U1 constructor, basis, and state API | PASS WITH PERFORMANCE DEBT | New names, stable basis shapes, restricted/full terminals, read-only arrays, and `parameters=None` behavior are present; R3 remains. |
| Mapping names and `mapping=` keyword | PARTIAL | Normal factory and compile paths use the new vocabulary, but the raw top-level mapping constructor remains exposed. |
| Ordinary/advanced export tiers | PARTIAL | Concrete advanced types are removed from top-level `__all__`, but the advanced reference page and exact manifest freeze are incomplete. |
| Full local correctness and quality gate | PASS | All formatter, lint, type, Rust, Python, doctest, and strict documentation-build checks passed. |

## Findings

### R1 — MAJOR: `SPPSCircuit` deterministic-only methods remain inherited and can be called through the class

Locations: `python/tencirpauli/propagation_circuit.py:205-253,388-397,527-559,581-623` and `python/tencirpauli/spps_circuit.py:138-153`; contract: `docs/vibe/phase-8-api-coherence-spec.md`, sections 3.1 and 12.

The new private `_CircuitBuilder` still contains `ptm()`, `propagate_operator()`, `profile()`, and the deterministic compile/terminal implementation. `SPPSCircuit` inherits that complete class and hides three names only through instance `__getattribute__()` and `__dir__()`. Consequently `hasattr(SPPSCircuit(1), "ptm")` is false, but `hasattr(SPPSCircuit, "ptm")` is true and the inherited descriptor can be invoked directly.

The following public-class call bypasses the intended capability boundary and serializes an unsupported PTM into an SPPS circuit:

```python
import numpy as np
import tencirpauli as tcp

circuit = tcp.SPPSCircuit(1)
tcp.SPPSCircuit.ptm(circuit, [0], np.eye(4))
assert circuit.to_qir()[0]["name"] == "ptm"
```

Failure is then delayed until stochastic engine construction, which is precisely the failure timing the specification prohibits. Static analyzers and class-level autocomplete also continue to expose `ptm`, `propagate_operator`, and `profile`. In addition, shared `from_circuit()` and `from_qir()` are annotated as returning `_CircuitBuilder`, not `Self` or an equivalent class-bound type, so a call through `SPPSCircuit` loses the exact facade type statically even though runtime construction uses `cls`.

Required resolution:

1. Restrict `_CircuitBuilder` to storage, wire/parameter validation, common gates, QIR mechanics, cache invalidation, and capability-aware conversion helpers.
2. Move `ptm()` and all deterministic terminals onto `PropagationCircuit`; keep only stochastic terminals on `SPPSCircuit`.
3. Type conversion classmethods with `Self` or a Python-3.9-compatible bound `TypeVar` so both runtime and static return types are exact.
4. Add regressions on both the instance and class surfaces, including `not hasattr(SPPSCircuit, "ptm")`, and verify that no unbound call can append PTM or dispatch a deterministic terminal.

Closure gate: neither instance lookup, class lookup, static typing, nor QIR/TensorCircuit conversion exposes PTM, propagation materialization, or profiling on `SPPSCircuit`; all genuinely shared gates and conversions still return the requested facade type.

### R2 — MEDIUM: empty bool, float, and object Pauli code arrays bypass dtype validation

Location: `python/tencirpauli/pauli.py:1052-1080`; contract: `docs/vibe/phase-8-api-coherence-spec.md`, section 3.3.

`_normalize_code_arrays()` guards both the dtype check and range check with `if code_array.size`. As a result, an invalid dtype is rejected only when the array contains at least one scalar. Empty arrays and zero-width arrays silently convert to `uint8` and are accepted:

```python
import numpy as np
import tencirpauli as tcp

for dtype in (np.bool_, np.float64, object):
    tcp.PauliOperator.from_code_arrays(
        np.empty((0, 2), dtype=dtype),
        np.empty(0),
    )
    tcp.PauliOperator.from_code_arrays(
        np.empty((1, 0), dtype=dtype),
        np.zeros(1),
    )
```

The specification explicitly requires dtype-based rejection of bool, float, and object arrays, independent of payload size. This is a contract inconsistency rather than a wrong nonzero operator result, but it makes validation depend on shape and lets invalid schemas enter large-array APIs.

Required resolution: validate `code_array.dtype.kind in ("i", "u")` unconditionally after the two-dimensional shape check, then perform the value-range scan only when the array is nonempty. Add empty-row and zero-qubit cases for bool, float, object, signed integer, and unsigned integer dtypes.

### R3 — MEDIUM: U1 gradient terminals repeat the exact-Hermiticity scan on every hot execution

Locations: `python/tencirpauli/u1_circuit.py:321-343,599-615` and `python/tencirpauli/pauli.py:591-604`; contract: `docs/vibe/phase-8-api-coherence-spec.md`, section 3.2.

Both `U1CircuitPlan.value_and_grad()` and `U1Circuit.value_and_grad()` call `observable.is_hermitian(tolerance=0.0)` on every execution. `PauliOperator.is_hermitian()` performs a native call over the operator arrays, so repeated U1 evaluations incur an extra full operator scan and PyO3 round trip before the actual gradient call. This contradicts the explicit Phase 8 requirement that exact Hermiticity be cached in the immutable operator/compiled handle or reused from construction-time information rather than rechecked on every hot execution.

Required resolution: cache exact Hermiticity on immutable `PauliOperator` construction or provide an equivalent immutable cached flag that the U1 terminals can read in O(1) without FFI. Tolerance-dependent public queries may still use the general validation path. Add an instrumented regression proving that repeated U1 gradient calls do not invoke the full Hermiticity kernel repeatedly.

### R4 — MEDIUM: the raw `FermionQubitMapping` constructor remains a top-level construction path

Locations: `python/tencirpauli/mapping.py:155-285` and `python/tencirpauli/__init__.py:22,79`; contract: `docs/vibe/phase-8-api-coherence-spec.md`, sections 2.1, 9, 10, and 11.

`FermionQubitMapping` remains top-level and its public `__init__()` accepts `mapping_name`, an arbitrary invertible encoding matrix, and positional `max_bytes`. A caller can therefore construct an object named `"jordan_wigner"` whose encoding matrix is actually parity-like or otherwise custom. The read-only `name` property then advertises provenance that does not match the stored transform.

This bypasses the named factory contract and also leaves one public resource budget parameter positional despite the Phase 8 keyword-only rule. The normal `from_name()`, `jordan_wigner()`, `parity()`, and `bravyi_kitaev()` factories are correct; the defect is the still-public raw route.

Required resolution: make raw construction private/factory-only, retain the class at top level for typing and `isinstance`, and ensure every public instance originates from a named validated factory. If custom encodings are desired later, add a separately named advanced factory with honest provenance and an independently reviewed schema rather than overloading the three standard names.

### R5 — MEDIUM: deterministic single-observable propagation lacks required `term_count` metadata

Locations: `python/tencirpauli/propagation.py:345-394`, `crates/tencirpauli-native/src/propagation.rs:16-57`, and `python/tencirpauli/_native.pyi:410-432`; contract: `docs/vibe/phase-8-api-coherence-spec.md`, section 4.

`SPPSEngine.observable_terms` was migrated to `SPPSEngine.term_count`, but `PropagationEngine` exposes no corresponding `term_count`. The specification explicitly requires both single-observable engines to use the canonical algebraic-term metadata name.

Required resolution: store or expose the canonical observable term count during deterministic engine construction without rescanning the operator, add the native getter/stub only if it is needed by the wrapper, and test exact-zero canonicalization cases so the value is the consumed canonical algebraic term count rather than a propagated transition or final term count.

### R6 — MINOR: the advanced reference and several public docstrings do not describe the implemented Phase 8 contract

Locations: `docs/api.md:1-19`, `mkdocs.yml:58-62`, `python/tencirpauli/hamiltonian.py:417-437`, `python/tencirpauli/u1_circuit.py:246-343,553-615`, and the scalar facade methods in `python/tencirpauli/propagation_circuit.py` and `python/tencirpauli/spps_circuit.py`; contract: `docs/vibe/phase-8-api-coherence-spec.md`, sections 3.2, 6, 8, 10, and 11.

The MkDocs navigation contains only one API page rendering `tencirpauli`; there is no separate `tencirpauli.advanced` reference with its stability boundary and complete signatures. A strict build succeeds because the missing page is not a build error, but advanced types are absent from the published reference.

Several source docstrings are also stale or incomplete. `BackendMVPPlan.apply()` still says direct-Weyl plans accept a mixed-radix tensor shape and return the same logical input shape, while the implementation correctly enforces and returns a flat `(dimension,)` vector. `U1CircuitPlan.state_full()` describes a restricted state rather than the full computational-basis vector. High-level scalar terminals generally omit the required exact-Hermitian precondition and `ValueError` contract, and U1 array terminal docstrings do not consistently state dtype, shape, ownership, restricted/full distinction, and ordering.

Required resolution: add an advanced API reference page and navigation entry, correct the stale flat-vector and U1 full-state descriptions, and complete the Hermitian/result/error documentation promised by the specification. Keep ordinary pages on top-level entry points only.

### R7 — MINOR: the explicit Phase 8 contract tests do not freeze the complete promised surface

Location: `tests/test_phase8_api.py:14-113`; contract: `docs/vibe/phase-8-api-coherence-spec.md`, sections 10 and 12.

The new test file checks that a selected set of advanced names is disjoint from top-level `__all__`, but it does not assert exact top-level and advanced manifests. The SPPS test checks only instance `hasattr()`, which is why R1 passes. Array validation uses only nonempty arrays, which misses R2. The MVP test exercises one native Pauli plan rather than the shared four-plan protocol, construction/execution budget independence, `__call__` default-budget behavior, output ownership, and all factory-only constructor cases. Hermitian tests cover representative facade calls but not the cached-hot-path requirement.

Existing module-specific tests provide substantial additional correctness coverage, so this is not a claim that all Phase 8 behavior is untested. It is a finding that the promised explicit contract freeze is incomplete and allowed multiple API regressions to pass every gate.

Required resolution: replace subset assertions with exact reviewed manifests and add focused regressions for R1–R6 plus the complete cross-plan and circuit-plan contract matrix. Keep numerical algorithm tests in their existing modules; the Phase 8 file should freeze spelling, signatures, capability surfaces, metadata, mutability, budgets, and error timing.

## Architecture assessment

The underlying architecture remains appropriate. The Rust core is independent from Python and TensorCircuit, the binding remains coarse-grained, public operators are immutable and deterministic, grouping and canonical term vocabulary are mostly coherent, MVP execution budgets are separated from construction budgets, and the U1 restricted/full-state split is materially clearer than before Phase 8.

The open issues do not justify merging the three circuit executors or adding a generic plan base class. The key correction is to make the Python surface structurally capability-aware instead of simulating capability removal through dynamic attribute masking, then close the smaller validation, metadata, caching, and documentation gaps.

## Recommended remediation order

1. Fix R1 first because it defeats the central capability-separation goal and affects runtime, static typing, conversion boundaries, and discoverability.
2. Fix R2 and R4 next to close public construction and validation routes before the API is released to users.
3. Cache exact Hermiticity for R3 and add `PropagationEngine.term_count` for R5 without introducing a new hot-path scan or FFI call.
4. Complete R6 documentation and R7 contract tests, then rerun the full quality gate and strict MkDocs build.
5. Request a narrow Phase 8 second-round review. Performance benchmarks are unnecessary unless the Hermiticity-cache change materially alters a hot path; if measured, include full Python/native boundary cost.

## Closure checklist

- [ ] `SPPSCircuit` has no instance-level, class-level, or statically visible `ptm`, `propagate_operator`, or `profile` surface.
- [ ] `from_qir()` and `from_circuit()` preserve exact subclass typing and reject unsupported capabilities at conversion boundaries.
- [ ] Empty and zero-width bool, float, and object Pauli arrays raise `TypeError`; integer arrays remain accepted.
- [ ] Repeated U1 gradient execution reads cached exact Hermiticity without a full scan or PyO3 round trip.
- [ ] Standard fermion mappings can be constructed only through honest named public factories, with keyword-only `max_bytes`.
- [ ] `PropagationEngine.term_count` reports the consumed canonical observable term count.
- [ ] Exact top-level and advanced export manifests are frozen by tests.
- [ ] All four MVP plans pass the common metadata, flat apply, budget, call, ownership, contiguity, mutability, and concurrency contract tests.
- [ ] The advanced API reference is published and stale flat/tensor, Hermitian, and U1 state docstrings are corrected.
- [ ] `python scripts/check.py --benchmark skip` and strict MkDocs build pass on the remediation commit.
