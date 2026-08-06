# Phase 9.5 Circuit Differentiation Implementation Review, 2026-08-06

Status: open focused remediation report. Phase 9.5 is not acceptance-closed at commit `9e07a2c`, but this review found no demonstrated numerical defect on valid inputs in the implemented native/JAX circuit paths.

## 1. Review scope and verdict

This review compares commit `9e07a2c` (`feat: implement phase 9.5 circuit differentiation boundary`) with its parent against `docs/vibe/phase-9.5-spec.md`. It covers the actual-angle public circuit contract, occurrence-space native gradients, JAX callback/VJP execution, removal of the public and Rust expression systems, private compilation, numerical acceptance evidence, release benchmarks, repeated-MVP scratch review, and live-document migration.

The main implementation direction is correct and should be preserved. `Parameter` and `ParameterExpr` are removed from the ordinary facade, the three circuit families expose observable-first `expectation()`, `value_and_grad()`, and `expectation_jax()`, gradient lowering assigns one private runtime slot per angle occurrence, forward execution retains static-angle plans, U1 uses a numerical `AngleRef`, and one shared JAX helper performs a coarse native value-and-gradient callback with a custom VJP. The full local quality and correctness gate passes.

Phase 9.5 nevertheless cannot be acceptance-closed under the current frozen specification because the independent numerical/JAX evidence, circuit performance package, and repeated-MVP decision package are incomplete. These are primarily acceptance-evidence gaps rather than evidence that the valid-input production implementation is numerically wrong. The review does not recommend speculative Rust refactors, per-execution defensive scans, exact allocator accounting, or broad benchmark cartesian products that are not tied to representative hot paths.

## 2. Owner triage principle recorded during review

The owner direction is to prioritize correctness and representative hot paths without over-defensive validation or over-engineering. Invalid-input handling is not a release blocker when tightening it would add material steady-state cost or substantial complexity without protecting a scientific workload. Cheap construction-time or JAX trace-time validation may be retained when it reuses metadata already computed by the native plan, but repeated Hermiticity scans, recovery layers, and defensive branches in numerical kernels are out of scope.

Accordingly, the JAX error-shape issue originally identified as P1 is downgraded to P2. It affects the exception type for invalid static inputs, not valid objective values or gradients. If the current specification continues to require stable pre-staging errors, the minimal repair is an O(1) trace-time check using existing engine metadata and existing scalar validators. Otherwise the owner may explicitly waive that clause rather than add runtime machinery.

## 3. Verification performed

- Inspected the complete `9e07a2c^..9e07a2c` diff and the live Phase 9.5 specification.

- Ran repository-wide residual searches for the removed symbolic parameter and public circuit-plan APIs, with historical `docs/vibe/` specifications and archived reviews treated separately from live documents.

- Ran focused circuit/JAX tests: 16 tests passed across `test_circuit_jax.py`, `test_circuit_facades.py`, `test_u1_circuit.py`, and `test_circuit_ir.py`.

- Ran `python scripts/check.py --benchmark skip`: Rust formatting, Black, stage-label validation, Clippy with warnings denied, Ruff, strict mypy, and `git diff --check` passed; Rust tests were 41/41, Python tests were 354/354, and doctests were 8/8.

- Ran direct probes for non-Hermitian Propagation JAX objectives, invalid propagation checkpoint intervals, invalid SPPS sample budgets and seeds, the current PyTree chain-rule fixture, the current U1 all-angle fixture, and boolean TensorCircuit U1 angles.

## 4. Completed implementation that should be preserved

- Ordinary circuit construction stores actual scalar `theta` values and exposes deterministic `angle_count`.

- Forward-only propagation and SPPS lower static angles into zero-parameter tapes and call value-only native terminals; U1 forward expectation reuses its cached final state.

- Direct native gradients return read-only contiguous `float64` occurrence-space arrays.

- U1 static and runtime angles share a small numerical `AngleRef` representation without retaining an expression DAG or reverse expression pass.

- The JAX route lazily imports JAX, requires float64 mode, stacks angle occurrences with JAX operations, performs one coarse callback in the custom-VJP forward rule, and reuses the callback gradient in the backward rule.

- Fixed-budget SPPS captures the explicit seed and sample budget and reuses the existing native joint value-and-gradient estimator.

- Public circuit `compile()` methods and ordinary/advanced Python circuit-plan exports are removed while operator-family compilation remains intact.

## 5. Open findings

### R1 — Independent numerical and JAX acceptance evidence is too weak

Priority: P1 acceptance blocker. This is a correctness-evidence finding, not a reproduced numerical failure.

Evidence: `tests/test_circuit_jax.py:17-39` compares eager and JIT results from the same implementation but does not compare the returned PyTree gradient with an independent chain-rule value. Its arithmetic and `sin`/`cos` leaves feed a final RZ gate while the objective is a Z expectation after RY rotations, so the `scale` and `smooth` gradients are exactly zero; the fixture therefore does not exercise the advertised nontrivial outer chain rule. `tests/test_u1_circuit.py:31-62` includes RZ, RZZ, CPhase, and iSWAP in one finite-difference loop, but the current state/observable fixture gives gradients approximately `[0, 0, 0, -1.7376]`, so only iSWAP is tested nontrivially. The commit also removed the prior independent dense U1 gate reference and did not replace it with equally strong actual-angle coverage. There is no callback-count instrumentation or JIT snapshot-mutation test.

Impact: the tests prove basic wiring and eager/JIT self-consistency, but a wrong occurrence order, missed outer cotangent, repeated callback, or incorrect non-iSWAP U1 derivative could pass. This matters because the phase changes the differentiation boundary itself.

Minimal closure: add one small independent nonzero value/gradient fixture for every gradient-supported gate in each applicable family, using dense or central finite differences for deterministic paths and the existing seeded/statistical reference for SPPS. Add one PyTree fixture whose repeated leaf, arithmetic leaf, and smooth-function leaf all have nonzero independently computed gradients. Instrument one representative JAX objective to assert one native forward execution and no native backward execution, and retain one persistent-circuit mutation test for immutable JIT snapshot semantics. Do not add large randomized suites or exhaustive circuit products unless a failure motivates them.

### R2 — The circuit performance evidence is incomplete and the JAX timing does not synchronize the gradient

Priority: P1 performance-evidence blocker. This is not a demonstrated runtime regression.

Evidence: `benchmarks/manual/circuit_differentiation_ab.py` contains only one deterministic propagation workload. The first and warm JAX timings call `runner(weights)[0].block_until_ready()`, synchronizing only the scalar value and not the returned gradient buffer. The benchmark does not separately report private plan construction, native first execution, native steady execution, callback-only transfer, or native kernel time; it has no U1 or SPPS workload, no native-facade-plus-caller-chain baseline, no TensorCircuit/JAX comparison, and no SPPS term-count/angle-count workspace matrix.

Impact: the recorded `jit(value_and_grad)` time may omit gradient completion or transfer, and the source cannot support the specification's claims about all three families, occurrence-gradient scaling, or comparison with the best applicable baseline.

Minimal closure: synchronize every leaf of the `(value, gradient)` result; record construction/trace, first execution, and warm execution separately; add one representative U1, deterministic propagation, and fixed-budget SPPS case; and include one matched native/caller-chain and one applicable TensorCircuit/JAX baseline. Keep the matrix small and workload-driven. Peak memory and callback bytes should be reported for SPPS because its workspace scales with observable terms times `angle_count`; no wall-time CI gate is needed.

### R3 — The repeated-MVP scratch decision package was not delivered

Priority: P1 phase-scope blocker under the current specification. No scratch optimization is required merely to close this finding.

Evidence: the commit adds only `benchmarks/manual/circuit_differentiation_ab.py`; it adds no release-mode repeated `apply_into` comparison for generic charge, U1-lazy, or structured MVP, no first-versus-steady or single-thread-versus-concurrent record, and no decision ledger stating which scratch changes were accepted, rejected, or deferred. Section 8 and the acceptance package in Section 10 of the specification therefore remain unaddressed.

Impact: there is no evidence that retained scratch would materially improve a representative Python-visible workload, but there is also no recorded decision to leave the current allocating internals unchanged. The implementation should not add synchronized pools, mutex-protected buffers, or retained memory without such evidence.

Minimal closure: either run the frozen representative matrix and record the no-change/implement decision, or amend the owner-approved specification to defer the scratch review because repeated generic charge/U1-lazy/structured MVP has not been established as a current hotspot. Under the hotspot-first owner direction, a documented defer/no-change decision is preferable to speculative scratch infrastructure; it must be an explicit scope amendment rather than an unrecorded omission.

### R4 — Some invalid JAX static inputs surface as callback runtime failures

Priority: P2 non-blocking API-quality issue; not a hot-path or numerical-correctness blocker.

Evidence: `PropagationCircuit.expectation_jax()` stages a callback without first checking `objective.engine.is_hermitian` or `checkpoint_interval`. `SPPSCircuit.expectation_jax()` captures `samples_per_term` and `seed`, but their existing validators run only inside the callback. Direct probes show that a non-Hermitian propagation observable, `checkpoint_interval=0`, `samples_per_term=1`, and `seed=-1` surface as `jax.errors.JaxRuntimeError` wrapping the original `ValueError` rather than as a pre-staging typed error.

Impact: only invalid-input error shape changes; valid hot-path execution and gradients are unaffected.

Minimal disposition: do not add repeated observable scans. If stable pre-staging errors remain required, check the already-computed `PropagationEngine.is_hermitian` flag once and invoke the existing O(1) checkpoint/sample/seed validators before constructing the callback. Otherwise amend the specification to permit callback-wrapped execution failures for these invalid static inputs and close this item without code changes.

### R5 — Live documentation still describes the superseded symbolic/JAX contract

Priority: P2 migration issue.

Evidence: `README.md:210` says direct symbolic QIR references produce parameter slots; `docs/vibe/implementation-status.md:266` says JAX tracers and `jax.jit` are outside the native contract; `docs/vibe/architecture.md:380` still describes public circuit `compile()`; and the U1 gate docstrings at `python/tencirpauli/u1_circuit.py:464-478` still describe symbolic angles.

Impact: users can follow live instructions that contradict the implemented public API, and the required zero-result migration search is not satisfied.

Minimal closure: update these live paragraphs/docstrings and record the exact residual search plus explicit historical exclusions. Do not rewrite archived specifications or historical review quotations.

### R6 — TensorCircuit U1 conversion accepts boolean angles as `1.0`

Priority: P2 narrow boundary-validation issue.

Evidence: `python/tencirpauli/integrations/tensorcircuit.py:145-153` converts U1 QIR angles with `float(theta)` without the boolean rejection already used by the propagation converter. A direct probe with `theta=True` produces QIR containing `theta: 1.0`.

Impact: one explicitly invalid public angle form is silently reinterpreted. This is not a hot-path cost issue.

Minimal closure: apply the existing `(bool, np.bool_)` boundary check before conversion and add one small regression test. No native or repeated runtime validation is needed.

### R7 — Required TensorCircuit integration tests are now optional skips

Priority: P2 packaging/test-policy issue.

Evidence: `tests/test_tensorcircuit_integration.py` and `tests/test_u1_tensorcircuit.py` now use `pytest.importorskip("tensorcircuit")` and describe TensorCircuit as an optional test-environment dependency, while `pyproject.toml` and the repository architecture make `tensorcircuit-ng` a required runtime dependency.

Impact: a broken required-dependency installation can skip the entire integration suite, including the test intended to prove the runtime dependency is available.

Minimal closure: keep optional JAX-specific tests skippable when appropriate, but let the required TensorCircuit runtime smoke fail normally when TensorCircuit is missing from the supported test environment.

## 6. Acceptance recommendation

No Rust numerical-kernel redesign is recommended from this review. Preserve the actual-angle facade, occurrence-slot lowering, static forward plans, native adjoints, shared JAX callback helper, and fixed-budget SPPS route.

Under the current frozen specification, R1-R3 must be resolved before acceptance closure because they are the correctness/performance/decision evidence explicitly required for the new boundary. Only R1 is primarily a test-coverage issue; R2 and R3 are benchmark and owner-decision evidence. R4-R7 are non-blocking cleanup items and may be fixed with narrow boundary/document changes or explicitly waived where the specification is amended.

If the owner chooses a narrower hotspot-only acceptance scope, the specification should be updated before closure: retain small nontrivial numerical/JAX proofs, retain correctly synchronized representative performance measurements, and explicitly defer repeated-MVP scratch work until profiling identifies one of those paths as a material workload. The project should not implement scratch pools or defensive execution layers merely to satisfy an unmeasured hypothetical case.
