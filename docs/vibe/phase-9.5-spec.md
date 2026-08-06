# Phase 9.5 Circuit Differentiation Boundary, JAX Execution, and Repeated MVP Specification

Status: owner-approved implementation contract. The circuit parameter, public circuit compilation, and JAX decisions in this document supersede the conflicting parameter-expression and public-plan contracts in the Phase 6, Phase Alpha, and Phase 8 design documents. Historical review documents remain historical records; live architecture, status, README, quickstart, API reference, tests, and examples must be migrated during implementation.

## 1. Purpose

This phase freezes one numerical differentiation boundary for `U1Circuit`, `PropagationCircuit`, and `SPPSCircuit`, adds an explicit JAX-compatible expectation terminal, removes the library-owned symbolic parameter system, and decides whether repeated matrix-free MVP calls justify internal scratch reuse.

The governing split is simple: users and JAX own the construction of gate-angle values from arbitrary outer parameters, while TenCirPauli owns native value and first-order gradient evaluation with respect to the individual gate-angle occurrences it executes. TenCirPauli does not parse, evaluate, serialize, or differentiate a user parameter-expression language.

Correctness remains the gate for performance work. Every numerical path must first agree with an independent reference, after which release-mode end-to-end benchmarks must include Python-to-Rust conversion, JAX callback overhead where applicable, native execution, output transfer, and any expansion from outer parameters to individual gate angles.

## 2. Frozen owner decisions

The following decisions are final for this implementation:

- Angle-bearing gate methods receive the actual angle through `theta=`. A native call receives a finite real scalar; a JAX-traced objective may supply a real scalar JAX value or tracer such as `weights[0]` or `2.0 * weights[1] + 0.1`.
- The ordinary circuit API has no `Parameter`, `ParameterExpr`, public parameter slot, public runtime parameter vector, `bind_parameters()`, `remap_parameters()`, or expression-aware `parameter_map`.
- The ordinary circuit API has no public `compile()` method and exposes no public circuit-plan type. Native plans, handles, angle-slot layouts, and caches are private implementation details.
- All three circuit facades provide `expectation()`, `value_and_grad()`, and `expectation_jax()` with the observable as the first argument and family-specific controls as keyword-only arguments.
- `expectation_jax()` is a circuit method. It uses one JAX `pure_callback` and a first-order `custom_vjp` rule backed by the native value-and-gradient engine.
- `value_and_grad()` returns derivatives with respect to independent gradient-supported gate-angle occurrences in deterministic circuit execution order. Reuse and arithmetic relationships among outer user parameters are handled only by JAX or by caller-side numerical chain rules.
- Forward-only native `expectation()` remains independent of JAX and must not compute or allocate a gradient.
- Explicit numerical runtime slots may remain inside Rust and advanced raw engines. They are an array-indexing ABI, not a symbolic parameter system, and they must not leak into the ordinary circuit API.

## 3. Scope and non-goals

This phase covers the common gate-angle contract, native value-and-gradient terminals, JAX `jit` plus reverse-mode differentiation, removal of public and Rust expression DAGs, internal circuit compilation and caching, concrete QIR migration, fixed-budget SPPS integration, and repeated MVP scratch review.

This phase does not add a general symbolic algebra system, arbitrary Python-callable tracing, expression simplification, higher-order native derivatives, Hessians, JAX accelerator kernels, or a second implementation of the native algorithms in JAX. It does not promise `vmap`, `pmap`, `pjit`, forward-mode `jax.jvp`, or differentiation of non-scalar outputs. These transformations require separate evidence and explicit contracts.

Adaptive SPPS remains on its current deterministic seeded implementation. Its direct native API remains available, but the first JAX integration is limited to fixed-budget SPPS because adaptive stopping and gradient-error proxies require an explicit decision about angle-space tolerances and stochastic optimizer behavior.

The repeated MVP review does not change the public MVP API. Any accepted scratch reuse is internal and must preserve concurrency and memory-budget semantics.

## 4. Unified public circuit contract

### 4.1 Gate construction

Every supported angle-bearing gate accepts its actual scalar angle through `theta=`. Existing gate-specific conventions remain unchanged: Pauli rotations use their current radian convention, while any gate with a documented normalized parameter such as the current U1 iSWAP keeps that convention.

```python
def objective(weights):
    circuit = tcp.PropagationCircuit(2)
    circuit.ry(0, theta=weights["shared"])
    circuit.ry(1, theta=weights["shared"])
    circuit.rzz(0, 1, theta=2.0 * weights["interaction"] + 0.1)
    return circuit.expectation_jax(hamiltonian)
```

This example has three independent native gate-angle occurrences. JAX sees that the first two angles depend on the same outer leaf and automatically sums their cotangents. TenCirPauli sees only the three numerical angles and never sees the outer dictionary or its dependency graph.

For ordinary native execution, `theta` must be a finite real scalar. For `expectation_jax()`, `theta` may additionally be a scalar real JAX array or tracer. Boolean, complex, non-scalar, and non-finite concrete angles are rejected. Arbitrary callables are not angle values.

All gradient-supported angle-bearing gate occurrences contribute one entry to the circuit's deterministic `angle_count`, ordered by gate execution order and then by the documented angle position for any future multi-angle gate. Fixed Clifford gates and static diagonal payloads do not contribute entries. `value_and_grad()` returns a read-only contiguous `float64` gradient with shape `(angle_count,)`.

Python numeric constants used as `theta` remain constants in the outer JAX program, so their returned native cotangents have no outer parameter leaf to update. The native engine may still compute their local angle derivatives as part of the uniform occurrence-space result. Performance benchmarks must measure this choice rather than reintroducing a trainable/static symbolic distinction in the public API.

### 4.2 Common terminals

The shared ordinary surface is:

```python
value_or_estimate = circuit.expectation(observable, **family_options)
result = circuit.value_and_grad(observable, **family_options)
jax_scalar = circuit.expectation_jax(observable, **family_options)
```

The normative terminal shapes are below. The private `_USE_CIRCUIT_DEFAULT` sentinel means that omitting `max_bytes` uses the budget stored by the circuit constructor; an explicit `int` or `None` overrides it for that call. The sentinel is not exported or accepted as user data.

```python
class U1Circuit:
    def expectation(self, observable: PauliOperator) -> complex: ...
    def value_and_grad(self, observable: PauliOperator) -> U1CircuitValueAndGradient: ...
    def expectation_jax(self, observable: PauliOperator) -> JaxScalar: ...

class PropagationCircuit:
    def expectation(
        self,
        observable: PauliOperator,
        *,
        initial_state: PropagationState | None = None,
        max_weight: int | None = None,
        max_bytes: object = _USE_CIRCUIT_DEFAULT,
    ) -> float: ...

    def value_and_grad(
        self,
        observable: PauliOperator,
        *,
        initial_state: PropagationState | None = None,
        max_weight: int | None = None,
        max_bytes: object = _USE_CIRCUIT_DEFAULT,
        checkpoint_interval: int | None = None,
    ) -> PropagationValueAndGradient: ...

    def expectation_jax(
        self,
        observable: PauliOperator,
        *,
        initial_state: PropagationState | None = None,
        max_weight: int | None = None,
        max_bytes: object = _USE_CIRCUIT_DEFAULT,
        checkpoint_interval: int | None = None,
    ) -> JaxScalar: ...

class SPPSCircuit:
    def expectation(
        self,
        observable: PauliOperator,
        *,
        samples_per_term: int,
        seed: int,
        initial_state: SPPSState | None = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_CIRCUIT_DEFAULT,
    ) -> SPPSValueEstimate: ...

    def value_and_grad(
        self,
        observable: PauliOperator,
        *,
        samples_per_term: int,
        seed: int,
        initial_state: SPPSState | None = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_CIRCUIT_DEFAULT,
    ) -> SPPSEstimate: ...

    def expectation_jax(
        self,
        observable: PauliOperator,
        *,
        samples_per_term: int,
        seed: int,
        initial_state: SPPSState | None = None,
        smoothing: float = 0.01,
        max_bytes: object = _USE_CIRCUIT_DEFAULT,
    ) -> JaxScalar: ...
```

The exact public typing name for a JAX scalar may use `Any` or a type-checking-only alias so JAX remains lazily imported; it must not add JAX to native import time. The argument positions, names, and semantics above are fixed.

The common behavior is:

| Contract | `U1Circuit` | `PropagationCircuit` | `SPPSCircuit` |
| --- | --- | --- | --- |
| `expectation()` | exact restricted-state expectation | exact or weight-projected native expectation | fixed-budget value estimate with diagnostics |
| `value_and_grad()` | exact adjoint value and occurrence-space gradient | frozen-support reverse value and occurrence-space gradient | fixed-budget stochastic value and occurrence-space gradient |
| `expectation_jax()` | real Hermitian scalar with custom VJP | real Hermitian scalar with custom VJP | fixed-budget real stochastic scalar with custom VJP |
| Common result fields | `value`, `gradient` | `value`, `gradient` | `value`, `gradient`, plus estimator diagnostics |

`expectation_jax()` always returns a scalar JAX array rather than a result container. JAX users obtain outer gradients with `jax.grad()` or `jax.value_and_grad()`. Direct `value_and_grad()` remains the route for inspecting the native occurrence-space gradient and, for SPPS, its diagnostics.

```python
value, gradient = jax.jit(jax.value_and_grad(objective))(weights)
```

All three families use the same observable-first call shape. Family-specific controls remain keyword-only: U1 keeps sector and restricted-state capabilities; deterministic propagation keeps weight projection and profiling controls; SPPS keeps sample budgets, seed, smoothing, error metadata, and its separate adaptive terminal.

The shared differentiable contract requires an exactly Hermitian observable and a real scalar objective. U1 may retain a direct native complex expectation capability for non-Hermitian observables, but that extension is outside `expectation_jax()` and does not weaken the common Hermitian contract.

### 4.3 Family-specific extensions

Family-specific methods remain additive rather than changing the shared terminals. `U1Circuit` retains restricted/full state and probability outputs and sector metadata. `PropagationCircuit` retains propagated-operator and profile outputs. `SPPSCircuit` retains value-error metadata and `value_and_grad_adaptive()`.

The shared API must not grow placeholder arguments merely to make signatures textually identical. A family-specific option appears only where it has real semantics, but common arguments retain the same names, positions, validation rules, and result-field meanings.

### 4.4 Public compilation and cache semantics

Circuit `compile()` methods and public circuit-plan classes are removed. Each terminal privately lowers and caches the native representation it needs. This does not remove native compilation; it removes a redundant user-facing execution model.

For deterministic propagation and SPPS, an internal objective plan is keyed by circuit generation, observable identity or immutable handle, initial-state configuration, algorithm-specific options, and resource budget. The implementation may keep a single most-recent objective cache. Therefore a native sequence `H1`, `H2`, `H1` may rebuild the first objective on the final call. Users who require independent long-lived caches may use independent circuit instances; multiple deterministic observables with shared execution should use `PropagationBatch` where its semantics apply.

U1 compilation remains observable-independent. Its internal circuit engine is keyed by circuit generation and sector configuration, while its final-state cache is keyed by concrete angle values. Switching from `H1` to `H2` at unchanged angles must reuse the same compiled circuit and final state.

A JIT-compiled `expectation_jax()` call captures an immutable native snapshot at JAX trace time. Mutating or extending a persistent circuit after tracing does not change an already compiled JAX executable; the user must retrace or rebuild the jitted objective. The recommended VQE style constructs the circuit inside the traced objective, as in Section 4.1, so Python circuit construction and native lowering occur during tracing rather than on every optimizer step.

Operator-family `compile(target=...)` APIs are unaffected. They remain meaningful explicit transformations to dense, sparse, native MVP, or backend MVP artifacts.

## 5. JAX execution and differentiation route

### 5.1 Trace-time lowering

`expectation_jax()` lazily imports JAX and fails with a clear dependency error when JAX is unavailable. It must never import JAX from ordinary native circuit construction or native `expectation()`.

At JAX trace time, the Python circuit facade performs only structural work:

1. Validate the circuit structure, observable, family options, and concrete static payloads.
2. Collect the scalar `theta` values from gradient-supported gate occurrences in deterministic execution order.
3. Lower the native tape or U1 program with private contiguous runtime angle indices `0..angle_count-1`.
4. Stack the collected values into one JAX `float64` vector using JAX operations without converting a tracer through `float()`, `np.asarray()`, or a Python sequence of host floats.
5. Capture the immutable native objective handle and static execution options in a private JAX primitive closure.

The private runtime indices are not public identities and are never serialized as user parameters. Two gates that receive the same JAX value still receive distinct native occurrence indices; JAX performs the outer accumulation.

### 5.2 `pure_callback` and `custom_vjp`

The scalar primitive uses one coarse `jax.pure_callback` whose callback invokes the native value-and-gradient path exactly once:

```python
@jax.custom_vjp
def native_expectation(angle_values):
    value, _ = jax.pure_callback(
        native_value_and_grad,
        (scalar_spec, gradient_spec),
        angle_values,
    )
    return value

def native_expectation_fwd(angle_values):
    value, gradient = jax.pure_callback(
        native_value_and_grad,
        (scalar_spec, gradient_spec),
        angle_values,
    )
    return value, gradient

def native_expectation_bwd(gradient, cotangent):
    return (cotangent * gradient,)
```

The illustrative code fixes the semantic contract, not helper names or decorator placement. `jax.value_and_grad()` must execute one native callback in the forward rule and no native callback in the backward rule. There must never be a callback per gate, Pauli term, basis state, path, or expression node.

The callback output specifications are a scalar `float64` value and a fixed-shape `(angle_count,)` `float64` gradient. The first implementation requires `jax_enable_x64=True` and fails clearly rather than silently truncating scientific calculations to `float32`.

`pure_callback` stages the native CPU call inside a JAX program; it does not compile the Rust kernel into XLA or fuse surrounding device operations into Rust. End-to-end benchmarks must include device-to-host angle transfer, host callback dispatch, native execution, and host-to-device value/gradient transfer.

The native objective handle captured by the callback must be immutable and safe for concurrent GIL-released execution. The callback must have no hidden random state or cache mutation that changes its result for the same explicit inputs.

### 5.3 Supported transformations and errors

Required JAX behavior is first-order reverse differentiation through `jax.grad()` and `jax.value_and_grad()`, both with and without `jax.jit()`, for array parameters and arbitrary outer PyTrees. Repeated outer leaves, arithmetic expressions, and ordinary smooth JAX functions such as `sin` and `cos` are verified in the outer JAX program rather than implemented by TenCirPauli.

Higher-order derivatives, `jax.jvp`, `vmap`, `pmap`, and `pjit` are unsupported in the first implementation. The wrapper must document this boundary and must not advertise a silent zero Hessian as a supported result. Batched execution should use an explicitly reviewed native batch path rather than relying on implicit callback vectorization.

Structural errors are raised before staging the callback where possible. Runtime non-finite angles, native memory-budget failures, and native numerical errors surface as callback execution failures with the original TenCirPauli context preserved as far as JAX permits.

### 5.4 SPPS-specific JAX contract

Fixed-budget `SPPSCircuit.expectation_jax()` requires an explicit sample budget and non-negative seed. The same angles, observable, budget, smoothing, seed, and circuit structure must produce the same callback result, satisfying the `pure_callback` purity requirement.

The callback invokes the existing joint stochastic value-and-gradient estimator and returns only its scalar value and occurrence-space gradient to JAX. Standard errors, gradient proxies, per-term proxies, replicate metadata, and convergence metadata remain available from direct native estimator terminals and are not differentiable JAX outputs.

The chained stochastic gradient is valid as the outer JAX VJP of the occurrence-space estimator. Its variance and any adaptive stopping proxy remain defined in occurrence-angle space unless a future design carries the required covariance information through the outer mapping. `expectation_jax()` therefore supports fixed-budget SPPS first; adaptive SPPS JAX integration is deferred.

## 6. Implementation changes and deletion boundary

### 6.1 Python removal

Delete the public `Parameter` and `ParameterExpr` classes, their arithmetic overloads, expression operand aliases, recursive evaluators, dense Jacobian construction, slot-set discovery, expression replacement, parameter binding/remapping, public runtime parameter vectors, and top-level exports. Delete or rewrite tests, benchmarks, examples, README sections, quickstart sections, type stubs, QIR helpers, and API manifests that depend on them.

`bind_parameters()` and `remap_parameters()` are removed because the circuit already stores actual angle values. `inverse()` may remain by negating actual angle values through ordinary Python or JAX arithmetic. `append()` may remain without a parameter map. No suppressive compatibility branch or deprecated parallel expression representation is retained.

Remove public circuit `compile()` methods and public `PropagationCircuitPlan`, `SPPSCircuitPlan`, and `U1CircuitPlan` exports. Equivalent private facades or native handles may remain when they reduce duplication, but users must not need to construct, type-check, or retain them.

### 6.2 Rust symbolic-expression removal

Delete `ParameterExprNode`, the parameter-program storage in `CircuitProgram`, expression-DAG validation, `evaluate_parameters()`, `reverse_parameter_program()`, expression-node adjoints, PyO3 expression opcode transport, and U1 static-expression analysis based on DAG nodes. Remove the corresponding re-export from the core crate.

The repository audit performed when this contract was frozen found Rust expression-DAG usage only in `crates/tencir-pauli-core/src/circuit_ir.rs`, `crates/tencir-pauli-core/src/u1_circuit.rs`, `crates/tencir-pauli-core/src/lib.rs`, and `crates/tencirpauli-native/src/u1_circuit.rs`. No Hamiltonian, mapping, grouping, symmetry, charge, propagation, SPPS, or structured-operator kernel consumes `ParameterExprNode`. The Rust symbolic-expression subsystem can therefore be deleted completely rather than retained for a hypothetical future consumer.

Retain the backend-neutral `CircuitProgram` and `CircuitGate` only if they still provide useful structural reuse after the expression fields are removed. Their angle representation must become numerical, for example a finite static angle or a private runtime-angle index. This representation performs no arithmetic beyond loading an angle value.

U1 adjoint execution accumulates each gate derivative directly into its independent runtime-angle entry. It no longer creates a node-adjoint vector or performs a second reverse pass through a parameter program.

### 6.3 Numerical runtime slots that remain

Propagation and SPPS already use `ParameterRef::Slot` and flat parameter arrays as their low-level numerical ABI. This mechanism remains valid because it selects an element of a concrete `f64` slice and accumulates a numerical derivative; it does not represent or evaluate a symbolic expression.

High-level circuit lowering assigns a unique private slot to every gradient-supported angle occurrence. The advanced raw `GateTape`, `PropagationEngine`, `PropagationBatch`, and `SPPSEngine` APIs may retain explicit shared numerical slots and `nparameters` because their contract is a low-level precompiled numerical tape. They remain outside the ordinary circuit facade and must not be used to reintroduce `Parameter` or `ParameterExpr` at the high level.

For U1, add the smallest numerical angle-reference representation needed by its compiler rather than copying an expression subsystem. Reuse an existing numerical primitive if its conventions fit; do not force U1's gate conventions into the propagation-specific precomputed sine/cosine representation merely for nominal type sharing.

### 6.4 Forward and gradient lowering modes

Forward-only native `expectation()` may lower concrete Python angles as static native values so existing constant folding, gate fusion, or precomputation remains available. Native `value_and_grad()` and `expectation_jax()` lower gradient-supported angle occurrences as independent runtime entries and pass their current values in one contiguous vector.

Maintaining separate private forward and gradient/JAX plan variants is acceptable when benchmarks justify the static forward optimization. It must not produce separate public execution models, duplicate numerical kernels, or divergent gate semantics. Gate order, angle convention, value, and error behavior must remain identical.

### 6.5 QIR and TensorCircuit conversion

Concrete finite-angle QIR import/export remains supported. QIR records carry actual angle values, not `Parameter` identities or expression DAGs. Object-identity symbol discovery, `parameter_order`, shared-symbol slot recovery, and expression round-tripping are removed.

JAX tracer-bearing circuits are trace-local execution objects, not serialization artifacts. `to_qir()` must reject non-concrete tracer angles with a clear error rather than leaking a tracer or inventing a symbolic encoding. A TensorCircuit conversion invoked inside a JAX trace may preserve supported scalar JAX angle values directly only if it can do so without symbol discovery or host conversion; otherwise the documented JAX route is to build the TenCirPauli circuit directly inside the objective.

## 7. Unified acceptance standard

### 7.1 Public API acceptance

- `U1Circuit`, `PropagationCircuit`, and `SPPSCircuit` expose the common `expectation(observable, ...)`, `value_and_grad(observable, ...)`, and `expectation_jax(observable, ...)` names.
- Their gate methods consistently use `theta=` for actual values. Ordinary high-level examples contain no `Parameter`, `ParameterExpr`, `parameter=slot`, runtime parameter vector, or public circuit plan.
- Circuit `compile()` and public circuit-plan types are absent from the ordinary and advanced Python API manifests. Operator `compile(target=...)` remains present.
- Each circuit exposes deterministic `angle_count`, and direct gradient shapes and ordering agree with gradient-supported gate occurrence order.
- Shared result fields use the same `value` and `gradient` names, dtypes, contiguity, and read-only rules. SPPS adds diagnostics without renaming the common fields.

### 7.2 Numerical acceptance

- Every gradient-supported gate in all three families has at least one known-number value test and an independent per-occurrence gradient comparison against finite differences or a trusted dense reference.
- Multi-gate circuits verify exact occurrence order, including multiple gates with equal numerical angles and multiple gates driven by the same outer JAX leaf.
- U1, deterministic propagation, and fixed-budget SPPS agree with their existing numerical semantics after removing the expression pass. Weight projection, frozen-support behavior, gate conventions, qubit ordering, stochastic seeding, and SPPS smoothing remain unchanged.
- Forward-only `expectation()` is proven not to call a value-and-gradient entry point or allocate an occurrence-gradient buffer.
- Invalid scalar shape, bool, complex angle, non-finite concrete angle, non-Hermitian differentiable objective, and family-specific invalid options fail with stable typed errors.

### 7.3 JAX acceptance

- `expectation_jax()` works under eager `jax.value_and_grad()` and `jax.jit(jax.value_and_grad(...))` for all three circuit families.
- A PyTree test uses one outer leaf in multiple gates and uses arithmetic plus at least one smooth JAX function to construct gate angles. The returned outer gradient agrees with an independent chain-rule reference.
- Instrumentation proves exactly one `pure_callback`/native value-and-gradient execution per scalar objective evaluation and no native execution in the VJP backward rule.
- Repeated warm JIT executions reuse the traced native snapshot. A mutation test documents that changing a persistent circuit after tracing requires retracing and does not silently mutate an existing executable.
- JAX-disabled and `jax_enable_x64=False` environments fail clearly at the JAX terminal without affecting native imports or native execution.
- Fixed-budget SPPS is deterministic for identical explicit inputs and seed, and its chained stochastic gradient agrees with the direct occurrence-space result after applying the independently computed outer JAX chain rule within statistical tolerance.

### 7.4 Repository-wide removal and migration acceptance

- Production Python and Rust source contains no `ParameterExpr`, `ParameterExprNode`, parameter-expression opcode, expression evaluator, expression reverse pass, or dense gate-angle-by-public-parameter Jacobian.
- Before acceptance, the implementer runs repository-wide searches over all tracked source, type stubs, tests, benchmarks, examples, scripts, README, quickstart, API documentation, architecture, implementation status, and other live documentation for `Parameter`, `ParameterExpr`, public circuit `compile()`, public circuit-plan types, runtime circuit parameter vectors, `parameter=slot`, `bind_parameters()`, `remap_parameters()`, and expression-aware `parameter_map` usage.
- Every executable occurrence and every current user-facing example, signature, test name, assertion, benchmark workload, script, type annotation, export manifest, and prose instruction found by that search is deleted or migrated to actual `theta` values and the common `expectation()` / `value_and_grad()` / `expectation_jax()` contract.
- README, quickstart, API reference, examples, and live design/status documents contain no instructions for the removed API. Tests and benchmarks must exercise the replacement API rather than retaining old calls merely as compatibility coverage.
- Historical specifications and archived reviews may retain quoted historical text only when their status and the design index clearly identify this specification as the superseding contract. Any exclusion of archived paths from the final zero-result search is explicit and recorded; there is no broad `docs/` exclusion.
- The acceptance evidence records the exact search commands, explicit historical exclusions, and final results so future reviews can reproduce the migration check.
- Dead helpers, dead FFI arguments, unused schema fields, compatibility probes, and abandoned expression tests are deleted rather than hidden behind casts or deprecated aliases.

### 7.5 Performance acceptance

- Benchmarks separate native plan construction, JAX tracing/compilation, first execution, warm execution, callback transfer, and native kernel time.
- Native forward-only expectation is no slower than the current concrete-angle path on representative small, medium, and large cases.
- JAX warm `value_and_grad` is measured end to end against the current native facade plus Python chain rule and against the best applicable TensorCircuit/JAX baseline with matched semantics.
- The benchmark matrix includes few outer parameters driving many gate occurrences because the new boundary returns an occurrence-space gradient of length `M` instead of a shared-slot gradient of length `P`.
- SPPS benchmarks explicitly report observable term count, `angle_count`, sample budget, gradient workspace, callback transfer bytes, and peak memory. Its current workspace contains terms proportional to term count times angle count, so the implementation must quantify the effect rather than assume the outer JAX chain is free.
- No wall-time CI gate is introduced. Reproducible release-mode benchmark sources and local comparison instructions are retained.

## 8. Repeated MVP scratch review

### 8.1 Current candidates

Generic charge `apply_into`, U1-lazy MVP `apply_into`, and structured MVP `apply_into` allocate reusable-looking buffers inside each call. The fast Fermion and eager U1 paths demonstrate that some repeated execution paths can avoid execution-time major allocations.

The public MVP APIs remain unchanged. The question is whether native internals should retain or borrow scratch across repeated calls after a release-mode benchmark demonstrates a material benefit.

### 8.2 Candidate implementation routes

1. Add core-level caller-owned `Scratch` structures and `apply_into_with_scratch` methods while retaining the existing allocating convenience methods. This gives the clearest ownership and concurrency semantics but may require a private native execution-context object before Python can benefit.
2. Keep scratch in a native plan behind a synchronized pool. This preserves the public API but adds locking, pool sizing, retained memory, and contention behavior to immutable plans.
3. Use small fixed buffers or `SmallVec` for narrow layouts. This is low-risk but only addresses small cases and does not remove variable-size destination aggregation.

The first route is preferred if the evidence warrants a change. A single mutable scratch buffer guarded by a mutex or a non-thread-safe interior-mutable cell is not the default because plans currently support concurrent calls and GIL-released execution. Any retained scratch must preserve concurrent correctness, explicit memory accounting, and the existing best-effort `max_bytes` contract.

### 8.3 Required benchmark matrix

The benchmark must be release-mode and Python-visible. It must include input conversion, the native boundary, output-buffer handling, and repeated execution rather than timing only an isolated Rust helper. Each case records plan construction separately from first apply and steady apply.

The minimum matrix contains:

- generic charge lazy MVP: conserved and non-conserved representative operators;
- U1-lazy MVP: low, medium, and wide sectors;
- structured MVP: representative fermion/boson/hybrid finite local dimensions;
- one, ten, one hundred, and one thousand repeated `apply_into` calls using caller-owned input/output arrays;
- small and medium cases where allocation overhead may matter, plus larger cases where rank/unrank or term traversal dominates;
- single-threaded repeated calls and concurrent independent calls where the plan is documented as safe;
- numerical equality against the current reference path before timing.

The benchmark record must report median runtime, first-versus-steady ratio, input term count, sector or matrix-free dimension, scratch estimate, output bytes, and any retained scratch bytes. Benchmark results remain local and informational; no wall-time CI gate is added by this specification.

### 8.4 Decision rule

If scratch reuse is below measurement noise or improves only a low-use corner case, retain the current implementation. If it produces a material steady-state improvement on a documented high-frequency workload without regressions in construction, memory, or concurrency, implement it internally without changing the Python API. A useful initial owner threshold is roughly 10% end-to-end improvement in repeated steady apply, subject to review of workload frequency and implementation complexity rather than treated as an automatic acceptance rule.

## 9. Implementation order

1. Add independent numerical reference coverage for every gradient-supported gate and preserve the current semantics before deleting any path.
2. Replace the U1 expression program with a numerical static/runtime angle representation and direct occurrence-gradient accumulation; verify native U1 values and gradients before removing the old DAG.
3. Refactor the three Python facades to store actual `theta` values, expose `angle_count`, remove runtime parameter arguments and public compile/plan APIs, and lower private occurrence slots automatically.
4. Delete Python `Parameter`/`ParameterExpr`, Rust `ParameterExprNode`, expression transport/evaluation/reverse code, binding/remapping APIs, and all abandoned tests and helpers.
5. Implement `expectation_jax()` first for deterministic propagation and U1, then for fixed-budget SPPS, using one common private callback/VJP helper where the contracts are genuinely identical.
6. Migrate concrete QIR/TensorCircuit conversion, documentation, examples, API manifests, benchmarks, architecture, and implementation status to the actual-angle contract.
7. Run the full numerical, API, JAX, concurrency, and release-mode performance acceptance matrix.
8. Run the repeated MVP scratch benchmark matrix and implement scratch reuse only if it clears the workload-first decision rule.

## 10. Acceptance evidence package

The completion package must include the final public API signatures for all three circuits, the gate-occurrence ordering contract, independent numerical gradients for every supported gate, JAX PyTree chain-rule tests, callback-count evidence, cold-versus-warm JAX timings, native forward and gradient benchmarks, the SPPS occurrence-gradient scaling matrix, and the repeated MVP scratch matrix.

The package must also contain a deletion ledger naming every removed Python and Rust expression component, a residual-slot ledger explaining each retained low-level numerical slot API, and a live-document migration checklist. It must explicitly record which scratch optimizations were implemented, which were rejected because measured benefit was insufficient, and which remain deferred.
