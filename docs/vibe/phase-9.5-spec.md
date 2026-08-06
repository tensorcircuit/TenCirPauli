# Phase 9.5 Gradient Boundaries and Repeated MVP Technical Review Specification

Status: owner discussion draft. This document records the proposed technical direction and decision gates for the next review; it is not acceptance-closed and does not authorize speculative implementation without the evidence required below.

## 1. Purpose

Phase 9 established native operator storage, native gate tapes, handle-consuming terminals, and thin Python facades. The next review concerns two cross-cutting questions that remain after the third audit: whether U1Circuit, PropagationCircuit, and SPPSCircuit expose one coherent gradient boundary, and whether repeated matrix-free MVP calls justify internal scratch reuse.

The governing rule is workload-first: implement a change only when a representative high-frequency path shows a material end-to-end benefit, including Python-to-Rust conversion and boundary costs. A kernel-level improvement without a meaningful public-workload improvement is not sufficient. Public API expansion, concurrency restrictions, memory-retention changes, and duplicated execution models count as real costs in the decision.

## 2. Scope and non-goals

This review covers:

- the relationship between public circuit plans and native engines for value-and-gradient execution;
- a common contract for public parameter expressions, per-gate numerical angles, and public-slot gradients;
- the boundary between Python expression handling and Rust numerical propagation;
- internal scratch reuse for generic charge, U1-lazy, and structured matrix-free MVP plans;
- release-mode benchmarks and acceptance gates for the scratch decision.

This review does not currently change the adaptive SPPS term-parallelism policy. Adaptive SPPS remains on the existing deterministic implementation until a separate workload demonstrates that term-level parallelism is a material end-to-end win after scratch memory, deterministic random streams, and small-workload overhead are accounted for.

This review does not introduce a general-purpose Rust symbolic algebra system. It does not add symbolic simplification, arbitrary Python-callable tracing, algebraic rewriting, or a SymPy-like expression engine.

## 3. Current state

The low-level `PropagationEngine.value_and_grad` already performs Pauli forward propagation, aggregation, reverse propagation, and gradient computation in Rust. Its numerical inputs are the independent dynamic gate-angle slots in the native tape, and its gradient is with respect to those slots.

`PropagationCircuitPlan.value_and_grad` is a higher-level adapter. It currently evaluates `Parameter`/`ParameterExpr` values in Python, creates a dense angle-by-public-parameter Jacobian, calls the native engine, and applies `jacobian.T @ native_gradient` in Python. Therefore the propagation kernel is native, but the public-parameter chain rule is not fully native and the dense Jacobian can be wasteful.

`SPPSCircuitPlan` has the same public-parameter shape around the stochastic native engine. Its adaptive sampling loop is a separate issue and remains unchanged by this document.

`U1Circuit` already has a native parameter-expression program in the common circuit IR. It evaluates parameter nodes and reverses their adjoints inside the U1 execution path. This is evidence for the mathematical contract, not a requirement to copy a second general symbolic subsystem into Propagation or SPPS.

The current public expression language is a small arithmetic DAG over parameter slots and finite real constants. It supports negation, addition, subtraction, multiplication, and division. Any future function such as `sin` or `cos` must be an explicitly supported expression operation; arbitrary Python functions are outside the trace contract.

## 4. Unified gradient contract

The common semantic contract is:

```text
public parameter values p
        |
        | Python-owned parameter-expression trace
        v
independent numerical gate angles theta
        |
        | one coarse native engine call
        v
value and dvalue/dtheta from Rust
        |
        | Python reverse trace / slot accumulation
        v
dvalue/dp in public parameter-slot order
```

The native engine contract is intentionally numerical: each parameterized gate receives an independent angle input, and the engine returns the gradient with respect to those independent angles. The circuit-plan contract is responsible for mapping public parameters to those angles and mapping the returned angle adjoints back to public slots. This is the eventual common boundary for U1Circuit, PropagationCircuit, and SPPSCircuit; unifying the boundary does not require their internal numerical kernels to become identical.

For an expression `theta_i = f_i(p)`, the public gradient is:

```text
dvalue/dp_j = sum_i (dvalue/dtheta_i) * (dtheta_i/dp_j)
```

Repeated public parameters are accumulated, and direct parameter references use a cheap slot scatter. Expressions are evaluated numerically; no symbolic simplification is required.

The target is one native propagation or sampling call per plan execution, not one PyO3 call per gate or per expression node. Python may perform a compact pre-call forward trace and a compact post-call reverse trace over parameter metadata. These operations must not construct an operator-sized intermediate, a dense gate-by-parameter Jacobian, or a Python object per Pauli term or gate execution.

## 5. Proposed parameter-expression implementation

### 5.1 Compile-time representation

At circuit compilation, Python validates the public expression objects and lowers each dynamic angle into a private compact trace. The trace may be a topologically ordered opcode list with node operands and one root per dynamic gate angle. It is a numeric autodiff trace, not a symbolic algebra engine: it does not rewrite expressions or attempt to prove identities.

The trace is cached with the structural circuit plan. Parameter values do not invalidate it. Circuit mutation, parameter-expression replacement, or QIR reconstruction invalidates and rebuilds it.

### 5.2 Runtime execution

At each execution, Python performs a forward evaluation of the compact trace to produce the independent gate-angle array. It calls the existing native engine once. The native engine returns the scalar value and one adjoint per gate angle. Python then performs a reverse traversal of the same trace and returns the public-slot gradient.

The implementation must replace the current dense Jacobian path before considering a native symbolic evaluator. The first implementation should keep the public circuit APIs unchanged and should preserve the low-level raw-angle `PropagationEngine` API.

### 5.3 Supported operations and extension policy

The first trace supports the existing arithmetic expression operations. A smooth function such as `sin` or `cos` can be added as one explicit opcode with a forward value and a reverse derivative rule; a parameter expression containing such a function is traceable only after that opcode is part of the supported expression language. Domain-sensitive operations such as division, logarithm, or tangent require explicit validation and error semantics. Branching, `abs`, `min`, and `max` require a separate differentiability decision and are not inferred automatically. Arbitrary Python callables are outside the trace contract.

U1's existing native parameter program is a transition implementation and a reference for node semantics during this review, not the target boundary for the other engines. The convergence route is to make the U1 plan accept independent numerical gate angles and return angle-space gradients like the propagation and SPPS engines, then use the same Python trace and reverse scatter for all three plans. If a common implementation is later justified, extract only the small parameter-node contract and evaluator primitives that are demonstrably shared. Do not expand the scope into a general symbolic engine merely to make the internal implementation identical.

### 5.4 Decision gates

The Python trace route is accepted as the common route when it satisfies all of the following for U1, Propagation, and SPPS:

- deterministic numerical agreement with the current independent gradient references for direct parameters, repeated parameters, nested arithmetic expressions, and invalid expression inputs;
- no dense Jacobian allocation proportional to dynamic-angle count times public-parameter count;
- one native execution call per plan evaluation;
- no operator-sized Python materialization or per-gate PyO3 crossing;
- release-mode end-to-end performance that is no worse than the current implementation on small and medium circuits and materially better on large parameterized circuits where the dense Jacobian is significant.

A native parameter evaluator becomes a separate owner decision only if profiling shows that the compact Python trace itself is a material fraction of representative optimizer-loop time. The default is to keep the trace in Python because this avoids introducing a second Rust symbolic subsystem.

## 6. Repeated MVP scratch review

### 6.1 Current candidates

Generic charge `apply_into`, U1-lazy MVP `apply_into`, and structured MVP `apply_into` allocate reusable-looking buffers inside each call. The fast Fermion and eager U1 paths demonstrate that some repeated execution paths can avoid execution-time major allocations.

The public Python APIs remain unchanged in this review. The question is whether native internals should retain or borrow scratch across repeated calls after a release-mode benchmark demonstrates a material benefit.

### 6.2 Candidate implementation routes

1. Add core-level caller-owned `Scratch` structures and `apply_into_with_scratch` methods while retaining the existing allocating convenience methods. This gives the clearest ownership and concurrency semantics but may require a private native execution-context object before Python can benefit.
2. Keep scratch in a native plan behind a synchronized pool. This preserves the public API but adds locking, pool sizing, retained memory, and contention behavior to immutable plans.
3. Use small fixed buffers or `SmallVec` for narrow layouts. This is low-risk but only addresses small cases and does not remove variable-size destination aggregation.

The first route is preferred if the evidence warrants a change. A single mutable scratch buffer guarded by a mutex or a non-thread-safe interior-mutable cell is not the default because plans currently support concurrent calls and GIL-released execution. Any retained scratch must preserve concurrent correctness, explicit memory accounting, and the existing best-effort `max_bytes` contract.

### 6.3 Required benchmark matrix

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

### 6.4 Decision rule

If scratch reuse is below measurement noise or improves only a low-use corner case, retain the current implementation. If it produces a material steady-state improvement on a documented high-frequency workload without regressions in construction, memory, or concurrency, implement it internally without changing the Python API. A useful initial owner threshold is roughly 10% end-to-end improvement in repeated steady apply, subject to review of workload frequency and implementation complexity rather than treated as an automatic acceptance rule.

## 7. Deferred adaptive SPPS parallelism

Adaptive SPPS remains serial across terms for now. Any future term-level parallel implementation must retain deterministic per-term random streams, combine results in canonical term order, account for one `PathScratch` per active worker, and keep a serial fallback for small term counts or budgets. It must demonstrate a material end-to-end gain on a representative adaptive workload before implementation.

## 8. Implementation order for further review

1. Add and retain the numerical and diagnostic test coverage identified by the third audit.
2. Implement the compact Python parameter trace for deterministic PropagationCircuit while keeping the raw-angle native engine unchanged.
3. Compare direct, repeated-parameter, and expression-gradient release benchmarks against the current dense-Jacobian path.
4. Run the repeated MVP `apply_into` benchmark matrix and decide whether native scratch reuse clears the workload-first gate.
5. Migrate U1 and SPPS onto the same angle-space native contract after the Propagation trace is validated; remove the U1-specific symbolic execution path once equivalent coverage and representative benchmarks are available.

## 9. Acceptance evidence

The review package must include the final parameter-boundary diagram, the public-to-gate parameter mapping contract, independent numerical gradient differentials for all three circuit families, a release-mode end-to-end comparison of the old and compact Python parameter paths, and the repeated MVP benchmark matrix from Section 6.3.

The package must explicitly record which optimizations were implemented, which were rejected because the measured benefit was insufficient, and which remain deferred. A rejected optimization is not a defect when its representative end-to-end benefit does not exceed its complexity, memory, concurrency, or maintenance cost.
