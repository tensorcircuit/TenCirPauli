# Phase 8.5 Explicit MVP Storage and Reusable Execution Specification

Status: frozen owner-approved implementation contract. This specification supersedes earlier Phase 8 descriptions wherever they conflict on MVP storage defaults, charge-restriction caching, or the public fixed-particle-number restriction entry point. It does not authorize JAX sparse/scatter execution, a JAX custom call for lazy plans, or an unbounded source-parallel memory multiplier.

## 1. Goals and owner decisions

Phase 8.5 makes CPU-native MVP storage explicit while keeping the ordinary path compact. Every CPU-native MVP construction defaults to `storage="lazy"`; eager storage is opt-in. A fixed MVP plan never changes storage after construction, while a restricted-operator facade may memoize an eager transition graph after the user explicitly requests CSR, COO, dense materialization, or an eager plan.

The phase also moves repeated execution behind immutable native handles, exposes caller-owned output-buffer execution, preserves the optimized packed U1 and spinful-fermion backends behind a unified restriction interface, and optimizes measured hot paths without weakening exact algebraic semantics.

The frozen decisions are:

1. Pauli, structured, generic charge-restricted, and packed U1 MVP plans all default to lazy storage.
2. `storage="eager"` is explicit and must never silently fall back to lazy when its selected retained representation exceeds `max_bytes`.
3. `ChargeRestrictedOperator` represents a mathematical restricted operator rather than one permanently fixed storage representation. It starts compact by default and may retain a thread-safe eager cache after an explicit materialization request.
4. The optimized packed U1 Rust engine remains an implementation backend. The separate `PauliOperator.restrict_u1()` public entry point is deprecated in favor of `restrict_charge()`, while `U1Sector` and `U1Circuit` remain supported.
5. Cross-strategy results must be mathematically equivalent and agree within the established complex128 tolerance; bitwise identity between eager, lazy, serial, and parallel kernels is not required.
6. `apply_into` is a strict zero-copy protocol and rejects overlapping input/output storage.
7. Source-parallel lazy execution with a state-sized output per worker remains out of scope. Destination-major parallel execution is allowed only after a serial cached baseline is correct and profiling demonstrates a representative benefit.
8. The committed QuSpin research example remains optional and may be used for manual matched A/B work, but QuSpin is not a project dependency, benchmark-suite dependency, or CI gate. The spinful Hubbard fast path is not performance-complete until a matched manual run demonstrates a steady MVP advantage over QuSpin on the approved target workload.

## 2. Scope

This phase covers CPU-native matrix-free MVP plans for Pauli, finite structured layouts, generic charge sectors, and the existing packed fixed-Hamming-weight U1 engine. It covers storage selection, restricted-facade representation caching, native plan construction, compact metadata caching, specialized hot paths, reusable output buffers, generic destination aggregation, correctness differentials, and local release-mode performance evidence.

The existing backend-array `BackendMVPPlan` remains a pure-array TensorCircuit/JAX-facing plan and does not accept `storage`. The phase does not add charge-sector JAX scatter, a lazy-plan custom call, full transition tables for unrestricted structured operators, time evolution, or a committed machine-specific result record.

## 3. Public storage and execution contract

### 3.1 Uniform lazy default

Every public CPU-native MVP construction API accepts `storage: Literal["eager", "lazy"] = "lazy"` when both retained-storage strategies exist. Representative calls are:

```python
pauli.compile("native_mvp")
pauli.compile("native_mvp", storage="eager")
structured.compile("native_mvp", boson_cutoffs=...)
structured.compile("native_mvp", storage="eager", boson_cutoffs=...)

restricted = hamiltonian.restrict_charge(sector)
lazy_plan = restricted.mvp_plan()
eager_plan = restricted.mvp_plan(storage="eager")
```

Omitting `storage` is exactly equivalent to `storage="lazy"`. Lazy construction retains compact immutable operator, sector, and layout metadata but no dimension-scale diagonal or complete transition graph. Eager construction retains the family-specific reusable representation selected below and fails before its dominant allocation if that representation exceeds `max_bytes`.

`storage` is a plan-construction property and is not an `apply()` or `apply_into()` argument. Every returned plan exposes fixed read-only `storage` and `strategy` metadata. A plan's storage, strategy, `estimated_bytes`, dimension, term count, and basis ordering never change.

### 3.2 Fixed plans versus adaptive restricted facades

A fixed `MVPPlan` is immutable and never upgrades from lazy to eager. `ChargeRestrictedOperator` is different: it is an immutable mathematical facade with thread-safe internal memoization and may hold both a compact lazy plan and an eager transition cache.

`restrict_charge(sector)` creates or retains the compact lazy representation only. `restrict_charge(sector, storage="eager")` is an explicit eager prewarm request and constructs the transition cache during restriction. The construction preference is fixed diagnostic metadata; it is not a claim that no additional representation can later be cached.

The facade's `mvp_plan()` defaults to a fixed lazy plan even when an eager cache already exists. `mvp_plan(storage="eager")` constructs or reuses the fixed eager plan. This keeps plan-level benchmarking and execution reproducible.

The facade's convenience `apply()` and `apply_into()` use the compact lazy plan until an eager graph exists. After the user explicitly requests `csr()`, `coo()`, `dense()`, or `mvp_plan(storage="eager")`, subsequent facade execution may reuse the cached eager graph. This is not a hidden storage fallback: the graph was authorized by an explicit materialization or eager-plan request. Users who require a permanently fixed strategy use the returned fixed plan directly.

The facade may expose a read-only current retained-byte estimate that increases after cache population. Fixed-plan `estimated_bytes` remains immutable. Internal memoization does not change operator semantics and must be safe under concurrent read-only calls.

## 4. On-demand materialization and eager-cache lifecycle

Calling `csr()`, `coo()`, or `dense()` on a default lazy restricted facade is valid. The call first validates the requested target size and budget, then constructs the eager transition graph if it is not already cached, and finally materializes or exposes the requested representation.

The transition graph is initialized at most once per restricted facade. Concurrent first requests must share one construction. A failed construction must not poison the cache: a later call with a larger `max_bytes` may retry. Once installed, the graph remains retained until the facade is released.

CSR is the authoritative cached eager layout when supported efficiently: destination-major row pointers, source columns, and complex coefficients allow direct parallel gather execution. COO and dense results may be derived from that graph. Implementations should share cached columns and coefficients with returned read-only sparse views where ownership and lifetime are safe; any unavoidable full-size copy must be included in the call-level memory estimate.

Materialization must never change an already returned plan. A lazy plan obtained before `restricted.csr()` remains lazy afterward; an eager plan is a separate immutable handle.

The memory-retention tradeoff is intentional. A user who requests CSR, COO, dense output, or an eager plan has explicitly authorized transition-graph construction and retention on that restricted facade. Documentation must state that dropping only the returned sparse object does not release the cached graph; releasing the facade does.

## 5. Plan-family behavior

### 5.1 Unrestricted Pauli MVP

The lazy Pauli plan retains canonical compact matrix-term metadata and evaluates terms directly. It has plan storage `O(T)` and apply work `O(TD)` for term count `T` and dimension `D`.

The eager Pauli plan retains one reusable diagonal per distinct X permutation mask. For `G` distinct masks it has retained storage `O(T + GD)`, construction work `O(TD)`, and apply work `O(GD)`. An explicit eager request that cannot retain this representation fails with a clear memory error; it never becomes `term_direct` silently.

The stable diagnostic strategies are initially `term_direct` for lazy and `x_mask_diagonal` for eager. Additional strategies may be added without becoming new public storage modes.

### 5.2 Unrestricted structured MVP

Both structured modes remain matrix-free and never retain a full dimension-scale transition graph. Both cache mixed-radix strides, canonical local operations, touched-axis lists, coefficients, and compact descriptors because these are part of the reusable plan rather than eager dimension-scale storage.

Lazy structured execution computes local transitions and factors from compact descriptors at application time. Eager mode may additionally retain bounded tables for unique local operations, such as boson ladder destinations/factors or direct-Weyl shifts/phases. Eager tables must be bounded by the involved local dimensions rather than the full Hilbert-space dimension.

If profiling shows no material benefit for a family-specific bounded cache, eager and lazy may use the same kernel and compact representation. For compatibility, the current structured plan preserves the requested `storage="eager"` metadata as an accepted alias even when `strategy` and `estimated_bytes` are identical to lazy; this metadata records the accepted request rather than claiming that a distinct cache was retained. The implementation must not allocate a meaningless cache merely to justify the eager label.

The generic structured kernel preserves mixed-radix basis ordering, projected-boson boundary behavior, direct-Weyl conventions, exact cancellation semantics, and finite-amplitude checks.

### 5.3 Generic charge-sector MVP

The lazy generic charge plan retains the reusable charge rank/unrank plan, validated native term descriptors, layout metadata, and bounded lookup tables. It enumerates sector states on application and retains no complete restricted transition graph.

For non-termwise-conserving operators, contributions reaching the same destination are aggregated before exact-zero removal and sector-leakage validation. For independently validated termwise-conserving descriptors, the specialized kernel may bypass the per-source destination map without weakening leakage checks.

The eager generic charge representation is the deterministic aggregated restricted transition graph. It is created only by an explicit eager prewarm, eager-plan request, or materialization call, and is cached by the restricted facade as specified above.

### 5.4 Packed U1 backend and unified restriction API

The existing packed fixed-Hamming-weight U1 engine remains authoritative for eligible pure-qubit sectors. It must not be replaced by the generic occupation-vector charge kernel. Its arbitrary-width packed representation, combinatorial rank/unrank semantics, active-particle/active-hole optimizations, X-mask grouping, destination-major ordering, deterministic aggregation, and wide-system memory guarantees remain intact.

Packed U1 gains a genuine lazy plan. The lazy representation retains the packed sector index and grouped Pauli descriptors but no complete destination-major transition graph. The eager representation is the existing destination-major graph. Both default and on-demand cache behavior follow the common restricted-facade contract.

`PauliOperator.restrict_charge()` accepts both `ChargeSector` and `U1Sector`. A `U1Sector`, or a generic charge sector proven equivalent to a single canonical qubit-number constraint, dispatches to the packed U1 backend. Eligibility must be derived and validated in native plan construction rather than trusted from an unchecked Python flag.

`PauliOperator.restrict_u1()` is deprecated in Phase 8.5 and emits `DeprecationWarning` with `restrict_charge(U1Sector(...))` as the replacement. `U1RestrictedOperator` and `U1MvpPlan` cease to be ordinary user-facing entry points and remain compatibility or advanced types during the deprecation window. Removal occurs no earlier than the next explicit breaking release after one released deprecation cycle.

`U1Sector` remains supported because it provides the fixed-particle basis contract used by `U1Circuit`, packed rank/unrank helpers, and direct restriction convenience. `U1Circuit`, `U1CircuitPlan`, their gate set, state terminals, observable semantics, and gradients are not deprecated or redesigned by this phase.

### 5.5 Pure spinful fermion backend

The pure spinful fermion charge backend handles raw fermion layouts such as the Hubbard `N_up`/`N_down` sector. It is separate from packed qubit U1 because it supports two independently fixed species and avoids a mandatory fermion-to-Pauli mapping.

The plan caches its eligible-sector index and, subject to checked internal thresholds and `max_bytes`, combination masks and rank tables. For the 4x4 half-filled spin sector the combination-mask and rank tables are approximately 0.20 MiB and 0.25 MiB, while one complex128 state is approximately 2.47 GiB. Larger sectors retain the bounded combinatorial fallback rather than forcing table construction.

Eligible terms compile once into compact bit descriptors containing occupation requirements, destination flips, parity masks, diagonal/off-diagonal classification, coefficients, and validated rank-preserving shortcuts. Common diagonal density interactions and quadratic hopping use specialized descriptors; unsupported quartic or general patterns remain on a validated generic fermion descriptor.

## 6. Native plan construction and cache policy

Plan construction performs Python-to-native term conversion, canonical structural validation, storage eligibility, termwise-conservation analysis, fast-path selection, and retained-memory estimation once. Repeated `apply()` and `apply_into()` calls reuse one immutable native handle and must not reconstruct Python lists, positions, operation codes, coefficients, sector indices, or fast-path tables.

Native constructors, not Python booleans, determine whether packed U1, spinful fermion, termwise-conserving, or generic aggregation kernels are valid. Python may perform friendly prevalidation, but the Rust core remains the final semantic authority.

Cache thresholds for combination masks, rank tables, and bounded local lookup tables are internal deterministic implementation choices. They must obey `max_bytes`, retain a bounded fallback, be reported in `estimated_bytes`, and be benchmarked, but they do not become additional public tuning parameters.

Plans are immutable and shareable. Per-call scratch is owned by the call, not the plan, so concurrent calls with distinct outputs cannot race through shared mutable workspace. Any internal once-cache on a restricted facade must be synchronized and must publish only fully validated immutable plans.

## 7. Reusable output-buffer execution

`apply_into` means “write the complete result into caller-provided output storage”; it never means in-place overwrite of the input. The public signature is:

```python
plan.apply_into(input_state, output_state, *, max_bytes=DEFAULT_MAX_BYTES) -> None
```

Input and output must be NumPy arrays with exact dtype `complex128`, shape `(dimension,)`, and C-contiguous storage. Output must be writable. `apply_into` performs no dtype conversion, shape normalization, contiguous copy, or hidden temporary output allocation.

Any input/output memory overlap is rejected before execution. The implementation must check overlapping byte ranges rather than only Python object identity, so overlapping slices are also rejected. Validation and budget failures occur before output modification.

The kernel overwrites every output element and does not accumulate over prior contents. `apply_into` returns `None`. If an execution-time numerical error can still occur after validation, the output contents after that error are documented as unspecified; the design should move all structural failures to construction so this case is exceptional.

The existing `apply()` remains the allocating convenience. It normalizes its accepted friendly input, allocates one owned writable C-contiguous complex128 output, calls the same core kernel, and returns the output. Its numerical semantics are identical to `apply_into` within the stated strategy contract.

`apply_into` is required for lazy/eager Pauli, structured, generic charge, packed U1, and spinful fermion CPU-native plans. It is out of scope for backend-array/JAX callables.

## 8. Numerical and determinism contract

All strategies represent the same mathematical operator in the same basis. Phase conventions, fermionic signs, projected-boson boundaries, Weyl conventions, charge selection, and basis ordering are exact semantic requirements.

Static canonicalization and destination-coefficient aggregation remove only exact complex zeros. No coefficient-magnitude cutoff may enter MVP construction or execution. Contributions that can cancel before leakage validation must be aggregated before that validation.

Eager, lazy, generic, specialized, serial, and parallel kernels are not required to produce bitwise-identical complex128 arrays because legal grouping and parallel row execution can change floating-point association. They must agree with independent references and with one another within the established complex128 tolerance. Every individual strategy must be deterministic across repeated runs; parallel scheduling must not change the per-output accumulation order.

Optimizations that deliberately change the accumulation order require an explicit differential test and recorded numerical-error evidence. They do not require a new owner decision when they remain within the frozen tolerance and preserve exact-zero/leakage semantics.

## 9. Parallelism policy

The initial cached lazy implementations are serial references. Automatic source-parallel execution that requires one state-sized output per worker remains prohibited because it multiplies the dominant memory of large sectors.

After the serial spinful-fermion and packed U1 kernels are stable, profiling may evaluate destination-major gather. Each worker must own disjoint output rows and only bounded local scratch; no state-sized per-worker accumulation buffer is allowed. Rayon activation requires a measured release-mode benefit on representative workloads, deterministic output, correctness differentials, and no material small-case regression.

The implementation may retain serial execution when destination-major input access, scheduling overhead, or descriptor inversion makes parallel execution slower. Parallelism is an optimization selected by evidence, not a phase-completion requirement by itself.

## 10. Generic destination aggregation

The generic lazy charge path retains per-source aggregation for layouts that are not independently termwise conserving. Repeated full occupation-vector cloning and per-entry heap allocation should be replaced only when profiling identifies them as a material bottleneck.

Allowed improvements include reusable scratch, compact bounded keys, packed small-axis keys, and slice-backed mixed-radix keys. Every packed representation requires overflow-safe eligibility checks and a general fallback. Generic aggregation optimization must be benchmarked separately from the spinful Hubbard path so it cannot regress the specialized backend unnoticed.

This workstream is profile-gated. A profile demonstrating that it is not a material bottleneck is sufficient to close it without speculative optimization.

## 11. Memory and error policy

Plan `estimated_bytes` reports logical major retained buffers, including compact descriptors and selected caches, but excluding caller-owned input/output, Python wrapper overhead, and allocator metadata. A restricted facade's current retained estimate may grow after its eager once-cache is populated; each fixed plan's estimate remains constant.

Construction `max_bytes` covers the requested plan-owned buffers and unavoidable major construction workspace. `apply()` covers its newly allocated output and per-call scratch. `apply_into()` covers only newly allocated scratch because both state buffers are caller-owned. A materialization call covers a missing eager cache, its major construction workspace, and the requested output that is not safely shared with the cache.

The checks remain best-effort major-buffer guards rather than exact peak-RSS quotas. Checked arithmetic and dimension overflow are mandatory. Invalid storage values, incompatible shapes/dtypes, non-contiguous or read-only output, overlap, invalid sector/backend eligibility, budget violations, and unsupported targets fail explicitly before the dominant allocation or execution.

No path may silently switch public storage mode, materialize a full-space state, use a dense fallback, alter basis conventions, or route an eligible packed U1 case through a slower generic backend without a documented reason and benchmark evidence.

## 12. Correctness gates

Tests compare lazy and eager outputs against independent dense or trusted small-system references for Pauli, fermion, boson, qudit, hybrid, generic charge, packed U1, and structured fixtures where applicable.

Restricted-facade tests must cover lazy construction, lazy execution, first CSR/COO/dense eager-cache construction, cache reuse, automatic facade execution reuse after materialization, fixed lazy-plan stability after cache creation, explicit eager-plan reuse, failed-build retry, concurrent first materialization, and retained-memory reporting.

`apply_into` tests verify exact dtype/shape/contiguity validation, rejection of read-only and overlapping outputs, unchanged input, complete overwrite of nonzero garbage, repeated two-buffer alternation, equality with allocating `apply()`, budget behavior, and concurrent calls on one immutable plan with distinct outputs.

Packed U1 dispatch tests cover `restrict_charge(U1Sector(...))`, recognition of an equivalent canonical generic charge sector, wide low-particle and low-hole systems, eager/lazy differentials, strict leakage behavior, and `restrict_u1()` deprecation warnings. `U1Circuit` regressions remain unchanged.

Spinful fermion tests cover diagonal density terms, hopping, sign-sensitive orderings, supported and fallback quartic terms, zero coefficients, exact cancellation, low/high filling, combination-table and combinatorial fallback paths, generic-versus-specialized differentials, and the 4x4-compatible `N_up=N_down` convention.

Structured tests cover mixed-radix ordering, boson cutoffs, Weyl shifts/phases, bounded eager tables, cache-disabled fallback, finite-amplitude failures, and generic-versus-specialized results.

## 13. Performance gates and QuSpin research comparison

Committed TenCirPauli benchmarks separate plan construction, eager-cache construction, first application, steady allocating `apply()`, steady `apply_into()`, sparse materialization, and repeated-buffer workflows. They report dimension, source and canonical term counts, transition count when present, storage, strategy, state/output bytes, retained plan bytes, scratch estimate, runtime, throughput, allocation behavior, peak memory where measured, thread count, and numerical error.

The standard benchmark matrix includes small eager/lazy crossover cases, the 2x4 spinful Hubbard case, a practical 4x3 spinful Hubbard case, at least one packed U1 case, one structured case, and one non-termwise-conserving generic aggregation case. A TenCirPauli-only 4x4 workload may live under committed manual/research tooling and is not part of routine smoke or CI.

QuSpin remains an optional research comparison. Its script may be committed under `examples/research/`, but it is excluded from package dependencies, project environment setup, the formal pytest-benchmark suite, CI, and automated release gates. QuSpin is installed and run in a separate temporary environment; machine-specific comparison results remain local or under `/tmp` and are not committed.

The spinful Hubbard fast path is not performance-accepted until a matched manual A/B on the approved machine shows that TenCirPauli steady MVP is faster than QuSpin `quantum_LinearOperator.dot` beyond ordinary timing noise. The primary target is the 4x4 half-filled open-boundary `N_up=N_down` workload; 4x3 is the development-scale checkpoint. Runs use the same Hamiltonian, complex128 dtype, state semantics, thread allocation, warmup policy, and process-level resource policy. Construction time and peak memory are reported separately and are not substituted for steady MVP throughput.

The matched comparison reports allocating `TenCirPauli.apply()` against allocating QuSpin `dot()` as the public-equivalent measurement and reports `apply_into()` separately to quantify iterative-solver buffer reuse. Small-system basis-mapped correctness is established before the large performance run.

Internal optimizations are accepted only when correctness passes and same-machine release measurements show a benefit at the intended bottleneck beyond measurement noise, with no material regression in runtime, memory, or scaling on the rest of the representative matrix. No single universal speedup ratio is imposed, and no wall-time CI threshold is added.

## 14. Complexity and intended tradeoffs

The intended leading-order tradeoffs are:

| Plan family | Lazy retained space | Lazy apply | Eager retained space | Eager apply |
| --- | ---: | ---: | ---: | ---: |
| Pauli | `O(T)` | `O(TD)` | `O(T + GD)` | `O(GD)` |
| Structured | `O(TK + A)` | `O(DTK)` | `O(TK + A + L)` | `O(DTK)` with smaller constants |
| Generic charge | `O(TK + P)` | approximately `O(DT(A + K))` | `O(E + P)` | `O(E)` |
| Packed U1 | `O(TW + P)` | approximately `O(DT)` | `O(E + P)` | `O(E)` |
| Spinful fermion | `O(TK + C + R)` | approximately `O(DT)` | `O(E + C + R)` | `O(E)` |

Here `D` is restricted or full dimension, `T` is canonical term count, `G` is distinct Pauli X-mask count, `E` is aggregated transition count, `A` is layout axis count, `K` is average touched-operation count, `W` is packed word count, `P` is sector-plan metadata, `L` is bounded local lookup storage, and `C`/`R` are optional combination-mask/rank tables.

Lazy is the ordinary default because it makes large valid MVPs constructible and preserves compact memory. Eager is valuable when a graph/cache fits and repeated applications amortize construction. The restricted facade's on-demand cache gives ordinary users both behaviors without requiring them to predict future materialization needs at initial restriction time.

## 15. Delivery order

1. Implement the uniform lazy default, fixed plan metadata, restricted-facade/cache distinction, and Phase 8.5 deprecation surface.
2. Move generic charge, packed U1, spinful fermion, Pauli, and structured repeated execution behind immutable native handles with one-time conversion and validation.
3. Implement thread-safe on-demand eager transition caching and CSR/COO/dense materialization reuse for restricted facades.
4. Expose strict `apply_into` across all compatible CPU-native plan families and establish allocation/steady baselines.
5. Implement and differentially test packed U1 lazy execution, cached spinful fermion descriptors/tables, and required structured compact metadata.
6. Profile destination-major parallelism and bounded structured lookup tables; retain only measured improvements.
7. Profile generic destination aggregation and optimize it only if it remains material.
8. Run the complete correctness and TenCirPauli benchmark matrix, then perform the separate matched QuSpin research A/B for the Hubbard fast path.

## 16. Completion criteria

Phase 8.5 is complete when all CPU-native MVP entry points default to lazy, explicit eager requests and on-demand materialization obey the frozen cache contract, fixed plans remain immutable, repeated calls perform no Python-to-native term reconstruction, packed U1 and spinful fermion specializations remain behind the unified restriction API, `apply_into` is available and validated, correctness gates pass, accepted optimizations have representative release evidence, and the manual matched Hubbard comparison demonstrates steady TenCirPauli MVP performance above QuSpin.

The local quality gate, strict documentation build, committed TenCirPauli benchmark definitions, and research-example smoke must pass. Machine-specific benchmark and QuSpin result files remain untracked. No JAX charge scatter, source-parallel state-sized memory multiplier, dense fallback, or weakening of exact algebraic semantics may enter under this phase.
