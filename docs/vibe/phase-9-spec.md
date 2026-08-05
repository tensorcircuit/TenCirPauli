# Phase 9 Rust-Native Data Plane and Python Thinning Specification

Status: frozen owner-approved implementation contract. No owner decision remains open. This specification supersedes earlier documents wherever they require Python `Fraction` fallback, arbitrary-integer-exact charge aggregation, post-operation coefficient overflow rejection, or Python materialization of operator-sized data that can remain in a native handle.

## 1. Goal

Phase 9 makes Python the public API and integration layer while moving the remaining symbolic-computation data plane into Rust. A Python operator facade contains an immutable native handle plus small layout and API metadata. Operator-sized terms, full tapes, pairwise compatibility data, mapping intermediates, compilation inputs, charge transitions, and propagation states must stay in Rust until the caller explicitly requests a public materialized result.

The phase also removes numerical defenses aimed only at unrealistic magnitudes. Native numerical algebra uses ordinary IEEE `f64`/`complex128` arithmetic. The implementation must not pay recurring complexity or runtime cost to preserve exactness beyond binary64, detect every internally produced infinity or underflowed zero, or rescue hundreds-of-operator products that exceed the practical workload range.

Correctness for representative scientific workloads remains mandatory. The simplification in this specification concerns implausible numerical extremes, not algebraic phase, qubit ordering, canonicalization, deterministic output, dimension safety, allocation safety, or normal small-integer charge semantics.

The unified read-back ABI belongs in Phase 9 because it is a cross-cutting private PyO3 and Python-thinning change: it removes operator-sized boundary traffic without changing the public mathematical results or requiring a new scientific capability. It must be implemented alongside handle-native producers and terminal compilation so that new native paths do not reintroduce a materialize-and-reparse route.

## 2. Frozen owner decisions

1. Python retains friendly input normalization, type hints, docstrings, public result wrappers, explicit `.terms` and `to_dict()` exports, SciPy conversion, TensorCircuit/backend integration, and the public circuit-building DSL.
2. Rust owns every traversal whose work scales with the number of operator terms, tape operations, term pairs, groups, charge transitions, symmetry rows, or finite-basis transitions.
3. Compatible same-layout addition across specialized structured families promotes directly to a native-backed `HybridOperator`. It must not materialize typed Python terms.
4. Pauli, Fermion, Boson, Hybrid/Qudit, and Majorana mapping and conversion chains remain handle-native until a public output format is requested.
5. Fermion, Boson, Hybrid/Qudit, and Majorana commutators and anticommutators use fused native aggregation. This is an internal production optimization with no new public method.
6. Native dense, COO, CSR, MVP, propagation, restriction, grouping, symmetry, and sampling entry points consume native operator handles directly. Numeric handle read-back and terminal outputs use the unified flat-array ABI in Section 3.1; a native producer returns a handle whenever the next consumer is native.
7. All material native O(n), O(term-pair count), mapping, compilation, conversion, and execution work releases the GIL. Only Python argument extraction and Python result construction run with the GIL held.
8. A lazily compiled native `GateTape` handle is cached by each Python circuit facade and invalidated by every tape mutation. The public circuit DSL remains Python.
9. Charge analysis uses deterministic native `f64`/`complex128` arithmetic and exact comparison with floating-point zero. `Fraction`, arbitrary-precision integers, custom wide accumulators, and Python fallback analysis are removed.
10. Internal coefficient arithmetic follows ordinary IEEE behavior. Direct public input validation continues to reject an explicitly supplied NaN or infinity once at the boundary, but native algebra does not repeatedly rescan trusted handles or reject an infinity, NaN, overflow, or underflow produced by later arithmetic.
11. Dimension, index, shape, FFI-length, packed-word, and major-allocation overflow checks remain required. They protect memory safety or prevent impossible allocations and are not numerical over-defense.
12. Native tensor products and `Arc`/copy-on-write structural sharing remain deferred.
13. Redundant `_native_data`, parallel Python arrays, compatibility fallbacks, and production Python reference kernels are deleted once their native path is complete. A fallback is not retained merely because it existed before.

## 3. Architecture invariant

For a handle-backed object, every ordinary operation follows this shape:

```text
friendly Python input -> small validated descriptor -> one coarse native call
native handle -> native handle -> native handle -> explicit public materialization
```

The following shape is forbidden on an ordinary production path:

```text
native handle -> Python terms/parallel arrays -> Python traversal -> native handle
```

Passing O(axis count), O(group count), or O(parameter count) control metadata once is allowed. Passing O(operator term count), O(tape length), O(term-pair count), or O(basis-transition count) data through Python is not.

The Rust core remains independent of PyO3, Python, NumPy, TensorCircuit, and SciPy. PyO3 bindings translate small descriptors and own Python result construction; they must not become a second implementation of the algebra.

### 3.1 Unified numeric read-back and terminal-output ABI

The private PyO3 boundary uses one rule for every native handle read-back path: operator-sized numeric data either stays in a native handle or crosses into Python as flat NumPy arrays. No handle-owned `materialize*`, dense, COO, CSR, MVP, backend-plan, or equivalent numeric export may return nested `Vec<Vec<...>>`, split Python lists that the wrapper immediately recombines, or a nested Python `Sequence[Sequence[...]]` that is later re-marshaled into Rust. This is an ABI rule, not a requirement that every operator family share one physical element layout.

Fixed-width Pauli term exports use one documented flat schema with explicit count and shape metadata. The schema may use flat `uint8` codes or flat packed `uint64` X/Z words, together with one flat `complex128` coefficient array; the implementation chooses one canonical numeric representation per read-back contract and deletes superseded parallel variants. Variable-length Fermion, Boson, Hybrid/Qudit, and Majorana words use flat payload arrays plus offsets, indptr, lengths, or an equivalent compact descriptor. These descriptors are numeric and contiguous; they must not be represented as Python lists of per-term lists or tuples.

Matrix and vector terminal outputs use direct NumPy ownership transfer: dense returns a flat `complex128` buffer plus dimension, COO returns flat row/column/value arrays, CSR returns flat indptr/indices/value arrays, and MVP returns a flat `complex128` vector. A single complex value array is preferred over separate real and imaginary Python sequences. Existing handle-accepting native entry points are extended or reused so that Python does not materialize an operator merely to call a sibling FFI function.

`PyArray1::from_vec` is an ownership-transfer boundary that removes intermediate Python list allocation; it is not advertised as an alias of the handle's internal storage. Native output construction may still allocate or copy into the requested contiguous result buffer. The contract is no Python object expansion and no redundant native reparse, not a false zero-copy guarantee from the immutable handle itself.

Textual exports such as `to_dict()` remain explicit public materialization boundaries. A private string helper may be retained only for that textual result and must not serve as the canonical numeric ABI or as an intermediate for another native call. Explicit `.terms`, string, dense, COO, CSR, and NumPy requests may create their documented Python result objects; the forbidden pattern is the incidental round trip `native handle -> Python nested data -> native rebuild`.

The required boundary shapes are therefore:

```text
allowed:   native handle -> native handle -> native terminal -> flat NumPy result
allowed:   native handle -> flat NumPy result for an explicit public export
forbidden: native handle -> nested Python data -> Python traversal -> native rebuild
```

## 4. Ordinary floating-point policy and removal of over-defense

### 4.1 Required numerical semantics

Static numeric coefficients use `Complex64` in Rust, corresponding to NumPy `complex128`. Charge-commutator accumulation uses `f64`/`Complex64`. Aggregation order is deterministic, and a canonical term is removed only when its final coefficient compares exactly equal to complex zero.

The supported charge workload is ordinary particle number, spin, excitation number, and similar charges with small integer weights. These values are exactly representable in binary64 and must continue to agree with trusted references. Losing distinctions for weights beyond `2**53`, underflow after implausibly long products, or overflow after extreme repeated scaling is accepted and is not a Phase 9 defect.

### 4.2 Checks and machinery that must be removed

The implementation must remove the following from production paths:

- Python `Fraction` accumulation and exact float round-trip checks in charge construction or conservation analysis.
- `BigInt`, rational, decimal, log-domain, custom wide-accumulator, or Python-integer fallbacks intended to extend coefficient or charge range beyond `f64`.
- Helpers whose purpose is to reject non-finite results after every coefficient addition, scale, canonical aggregation, mapping contribution, commutator contribution, propagation update, or structured finite-basis transition.
- Repeated full-buffer `is_finite` scans of data already validated at a public input boundary.
- Branches that attempt to distinguish ordinary underflow-to-zero from an algebraic zero in extreme products.
- Special rescue paths introduced only for very high Boson powers, huge charge weights, hundreds of repeated scalings, or similarly non-representative numerical extremes.
- Tests whose sole contract is that one of these extreme internal overflows raises a particular exception.

Shared helpers such as `checked_added_coefficient` and `checked_scaled_coefficient` are deleted; their callers use ordinary addition or multiplication and fuse exact-zero removal into the final aggregate traversal without a second pass.

### 4.3 Checks that remain

The cleanup must not remove:

- Validation of public shapes, integer indices, Pauli codes, layout compatibility, local dimensions, sector consistency, and unsupported gates.
- Checked arithmetic used to size a vector, matrix, packed representation, FFI buffer, index space, or memory estimate.
- `max_bytes` preflights for major outputs and reusable workspaces under the existing best-effort policy.
- Cheap one-time rejection of explicitly supplied non-finite public coefficients, angles, tolerances, matrices, or states where accepting them would make an API call immediately meaningless.
- Algebraic checks for exact discrete phases, fermionic signs, canonical word validity, Hermiticity when explicitly requested, and conservation on the supported numeric range.
- A simple numerically stable expression when it is no more complex than the unstable expression. For example, multiplying Boson ladder square-root factors directly is ordinary `f64` arithmetic and need not be replaced by a deliberately worse factorial-then-square-root formulation. It must not grow a log-space or arbitrary-precision fallback.

The implementation audit must classify every existing overflow or finite-value check into one of these two categories. Removing a safety-related checked dimension calculation is not authorized; retaining internal coefficient overflow machinery requires evidence of a representative workload, not a hypothetical extreme.

## 5. Native operator algebra and construction

### 5.1 Direct construction and canonical results

Contiguous Pauli code and coefficient arrays construct `NativePauliOperatorHandle` in one native call. The constructor canonicalizes once and returns the handle directly. Native producers return a handle rather than arrays whenever the public contract does not request arrays.

Fermion, Boson, Hybrid/Qudit, and Majorana constructors likewise retain only canonical native storage after initial friendly-input normalization. Python must not retain `_native_data`, duplicate typed terms, or parallel structural arrays beside a handle.

### 5.2 Addition and promotion

Same-family addition uses a deterministic linear merge of already canonical sorted handles. Equal keys are aggregated once, exact zeros are removed, and unrelated structures are moved or cloned only as required by ordinary owned-handle storage.

Compatible different structured families with identical `OperatorSpace` promote to a native Hybrid handle. Required operand-order coverage includes Fermion/Hybrid, Boson/Hybrid, Qudit/Hybrid, and specialized/factory-produced objects. Incompatible layouts fail before operator-sized work begins.

### 5.3 Fused commutators

Fermion and Boson kernels feed both directional CAR/CCR expansions into one deterministic aggregate. Hybrid/Qudit reuses its domain-product machinery in a binary-operation mode. Majorana computes graded commutation parity before emitting a surviving canonical contribution. No family constructs complete `AB`, `BA`, scaled `BA`, and final-sum handles as four separate intermediates.

Python dispatches compatible native handles through one private native call. The composed Python recurrence is retained only as a test oracle outside the production package.

### 5.4 Equality, hashing, and Hermiticity

Large handle-backed equality and content hashing compare canonical native storage without typed-term materialization. Structured and Majorana Hermiticity checks run in Rust and return only a scalar result. Python performs only trivial identity and type/layout checks before the native call.

## 6. Handle-native mapping, conversion, embedding, and compilation

### 6.1 Mapping and conversion

`NativeMappingPlan` accepts and returns Pauli handles. Fermion Jordan-Wigner output chains directly into parity or Bravyi-Kitaev handle transforms. Hybrid-to-Pauli projection receives a compact axis-order descriptor and produces a Pauli handle. Fermion-to-Majorana and Majorana-to-Fermion are symmetric handle-to-handle conversions.

Term counts and memory estimates come from native handle metadata. Calling `.terms`, `materialize()`, or an array export only to calculate an estimate is forbidden.

### 6.2 Native embedding

`OperatorSpace.embed` passes a compact source-to-target axis mapping into Rust. Rust remaps canonical factors, applies fermionic permutation signs, validates domain/dimension compatibility, aggregates collisions, and returns the appropriate native handle. Python constructs the output facade but does not walk terms.

### 6.3 Terminal compilation

Pauli dense, COO, CSR, backend-plan, and native-MVP compilation consume a Pauli handle. Structured compilation consumes a structured handle plus compact ordered axis descriptors and Boson cutoffs; Rust performs term-to-local-operation lowering.

Final public NumPy arrays cross into Python only when they are the requested terminal result, and handle variants return the flat NumPy schema from Section 3.1 directly. Reusable MVP plans remain native handles. SciPy object construction remains Python-facing and consumes only final native arrays. A non-handle array entry point may remain only for a genuine external array input boundary, not as a workaround for a native handle that lacks a direct terminal method.

## 7. Grouping, sampling, and symmetry

### 7.1 Pauli grouping

QWC and general-commuting grouping consume a Pauli handle. Compatibility matrices or edge lists, deterministic coloring, group membership, QWC bases, and reconstruction masks are produced in Rust without exporting all Pauli terms to Python.

The public grouping result continues to expose its documented Python group metadata. Construction of that result is one terminal materialization; internal grouping algorithms must not make per-pair or per-term PyO3 calls.

### 7.2 Native sample reconstruction

Measurement reconstruction accepts the native grouping/reconstruction handle and batched sample arrays. Eigenvalue parity reduction, group accumulation, and coefficient combination run in Rust with the GIL released. Python returns the documented scalar or final result arrays.

### 7.3 Z2 symmetry and tapering

Z2 generator discovery, GF(2) nullspace work, Clifford/tapering transforms, and tapered-operator construction consume and return Pauli handles. Only small generator, sector, and qubit-selection metadata is materialized for the public result.

### 7.4 U1 restriction

U1 and generic additive-charge restriction consume Pauli or Structured handles directly. Python parallel-list serialization of every term is removed. Sector basis construction, transition compilation, fast-path detection, conservation analysis, and reusable eager/lazy plans remain native.

## 8. Native charge semantics

`analyze_charge()` performs one native deterministic selection-rule aggregation over the operator handle. Integer local charge deltas use the existing bounded native integer representation while they remain indices or small weights, but coefficient multiplication and commutator cancellation use `f64`/`Complex64`. No Python `Fraction` or exact binary-float decomposition participates.

Conservation means the final deterministic native aggregate is exactly zero in ordinary binary64 arithmetic. There is no hidden tolerance. The analysis metadata reports `method="native_float_selection_rules"`; the former `exact_integer_selection_rules` value is retired because it would make a false arbitrary-precision claim.

The fast Fermion-particle detector and termwise-conservation detector inspect native canonical structures. They do not call `_materialized_terms()`, `_arrays()`, or `.terms`. Common small-integer particle-number and spin cases require differential tests against the existing mathematical reference before the Python exact path is deleted.

## 9. Propagation, SPPS, and circuit tape

`PropagationCircuit`, SPPS, and `U1Circuit` accept Pauli operator handles for observables and initial symbolic operators. Propagation returns a Pauli handle when the public result is an operator; it does not return arrays that Python immediately sends back into Rust.

Each mutable Python circuit facade stores a monotonically increasing tape version. The first native terminal request compiles the Python gate descriptors into an immutable native `GateTape` handle and caches `(version, handle)`. Every method that appends, removes, replaces, or otherwise mutates a gate increments the version and clears the cache. Parameter values are execution inputs and do not invalidate a structurally compatible tape.

Tape compilation is one coarse call and releases the GIL for native validation and lowering. Execution receives the cached tape handle, compact parameter arrays, and operator handles. The public Python DSL, parameter-expression objects, TensorCircuit conversion, and backend-specific tensor execution stay in Python.

## 10. Python fallback and legacy-storage removal

Once a native path in this specification passes its independent differential tests, delete the corresponding production Python implementation in the same implementation slice. Do not leave `hasattr(_native, ...)`, size-threshold dispatch, environment switches, or exception-based fallback to the old path.

Required cleanup includes `_native_data`, redundant array-backed operator storage, private array-returning FFI used only as a native round trip, nested numeric handle read-back variants, split-list terminal outputs that are immediately recombined into NumPy arrays, Python exact-charge aggregation, Python termwise restriction serialization, Python compatibility-pair loops, Python operator embedding loops, and materialized Structured/Majorana Hermiticity traversal.

Independent Python or NumPy algorithms remain only under `tests/` as correctness oracles. Public explicit `.terms`, `to_dict()`, string export, dense/sparse output, and backend conversion are not fallbacks and remain supported.

The current Python tensor-product helpers remain because native tensor product is deferred. Their presence must not be used to retain unrelated general-purpose array-backed algebra infrastructure.

## 11. GIL contract

Every binding must separate Python extraction from pure Rust work explicitly. `Python::allow_threads` wraps construction, canonicalization, merge, multiplication, fused commutator, mapping, embedding, equality/hash traversal, Hermiticity, grouping, symmetry, charge analysis, restriction, tape compilation, propagation, sampling reconstruction, and terminal compilation whenever work is material in input size.

It is insufficient for a high-level call to release the GIL if it first clones or serializes the complete operator while holding it. Complete native preparation belongs inside the detached section. Python object creation and NumPy ownership transfer occur afterward with the GIL held.

Small metadata getters do not need to release the GIL. The code does not add thresholds for deciding whether a pure-Rust operator traversal is “large enough”; scalable operations use one simple detached implementation.

## 12. Public compatibility

Ordinary public operator class names, constructors, algebraic methods, result types, `.terms`, `to_dict()`, grouping APIs, symmetry APIs, restriction APIs, and circuit APIs remain unchanged unless this specification explicitly changes diagnostic metadata.

Private native handle classes and private FFI signatures may change freely. Materialization timing changes where required: properties that explicitly expose terms or arrays still materialize, while algebra, analysis, mapping, and compilation no longer do so incidentally.

The two deliberate semantic changes are ordinary IEEE behavior for internally produced numerical overflow/underflow and `AdditiveSymmetryAnalysis.method="native_float_selection_rules"`. They must be recorded in release notes. No compatibility shim recreates the retired exact or overflow-checking behavior.

## 13. Deferred and out of scope

The following are not Phase 9 acceptance requirements:

- Native tensor products.
- `Arc`, copy-on-write, persistent data structures, or structural sharing between operator results.
- Arbitrary-precision coefficient or charge arithmetic.
- Guaranteed detection or recovery of internal floating-point overflow, underflow, NaN, or cancellation beyond ordinary binary64 behavior.
- Moving TensorCircuit, JAX, SciPy, or backend tensor execution into Rust.
- Moving the entire circuit/program DSL into Rust.
- Hiding the cost of an explicitly requested `.terms`, `to_dict()`, dense, COO, CSR, or NumPy export.
- Making all operator families use one identical physical flat schema; each family may use a documented fixed-width or offset-based schema under the common no-nested-Python-object rule.
- A public API redesign unrelated to removing false exactness metadata.

## 14. Implementation slices

1. Establish direct array-to-handle construction, the family-specific flat NumPy read-back schemas, handle-accepting terminal methods, handle-native same/cross-family addition, fused commutators, direct conversions, terminal compilation, and complete GIL release for the current lazy-operator remediation.
2. Replace `Fraction` and checked coefficient-result machinery with deterministic ordinary floating-point aggregation. Remove obsolete extreme-overflow tests and update the documented coefficient and charge semantics.
3. Make grouping, compatibility construction, basis/mask generation, and sample reconstruction handle-native.
4. Make Z2 discovery/tapering and U1/generic charge restriction consume handles and return handles or native plans.
5. Add cached native `GateTape`; route propagation, SPPS, and U1Circuit observables and operator results through handles.
6. Implement native embedding and native Structured/Majorana Hermiticity, equality, and hashing.
7. Remove legacy storage, dead private FFI, production fallbacks, compatibility probes, and reference kernels made unreachable by slices 1–6.
8. Run the full correctness, residency, GIL, memory-safety, and release-benchmark gates and update architecture/status/release documentation.

Each slice must remove its superseded path rather than postponing cleanup to an indefinite final rewrite. Slices can be split into reviewable commits, but the architecture invariant is tested at each completed boundary.

## 15. Correctness and residency gates

Required tests include:

- Independent dense or dictionary differentials for construction, addition, multiplication, fused commutator/anticommutator, mapping, embedding, Hermiticity, grouping, tapering, charge restriction, and propagation on small systems.
- Operand-order and incompatible-layout tests for every cross-family Hybrid promotion.
- Canonical ordering, exact discrete phase, fermionic permutation sign, Boson/fermion normal ordering, qubit endianness, and exact-zero-after-aggregation regressions.
- Common particle-number, spin, and excitation-number charge differentials using small integer weights. Tests beyond binary64 integer exactness are removed or explicitly assert that no stronger guarantee exists.
- Handle-residency tests proving algebra, mapping, grouping, symmetry, restriction, propagation, and compilation do not populate `.terms` or invoke array/materialization exports.
- Read-back ABI tests proving every numeric handle materializer and dense/COO/CSR/MVP/backend-plan terminal returns flat NumPy arrays or consumes/returns a native handle, with explicit dtype, contiguity, shape metadata, deterministic ordering, and no nested Python sequence.
- Producer-chain tests proving propagation, tapering, mapping, embedding, and other native-to-native paths return handles directly instead of materializing and reparsing operator-sized data through Python.
- Tape-cache tests for reuse, mutation invalidation, parameter-only reuse, and independent circuit instances.
- Concurrent-observer tests showing scalable native work releases the GIL, without wall-time CI thresholds.
- Dimension/index/shape and `max_bytes` failure tests to ensure the numerical simplification did not remove memory-safety checks.

Self-comparison against the same native implementation is not independent evidence. Python reference kernels needed for differential tests live only in test code.

## 16. Performance gates

Release-mode benchmarks must measure complete Python-visible calls, including conversion and PyO3 cost. At minimum they cover array construction, numeric handle read-back, same- and cross-family addition, BCH-style fused commutators, parity/BK mapping, Hybrid-to-Pauli projection, embedding, grouping, sample reconstruction, Z2 tapering, charge analysis/restriction, tape compilation and cache reuse, propagation, and dense/sparse/native-MVP compilation.

Each benchmark records input term count, output term count, qubit or mode count, and whether explicit materialization is included. Benchmarks must prevent accidental canonical collapse from appearing as a speedup. Peak memory or retained bytes are recorded for paths that previously built complete Python intermediates.

The primary acceptance evidence is removal of operator-sized Python round trips and material intermediates plus representative end-to-end improvement. No wall-time CI gate or external benchmark service is added.

## 17. Completion criteria

Phase 9 is complete only when all required paths operate handle-to-handle or handle-to-terminal, every numeric handle read-back and matrix/vector terminal follows the flat NumPy or direct-handle ABI, scalable Rust calls release the GIL, ordinary charge cases pass independent differentials, numerical over-defense and `Fraction` machinery are removed, legacy Python storage/fallbacks are deleted, and the full repository quality gate passes after a release build.

Repository search must find no production use of `fractions.Fraction`, `_native_data`, exact-charge Python aggregation, nested numeric handle read-back for the Section 3.1 paths, or fallback dispatch for a completed native capability. Any remaining operator-sized `_arrays()`, `_materialized_terms()`, `.terms`, or native `materialize()` call must correspond to an explicit public materialization boundary or the deferred tensor-product helper and be documented at the call site.

There are no unresolved technical choices in this specification. Reopening native tensor product, structural sharing, arbitrary precision, or internal overflow recovery requires a separate owner decision backed by a representative workload or profile.
