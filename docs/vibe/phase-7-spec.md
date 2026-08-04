# Phase 7 Spec: structured Hamiltonian algebra and compilation

Status: frozen implementation contract; the first vertical slice is implemented and under acceptance review. The owner approved the Python construction direction on 2026-08-03 and approved the remaining algebra, finite-basis, memory, compilation, and delivery decisions later the same day. The remaining acceptance work is tracked in `docs/vibe/implementation-status.md`.

> API note: this historical specification predates the breaking Phase 8 API contract; current public names and signatures are defined in [`phase-8-api-coherence-spec.md`](phase-8-api-coherence-spec.md).

## 1. Purpose

Phase 7 expands TenCirPauli from a qubit-Pauli Hamiltonian utility into a structured Hamiltonian algebra and compilation layer for the TensorCircuit ecosystem. It adds exact structural manipulation for fermionic, bosonic, hybrid, and finite-dimensional qudit/Weyl operators, followed by bounded dense, sparse, or matrix-free compilation where a finite basis has been selected.

The phase does not provide a large library of model-specific Hamiltonian factories. It provides the lower-level infrastructure from which users or a later Python models layer can construct Hubbard, Holstein, electron-phonon, spin-boson, Bose-Hubbard, Potts, and related Hamiltonians.

TensorCircuit remains responsible for circuit construction, backend tensors, sampling, JIT, automatic differentiation, and accelerator execution. TenCirPauli owns canonical sparse operator algebra, deterministic mapping, finite-basis compilation, measurement-ready Pauli structure, and CPU structured matrix-vector-product kernels.

## 2. Frozen decisions

1. The public construction interface uses `OperatorSpace`, readable domain factories, overloaded `+`, `-`, and `*`, explicit `tensor_product()`, low-level `from_terms()`, and `OperatorBuilder`.
2. All operator families use the same `compile(target=...)` target vocabulary and the same target-dependent Python result types.
3. All public Phase 7 memory guards use the existing `DEFAULT_MAX_BYTES = 16 * 1024**3`. `max_bytes=None` explicitly disables the byte guard while preserving checked arithmetic and dimension validation.
4. Phase 7 coefficients are finite IEEE complex128-compatible values. Structural phases and fermionic signs are discrete, but arbitrary coefficient arithmetic is not claimed to be symbolic or algebraically exact.
5. The first fermion mapping is Jordan-Wigner only. Parity and Bravyi-Kitaev mappings are deferred.
6. `BosonOperator` is an infinite-Fock-space symbolic CCR operator. Finite compilation applies the projection `P O P` after symbolic canonicalization, uses an open boundary, and does not introduce a second public finite-boson algebra.
7. Qudit words use the direct `X^a Z^b` convention. Weyl multiplication phases are represented as modular integer exponents before conversion to complex128.
8. Mixed-dimension boson and hybrid execution is native-only in the first release. TensorCircuit backend MVP is required for Pauli-compatible and uniform-dimension qudit plans, not for mixed local dimensions.
9. `OperatorSpace` has a small public constructor and an automatically generated immutable ordered layout. Layout fingerprints, basis descriptors, and mapping metadata are internal or read-only outputs, not required user inputs.
10. Default public metadata is compact. Full per-source provenance is not a Phase 7 public requirement; independent references and diagnostic tests may retain it internally.
11. Expansion protection uses cheap checked upper bounds followed by a running best-effort byte estimate. Phase 7 does not add a public `AlgebraLimits` configuration object or a small global term-count cutoff.

## 3. Design principles

1. **Correct structural algebra comes first.** Normal ordering, canonical ordering, CAR/CCR contractions, identity elimination, fermionic nilpotency, modular Weyl arithmetic, duplicate aggregation, and exact-zero removal must be correct before optimization.
2. **Approximation and finite representation are explicit.** Boson occupation cutoffs, fermion mappings, symmetry sectors, and Pauli-weight projections are visible in method arguments or result metadata. A boson cutoff is not a coefficient cutoff.
3. **Separate domain algebras, common user surface.** Pauli, fermion, boson, hybrid, and qudit implementations may use different Rust data structures and algorithms. They share Python construction conventions, deterministic outputs, compilation target names, result protocols, error behavior, and memory policy.
4. **Optimize common vertical workflows.** The implementation should first make one- and two-body fermion Hamiltonians, low-degree boson Hamiltonians, Holstein/spin-boson hybrids, and uniform-dimension Weyl chains correct and fast. It must not build a speculative universal public algebra trait.
5. **Keep FFI coarse.** Construction, canonicalization, mapping, compilation, and plan application cross PyO3 in batches rather than once per factor, term, transition, or matrix entry.
6. **Guard finite targets.** Checked dimensions, indices, output sizes, major workspaces, and expansion buffers must be validated before obviously excessive allocations. `max_bytes` remains a best-effort guard, not an exact peak-RSS promise.
7. **Keep TensorCircuit at the Python boundary.** Pure Rust code must not depend on PyO3, NumPy, TensorCircuit, or backend tensor types.

## 4. Public object model

The public domain objects are:

- `PauliWord` and `PauliOperator`: the existing phase-free qubit-Pauli word and deterministic Pauli sum.
- `FermionWord` and `FermionOperator`: canonical fermionic monomials and deterministic sums governed by CAR.
- `BosonWord` and `BosonOperator`: canonical normal-ordered bosonic monomials and deterministic sums governed by CCR.
- `QuditWeylWord` and `QuditWeylOperator`: direct-convention uniform-dimension Weyl words and deterministic sums.
- `HybridOperator`: deterministic sums whose terms may contain compatible fermion, boson, Pauli, and qudit factors.
- `OperatorSpace`: an immutable logical subsystem layout and the preferred factory for hybrid construction.
- `OperatorBuilder`: a mutable, single-owner batched input builder whose `finish()` method performs one canonicalization pass and returns an immutable operator.
- `NativeMVPPlan` and `BackendMVPPlan`: reusable target plans with a common Python execution surface and domain-specific private implementations.

All operator values are immutable after construction. Public operators always contain canonical terms; there is no required public `canonicalize()` mutator or second noncanonical operator state.

## 5. Numeric, aggregation, and error contract

### 5.1 Coefficients

Python accepts real or complex scalar values that can be safely converted to one finite complex128 value. Booleans, arrays, backend tracers, NaN, and infinity are rejected as coefficients. The Rust core stores the existing two-`f64` complex representation.

Equal canonical keys are aggregated in a deterministic contribution order. A static operator removes a term only when both aggregated IEEE components compare equal to zero. No implicit tolerance or magnitude cutoff is applied. Signed zero is normalized by the ordinary exact-zero removal path. Overflow to a non-finite coefficient is an error.

Stored operator terms are sorted by their canonical public tuple encodings. Fermion keys compare `(creation_modes, annihilation_modes)`, boson keys compare their ascending `(mode, creation_power, annihilation_power)` blocks, qudit keys compare their ascending `(site, a, b)` triples, and hybrid keys compare the ordered tuple of present domain keys. The empty identity key sorts before non-empty keys. Hash-map iteration order must never reach a public result.

“Exact algebra” in this specification means exact structural CAR/CCR rewriting, exact fermionic signs, exact modular Weyl phase exponents, and no hidden numerical pruning. It does not mean arbitrary complex128 sums are associative or symbolically exact.

### 5.2 Stable failures

Invalid modes, dimensions, factor tokens, shapes, incompatible spaces, missing boson cutoffs, unsupported targets, and non-finite coefficients fail explicitly. Checked integer or dimension overflow becomes `OverflowError`; a rejected best-effort allocation or expansion becomes `MemoryError`; invalid values and incompatible layouts become `ValueError`; a recognized target intentionally unavailable for a domain becomes `NotImplementedError`. Rust core failures use typed errors and must not panic at the Python boundary.

An operation that fails must not publish a partially constructed operator or plan. A failed `OperatorBuilder.finish()` may consume or retain the builder at the implementation's convenience, but this behavior must be documented and tested; no partially canonical operator is returned.

## 6. Canonical domain schemas

### 6.1 Fermions

A raw fermion factor is `(mode, action)`, where `mode` is a non-negative integer below `n_modes` and `action` is exactly `"create"` or `"annihilate"`.

A canonical `FermionWord` contains all creation factors first with strictly increasing mode indices, followed by all annihilation factors with strictly decreasing mode indices. The empty word is identity. A repeated creation or repeated annihilation on the same mode is zero by nilpotency and therefore does not form a stored word. The ordering is chosen so that taking the adjoint of a canonical word and swapping creation with annihilation produces canonical factor order without an additional reordering sign.

Raw factor sequences are accepted by `FermionOperator.from_terms()` and by `OperatorBuilder`; they may normalize to zero, one word, or a sum of words. A public `FermionWord` constructor accepts canonical data only and rejects a factor sequence that would require expansion. Multiplication of two fermion words may expand and therefore returns a `FermionOperator`, not a single word.

The algebra obeys:

```text
{a_p, a_q} = 0
{a_p†, a_q†} = 0
{a_p, a_q†} = δ_pq
```

Products, adjoints, commutators, anticommutators, number operators, and Hermiticity validation use this canonical order and aggregate generated identity and contraction terms before exact-zero removal.

### 6.2 Bosons

A raw boson factor uses the same `(mode, action)` token shape. A canonical `BosonWord` is stored as ascending non-identity mode blocks `(mode, creation_power, annihilation_power)`, representing:

```text
∏_mode (b_mode† ** creation_power) (b_mode ** annihilation_power)
```

Powers are non-negative checked integers; a block with both powers zero is omitted. The empty word is identity. Different modes commute. Raw factor sequences are accepted by `BosonOperator.from_terms()` and `OperatorBuilder`, while a `BosonWord` constructor accepts canonical power blocks only. Multiplication of words may generate contraction sums and therefore returns a `BosonOperator`.

The algebra obeys:

```text
[b_p, b_q] = 0
[b_p†, b_q†] = 0
[b_p, b_q†] = δ_pq
```

The implementation uses exact integer combinatorial factors for normal ordering and converts them to complex128 only when multiplying the user coefficient. For one mode it must agree with:

```text
b^m (b†)^n = Σ_{k=0}^{min(m,n)} k! C(m,k) C(n,k) (b†)^(n-k) b^(m-k)
```

### 6.3 Qudit Weyl words

Phase 7 public qudit sites have a uniform local dimension `d` satisfying `3 <= d <= u32::MAX`. Numeric targets remain subject to dimension and memory guards. The TensorCircuit `QuditCircuit` adapter additionally accepts only the dimension range supported by the installed TensorCircuit version; the initial compatibility target is `3 <= d <= 36`.

A `QuditWeylWord` stores non-identity site triples `(site, a, b)` in strictly increasing site order. Exponents are canonical residues in `0 <= a,b < d`, and `(a,b)=(0,0)` is omitted. It denotes the direct-convention product `X^a Z^b` with:

```text
X|j> = |j+1 mod d>
Z|j> = ω^j |j>
ω = exp(2π i / d)
```

For one site, multiplication and adjoint are frozen as:

```text
(X^a Z^b)(X^c Z^e) = ω^(b*c) X^(a+c) Z^(b+e)
(X^a Z^b)† = ω^(a*b) X^(-a) Z^(-b)
```

All exponents are reduced modulo `d`. A word product returns a canonical word plus a modular phase exponent; multisite phase exponents add modulo `d`. Two words commute exactly when the summed exponent `Σ_i (b_i*c_i - e_i*a_i)` is zero modulo `d`.

The modular phase exponent remains an integer during word algebra and provenance. When a word product enters a numeric operator, `ω^k` is converted once to complex128 and absorbed into the coefficient. Centered Weyl conventions, mixed qudit dimensions, qudit stabilizer algorithms, and qudit propagation are not part of Phase 7.

## 7. `OperatorSpace`, layout, and hybrid semantics

### 7.1 Public constructor

The Phase 7 constructor is:

```python
space = tcp.OperatorSpace(
    fermions=4,
    bosons=2,
    qubits=2,
    qudits=(3, 3),
)
```

`fermions`, `bosons`, and `qubits` are non-negative integers and reject booleans. `qudits` is a tuple of dimensions. In Phase 7 every qudit dimension in one space must be equal and at least three; mixed qudit dimensions are rejected. Empty domains are allowed, and the completely empty space represents a scalar operator space.

The constructor automatically builds an ordered immutable axis descriptor list. Its default order is increasing fermion mode, increasing boson mode, increasing qubit index, then increasing qudit site. This descriptor list, not Python object identity, defines compatibility. Separately constructed spaces with equal descriptors are compatible.

The detailed descriptors and deterministic layout fingerprint are generated automatically and may be exposed as read-only diagnostics. Users do not supply fingerprints, namespaces, basis strides, or serialization identifiers in ordinary construction.

### 7.2 Factories

The space exposes:

```python
a = space.fermion.annihilate
adag = space.fermion.create
b = space.boson.annihilate
bdag = space.boson.create
X = space.qubit.x
Y = space.qubit.y
Z = space.qubit.z
W = space.qudit.weyl
```

Each factory returns a one-term immutable operator carrying the space. `W(site, a, b)` validates and reduces exponents modulo the common qudit dimension. Ordinary hybrid construction through these factories requires no explicit embedding metadata.

### 7.3 Embedding and tensor products

Low-level single-domain operators created with `from_terms()` carry their own single-domain space. They must be explicitly embedded when the target layout is ambiguous. `space.embed(operator, ...)` accepts an explicit source-to-target mode or site map; a convenience identity embedding is allowed only when the domain counts and ordered identities match exactly.

`left_operator.tensor_product(right_operator)` returns an operator on a new space whose ordered axis descriptors are the left descriptors followed by the right descriptors. Domain-local indices from the right operand are deterministically offset. Non-fermion factors use the ordinary tensor product. Fermionic terms use the graded tensor product induced by the combined global fermion-mode order, with signs computed term by term for odd-parity factors. Under Jordan-Wigner compilation, an odd right-fermion term carries the parity operator of all left fermion modes; therefore a fermionic graded tensor product is not in general the plain matrix Kronecker product. It reduces to the ordinary Kronecker product when the right fermion factor is parity-even or when no fermion domain is present. Boson, qubit, and qudit factors commute with fermion factors and with factors from other non-fermion domains. If both operands contain qudits with different local dimensions, Phase 7 rejects the tensor product because the resulting mixed-qudit Weyl algebra is deferred.

### 7.4 Hybrid canonical terms

A canonical hybrid term contains at most one canonical word per present domain plus one coefficient. Cross-domain multiplication first combines the domain factors and then canonicalizes the fermion or boson factors that may expand. Equal complete hybrid keys aggregate deterministically. Addition of compatible same-layout domain operators promotes missing domains to identity and returns a `HybridOperator` when more than one domain is present.

## 8. Python construction and algebra API

### 8.1 Readable expressions

The required overloads are:

```python
H = 2.0 * operator_a + operator_b - operator_c
H = -operator_a
H = operator_a * operator_b
```

The rules are:

- `+` and `-` form exact structural sums and require compatible spaces;
- unary `-` negates all coefficients;
- scalar `*` scales coefficients on either side;
- same-domain operator `*` performs the domain algebra and may expand;
- compatible cross-domain `*` forms a hybrid product;
- incompatible layouts fail rather than guessing an embedding;
- `@` is not overloaded;
- `tensor_product()` is the only operation that combines independent spaces;
- `+=` and `-=` are ordinary rebinding to new immutable values and do not mutate shared plans.

For example:

```python
t = 1.0
U = 2.0
omega = 0.5
g = 0.3

n0 = adag(0) * a(0)
n1 = adag(1) * a(1)

H = (
    -t * (adag(0) * a(1) + adag(1) * a(0))
    + U * n0 * n1
    + omega * bdag(0) * b(0)
    + g * n0 * (b(0) + bdag(0))
    + 0.2 * Z(0)
)
```

The result is a canonical immutable `HybridOperator`.

### 8.2 Low-level construction

The stable raw fermion input shape is:

```python
f = tcp.FermionOperator.from_terms(
    n_modes=4,
    terms=[
        (((0, "create"), (1, "annihilate")), 1.0),
    ],
    max_bytes=tcp.DEFAULT_MAX_BYTES,
)
```

`BosonOperator.from_terms()` uses the same raw factor token shape. `QuditWeylOperator.from_terms()` accepts site triples `(site, a, b)`. These tuple forms are deterministic bulk/data-exchange inputs, not the preferred notation for formulas. Persistent serialized artifacts, if added, must carry a schema version, domain, space layout, ordering, and complex coefficients as explicit real/imaginary pairs; compiled-plan serialization is not required in Phase 7.

`OperatorBuilder` provides a single batched path:

```python
builder = space.builder()
builder.add_product(
    coefficient=1.0,
    fermions=((0, "create"), (1, "annihilate")),
)
H = builder.finish(max_bytes=tcp.DEFAULT_MAX_BYTES)
```

The builder accepts domain-specific raw factors, hybrid products, and optional diagnostic source indices. It sends flattened buffers through one coarse PyO3 call at `finish()` and avoids constructing one Python operator per input term.

### 8.3 Named algebraic methods

All operator classes expose compatible named methods:

```python
operator.add(other, *, max_bytes=DEFAULT_MAX_BYTES)
operator.scale(coefficient, *, max_bytes=DEFAULT_MAX_BYTES)
operator.multiply(other, *, max_bytes=DEFAULT_MAX_BYTES)
operator.commutator(other, *, max_bytes=DEFAULT_MAX_BYTES)
operator.anticommutator(other, *, max_bytes=DEFAULT_MAX_BYTES)
operator.adjoint(*, max_bytes=DEFAULT_MAX_BYTES)
operator.is_hermitian(tolerance=0.0)
operator.tensor_product(other, *, max_bytes=DEFAULT_MAX_BYTES)
```

Arithmetic overloads use `DEFAULT_MAX_BYTES`. The named methods allow `max_bytes=None` or a caller-selected non-negative limit without adding configuration objects to ordinary expressions. `is_hermitian()` uses exact coefficient comparison by default; a non-negative finite tolerance is explicit and affects validation only.

## 9. Expansion and memory policy

Every Phase 7 public `max_bytes` defaults to 16 GiB and accepts `None` as unbounded. The guard covers cheaply estimated major outputs and workspaces, including term buffers, mapping expansion buffers, primary aggregation storage, matrix arrays, and native plan storage. It does not promise exact accounting for allocator overhead, Python objects, temporary FFI conversion buffers, backend copies, or peak RSS.

Expansion protection has two stages:

1. Perform checked cheap bounds before major allocation, such as `left_term_count * right_term_count`, `2**ladder_count` for one Jordan-Wigner monomial, and the product of per-mode bosonic contraction counts.
2. During generation, maintain a running estimate for the actual intermediate term buffers and primary aggregation storage. Stop with `MemoryError` before the estimate exceeds `max_bytes`.

The implementation must not perform a complete dry-run solely to count the exact final number of terms. It must not expose a low default `max_terms` that rejects ordinary aggregation-heavy workloads. The counter and guard belong inside the generation hot loop and must be included in release benchmarks.

Common one- and two-body fermion terms are the optimization priority: a monomial with `k` ladder factors has a Jordan-Wigner raw upper bound of `2**k`, so the usual `k <= 4` cases generate at most sixteen raw Pauli contributions before aggregation. High-body or adversarial expressions may still grow combinatorially and are allowed to fail the explicit memory guard.

## 10. Fermion mapping contract

### 10.1 Jordan-Wigner convention

Phase 7 implements only `mapping="jordan_wigner"`. Mode `p` maps to the qubit axis replacing fermion mode `p`, with computational `|0>` meaning unoccupied, `|1>` meaning occupied, and lower fermion modes supplying the parity string:

```text
a_p  = (∏_{j=0}^{p-1} Z_j) (X_p + i Y_p) / 2
a_p† = (∏_{j=0}^{p-1} Z_j) (X_p - i Y_p) / 2
```

The Pauli convention and matrix ordering are the existing TenCirPauli conventions. For a pure fermion space, mode zero becomes qubit zero and therefore the most-significant matrix axis. In a hybrid layout, each fermion axis is replaced in place by its mapped qubit axis; unrelated boson, physical-qubit, and qudit axes retain their relative positions.

`operator.map_fermions(mapping="jordan_wigner", max_bytes=...)` maps all fermion factors. A pure fermion operator, or a hybrid containing only fermions and qubits after mapping, becomes a `PauliOperator` with qubits in the resulting axis order. If boson or qudit factors remain, the result remains a `HybridOperator` with Pauli factors replacing the fermion factors. Its Pauli sector numbers all mapped-fermion and original-qubit axes in their global descriptor order and records the corresponding global axis positions; Jordan-Wigner parity strings act only on lower mapped-fermion axes, not on unrelated physical-qubit axes.

`FermionOperator.compile()` and `HybridOperator.compile()` default to Jordan-Wigner when fermion factors remain and record the mapping label in plan metadata. Parity and Bravyi-Kitaev strings are rejected as unsupported in Phase 7 rather than accepted as aliases.

## 11. Boson finite-basis contract

### 11.1 Symbolic operator and cutoff

`BosonOperator` always denotes an operator on the infinite Fock space and is canonicalized using the exact CCR identities before finite compilation. It does not store an occupation cutoff. Every finite target for a space containing boson modes requires `boson_cutoffs={mode: nmax}` with exactly one entry for every boson mode in the space. `nmax` is an inclusive non-negative integer, so the local dimension is `nmax + 1`.

The same symbolic operator may be compiled repeatedly at different cutoffs. Phase 7 does not expose `FiniteBosonOperator`, `BosonOperator.truncate()`, or a second cutoff-dependent boson algebra.

### 11.2 Projection semantics

Finite compilation is defined as `P O P`, where `O` is the already canonical infinite-space operator and `P` projects every boson mode onto occupations `0..nmax`. For a canonical local monomial `(b†)^p b^q`, the compiler applies the infinite-space ladder amplitude directly and retains the transition only when both source and destination occupations are inside the retained basis.

Consequently, an isolated raising operator obeys:

```text
P b† P |nmax> = 0
```

There is no wraparound and no compile-time leakage error. The plan records `boson_boundary="projected_fock"` and the inclusive cutoffs.

Projection is linear but is not an algebra homomorphism. In general:

```text
compile(A * B) != compile(A) @ compile(B)
```

For local dimension `d = nmax + 1`, separately truncated ladder matrices obey `[b_d, b_d†] = I - d |d-1><d-1|`, while symbolic CCR canonicalization still gives `b b† = b† b + 1` before projection. This distinction is required behavior and must have an explicit regression test.

If a future workload requires algebra after truncation, it should use a separately approved generic finite local transition operator rather than silently changing `BosonOperator` semantics.

## 12. Finite basis and compilation targets

### 12.1 Basis ordering

After fermion mapping and boson cutoff selection, every finite plan has an ordered `local_dimensions` tuple following the `OperatorSpace` axis descriptors. Axis zero is the most-significant digit. For digits `s_i` and local dimensions `d_i`, the flat index is:

```text
index = Σ_i s_i ∏_{j>i} d_j
```

Mapped fermion and qubit digits are `0` or `1`, boson digits are occupations, and qudit digits are computational-basis levels. Dense, COO, CSR, native MVP, backend MVP, TensorCircuit adapters, and independent references use this same ordering.

### 12.2 Unified target API

All operators use:

```python
operator.compile(target, *, max_bytes=DEFAULT_MAX_BYTES, ...)
```

The domain-specific signatures add only representation choices that are meaningful for that domain:

```python
pauli.compile(target, *, max_bytes=DEFAULT_MAX_BYTES)
fermion.compile(
    target,
    *,
    mapping="jordan_wigner",
    max_bytes=DEFAULT_MAX_BYTES,
)
boson.compile(
    target,
    *,
    boson_cutoffs={0: 7, 1: 7},
    max_bytes=DEFAULT_MAX_BYTES,
)
hybrid.compile(
    target,
    *,
    fermion_mapping="jordan_wigner",
    boson_cutoffs={0: 7},
    max_bytes=DEFAULT_MAX_BYTES,
)
qudit.compile(target, *, max_bytes=DEFAULT_MAX_BYTES)
```

`boson_cutoffs` is required whenever the operator space contains boson modes, including modes on which the current operator acts as identity. The default mapping has no effect when the space has no fermion modes, but any explicitly supplied unsupported mapping string still fails. Implementations use typed overloads or domain-specific signatures rather than accepting and silently ignoring unrelated keyword arguments.

The target-dependent return contract is:

| Target | Result | Required behavior |
| --- | --- | --- |
| `"dense"` | `numpy.ndarray[complex128]` | Shape `(D, D)`; fail before major allocation when the checked estimate exceeds `max_bytes`. |
| `"coo"` | existing `COOMatrix` | Deterministic row-major entries with duplicates aggregated and exact zeros removed. |
| `"csr"` | existing `CSRMatrix` | Deterministic CSR equivalent to COO and dense. |
| `"native_mvp"` | `NativeMVPPlan` | Reusable Rust-native plan; callable and exposes `apply()`. It must not be returned as an anonymous lambda. |
| `"backend_mvp"` | `BackendMVPPlan` | Versioned pure-array backend plan for supported finite layouts. |

`NativeMVPPlan` provides at least `dimension`, `local_dimensions`, `term_count`, `estimated_bytes`, `basis_ordering`, `strategy`, `apply(state, max_bytes=...)`, and `__call__(state)`. Existing Pauli fields such as `nqubits` remain available where meaningful. The private native handle and kernel strategy may differ by domain.

`BackendMVPPlan` provides the same dimension/layout metadata plus versioned structural arrays, required backend operations, and default coefficients. The existing optional backend coefficient override remains in canonical plan-term order. Phase 7 does not add a universal parameterized-operator or `bind()` abstraction.

Materialized dense, COO, and CSR outputs retain their existing lightweight result forms. They do not need a wrapper solely to carry compilation metadata; the compile arguments and shape define the finite representation. Reusable plans carry the full compact metadata needed for repeated execution.

### 12.3 Domain availability

The required Phase 7 target matrix is:

| Domain after required mapping | dense | COO | CSR | native MVP | backend MVP |
| --- | --- | --- | --- | --- | --- |
| Pauli | required, existing | required, existing | required, existing | required, existing | required, existing |
| Fermion via Jordan-Wigner | required via Pauli | required via Pauli | required via Pauli | required via Pauli | required via Pauli |
| Boson with cutoffs | required | required | required | required | deferred |
| Hybrid with mapped fermions and cutoffs | required when guarded | required when guarded | required when guarded | required | deferred for mixed local dimensions |
| Uniform-dimension qudit Weyl | required | required | required | required | required |

Requesting `backend_mvp` for a finite boson or mixed-dimension hybrid plan raises `NotImplementedError` with a direct explanation. No silent native or NumPy fallback is allowed.

## 13. TensorCircuit integration

TensorCircuit integration remains in `python/tencirpauli/integrations/tensorcircuit.py`.

- Pauli and Jordan-Wigner-mapped pure fermion operators continue to use the existing TensorCircuit backend MVP path.
- Uniform-dimension `QuditWeylOperator` plans provide a pure-array backend executor and differential tests against the independent dense target for supported dimensions and basis ordering. The executor uses TensorCircuit NumPy/JAX backend operations directly; conversion to `QuditCircuit` is outside the Phase 7 acceptance contract.
- Boson and mixed-dimension hybrid plans are native-only in Phase 7. A state with local dimensions such as `(2, 8, 3)` is handled as a flat mixed-radix vector by the native plan; Phase 7 does not promise a TensorCircuit circuit or backend executor for that layout.
- Phase 7 does not provide a `QuditCircuit` conversion for sparse Weyl Hamiltonians and does not accept arbitrary circuit unitaries as sparse Weyl Hamiltonians.

Phase 7 does not add a generic TensorCircuit `Hamiltonian` class, modify TensorCircuit source, introduce a JAX custom call, or promise backend AD for native plans.

## 14. Metadata and provenance

Reusable plans expose compact immutable metadata:

- target and schema version;
- dimension, ordered local dimensions, and basis ordering;
- canonical source term count and compiled transition or plan-term count;
- estimated plan bytes and selected native strategy;
- Jordan-Wigner mapping label when applied;
- boson cutoffs and `projected_fock` boundary label when present;
- qudit dimension and direct-Weyl convention when present.

Ordinary operators and plans do not retain an unbounded many-to-many source lineage through every multiplication and contraction. Full source-term provenance is not part of the stable Phase 7 public API. Deterministic internal/reference paths must still be able to validate input-to-canonical mappings, signs or modular phase multipliers, expansion counts, and mapping correctness without placing that storage on optimized user paths.

## 15. Correctness requirements

Independent small-system references must not call the implementation under test and must cover:

1. Fermionic CAR identities, the frozen canonical order, nilpotency, contraction-generated identities, adjoints, products, commutators, and Hermiticity.
2. Jordan-Wigner mapping against explicit Fock-space matrices, including mode-zero/MSB ordering, complex phases, number operators, hopping terms, and four-factor interactions.
3. Bosonic CCR normal ordering and the closed-form contraction coefficients for several powers and modes.
4. `P O P` finite-boson matrices, open-boundary raising, the top-state `b b†` distinction, multiple cutoffs for the same symbolic operator, and no cyclic wraparound.
5. Hybrid fermion-boson/qubit products, graded signs, in-place fermion-axis replacement, mixed-radix indexing, and Holstein or spin-boson fixtures.
6. Direct-Weyl multiplication, modular phases, adjoints, commutation, Hermiticity, and dense matrices for dimensions 3, 4, 5, and at least one additional composite dimension.
7. `OperatorSpace` structural compatibility across separately constructed values, explicit embedding, incompatible-layout failures, ordinary Kronecker behavior for non-fermion and parity-even-right tensor products, and graded dense references for odd right-fermion tensor products.
8. Dense, COO, CSR, and native MVP equivalence for every finite domain; backend MVP equivalence for Pauli-compatible and uniform-qudit domains.
9. Deterministic canonical ordering and bitwise-stable structural outputs across repeated runs and supported thread counts.
10. Checked overflow, invalid cutoffs, missing cutoff entries, non-finite coefficients, unsupported targets, expansion guard failures, the 16 GiB default, explicit lower limits, and `max_bytes=None` without allocating an unsafe test workload.

Property tests must cover algebraic identities where floating coefficients do not make exact comparison inappropriate. Dense differential tests use documented complex128 tolerances. Every critical boundary convention must also have a deterministic regression vector.

## 16. Performance requirements

Release benchmarks separate Python input conversion, symbolic construction, canonicalization, mapping, finite-plan construction, first execution, steady execution, output materialization, and memory. Debug timings are not evidence.

Required representative workloads include:

- sparse fermionic one- and two-body operators and aggregation-heavy duplicates;
- Jordan-Wigner mapping for hopping, density-density, and molecular-style terms with raw and canonical term counts;
- low-degree single- and multi-mode boson expressions at several cutoffs;
- Holstein or spin-boson mixed native MVP scaling;
- uniform-dimension Weyl chains for several `d` values;
- guarded dense/COO/CSR construction at small finite dimensions;
- native plan first and steady apply, including caller-visible conversion and PyO3 overhead;
- expansion-heavy cases that exercise the running byte guard without making the benchmark itself memory-dangerous.

Benchmarks record runtime, throughput, input terms, canonical terms, generated contributions, finite dimension, nonzeros or transitions, plan/output bytes, thread count, and numerical error. No predetermined speedup multiplier or wall-time CI gate is imposed. Optimization follows profiles and preserves the independent correctness gate.

Hot implementations should use compact contiguous term storage, borrowed slices, reusable scratch buffers, checked preallocation, coarse FFI, and deterministic aggregation. Shared internal infrastructure should be extracted only after at least two domain slices demonstrate the same requirement; the phase must not begin with a public universal algebra trait or a generic boxed factor graph.

## 17. Non-goals

- No automatic coefficient-magnitude pruning or hidden numerical tolerance.
- No public finite-boson algebra, boson `truncate()` method, or cyclic interpretation of a truncated boson.
- No parity or Bravyi-Kitaev fermion mapping.
- No mixed-qudit-dimension Weyl algebra in one operator space.
- No boson or mixed-dimension hybrid backend MVP in Phase 7.
- No generic eigensolver, time-evolution engine, ODE solver, or automatic Trotterization.
- No arbitrary Python callbacks in Rust compilation or execution.
- No new native/general symbolic coefficient AD, symbolic coefficient expression system, or universal plan-binding API; supported backend-plan coefficient overrides retain the backend differentiation behavior they already provide.
- No arbitrary TensorCircuit circuit-to-Weyl decomposition.
- No mandatory high-level model library in the Rust core.
- No promise that dense or CSR materialization scales beyond the guarded finite dimension.
- No public user-implementable Rust trait unifying all domain algebras.

## 18. Implementation slices

### P0: references, Pauli API alignment, and frozen schemas

Add independent Python reference helpers for CAR, CCR projection, mixed-radix indexing, and direct Weyl matrices. Freeze test vectors for every convention in Sections 6, 10, 11, and 12. Align existing `PauliOperator.compile(target="native_mvp")` to return its callable `NativeMVPPlan` rather than a lambda, extend plan metadata compatibly, and add the shared 16 GiB/max-bytes tests. Do not introduce a universal algebra layer.

Acceptance gate: existing Pauli tests remain green; the target return types, callable plan compatibility, memory behavior, and new independent references are tested before a new domain is implemented.

### P1: fermion-to-Pauli vertical slice

Implement canonical `FermionWord`, `FermionOperator`, raw normal ordering, immutable arithmetic, low-level batched construction, Jordan-Wigner mapping, and compilation through the existing Pauli targets. Optimize common one- and two-body terms and retain mapping benchmarks.

Acceptance gate: CAR/property tests, explicit Fock-matrix differentials, all mapped Pauli target equivalence, deterministic ordering, expansion failures, Python typing, and release construction/mapping benchmarks pass.

### P2: symbolic boson and finite native slice

Implement canonical power-block `BosonWord`, `BosonOperator`, CCR normal ordering, explicit inclusive cutoffs, direct `P O P` transition generation, dense/COO/CSR, and reusable native MVP. Do not add `FiniteBosonOperator` or backend MVP.

Acceptance gate: closed-form contraction references, top-boundary regressions, multiple-cutoff reuse, finite target equivalence, memory failures, and release plan/apply benchmarks pass.

### P3: hybrid native slice

Implement `OperatorSpace`, domain factories, explicit embedding, graded tensor products, hybrid canonical terms, in-place Jordan-Wigner axis replacement, mixed-radix dense/COO/CSR, and native MVP for mapped-fermion/boson/qubit systems. Add `OperatorBuilder` once the actual flattened schemas of P1 and P2 are available.

Acceptance gate: layout/Kronecker tests, graded-sign references, Holstein or spin-boson differentials, native target equivalence, coarse-FFI construction, and mixed-dimension performance benchmarks pass.

### P4: direct-Weyl vertical slice

Implement uniform-dimension `QuditWeylWord`, `QuditWeylOperator`, modular phase exponents, arithmetic, dense/COO/CSR/native MVP, versioned backend MVP, and TensorCircuit-compatible ordering tests. Reuse only shared target and error components already justified by P1-P3.

Acceptance gate: modular property tests, dimensions 3/4/5/composite dense references, all target equivalence, TensorCircuit NumPy/JAX smoke tests where available, and release qudit benchmarks pass.

### P5: public delivery and handoff

Complete exports, strict type hints, docstrings, examples, compatibility notes, target support tables, benchmark registration, and release notes. Run formatting, linting, strict mypy, release Rust tests, Clippy, maturin development install, Python tests, examples, and benchmark smoke through the repository workflow.

Acceptance gate: `python scripts/check.py --benchmark smoke` passes in the project environment, representative release benchmark records are retained locally through the benchmark harness, no `.benchmarks/` result is committed, and `implementation-status.md` records exact evidence and remaining limitations.

## 19. Final phase acceptance

Phase 7 is complete only when P0-P5 pass, the public target matrix in Section 12 is implemented without silent fallback, all common-path performance claims have release-mode evidence, and the documentation examples execute against the installed package. Deferred mappings, mixed-dimension backend MVP, finite-boson algebra, and model factories remain deferred rather than becoming implicit completion blockers.
