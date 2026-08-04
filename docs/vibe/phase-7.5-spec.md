# Phase 7.5 specification: Majorana algebra, fermion mappings, and additive-charge sectors

Status: frozen implementation contract, approved by the owner on 2026-08-03. P0–P5 implementation and local acceptance evidence are complete; the clean Phase 7 prerequisite and Phase 7.5 handoff are recorded in `implementation-status.md`.

> API note: this historical specification predates the breaking Phase 8 API contract; current public names and signatures are defined in [`phase-8-api-coherence-spec.md`](phase-8-api-coherence-spec.md).

## 1. Purpose

Phase 7.5 extends the structured-operator layer in three connected directions: a public Majorana algebra at the same level as the existing fermion and boson algebras; deterministic parity and Bravyi–Kitaev fermion-to-qubit mappings alongside Jordan–Wigner; and exact integer-valued additive charges that can be validated as symmetries and used to reduce a finite Hamiltonian basis.

The phase preserves TenCirPauli's division of responsibility. Rust owns canonical sparse algebra, exact discrete signs and phases, deterministic mapping, guarded sector enumeration, and native restricted plans. Python owns the typed public API and friendly validation. TensorCircuit remains a consumer of compiled qubit/backend plans rather than a dependency of the Rust core.

Phase 7.5 is not a general computer-algebra phase. It does not add arbitrary-order BCH, generic Lie closure, reference-dependent Wick expansion, a public finite-boson algebra, boson-to-qubit encoding, qudit symmetry, or a universal public operator trait. Those ideas are tracked in `feature-incubator.md`.

## 2. Prerequisites

Phase 7 must close the two gates retained by `phase-7-third-round-review-2026-08-03.md` before Phase 7.5 is marked complete:

1. Add an independent Holstein or spin-boson dense/COO/CSR/native-MVP differential.
2. Correct structured-plan basis/domain metadata and freeze exact public metadata regressions.

Implementation may begin in isolated branches while those fixes are in flight, but Phase 7.5 acceptance evidence must be based on a Phase 7-clean baseline.

## 3. Agreed scope

Phase 7.5 includes:

1. Public `MajoranaWord`, `MajoranaOperator`, `MajoranaTerm`, and a coefficient-free word-product result carrying the exact sign.
2. Exact conversions between canonical fermion and Majorana operators, with explicit expansion guards.
3. Reusable `FermionQubitMapping` plans for `jordan_wigner`, `parity`, and `bravyi_kitaev`.
4. Fermion and Majorana mapping through one frozen occupation-bit convention and deterministic GF(2)/Clifford transforms.
5. Mapping support for pure fermion operators and compatible hybrid operators without changing boson, physical-qubit, or qudit axes.
6. Exact integer-valued `AdditiveCharge` definitions over fermion, boson, and two-level qubit axes.
7. Exact additive-symmetry analysis, explicit charge-sector selection, guarded finite sector bases, and restricted dense/COO/CSR/native-MVP execution.
8. Compatibility with the existing `U1Sector`, Z2 tapering, Phase 7 finite-target, and `max_bytes` contracts without silently changing their semantics.

## 4. Frozen algebra and numeric principles

1. Operator coefficients remain finite complex128-compatible values. Majorana signs and fermion-mapping phases are exact discrete data before absorption into coefficients.
2. Static operators aggregate equal canonical keys in a deterministic order and remove exact zeros only. No coefficient-magnitude cutoff is introduced.
3. Every public operator is immutable and canonical. Raw Majorana sequences are canonicalized during construction; there is no public mutable noncanonical operator state.
4. Every mapping is deterministic, schema-versioned, and explicit in reusable-plan metadata. String mapping names remain convenience inputs, not the only provenance.
5. Additive charges use exact integers. User-facing names such as particle number, electric charge, `2Sz`, or excitation number are metadata; the algebra is defined by the integer weights and layout.
6. Restriction means exact action on an invariant sector. It never means silently replacing an operator by `P H P` after discarding leakage.
7. A restricted boson-containing sector must be provably finite, either from the charge constraints or from explicit inclusive boson cutoffs.
8. All major expansions, finite dimensions, sector counts, transition counts, and allocations use checked arithmetic and the existing best-effort `max_bytes` policy.

## 5. Majorana algebra

### 5.1 Convention

For fermion mode `p`, Phase 7.5 uses:

```text
gamma_(2p)   = a_p + a_p†
gamma_(2p+1) = i (a_p† - a_p)

a_p  = (gamma_(2p) + i gamma_(2p+1)) / 2
a_p† = (gamma_(2p) - i gamma_(2p+1)) / 2
```

The generators are Hermitian and obey:

```text
gamma_i† = gamma_i
gamma_i^2 = I
{gamma_i, gamma_j} = 2 delta_ij I
```

This convention must be checked against explicit Fock-space matrices with fermion mode zero as the most-significant occupation axis, matching Phase 7.

### 5.2 Canonical word

A `MajoranaWord` belongs to `n_modes` fermion modes and stores a strictly increasing tuple of indices in `0 <= index < 2*n_modes`. The empty tuple is identity. Repeated indices are removed during raw multiplication through `gamma_i^2 = I`; reordering contributes the exact sign from the permutation parity.

The public canonical constructor accepts only sorted unique indices. A separate `from_indices()` or operator constructor accepts an arbitrary raw sequence and returns the resulting canonical operator. Multiplication of two canonical words produces exactly one canonical word and one sign in `{+1,-1}`; it does not branch.

The adjoint of a length-`k` canonical word reverses generator order and therefore contributes:

```text
(-1) ** (k * (k - 1) / 2)
```

The word remains phase-free; the sign is carried by a `MajoranaProduct` result or absorbed into an operator coefficient.

### 5.3 Public API

```python
word = tcp.MajoranaWord(n_modes=4, indices=(0, 3, 6))
product = word.multiply(other_word)
product.word
product.sign

operator = tcp.MajoranaOperator.from_terms(
    n_modes=4,
    terms=[((0, 3, 6), 0.5j), ((2, 1, 2), 1.0)],
)

fermion = operator.to_fermion(max_bytes=...)
majorana = fermion.to_majorana(max_bytes=...)
```

`MajoranaOperator` supports deterministic terms, addition, scaling, multiplication, commutator, anticommutator, adjoint, Hermiticity testing, exact conversion to fermions, and mapping to all Phase 7.5 qubit mappings. Compile targets that require qubits route through an explicit mapping rather than inventing a separate Majorana matrix convention.

### 5.4 Representation and kernels

The Rust core should store Majorana support in packed `u64` limbs for large mode counts. Word multiplication uses support XOR plus the parity of cross inversions. The first correct implementation may use an indexed parity scan; a prefix-parity or limb-popcount optimization should be added only after release profiling identifies multiplication as a bottleneck.

Operator multiplication, conversion, and mapping cross PyO3 once per complete batch. Public outputs sort lexicographically by canonical Majorana indices, independent of hash seed or thread count.

## 6. Fermion-to-qubit mappings

### 6.1 Supported mappings

Phase 7.5 supports exactly:

```text
jordan_wigner
parity
bravyi_kitaev
```

Graph-dependent Bravyi–Kitaev superfast, auxiliary-fermion, ternary-tree, and user-defined arbitrary encodings are not part of this phase.

### 6.2 Occupation encoding

For `n` fermion modes, an encoding is defined by an invertible binary matrix `B`:

```text
q = B n  (mod 2)
```

Here `n` is the occupation vector in increasing fermion-mode order and `q` is the computational-basis qubit vector in increasing qubit order. Vector entry zero is mode/qubit zero and is rendered as the most-significant computational-basis bit. The convention identifier is `tencirpauli.gf2_occupation.v1`.

The three encoding matrices are frozen as:

```text
Jordan-Wigner:
    B[j,k] = 1 iff k == j

Parity:
    B[j,k] = 1 iff 0 <= k <= j

Bravyi-Kitaev Fenwick:
    lowbit(r) = r & (-r)
    B[j,k] = 1 iff j + 1 - lowbit(j + 1) <= k <= j
```

Thus the parity qubit `j` stores the prefix parity `n_0 xor ... xor n_j`. The BK qubit `j` stores the Fenwick interval ending at mode `j`. For four modes, the BK encoded bits are `(n_0, n_0 xor n_1, n_2, n_0 xor n_1 xor n_2 xor n_3)`. These constructive definitions, rather than an external library's mapping name, are the semantic authority.

The implementation constructs a deterministic linear-reversible Clifford/CNOT transform `C_B` satisfying:

```text
C_B |n> = |B n>
```

The encoded operator is obtained from the frozen Jordan–Wigner representation by Clifford conjugation:

```text
O_B = C_B O_JW C_B†
```

This is the correctness authority for parity and BK. Specialized direct generator formulas may replace the conjugation path only after dense differentials prove exact equivalence and release benchmarks show a material end-to-end gain.

The canonical CNOT provenance is obtained by reducing the lower-triangular unit-diagonal matrix `B` to identity with GF(2) row additions. Pivot columns and target rows are visited in increasing order; the recorded row additions are reversed to synthesize `B` from identity. A row addition `row[target] ^= row[control]` corresponds to `CNOT(control, target)`. Equivalent shorter circuits may be used internally only if the public mapping result, encoding matrices, and convention identifier remain unchanged; the canonical provenance remains available for deterministic tests.

### 6.3 Reusable mapping plan

The public object is:

```python
mapping = tcp.FermionQubitMapping.bravyi_kitaev(n_modes=16)

mapped = fermion.map_fermions(mapping, max_bytes=...)
mapped_hybrid = hybrid.map_fermions(mapping, max_bytes=...)
mapped_majorana = majorana.map_fermions(mapping, max_bytes=...)
```

Existing string calls remain valid:

```python
fermion.map_fermions("jordan_wigner")
fermion.map_fermions("parity")
fermion.map_fermions("bravyi_kitaev")
```

The reusable plan exposes immutable metadata:

- schema version and mapping name;
- `n_modes`, output qubit count, mode ordering, and basis ordering;
- canonical packed encoding and inverse-encoding matrices;
- deterministic Clifford/CNOT transform provenance;
- estimated plan bytes;
- convention identifier distinguishing the frozen Phase 7.5 parity/BK definitions.

User-supplied arbitrary GF(2) matrices are deferred. Public matrices are read-only diagnostics for the three supported mappings, not an invitation to bypass validation.

### 6.4 Hybrid mapping

Mapping replaces the fermion axes in place with the mapping plan's qubit axes while preserving the relative order of boson, physical-qubit, and qudit axes. Original physical qubits remain distinguishable from mapped fermion qubits in layout metadata. Raw/mapped operand-order rules from Phase 7 remain in force: algebra that would lose operand order maps eagerly or fails explicitly, never silently reorders factors.

## 7. Additive charges and symmetry analysis

Status: frozen.

### 7.1 Mathematical definition

An `AdditiveCharge` is an exact integer-valued diagonal operator on a structured layout. Qudit axes are always uncharged. The charge has the form:

```text
Q = offset * I
  + sum_p fermion_weight[p] * a_p† a_p
  + sum_r boson_weight[r] * b_r† b_r
  + sum_j (qubit_level[j,0] |0><0| + qubit_level[j,1] |1><1|)
```

Fermion and boson weights may be positive, zero, or negative integers. Qubit level values are integer pairs. Unspecified axes carry zero charge. `name` is descriptive metadata and does not change algebraic equality; two definitions with the same layout, offset, and canonical weights represent the same charge even if their display names differ.

Examples include:

```python
particle_number = tcp.AdditiveCharge(
    space,
    name="particle_number",
    fermions={mode: 1 for mode in range(space.fermions)},
)

spin_z2 = tcp.AdditiveCharge(
    space,
    name="2Sz",
    fermions={up_mode: +1, down_mode: -1},
)

excitation_number = tcp.AdditiveCharge(
    space,
    name="excitation_number",
    fermions={0: 1, 1: 1},
    bosons={0: 1},
    qubits={0: (0, 1)},
)
```

The name does not decide whether a quantity is energy, electric charge, spin, or particle number. The user-supplied integer weights define the generator. A free energy such as `sum_i omega_i n_i` fits this interface only when the frequencies are represented exactly as commensurate integer weights after choosing a unit. A general interacting Hamiltonian is not an additive charge merely because it is called energy.

Phase 7.5 does not assign charge to qudit basis levels. Qudit axes may remain as uncharged spectators during analysis and restriction under Section 7.4.

### 7.2 Symmetry definition

For a Hamiltonian `H`, an additive charge is a continuous symmetry exactly when:

```text
[H, Q] = 0
```

The public analysis must validate the complete aggregated operator. It must not reject a conserved Hamiltonian merely because individual Pauli terms appear to leak before cancellation, as in `XX + YY` hopping. A simple independent reference constructs `Q` as an operator and computes the exact canonical commutator. Optimized domain-specific analysis may use charge selection rules or grouped transitions, but it must agree with this reference.

The public API is:

```python
generator = particle_number.as_operator(max_bytes=...)
analysis = h.analyze_charge(particle_number, max_bytes=...)
analysis.charge
analysis.is_conserved
analysis.commutator_term_count
analysis.method

h.conserves(particle_number, max_bytes=...)
```

The result type is `AdditiveSymmetryAnalysis`. Its main contract is intentionally lightweight: it exposes the charge, `is_conserved`, canonical commutator term count, and analysis method without retaining a potentially large full commutator. An optional bounded witness or an explicitly requested commutator may be added as diagnostics, but neither is part of the main symmetry/restriction workflow.

Phase 7.5 supports only exact integer-valued continuous additive charges and the symmetry condition `[H,Q] = 0`. It does not accept floating, rational-object, symbolic, or modular weights. Modular charges such as fermion parity require `[H, exp(2*pi*i*Q/m)] = 0` rather than `[H,Q] = 0` and continue to use the existing Z2 symmetry API in this phase.

### 7.3 Charge construction API

The constructor accepts sparse domain maps and stores canonical full tuples internally:

```python
charge = tcp.AdditiveCharge(
    space,
    name="N",
    fermions={0: 1, 1: 1},
    bosons={0: 2},
    qubits={0: (0, 1)},
    offset=0,
)
```

Alternative convenience factories such as `space.additive_charge(...)`, `AdditiveCharge.particle_number(space)`, or model-specific spin constructors are not required for the first slice. They may be added only after the canonical low-level form is stable.

### 7.4 Uncharged qudit spectator option

An uncharged spectator means that the charge acts as identity on the qudit factor rather than assigning a qudit quantum number. For a space with two fermion modes and one qutrit:

```text
Q = (n_0 + n_1) tensor I_3
```

A Hamiltonian term such as `n_0 tensor X_qutrit` conserves `Q`: the qutrit can change state, but fermion number does not. Selecting the fermion-number-one sector would retain both one-fermion basis states and all three qutrit levels, giving dimension `C(2,1) * 3 = 6`. This is not qudit symmetry analysis; the qutrit is simply carried through the reduced basis unchanged.

This zero-charge spectator behavior is required. Charge analysis treats every qudit factor as commuting with `Q`, while sector enumeration retains every qudit basis level and includes its local dimension in rank, unrank, dimension, and restricted execution. Phase 7.5 still does not define qudit charge weights or discover qudit symmetries.

## 8. Charge sectors and Hamiltonian-space reduction

Status: frozen.

### 8.1 Sector semantics

A charge sector selects computational/Fock basis states satisfying one or more exact charge constraints. Restriction is allowed only after the operator has been proven to conserve every selected charge. The implementation must never silently delete sector-changing contributions.

For one charge, the convenience shape is:

```python
sector = particle_number.sector(
    value=4,
    boson_cutoffs=None,
    max_bytes=...,
)

restricted = h.restrict_charge(sector, max_bytes=...)
```

For simultaneous particle-number and spin sectors, the general shape is:

```python
sector = tcp.ChargeSector(
    constraints=((particle_number, 4), (spin_z2, 0)),
    boson_cutoffs=None,
    max_bytes=...,
)
```

The first implementation supports one or more simultaneous commuting `AdditiveCharge` constraints. All charges are diagonal on the same compatible layout and therefore commute by construction. The canonical tuple form is part of the public contract; `charge.sector(value=...)` is a single-charge convenience wrapper around it.

### 8.2 Finite-basis condition

Fermion and qubit axes are finite. Boson axes require proof that the selected constraints bound every occupation, or an explicit inclusive cutoff for each otherwise-unbounded mode.

For example, a fixed non-negative total boson number:

```text
Q = sum_r b_r† b_r
Q = N
```

implies `0 <= n_r <= N` and therefore defines a finite sector without explicit cutoffs. Conversely, a zero-weight spectator boson, an unconstrained boson, or canceling positive/negative weights can leave infinitely many basis states and must require an explicit cutoff or fail.

The first implementation must infer simple exact boson bounds from non-negative integer weights and non-negative target values. More general mixed-sign or multi-constraint finiteness proofs are optional; when the implemented proof cannot establish finiteness, the error must request explicit cutoffs rather than claim that the mathematical sector is necessarily infinite.

### 8.3 Basis ordering and combinatorics

The restricted basis is the subsequence of the full finite mixed-radix computational/Fock basis that satisfies all charge constraints, preserving Phase 7's axis and most-significant-axis ordering. The implementation must not materialize the full basis to filter it.

`ChargeSector` is an immutable reusable basis plan. Its constructor, including the `charge.sector(...)` convenience path, validates finiteness, resolves inferred or explicit boson bounds, builds the checked rank/unrank dynamic-programming state, and charges that storage to `max_bytes`. No separate public `ChargeBasisPlan` is introduced. The object exposes:

```python
sector.dimension
sector.local_dimensions
sector.basis_ordering
sector.estimated_bytes
sector.rank(occupations)
sector.unrank(index)
sector.basis_states(max_bytes=...)
```

`occupations` is a tuple in `OperatorSpace.axes` order rather than a qubit-only integer. Rank, unrank, and dimension use checked dynamic programming over suffix axes and remaining charge values. Simultaneous charges use tuple-valued DP keys. Negative weights, offsets, inferred boson bounds, explicit cutoffs, and platform-index overflow require dedicated tests.

### 8.4 Restricted operator

The `ChargeRestrictedOperator` result is parallel to, but does not replace, `U1RestrictedOperator`:

```python
restricted = h.restrict_charge(sector, max_bytes=...)

restricted.sector
restricted.dimension
restricted.apply(state, max_bytes=...)
restricted.mvp_plan(max_bytes=...)
restricted.dense(max_bytes=...)
restricted.coo(max_bytes=...)
restricted.csr(max_bytes=...)
```

Its reusable matrix-free result is `ChargeMvpPlan`. The restricted action must equal `P† H P` only after exact conservation has been established. Construction and apply must work directly in the restricted basis without allocating a full-space state or full matrix. Dense, COO, and CSR remain guarded materializations; native MVP is the primary scalable target.

The existing `U1Sector`, `U1RestrictedOperator`, and `PauliOperator.restrict_u1()` remain stable. Phase 7.5 may share private combinatorial or transition infrastructure only after benchmarks show no regression in the mature arbitrary-width U1 path. It must not replace the existing optimized fixed-Hamming-weight engine with a slower generic implementation.

### 8.5 Mapping interaction

Charge analysis and restriction are defined on the physical structured layout before fermion-to-qubit mapping. A `FermionQubitMapping` may map the charge generator itself to a Pauli operator and preserve provenance, but Phase 7.5 does not require a BK-encoded fixed-charge basis to masquerade as a fixed-Hamming-weight `U1Sector`.

The workflow is:

```python
charge = tcp.AdditiveCharge(...)
analysis = h.analyze_charge(charge)
sector = charge.sector(value=...)
restricted = h.restrict_charge(sector)
```

Mapping-first and restriction-first equivalence must be validated on small systems, but the optimized restricted implementation may remain native to the structured occupation basis.

## 9. Error and memory contract

`max_bytes` is a cheap, best-effort guard for the dominant new output, retained plan storage, and directly predictable workspace of the current public call. It is intentionally not an exact peak-memory quota. Implementations may use a deliberately loose checked upper bound, but must not perform a symbolic second traversal, allocator/RSS query, per-element budget accounting, or other work whose runtime is material relative to the operation merely to tighten the estimate. Python/Rust object headers, allocator slack, conversion temporaries, and pre-existing caller-owned state remain outside this logical estimate.

The phase must fail explicitly for:

- invalid, non-integer, or out-of-layout charge weights;
- incompatible charge/operator/sector layout fingerprints;
- duplicate or inconsistent simultaneous constraints;
- a selected charge that is not conserved;
- a boson-containing sector whose finiteness cannot be proved and whose required cutoffs are absent;
- invalid or insufficient explicit cutoffs;
- invalid Majorana indices or noncanonical public word construction;
- incompatible mapping mode counts or unknown mapping names;
- singular or malformed internal encoding matrices;
- expansion, sector dimension, rank/unrank, transition, or output arithmetic overflow;
- a state shape different from the restricted dimension;
- a cheaply estimated major expansion, plan, workspace, or output exceeding `max_bytes`.

Mapping or restriction failure must not return a partial operator or silently fall back to Jordan–Wigner, a full-space plan, NumPy execution, or projected leakage removal.

## 10. Correctness requirements

Independent references must cover:

1. Majorana anticommutation, squared identity, multiplication signs, adjoints, Hermiticity, canonical ordering, deterministic aggregation, and large packed-index boundaries.
2. Fermion/Majorana conversion in both directions against explicit Fock matrices, including all words through a bounded degree on small mode counts.
3. JW, parity, and BK occupation-basis matrices, Clifford transforms, mapped ladder/Majorana generators, number operators, hopping, quartic Hubbard terms, and random bounded operators for small `n`.
4. Equality of mapped dense/COO/CSR/native/backend MVP targets where supported, with frozen mode-zero/MSB ordering.
5. Additive-charge generator matrices, exact commutator-based conservation, conserved and broken fermion/boson/qubit/hybrid examples, and cancellation-dependent Pauli examples.
6. Charge-sector dimension/rank/unrank against independent enumeration, including fixed fermion number, `2Sz`, mixed fermion-boson excitation, inferred finite boson sectors, explicit cutoffs, negative/zero weights, and invalid infinite sectors.
7. Restricted dense/COO/CSR/MVP equivalence to independently constructed `P† H P`, without using the implementation under test to generate `P`.
8. Simultaneous constraints, especially particle number plus `2Sz` for a spinful Hubbard fixture.
9. Mapping/restriction ordering differentials and preservation of charge-generator spectra under all three fermion mappings.
10. Deterministic structural results, checked overflow, exact-zero behavior, explicit `max_bytes=None`, low memory limits, and no per-term or per-basis-state FFI.

## 11. Performance requirements

Release benchmarks must separate:

- Majorana raw construction, canonical multiplication, fermion conversion, and mapping;
- JW, parity, and BK plan construction plus one-shot and reused mapping;
- input term count, mapped canonical term count, Pauli weight distribution, plan bytes, runtime, throughput, and numerical error;
- charge analysis setup;
- sector dimension/rank/unrank and basis-plan construction;
- restricted plan construction, first apply, steady apply, COO/CSR materialization, transition count, workspace bytes, and numerical error;
- comparison with the existing optimized `U1Sector` path on equivalent fixed-Hamming-weight workloads to prevent a generic-path regression.

Representative workloads include one-/two-body fermion Hamiltonians, quartic Hubbard terms, Majorana chains, mapping-sensitive long parity strings, fixed particle number, simultaneous particle number plus `2Sz`, Bose-Hubbard fixed total occupation, and a mixed excitation-conserving fermion-boson fixture.

No fixed speedup threshold is imposed. Every material optimization claim requires release-mode measurement including Python/Rust boundary costs.

## 12. Implementation slices

### P0: close Phase 7 and freeze reference conventions

Close the retained Phase 7 Holstein/metadata gates. Freeze Majorana matrices, parity/BK encoding matrices, basis permutations, mapping schema identifiers, additive-charge normalization, sector basis ordering, and the final owner decisions in Section 14 as executable references and public signatures.

Acceptance gate: independent references and public signatures are approved before native optimization or broad implementation.

### P1: Majorana vertical slice

Implement pure Rust packed Majorana words/operators, PyO3 batching, typed Python objects, algebra, fermion conversion, deterministic tests, and construction/multiplication benchmarks.

Acceptance gate: algebraic properties and Fock-matrix differentials pass; outputs are deterministic and expansion guards work.

### P2: reusable JW/parity/BK mapping plans

Implement the three frozen mapping plans through the GF(2)/Clifford reference path, route fermion and Majorana operators through them, support compatible hybrid mapping, expose metadata, and benchmark plan reuse and mapped Pauli weight.

Acceptance gate: all three mappings agree with independent encoded-basis matrices and all supported Pauli targets.

### P3: additive-charge definitions and symmetry analysis

Implement exact immutable charges, materialized/reference generators, complete-operator conservation analysis, cancellation regressions, diagnostics, and the final approved public analysis API.

Acceptance gate: conserved and broken structured/Pauli fixtures agree with exact commutator references and never rely on term-wise false positives.

### P4: charge-sector basis and restricted native execution

Implement finite-sector proof/cutoff validation, DP dimension/rank/unrank, direct restricted transition compilation, native MVP, guarded dense/COO/CSR, and explicit leakage failures. Preserve the existing U1 engine.

Acceptance gate: restricted targets agree with independent `P† H P` references across fermion, boson, qubit, and hybrid fixtures; no full-space state is allocated.

### P5: delivery and handoff

Complete exports, type hints, docstrings, examples, compatibility notes, mapping/sector metadata tables, release notes, benchmark registration, and the repository quality workflow.

Acceptance gate: the full quality gate, focused release benchmark record, executable examples, and clean status handoff pass on a committed tree.

## 13. Non-goals

- No public finite-boson operator algebra or boson-to-qubit encoding.
- No qudit charge or generalized qudit stabilizer symmetry.
- No automatic discovery of arbitrary continuous, non-Abelian, or model-specific symmetries.
- No generic Lie closure for Pauli sums or other operator polynomials.
- No BCH, Magnus, formal exponentials, or time-evolution algorithm.
- No reference-dependent normal ordering, Wick expansion, Gaussian-state engine, Pfaffian, or hafnian API.
- No graph-dependent BKSF or auxiliary-fermion mapping.
- No arbitrary user-defined GF(2) fermion encoding in the first release.
- No hidden projection of a nonconserving Hamiltonian into a selected sector.
- No replacement of the optimized existing Z2 or arbitrary-width `U1Sector` implementations.
- No universal public algebra protocol introduced solely to share method names.

## 14. Final owner decisions

The following decisions are frozen for Phase 7.5:

1. `AdditiveCharge.name` is display metadata and is excluded from semantic equality and hashing.
2. Charge offsets, fermion/boson weights, qubit level values, and sector values are exact Python/Rust integers. Floating, rational-object, symbolic, and modular charges are rejected.
3. The first `ChargeSector` supports several simultaneous commuting charge constraints; the single-charge constructor is a convenience wrapper.
4. `AdditiveSymmetryAnalysis` remains lightweight and does not retain a full commutator by default.
5. Simple finite boson bounds implied by non-negative integer charge constraints are inferred without requiring redundant explicit cutoffs.
6. Mapping methods accept both stable string names and validated reusable `FermionQubitMapping` plans.
7. Qudit axes are permitted only as zero-charge spectators; their complete local basis is retained in every selected charge sector.
8. `ChargeSector` owns the immutable checked rank/unrank dynamic-programming plan and exposes `estimated_bytes`; there is no separate public `ChargeBasisPlan`.
9. Jordan–Wigner, parity, and BK use the exact matrices and canonical mode-zero/MSB convention in Section 6.2.
