# Feature Incubator

Status: living design ledger for ideas that are not mature enough to enter a frozen phase specification.

## Purpose and maintenance policy

This document preserves feature ideas, partial designs, unresolved questions, and deferred directions that arise during TenCirPauli development but do not yet have a committed implementation phase. It prevents useful reasoning from being lost and prevents speculative features from leaking into an active specification.

Every new concept should enter this document before implementation unless it is already covered by a frozen phase spec. Each entry records its status, motivation, current decisions, unresolved questions, dependencies, and promotion criteria.

Allowed statuses are:

| Status | Meaning |
| --- | --- |
| `seed` | A potentially useful idea with little design work. |
| `discussing` | Active design work; semantics or scope remains open. |
| `deferred` | Understood well enough to postpone intentionally. |
| `promoted` | Moved into a named phase spec; this entry remains as historical context and links to that spec. |
| `rejected` | Deliberately excluded, with the reason retained. |

When an idea is promoted, do not delete its entry. Mark it `promoted`, summarize the owner decision, and link to the authoritative spec. The phase spec becomes the source of truth; this ledger should not duplicate changing implementation details.

## Promoted ideas

### Majorana algebra, parity/BK mappings, and additive-charge sectors

Status: `promoted` to the frozen implementation contract `phase-7.5-spec.md`.

The promoted scope includes public Majorana words/operators, fermion-to-Majorana conversion, reusable Jordan–Wigner/parity/Bravyi–Kitaev mapping plans, exact integer-valued additive charges, simultaneous commuting-charge sectors, zero-charge qudit spectators, symmetry validation, and guarded Hamiltonian-space reduction.

## Deferred ideas

### Public finite-boson algebra

Status: `deferred`.

Motivation: support exact algebra after choosing finite occupation cutoffs, make projected boson operators first-class, and provide a natural input to boson-to-qubit encodings.

Current design: do not reuse infinite-Fock CCR power blocks as the canonical finite algebra. For a one-mode cutoff `N`, use matrix units `E_mn = |m><n|` with `E_mn E_pq = delta_np E_mq`; multimode words are tensor products of local matrix units. This is exact, finite, and closed but is mathematically a structured finite local-matrix algebra rather than a second CCR algebra.

Reason for deferral: Phase 7 already supports explicit projected finite targets, while no current user workflow requires symbolic multiplication of projected operators. Public exposure would add a broad local-operator abstraction and potentially large term counts without an immediate consumer.

Promotion criteria: a concrete boson-to-qubit, finite-boson circuit, or restricted finite-boson workflow that benefits from reusable finite algebra rather than direct compilation.

### Boson-to-qubit encodings

Status: `deferred`.

Motivation: compile finite boson modes to Pauli operators for qubit hardware, VQE, and TensorCircuit qubit backends.

Current design: support explicit binary, Gray, and unary/one-hot encodings. Return both the Pauli operator and immutable encoding metadata, including the code-space projector and the action on unused computational states. Non-power-of-two compact encodings must define illegal-state behavior explicitly; penalty terms are opt-in and never hidden.

Dependencies: a frozen finite-boson transition representation, exact isometry convention, analytic or sparse matrix-unit-to-Pauli decomposition, and representative term-count/Pauli-weight benchmarks.

Promotion criteria: an identified qubit-simulation workload and owner decisions on encoding set, illegal-state semantics, projector/penalty API, and acceptable term-growth boundaries.

### Lie closure for Pauli-polynomial generators

Status: `deferred`.

Motivation: analyze dynamical Lie algebras, controllability, reachable operator spaces, and symmetry-induced reductions from user-supplied Hamiltonian generators.

Current design: closure of individual Pauli words is comparatively simple because a nonzero commutator produces one Pauli word up to scalar, allowing a packed symplectic set closure. That restricted feature is not the desired endpoint. The target use case accepts Pauli sums or general operator polynomials, for which commutators are sparse vectors and new generators must be tested for linear independence in the real anti-Hermitian Lie algebra.

Main difficulties: sparse-vector fill-in, exact versus numerical rank, real versus complex scalar conventions, coefficient tolerance, deterministic basis selection, exponential dimension up to `4**n - 1`, and the fact that a `max_dimension` cap yields an incomplete rather than closed result.

Promotion criteria: a concrete control workload, a frozen coefficient/rank contract, a deterministic sparse elimination design, explicit completeness/truncation semantics, and bounded benchmark cases with known Lie dimensions.

### Arbitrary-order BCH and formal operator series

Status: `deferred`.

Motivation: effective Hamiltonians, composition of exponentials, canonical transformations, Floquet/Magnus calculations, and symbolic perturbation theory.

Current design: treat BCH as a formal truncated Lie series, not as a numerical approximation with an automatic error bound. Arbitrary order requires a free-Lie representation such as Hall/Lyndon bases or another audited coefficient generator, exact rational universal coefficients, and explicit `order`, `max_terms`, and `max_bytes` limits. The universal BCH term growth is compounded by term growth and CAR/CCR contractions in the evaluated operator algebra.

Related easier feature: truncated adjoint action `exp(A) B exp(-A) = sum_k ad_A**k(B)/k!` follows one commutator chain and may be worth promoting independently if a concrete canonical-transformation use case appears.

Reason for deferral: BCH is not required by the Phase 7.5 Majorana/mapping/symmetry workflows and would pressure the project into a premature universal operator protocol.

Promotion criteria: a concrete effective-Hamiltonian or formal-series workload, a decision between bounded fixed-order tables and arbitrary-order generation, a coefficient-exactness contract, and representative term-growth evidence.

### Reference normal ordering and Wick expansion

Status: `deferred`.

Motivation: many-body perturbation theory, Gaussian fermion expectations, reference-dependent normal ordering, diagrammatic contractions, and reduced-density calculations.

Current design: ordinary vacuum CAR/CCR normal ordering is already implemented and is not a missing feature. A future API must distinguish reference-normal-ordered objects from the current vacuum-canonical `FermionOperator`/`BosonOperator`. Fermionic Gaussian expectations should preferably use Majorana covariance matrices and Pfaffians rather than enumerate all pairings. Explicit Wick expansion, if required, must retain reference metadata and contraction provenance.

Promotion criteria: a concrete Gaussian/reference-state workload and owner decisions on Slater versus general Gaussian references, expectation-only versus explicit expansion, anomalous contractions, and result representation.

### Qudit symmetry and sector reduction

Status: `deferred`.

Motivation: discover generalized Pauli/Weyl symmetries and reduce uniform-qudit Hamiltonians.

Current design: prime local dimension permits symplectic linear algebra over a finite field. Composite dimensions such as `d=4` or `d=6` require module methods over `Z_d`, potentially Smith normal form, and cannot reuse prime-field Gaussian elimination unchanged.

Reason for deferral: Phase 7.5 additive charges intentionally exclude charged qudit axes, and the current Weyl API supports both prime and composite dimensions. A partial implementation that silently treats every `Z_d` as a field would be incorrect.

Promotion criteria: a target qudit workload, a decision on prime-only versus general composite support, a frozen stabilizer/sector convention, and independent small-dimension references.

### Graph-dependent fermion-to-qubit mappings

Status: `deferred`.

Motivation: reduce Pauli weight or improve locality beyond JW/parity/BK for structured interaction graphs.

Candidates include Bravyi–Kitaev superfast and auxiliary-fermion mappings. These may change qubit count, introduce stabilizer constraints, and depend on graph orientation or auxiliary choices, so they do not fit the Phase 7.5 invertible occupation-encoding plan contract.

Promotion criteria: a representative interaction graph, a target metric, explicit stabilizer/code-space semantics, and evidence that JW/parity/BK are insufficient.

## Seed ideas

### Specialized quadratic-algebra analysis

Status: `seed`.

Quadratic Majorana Hamiltonians correspond to real antisymmetric matrices and admit specialized symmetry, canonical-form, and Lie-closure algorithms. Quadratic boson Hamiltonians similarly connect to symplectic linear algebra. These specialized paths may be more useful and much safer than a universal polynomial Lie engine, but no concrete user workflow has yet selected the required outputs.

### Formal selection rules and operator grading

Status: `seed`.

Additive-charge analysis can later expose operator homogeneity and selection rules, not only Hamiltonian conservation. The frozen Phase 7.5 contract intentionally excludes a public `charge_deltas` API. Possible future outputs include a charge-homogeneous decomposition and rectangular maps between different sectors.

### Model-library layer

Status: `seed`.

Reusable Hubbard, Bose-Hubbard, Holstein, spin-boson, and related model factories could demonstrate the structured algebra and symmetry APIs. The current project deliberately prioritizes lower-level operator infrastructure; model factories should be promoted only when their conventions, boundary conditions, and parameter schemas can be kept small and stable.
