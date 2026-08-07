# Phase 10 Chemistry and Scientific-Interop Adapter Specification

Status: frozen implementation specification after owner review. The first public scope is the standardized PySCF import path and the SciPy `LinearOperator` path. SciPy is a required Python dependency; PySCF is optional and remains lazily imported. Numerical convention, ordering, implementation sequencing, and acceptance gates are fixed by this specification.

## 1. Goal

Phase 10 adds two thin Python integration paths: importing quantum-chemical molecular Hamiltonians from optional PySCF and exposing existing native matrix-free plans as SciPy `LinearOperator` objects. Phase 10 promotes SciPy from an examples-only dependency to the required numerical-interoperability and iterative-solver boundary.

The phase deliberately keeps quantum chemistry outside the Rust core. PySCF owns molecule construction, basis functions, SCF, AO integrals, and orbital coefficients. TenCirPauli owns canonical fermionic algebra, batched native construction, fermion-to-qubit mapping, and matrix-free execution. SciPy owns the external iterative-linear-algebra protocol, while the actual matrix-vector product remains in the existing native plan.

The intended user experience is:

```python
from pyscf import gto, scf
from tencirpauli.integrations.pyscf import from_scf

mol = gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    unit="Angstrom",
)
mf = scf.RHF(mol).run()

fermion_h = from_scf(mf)
qubit_h = fermion_h.map_fermions("jordan_wigner")
plan = qubit_h.compile("native_mvp")
linear_h = plan.to_scipy_linear_operator()
```

The chemistry adapter must not require OpenFermion. OpenFermion is not a Phase 10 dependency or compatibility target; its PySCF bridge would add an unnecessary intermediate representation for the primary path.

## 2. Scope and owner decisions

### 2.1 Required scope

1. Add `from_scf()` as the primary lazily imported PySCF adapter. Calling it performs eager integral conversion; it accepts a converged RHF or UHF object and returns a native-backed `FermionOperator`.
2. Add `from_molecule()` as a convenience PySCF adapter that constructs and runs the explicitly selected RHF or UHF method from a `pyscf.gto.Mole`.
3. Add `FermionOperator.from_integrals()` as the public canonical spin-orbital construction API and use the same native integral-ingestion primitive for this API and the compact RHF/UHF blocks supplied by the PySCF adapter. This is a reusable construction capability, not a third external framework adapter.
4. Add `to_scipy_linear_operator()` to every public native MVP plan that implements the common `MVPPlan` execution contract, including ordinary, charge-restricted lazy/eager, and U1-restricted plans. Add a `PauliOperator.to_scipy_linear_operator()` convenience method that compiles one ordinary native MVP plan internally and delegates to the shared adapter implementation.
5. Keep fermion-to-qubit mapping separate. A `FermionOperator` must first be mapped to a `PauliOperator`; the SciPy adapter does not accept a mapping argument and does not perform mapping implicitly.
6. Keep all scalable operator construction and MVP execution in the native path. Python adapters may normalize small control metadata but must not loop through one PySCF term or one MVP basis vector at a time.
7. Add independent numerical tests against PySCF and dense references for small molecules, including the nuclear-repulsion constant and the selected spin-orbital ordering.
8. Replace the manual SciPy `LinearOperator` construction in the existing ED research examples with the new `to_scipy_linear_operator()` API. Do not add a second standalone SciPy example.
9. Add one optional PySCF chemistry example that reaches a Pauli operator and a TensorCircuit backend-MVP/VQE energy evaluation.

### 2.2 Explicitly out of scope

- OpenFermion import or export adapters.
- A Rust-core dependency on PySCF, SciPy, NumPy, TensorCircuit, or any quantum-chemistry package. The Python distribution requires SciPy, and the PyO3 binding may accept NumPy arrays, but `tencir-pauli-core` remains independent of all Python packages and frameworks.
- A new chemistry engine, FCI/CCSD/MP2 implementation, orbital optimizer, or SCF implementation.
- Implicit selection among RHF, UHF, ROHF, DFT, or correlated PySCF objects.
- Direct support for nonorthogonal fermionic CAR. Standard `FermionOperator` construction requires canonical orthonormal orbitals.
- Automatic density-fitting, Cholesky, tensor-hypercontraction, or other factorized two-body representations in the first implementation.
- A general-purpose matrix-free eigensolver. The SciPy adapter only implements the `LinearOperator` protocol.
- FCIDUMP, Psi4, Qiskit, PennyLane, QuTiP, QuSpin, and other future framework adapters in the first implementation batch.
- Exact peak-RSS accounting for adapter conversion or SciPy callback allocations.

## 3. Package boundary and optional dependencies

New adapters belong under `python/tencirpauli/integrations/`:

```text
python/tencirpauli/integrations/
├── __init__.py
├── pyscf.py
├── scipy.py
└── tensorcircuit.py
```

`pyscf.py` must import PySCF inside the public adapter function or a private dependency-check helper. Importing `tencirpauli`, `tencirpauli.integrations`, or unrelated integration modules must not import PySCF. PySCF belongs to the unpinned `chemistry` optional dependency group so the environment resolver selects a compatible release. A missing installation must raise a clear `ImportError` naming PySCF and `pip install "tencirpauli[chemistry]"`.

SciPy is a core Python dependency and must be declared in the main project dependency set. `scipy.py` may import `scipy.sparse.linalg.LinearOperator` directly; it does not need a missing-SciPy fallback or an optional-dependency error branch. Existing `COOMatrix.to_scipy()` and `CSRMatrix.to_scipy()` retain local imports for import hygiene but remove their obsolete missing-SciPy recovery branches and optional-dependency wording. The former examples-only SciPy requirement is removed from that extra rather than duplicated. PySCF is never imported by ordinary package import.

The Rust core and PyO3 layer must not know that either adapter exists. They may expose a framework-neutral integral-ingestion primitive over NumPy arrays and small spin-block descriptors. The production path must not serialize dense integrals into Python term tuples or route them through the existing nested-sequence `FermionOperator.from_terms()` ABI.

## 4. Canonical molecular Hamiltonian contract

### 4.1 Mathematical form

The canonical spin-orbital operator is:

\[
\hat H = E_{\mathrm{nuc}} + \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_s a_r.
\]

The canonical two-body tensor is defined by:

\[
g_{pqrs} = \int \!\!\int \psi_p^*(1)\psi_q^*(2)r_{12}^{-1}\psi_r(1)\psi_s(2)\,d1\,d2.
\]

For the public integral constructor, `two_body[p, q, r, s]` is exactly the spin-orbital tensor $g_{pqrs}$ in this formula. The constructor emits the raw word $a_p^\dagger a_q^\dagger a_s a_r$ with coefficient `0.5 * two_body[p, q, r, s]`; callers must not divide the tensor by two beforehand. The constructor does not infer an alternate convention from tensor symmetry or silently accept an ambiguous four-index array.

The nuclear-repulsion energy is represented by the empty fermion word unless the caller explicitly sets `include_nuclear_repulsion=False`. Electronic one- and two-body terms are represented with canonical CAR aggregation. The default operator is therefore a complete molecular Hamiltonian, not only its electronic part.

### 4.2 Orbitals and overlap

AO overlap is an orbital-construction metric, not a coefficient tensor in the final canonical operator. If a supplied coefficient matrix is used, the adapter must verify or construct:

\[
C^\dagger S C = I.
\]

The adapter accepts real or complex `f64`-precision orbital coefficients and uses the conjugate transpose shown above. It requires `max(abs(C.conj().T @ S @ C - I)) <= 1e-8` separately for the alpha and beta coefficient matrices. Non-finite or nonorthogonal orbitals fail before two-electron integral transformation or operator-sized construction.

### 4.3 Spin-orbital ordering

The public adapter must expose an explicit `spin_ordering` option with exactly two first-version values:

- `"interleaved"`: `(orbital_0, alpha), (orbital_0, beta), ...`;
- `"alpha_then_beta"`: all alpha orbitals followed by all beta orbitals.

`"interleaved"` is the default. The choice is part of the adapter call contract and determines the mode indices in the returned second-quantized operator. A generic `FermionOperator` does not retain chemistry-specific ordering, molecule, electron-count, or orbital metadata; after construction, the returned operator is exactly the canonical fermionic expression encoded by its modes and coefficients. Tests compare both orderings using a fermionic mode permutation, including the occupied-mode inversion sign implemented by an FSWAP-equivalent basis transform rather than a sign-free qubit-bit permutation.

### 4.4 Integral conventions

The public constructor accepts one fixed, conventional spin-orbital Hamiltonian representation. It does not expose a `convention` switch. Antisymmetrized tensors, any four-index layout whose indices do not match the formula above, and raw PySCF spatial chemist-order tensors are not accepted through this API.

```python
FermionOperator.from_integrals(
    one_body,
    two_body,
    *,
    constant=0.0,
    max_bytes=DEFAULT_MAX_BYTES,
)
```

`one_body` has shape `(n_modes, n_modes)` and `two_body` has shape `(n_modes, n_modes, n_modes, n_modes)` in the spin-orbital formula of Section 4.1. Inputs are C-contiguous NumPy arrays with dtype `float64` or `complex128`; unsupported dtypes or layouts fail instead of triggering an unbudgeted full-size compatibility copy. `constant` is the scalar identity coefficient. The Hamiltonian contract requires a real constant, `one_body[p, q] = conj(one_body[q, p])`, and `two_body[p, q, r, s] = conj(two_body[r, s, p, q])`, checked with `rtol=1e-10` and `atol=1e-12`. After validation, the native constructor removes permitted floating-point asymmetry by pairwise Hermitian averaging and uses the real part of the constant, so the resulting operator is Hermitian by construction. It also validates rank, equal mode dimensions, finite values, and checked major-buffer sizes. It performs exact-zero removal only; it does not apply a numerical coefficient cutoff.

PySCF commonly exposes spatial molecular-orbital electron-repulsion integrals in chemist order, `eri[i, j, k, l] = (ij|kl)`. The adapter converts these internally according to

\[
g_{m(p,\sigma),m(q,\tau),m(r,\sigma),m(s,\tau)} = (p_\sigma r_\sigma|q_\tau s_\tau),
\]

with all spin-mismatched elements equal to zero and `m` determined by `spin_ordering`. In array-index terms, an appropriate transformed chemist block contributes `g[p_sigma, q_tau, r_sigma, s_tau] = eri[p, r, q, s]`. This conversion is private to the PySCF adapter and is not a second public convention accepted by `from_integrals()`.

Both public spin-orbital construction and private PySCF spin-block construction must use one framework-neutral native ingestion implementation. The PyO3 boundary accepts contiguous numerical arrays plus only O(1) block metadata; Rust performs index mapping, the one-half multiplication, exact-zero skipping, CAR aggregation, deterministic ordering, checked output estimates, and construction while the GIL is released. Building a Python term dictionary or nested factor list is forbidden for production, although `from_terms()` may be used in a tiny independent test oracle.

## 5. PySCF adapter

### 5.1 Public surface

The first public surface is small and function-oriented:

```python
from tencirpauli.integrations.pyscf import from_scf, from_molecule

fermion_h = from_scf(
    mf,
    *,
    spin_ordering="interleaved",
    include_nuclear_repulsion=True,
    max_bytes=DEFAULT_MAX_BYTES,
)

fermion_h = from_molecule(
    mol,
    *,
    method="rhf",
    scf_kwargs=None,
    spin_ordering="interleaved",
    include_nuclear_repulsion=True,
    max_bytes=DEFAULT_MAX_BYTES,
)
```

The settled public names are `from_scf()` and `from_molecule()`. `from_scf()` requires `mf.converged is True`; missing orbitals, a false convergence flag, or an absent molecule fails before overlap or two-electron work. `from_molecule()` accepts only `method="rhf"` or `method="uhf"`, constructs that exact PySCF object, calls `mf.run(**dict(scf_kwargs or {}))` without mutating the caller's mapping, verifies convergence, and then delegates to `from_scf()`. It must not inspect the molecule and silently choose a method.

RHF is the first correctness slice and UHF is the immediate mandatory follow-on slice within the same PySCF adapter. UHF requires equal alpha and beta molecular-orbital counts in the first implementation so both supported orderings remain unambiguous. ROHF, generalized Hartree-Fock, DFT objects, and correlated wavefunction objects are later adapter work or explicit `NotImplementedError` cases with tests. Type checks must reject DFT subclasses and other unsupported SCF families before operator-sized conversion.

### 5.2 Conversion steps

`from_scf()` performs the following steps:

1. Lazily import PySCF, validate the exact supported SCF family, require convergence, and confirm that SCF orbitals and the molecule are available.
2. Read the AO overlap, core Hamiltonian, and nuclear-repulsion energy through PySCF APIs; validate finite orthonormal orbital coefficients before two-electron work.
3. Use PySCF `ao2mo` APIs to transform electron-repulsion integrals. Do not implement the AO-to-MO contraction as Python loops or a new Rust chemistry kernel. RHF retains one shared spatial block; UHF retains only the nonzero alpha/alpha, alpha/beta, beta/alpha, and beta/beta spatial blocks rather than a zero-padded `(2n)^4` tensor.
4. Form the alpha and beta one-body blocks and pass the compact numerical blocks, ordering descriptor, constant, and memory budget through one coarse framework-neutral native construction call. Rust expands spin-mode indices and converts chemist axes to the fixed operator convention.
5. Return the native-backed `FermionOperator` without retaining the integral arrays or a parallel term dictionary.

The first API returns `FermionOperator` directly and does not attach chemistry metadata. A future metadata-bearing result type requires a separate reviewed API and must not be simulated by attaching mutable or lossy attributes to the generic operator.

### 5.3 Deferred active space and frozen core

Active-space and frozen-core support is deferred outside Phase 10 acceptance. When separately promoted, it must be implemented as an explicit transformation before canonical construction, updating the scalar constant, one-body tensor, and two-body tensor together rather than merely deleting inactive terms. Automatic chemical heuristics, localization, orbital ordering optimization, and correlated active-space solvers remain out of scope. The Phase 10 public PySCF API must not expose placeholder active-space options.

### 5.4 Correctness requirements

The PySCF adapter must include numerical tests for:

- a minimal closed-shell molecule with a known nuclear-repulsion constant;
- the one-body and two-body tensor values before and after spin expansion;
- Hermiticity and particle-number conservation;
- both spin-orbital orderings related by an explicit basis permutation;
- direct fermionic dense matrices against a trusted small-system construction;
- mapped Pauli matrices under Jordan–Wigner against the same fermionic reference;
- RHF and UHF cases, including a case where alpha and beta orbitals differ;
- the expectation of the complete imported Hamiltonian in the PySCF RHF or UHF determinant against `mf.e_tot`, including nuclear repulsion.

Tests must assert numerical coefficients or energies, not only mode counts, term counts, or laziness metadata.

The PySCF coverage belongs in `tests/test_pyscf_integration.py`. The normal test environment may skip tests requiring an installed PySCF runtime, but the supported `chemistry` optional-dependency environment must run them. The module must cover the lazy missing-PySCF error, convergence and unsupported-family failures before integral work, RHF H2 one- and two-body coefficients, the nuclear-repulsion identity term, FSWAP-correct ordering permutations, mapped Pauli numerical agreement, determinant-energy agreement, and a small UHF/open-shell case whose alpha and beta orbitals differ.

## 6. SciPy `LinearOperator` adapter

### 6.1 Public surface

The primary API is a method on every public native plan satisfying the common `MVPPlan` execution contract:

```python
plan = operator.compile("native_mvp")
linear_operator = plan.to_scipy_linear_operator(max_bytes=DEFAULT_MAX_BYTES)
```

For convenience, `PauliOperator` exposes the same explicit naming:

```python
linear_operator = pauli_operator.to_scipy_linear_operator(
    max_bytes=DEFAULT_MAX_BYTES
)
```

Fermion mapping remains a separate operation:

```python
fermion_h = from_scf(mf)
pauli_h = fermion_h.map_fermions("jordan_wigner")
linear_operator = pauli_h.to_scipy_linear_operator()
```

`PauliOperator.to_scipy_linear_operator()` compiles one ordinary native MVP plan internally and delegates to the same shared SciPy helper used by the plan methods. Neither the Pauli convenience method nor any plan method accepts a fermion-mapping argument or performs fermion-to-qubit mapping. `NativeMVPPlan`, `ChargeMvpPlan`, `ChargeLazyMvpPlan`, `U1MvpPlan`, and any other public native plan with the same `dimension` and `apply()` contract must not grow parallel adapter implementations. The wrapper returns `scipy.sparse.linalg.LinearOperator` with `shape=(plan.dimension, plan.dimension)` and `dtype=np.complex128`.

### 6.2 Execution semantics

The wrapper implements `matvec` by forwarding one vector to `plan.apply()` and never materializes a dense or sparse matrix. SciPy permits inputs with shape `(dimension,)` and `(dimension, 1)`; the callback accepts both, extracts the sole column as `x[:, 0]` for the latter, converts to contiguous `complex128` as needed, calls the plan with a one-dimensional vector, and lets the `LinearOperator` wrapper restore the public output shape. All other ranks and shapes fail. The callback captures the requested best-effort memory budget, and the native plan remains immutable and reusable across calls.

`matmat` uses SciPy's standard column-wise behavior initially; supporting `(dimension, 1)` in `matvec` is required for that default implementation. A custom multi-column native path is out of scope unless profiling shows that iterative workloads materially depend on it. `rmatvec` must not be advertised as the same operation unless the operator is known Hermitian. A later API may accept an explicit adjoint plan; the first wrapper exposes only mathematically justified callbacks.

The wrapper must preserve exceptions from invalid dimensions and native memory guards as useful Python exceptions. It must not catch native failures and silently fall back to a dense matrix. Repeated calls must reuse the same plan and must not rebuild native state.

### 6.3 SciPy compatibility tests

SciPy is required in the supported Python environment, so the adapter has one unconditional test path rather than a missing-dependency branch. Tests cover:

- `LinearOperator.shape` and `dtype`;
- `matvec` against `plan.apply()` for deterministic and random complex vectors in both `(dimension,)` and `(dimension, 1)` forms;
- default column-wise `matmat` behavior;
- a small dense matrix reference;
- repeated calls and plan reuse;
- invalid vector dimensions and memory-budget errors inherited from the plan;
- no dense or sparse matrix materialization during wrapper construction;
- `scipy.sparse.linalg.eigsh` on a small explicitly Hermitian Hamiltonian;
- the existing ED research examples after replacing their manual `LinearOperator` lambdas.

The test must also verify that the Pauli convenience method compiles once, all native-plan methods delegate to one adapter implementation, and no SciPy entry point accepts or performs a fermion mapping argument.

These cases belong in the unconditional module `tests/test_scipy_linear_operator.py` because SciPy is a required dependency. They must cover an ordinary Pauli native plan, both charge-restricted plan storage strategies, and a U1-restricted plan, in addition to the `PauliOperator` convenience method.

## 7. First adapter priorities

The first implementation batch contains exactly two public adapter capabilities:

1. **Standardized PySCF import.** `from_scf()` and `from_molecule()` return a native-backed `FermionOperator`. `FermionOperator.from_integrals()` and the compact PySCF block path share the same native integral-ingestion implementation. RHF is the first correctness slice and UHF is the immediate mandatory follow-on slice in the same PySCF module. Nuclear repulsion and explicit spin ordering are part of the first contract.
2. **SciPy native-MVP `LinearOperator`.** Every public native MVP plan exposes the reusable-plan method, including charge-restricted lazy/eager and U1 plans. `PauliOperator.to_scipy_linear_operator()` is the one-call convenience path that compiles once internally. No path performs fermion mapping.

FCIDUMP, Psi4, density-fitting factorized tensors, generic HDF5 chemistry formats, and framework-specific chemistry objects are deferred. If a later workload requires a file-level chemistry interchange, FCIDUMP is preferred to an OpenFermion bridge because it is a direct integral artifact and does not add another operator algebra dependency.

## 8. Performance and memory requirements

PySCF conversion may require dense spatial AO/MO tensors for the first implementation, but it must not allocate a zero-padded complete spin-orbital `(2n)^4` tensor. RHF passes one shared spatial two-electron block; UHF passes only its nonzero spin-pair spatial blocks. The adapter performs one coarse conversion into the native operator path and never creates a complete Python term dictionary merely to hand the data across the boundary.

The adapter and canonical constructor must apply checked shape and major-allocation estimates before requesting the largest predictable MO blocks, spin expansion, or native term buffers. PySCF's internal AO-to-MO workspace remains outside exact accounting. `max_bytes` is a best-effort guard rather than an exact peak-RSS guarantee. For realistic larger molecules, density fitting or low-rank two-body ingestion may later become necessary; Phase 10 records that as a future representation decision rather than adding an untested factorized branch.

The SciPy wrapper must have negligible setup cost relative to plan compilation and must not alter steady-state native MVP performance beyond the unavoidable Python/SciPy callback boundary. Benchmarks must separate plan construction, first `matvec`, steady `matvec`, and dense/sparse materialization baselines.

## 9. Documentation and examples

Update the existing ED research examples rather than adding a separate SciPy demonstration. In particular, replace the hand-written `scipy.sparse.linalg.LinearOperator` construction in:

- `examples/research/fermi_hubbard/run_tencirpauli.py`;
- `examples/research/syk_majorana/run_tencirpauli.py`.

Each must retain its existing `eigsh` workflow and call `plan.to_scipy_linear_operator()` so the example exercises the public adapter without changing the scientific workload. In particular, the Fermi–Hubbard example exercises the charge-restricted lazy-plan method rather than bypassing it through `restricted.apply`.

Add one optional PySCF chemistry example, such as `examples/quantum_chemistry_pyscf.py`, covering:

- PySCF H2 construction and RHF import through `from_scf()`;
- explicit `FermionOperator` to `PauliOperator` mapping;
- Pauli `backend_mvp` compilation and execution through the TensorCircuit backend;
- a small fixed-parameter TensorCircuit VQE-style energy evaluation using the backend MVP plan, with the resulting energy checked against a dense reference; adding an optimizer is not required.

The chemistry example is optional because it requires PySCF, but when run it must use the normal installed TensorCircuit runtime and must not introduce OpenFermion. Public documentation must state that SciPy is required, PySCF is optional, OpenFermion is not required, standard fermionic CAR requires orthonormal orbitals, and spin-orbital ordering and integral convention are part of the input contract.

`examples/README.md` must list the chemistry example separately from the default examples and identify its PySCF requirement. The existing ED research README entries remain the references for the migrated SciPy wrapper examples.

## 10. Acceptance gates

Phase 10 is complete only when:

- SciPy is declared once as a required Python dependency, its former examples-extra entry and obsolete sparse-conversion missing-dependency branches are removed, and the SciPy adapter has no missing-dependency fallback;
- the approved PySCF API imports without PySCF installed and fails clearly only when called;
- `FermionOperator.from_integrals()` accepts only the fixed spin-orbital contract, applies the one-half prefactor exactly once, normalizes permitted Hermitian roundoff, and has independent numerical coverage for one- and two-body coefficients;
- RHF produces numerically correct one-body, two-body, constant, fermionic, and mapped Pauli results, and the UHF follow-on slice passes the corresponding separate-alpha/beta checks;
- the public and PySCF integral paths share one batched native ingestion implementation, do not use per-term Python-to-native calls, and keep O(term-count) construction inside the GIL-released Rust section;
- RHF does not materialize a complete spin-orbital two-body tensor, and UHF materializes only nonzero spin-pair spatial blocks before native ingestion;
- both spin-orbital orderings have deterministic differential coverage;
- every public native MVP plan, including charge-restricted lazy/eager and U1 plans, and `PauliOperator.to_scipy_linear_operator()` return a working `LinearOperator` without matrix materialization;
- SciPy `matvec` accepts both `(dimension,)` and `(dimension, 1)` and default `matmat` works through the shared adapter;
- the SciPy convenience method accepts only a `PauliOperator` and does not accept a mapping argument;
- both existing ED research examples use the new SciPy wrapper while preserving their `eigsh` numerical results;
- the optional PySCF chemistry example reaches Pauli compilation and a TensorCircuit backend-MVP/VQE energy evaluation;
- Black, Ruff, strict mypy, Rust quality checks, release native build, and the relevant Python tests pass;
- benchmarks record PySCF transformation separately from native ingestion, complete conversion, first execution, steady MVP, peak memory, and numerical accuracy on representative small and medium workloads.

## 11. Implementation slices

Implement as the following bounded slices. Chemistry slices 1–3 are ordered; the SciPy slice 4 is independent and may land alongside them. Each slice lands with its own numerical tests and leaves no abandoned parallel representation:

1. Add the framework-neutral native integral-ingestion primitive, the flat NumPy/block PyO3 ABI, and `FermionOperator.from_integrals()` with independent dense fixtures.
2. Add RHF `from_scf()` and `from_molecule()`, including lazy dependency behavior, orthonormality checks, compact spatial-block ingestion, determinant-energy agreement, and the optional H2 example through the mapped Pauli result.
3. Add UHF ingestion with separate alpha/beta orbitals and all nonzero spin-pair blocks, then close the open-shell numerical and ordering tests. Phase 10 chemistry acceptance is not complete after RHF alone.
4. Add the single shared SciPy helper, delegate every native-plan method and the Pauli convenience method to it, and migrate both ED research examples.
5. Complete documentation, TensorCircuit backend-MVP/VQE-style example coverage, synchronized release benchmarks, and the full repository quality gate.
