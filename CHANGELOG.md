# Changelog

All notable changes to TenCirPauli will be documented in this file.

The project follows Semantic Versioning.

## Unreleased

### Fixed

- Fixed logical-witness search panicking for valid codes without stabilizer generators in the automatic, X, and Z sectors, including codes spanning multiple packed words.

## 0.5.0 - 2026-08-21

This release adds native classical-shadow snapshots and makes randomized-measurement estimation a first-class public workflow.

### Added

- Added `Snapshots.sample()` for local-Pauli and global-Clifford sampling from complete statevectors or native `StabilizerState` inputs, with deterministic seeded replay and setting-level memory-bounded execution.
- Added snapshot import and read-back through `Snapshots.from_pauli_measurements()`, `Snapshots.from_clifford_measurements()`, and `Snapshots.to_arrays()` with explicit shapes, signed packed frames, and `q0_first` ordering.
- Added native estimators for single and batched Pauli observables, Hamiltonian energy, reduced density matrices, Rényi-2 entropy, energy variance, stabilizer-target fidelity, arbitrary dense-target fidelity, and `shadow_bound()`.
- Added immutable stabilizer-state construction from zero states, generators, tableaux, and supported Clifford tapes.
- Added a reproducible classical-shadow research workflow and release benchmark coverage for dense statevector, stabilizer, local-Pauli, global-Clifford, estimator, RDM, entropy, and fidelity workloads.

### Documentation

- Documented the classical-shadow public surface in the README, quickstart, concepts, API, performance, and architecture documentation.

## 0.4.1 - 2026-08-11

This patch release improves backend compatibility and tightens native structured-operator execution while the project remains pre-1.0.

### Fixed

- Fixed TensorCircuit and JAX integration to preserve the active `complex64` or `complex128` state dtype, including JIT-compiled backend MVP and value-and-gradient paths.
- Fixed repeated backend MVP calls to reuse static coefficient and phase conversions without freezing caller-provided backend tensors.

### Changed

- Improved structured eager CSR execution for large transition graphs with thresholded native parallel application while retaining serial behavior for smaller workloads.
- Tightened native charge, symmetry, Pauli, and structured-plan metadata, typing, budget checks, and numerical regression coverage.

### Documentation

- Added reproducible Hubbard time-evolution, measurement-grouping, and Pauli-propagation research workflows and expanded the performance documentation.

## 0.4.0 - 2026-08-10

This release extends structured Hamiltonian execution and adds chemistry and scientific-computing integrations while the project remains pre-1.0.

### Added

- Added SciPy `LinearOperator` adapters for Pauli operators and native MVP plans, allowing iterative solvers to use matrix-free execution.
- Added optional PySCF RHF/UHF and molecular Hamiltonian ingestion through `tencirpauli.integrations.pyscf`, producing canonical `FermionOperator` instances with explicit orbital and integral conventions.
- Added native raw-Fermion `compile("native_mvp")` execution in occupation-basis order while retaining explicit Jordan–Wigner, parity, and Bravyi–Kitaev mapping paths.
- Added factorized native execution for separable additive-charge constraints; genuinely coupled constraints continue to use the general restricted execution path.

### Changed

- SciPy is now a required runtime dependency for the public iterative-solver and `LinearOperator` integration paths; PySCF, JAX, and Optax are available through the `chemistry` extra.
- Expanded and optimized native lazy/eager MVP execution for structured and charge-restricted operators, including reusable eager sparse representations and direct native sparse-data paths.
- Improved raw-Fermion, charge-sector, and structured-operator execution and memory behavior without changing the public circuit boundary introduced in 0.3.0.

## 0.3.0 - 2026-08-06

This release intentionally changes the public circuit boundary while the project is still pre-1.0. The API remains subject to change while the project has no stable user base.

### Breaking changes

- Circuit facades now accept concrete gate angles through `theta=`. The public `Parameter` and `ParameterExpr` APIs, parameter binding/remapping helpers, circuit `compile()` methods, and public circuit-plan types have been removed.
- `U1Circuit`, `PropagationCircuit`, and `SPPSCircuit` now share observable-first `expectation()`, `value_and_grad()`, and `expectation_jax()` terminals. Direct gradients are indexed by gate occurrence order; outer parameter sharing and arithmetic belong to JAX or the caller.
- QIR and TensorCircuit conversion use concrete numeric angles. Symbol discovery and symbolic circuit round-tripping are no longer part of the public contract. Low-level `GateTape`/engine numerical slots remain available under `tencirpauli.advanced` for specialized code.
- The ordinary and advanced namespaces were narrowed: circuit facades remain in `tencirpauli`, while stability-sensitive native plans and low-level engines remain under `tencirpauli.advanced`. Operator `compile(target=...)` is unchanged.

### Added

- First-order JAX value-and-gradient terminals for deterministic propagation, U(1) circuits, and fixed-budget SPPS through one coarse native callback, with immutable traced circuit snapshots and explicit x64 requirements.
- Native-resident lazy Pauli and structured algebra results, flat NumPy/native-handle data paths, native grouping and charge analysis, reusable gate-tape compilation, and direct complex128 terminal outputs.
- Release-mode evidence and executable research workflows covering structured algebra, charge-restricted Hamiltonians, Majorana mappings, circuit differentiation, and repeated MVP execution.

### Changed

- CPU-native MVP plans default to lazy storage, while explicit eager storage and `apply_into()` provide reusable execution with best-effort memory guards.
- Structured fermion, boson, qudit, hybrid, Majorana, mapping, additive-charge, and restricted-sector workflows now follow the same native-backed operator and finite-target contracts.
- Required runtime dependency and supported integration boundary remain `tensorcircuit-ng>=1.8,<2`; the public package now targets CPython 3.10+ and the existing wheel platforms.

## 0.2.0 - 2026-08-04

- Added structured algebra workflows for fermion/boson/Weyl operators, Majorana conversion, Jordan–Wigner/parity/Bravyi–Kitaev mappings, additive-charge sectors, restricted operators, and native matrix-free plans.
- Improved API coherence with capability-separated circuit facades, canonical term metadata, unified validation, stable U1 state/basis APIs, factory-only mappings, ordinary/advanced exports, and flat MVP execution contracts.
- Added Rust-native propagation metadata, exact-Hermiticity caching, advanced API documentation, and contract regressions covering public names, error timing, ownership, and conversion boundaries.

## 0.1.0 - 2026-08-03

- Initial alpha release of Rust-native Pauli algebra and a TensorCircuit-facing Python API for Pauli operators, Hamiltonians, measurement grouping, symmetry reduction, U(1) circuits, deterministic propagation, and stochastic Pauli-path estimation.
