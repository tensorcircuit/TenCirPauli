# Changelog

All notable changes to TenCirPauli will be documented in this file.

The project follows Semantic Versioning.

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
- Required runtime dependency and supported integration boundary remain `tensorcircuit-ng>=1.8,<2`; the public package still targets CPython 3.9+ and the existing wheel platforms.

## 0.2.0 - 2026-08-04

- Added structured algebra workflows for fermion/boson/Weyl operators, Majorana conversion, Jordan–Wigner/parity/Bravyi–Kitaev mappings, additive-charge sectors, restricted operators, and native matrix-free plans.
- Improved API coherence with capability-separated circuit facades, canonical term metadata, unified validation, stable U1 state/basis APIs, factory-only mappings, ordinary/advanced exports, and flat MVP execution contracts.
- Added Rust-native propagation metadata, exact-Hermiticity caching, advanced API documentation, and contract regressions covering public names, error timing, ownership, and conversion boundaries.

## 0.1.0 - 2026-08-03

- Initial alpha release of Rust-native Pauli algebra and a TensorCircuit-facing Python API for Pauli operators, Hamiltonians, measurement grouping, symmetry reduction, U(1) circuits, deterministic propagation, and stochastic Pauli-path estimation.
