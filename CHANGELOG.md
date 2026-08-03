# Changelog

All notable changes to TenCirPauli will be documented in this file.

The project follows Semantic Versioning.

## 0.1.0 - 2026-08-03

- Initial alpha release of Rust-native Pauli algebra and a TensorCircuit-facing Python API for Pauli operators, Hamiltonians, measurement grouping, symmetry reduction, U(1) circuits, deterministic propagation, and stochastic Pauli-path estimation.
- Phase 7 review remediation: preserve partially mapped hybrid Pauli factors, validate explicit embeddings, use checked CCR expansion and overflow-safe finite-boson amplitudes, provide matrix-free structured native MVP plans, and add direct uniform-qudit TensorCircuit backend MVP execution.
- Phase 7 second-round remediation: preserve raw/mapped fermion operand order, add global Jordan–Wigner graded tensor handling, use structure-aware CAR fast paths with running guards, cache checked finite dimensions, and publish direct-Weyl plan metadata and multi-backend regression coverage.
- Phase 7.5 initial slice: add exact Majorana algebra and fermion conversion, reusable Jordan–Wigner/parity/Bravyi–Kitaev occupation plans, integer additive-charge analysis, simultaneous finite sectors, qudit spectators, restricted targets, and a native restricted MVP kernel.
- Phase 7.5 P4/P5 remediation: compile restricted transitions in the pure Rust core with aggregate-before-leakage semantics, add cancellation and memory-boundary regressions, and extend release benchmarks across restricted targets and the existing U1 reference.
- Phase 7.5 P1 completion: move Majorana canonicalization and multiplication into pure Rust with one coarse PyO3 batch boundary and retain the independent Fock-space differential.
