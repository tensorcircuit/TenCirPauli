# Changelog

All notable changes to TenCirPauli will be documented in this file.

The project follows Semantic Versioning once the public API reaches its first tagged release.

## Unreleased

- Optimize Rust matrix targets with precomputed packed phase masks, deterministic X-mask grouped COO aggregation, direct CSR construction, parallel native MVP, and a reusable zero-copy NumPy/PyO3 native MVP plan. Add typed batch canonicalization mapping and exact phase metadata.
- Add release-mode cross-implementation sparse benchmarks for TensorCircuit NumPy COO and JAX BCOO first/warm construction, duplicate canonicalization, storage, and warm matvec semantics.
- Complete the Phase 1 PauliWord, PauliOperator, deterministic grouping, and Hamiltonian compiler vertical slices.
- Add an independent NumPy dense reference, fixed regression vectors, Rust/PyO3 batch conversion, and deterministic release-mode benchmarks.
- Add dense/COO/CSR/native MVP/backend MVP targets with explicit allocation guards and a lazy optional TensorCircuit adapter.
- Refresh the public README, typing surface, examples, and implementation-status evidence.
