# Changelog

All notable changes to TenCirPauli will be documented in this file.

The project follows Semantic Versioning once the public API reaches its first tagged release.

## Unreleased

- Close the Phase 1 acceptance blockers: make caller-owned MVP output overwrite-safe, enforce canonical packed words and finite/nonzero operator invariants, remove per-term PyO3 round trips and repeated canonicalization, release the GIL on long native paths, use zero-copy shared complex buffers, bound grouping/backend allocations, and accelerate deterministic canonicalization and QWC grouping. Add contiguous NumPy canonicalization APIs and matched Python performance baselines.
- Optimize Rust matrix targets with precomputed packed phase masks, deterministic X-mask grouped COO aggregation, row-parallel sparse generation, direct singleton-X-mask COO/CSR output, direct CSR construction, parallel native MVP, and a reusable zero-copy NumPy/PyO3 native MVP plan. Add typed batch canonicalization mapping and exact phase metadata.
- Add release-mode cross-implementation sparse benchmarks for TensorCircuit NumPy COO and JAX BCOO first/warm construction, duplicate canonicalization, storage, and warm matvec semantics.
- Add synchronized 20-qubit COO/CSR/native-MVP/JAX-MVP benchmarks, explicit materialization memory-guard cases, and asynchronous JAX timing safeguards.
- Add a paired Python-to-sparse canonical benchmark for local Heisenberg chains, including JAX BCOO duplicate aggregation and in-call synchronization so native and TensorCircuit endpoint timings use the same contract.
- Set the public materialization safety budget to 4 GiB by default, expose `DEFAULT_MAX_BYTES`, and add local Heisenberg nearest/next-nearest chain benchmarks with explicit larger-budget coverage.
- Complete the Phase 1 PauliWord, PauliOperator, deterministic grouping, and Hamiltonian compiler vertical slices.
- Add an independent NumPy dense reference, fixed regression vectors, Rust/PyO3 batch conversion, and deterministic release-mode benchmarks.
- Add dense/COO/CSR/native MVP/backend MVP targets with explicit allocation guards and a lazy optional TensorCircuit adapter.
- Refresh the public README, typing surface, examples, and implementation-status evidence.
