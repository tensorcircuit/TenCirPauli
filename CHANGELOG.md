# Changelog

All notable changes to TenCirPauli will be documented in this file.

The project follows Semantic Versioning once the public API reaches its first tagged release.

## Unreleased

- Add `PropagationBatch` for deterministic multi-observable expectations and row-wise frozen-support gradients with shared compiled programs and observable-level parallelism.
- Implement Phase 5 arbitrary-width packed-`u64` U1 restricted sectors, including checked combinatorial rank/unrank, wide native basis materialization, aggregated leakage validation, deterministic restricted MVP/COO/CSR plans, and 63/64/65, 127/128/129, and 256-qubit coverage.
- Add Phase 4 Rust-native frozen-support reverse gradients with analytic local VJPs, shared parameter slots, checkpoint replay, static PTM transpose action, typed Python results, and deterministic differential tests.
- Add the independent Rust-native SPPS engine with smoothed importance sampling, stable prefix/suffix PAD, fixed and adaptive A/B budgets, counter-derived seeded replay, streamed fixed-chunk Rayon batching, typed estimates, and explicit unsupported-gate validation.
- Add Phase 4 release Criterion/Python workloads and optional end-to-end comparisons against the TensorCircuit `spps_pauli_path_vqe.py` example and `PauliPropagationEngine` plus JAX `value_and_grad`.
- Add the optional TensorCircuit QIR/SymbolCircuit-to-`GateTape` adapter for supported numeric and direct-symbol Clifford/Pauli-rotation circuits.
- Add the Rust-native `GateTape`/`PropagationEngine` with exact and per-gate Pauli-weight-projected Heisenberg propagation, supported Clifford and Pauli-rotation gates, finite real custom PTMs, product-state expectations, explicit operator materialization, typed parameter slots, and profile metadata.
- Add the independent Phase 3 dense propagation reference, differential/boundary tests, 100-qubit packed-key coverage, release Criterion/Python workloads, and synchronized complex128 JAX warm-reference comparison.
- Implement Phase 2 Pauli Z2 symmetry analysis, reusable Clifford tapering plans, explicit U(1) fixed-particle sectors, restricted MVP/CSR operators, public typed APIs, and deterministic setup/apply benchmark workloads.
- Add symmetry-aware JAX reduced-space baselines with matched U(1) transition aggregation and Z2 tapered TFIM semantics, including setup, steady-state, first-JIT, and end-to-end measurements at larger qubit counts.
- Close the Phase 1 acceptance blockers: make caller-owned MVP output overwrite-safe, enforce canonical packed words and finite/nonzero operator invariants, remove per-term PyO3 round trips and repeated canonicalization, release the GIL on long native paths, use zero-copy shared complex buffers, bound grouping/backend allocations, and accelerate deterministic canonicalization and QWC grouping. Add contiguous NumPy canonicalization APIs and matched Python performance baselines.
- Optimize Rust matrix targets with precomputed packed phase masks, deterministic X-mask grouped COO aggregation, row-parallel sparse generation, direct singleton-X-mask COO/CSR output, direct CSR construction, parallel native MVP, and a reusable zero-copy NumPy/PyO3 native MVP plan. Add typed batch canonicalization mapping and exact phase metadata.
- Add release-mode cross-implementation sparse benchmarks for TensorCircuit NumPy COO and JAX BCOO first/warm construction, duplicate canonicalization, storage, and warm matvec semantics.
- Add synchronized 20-qubit COO/CSR/native-MVP/JAX-MVP benchmarks, explicit materialization memory-guard cases, and asynchronous JAX timing safeguards.
- Add a paired Python-to-sparse canonical benchmark for local Heisenberg chains, including JAX BCOO duplicate aggregation and in-call synchronization so native and TensorCircuit endpoint timings use the same contract.
- Set the public materialization safety budget to 16 GiB by default, expose `DEFAULT_MAX_BYTES`, accept `None` for an unbounded best-effort guard, and add local Heisenberg nearest/next-nearest chain benchmarks with explicit larger-budget coverage.
- Complete the Phase 1 PauliWord, PauliOperator, deterministic grouping, and Hamiltonian compiler vertical slices.
- Add an independent NumPy dense reference, fixed regression vectors, Rust/PyO3 batch conversion, and deterministic release-mode benchmarks.
- Add dense/COO/CSR/native MVP/backend MVP targets with explicit allocation guards and a lazy optional TensorCircuit adapter.
- Refresh the public README, typing surface, examples, and implementation-status evidence.
