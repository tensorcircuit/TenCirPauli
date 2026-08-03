# Vibe Design Index

TenCirPauli is an experimental, vibe-coded project. Working specifications, architecture decisions, implementation plans, acceptance gates, and operational design notes live in this directory so that the repository root remains a stable open-source project entry point.

## Documents

- [Phase Alpha Python facade specification](phase-alpha-spec.md): implemented unified user-facing circuit contract for `U1Circuit`, `PropagationCircuit`, `SPPSCircuit`, parameter expressions, TensorCircuit conversion, and backend MVP execution; native executors remain independent.
- [Phase Alpha review, 2026-08-03](phase-alpha-review-2026-08-03.md): archived review findings, remediation scope, deferred items, and the local disposition record.
- [Pre-release review, 2026-08-03](pre-release-review-2026-08-03.md): archived open-source and PyPI release-readiness audit, verification evidence, blockers, and manual release actions.
- [Architecture and roadmap](architecture.md): scope, module boundaries, algorithms, differentiation strategy, risks, benchmarks, and go/no-go gates.
- [Core semantics](semantics.md): Pauli representation, phase, ordering, coefficients, canonicalization, matrix conventions, and owner decisions that must be frozen before autonomous implementation.
- [P0 reference vectors](reference-vectors.md): the independent NumPy dense oracle, fixed regression vectors, random seed, dtype, and tolerances.
- [Phase 1 implementation specification](phase-1-spec.md): bounded milestones, required deliverables, non-goals, acceptance gates, and completion checklist for the first implementation goal.
- [Phase 2 symmetry/sector Spike](phase-2-spec.md): background, concrete Python API, Z2 analysis and tapering, explicit U1 sector reduction, implementation slices, and end-to-end acceptance semantics.
- [Phase 3 Rust-native propagation specification](phase-3-spec.md): frozen propagation semantics, GateTape and Python API, real PTM contract, dynamic Rust implementation plan, performance requirements, benchmarks, milestones, and handoff boundaries.
- [Phase 4 frozen-support reverse and SPPS specification](phase-4-spec.md): frozen deterministic sparse-trace reverse semantics, SPPS sampling/PAD contracts, public APIs, checkpointing, seeded parallelism, benchmarks, milestones, and integration boundaries.
- [Phase 5 arbitrary-width packed U1 specification](phase-5-spec.md): multiword occupation representation, bounded restricted indices, combinatorial rank/lookup, aggregated leakage validation, MVP/CSR construction, wide-system tests, benchmarks, and implementation slices.
- [Phase 5.5 multiple-observable propagation specification](phase-5.5-spec.md): optional batched deterministic expectations and row-wise frozen-support gradients using a shared compiled program and observable-level parallelism, while preserving the existing Pauli-sum engine.
- [Phase 6 common circuit IR and U1Circuit specification](phase-6-spec.md): frozen implementation contract for a backend-neutral gate/parameter/serialization layer plus a TensorCircuit-compatible lazy facade, fused Rust-native restricted-state execution, bounded dense-state terminals, projected observables, arbitrary-width pair maps, and exact adjoint gradients; no generic full-state simulator is included.
- [Phase 6 implementation review, 2026-08-02](phase-6-review-2026-08-02.md): archived remediation findings and the performance/correctness handoff that drives the current checkpoint.
- [Deferred Phase 6.5 generic matrix-free time-evolution proposal](phase-6.5-spec.md): inactive research proposal for possible all-Rust Taylor, Hermitian Krylov/Lanczos, Chebyshev evolution and native observable reducers; it requires a new owner decision and dependency/accuracy spike before implementation.
- [Phase 7 structured Hamiltonian algebra and compilation specification](phase-7-spec.md): frozen implementation contract for fermion, symbolic boson, native mixed-dimension hybrid, direct-Weyl, and Pauli-compatible construction and finite target compilation; the first vertical implementation slice is under acceptance review.
- [Phase 7 implementation review, 2026-08-03](phase-7-review-2026-08-03.md): archived correctness, target-availability, finite-numeric, matrix-free-plan, performance, and P0–P5 delivery findings.
- [Phase 7 second-round remediation review, 2026-08-03](phase-7-second-round-review-2026-08-03.md): adversarial follow-up findings for hybrid operand order, CAR expansion guards, checked Weyl dimensions, correctness coverage, plan metadata, and release handoff.
- [Phase 7 third-round acceptance review, 2026-08-03](phase-7-third-round-review-2026-08-03.md): narrowed final triage accepting the CAR and benchmark handoffs while retaining the independent Holstein/spin-boson differential and public plan-metadata correction as the two remaining gates.
- [Phase 7.5 Majorana, fermion-mapping, and additive-charge specification](phase-7.5-spec.md): frozen implementation contract for public Majorana algebra, reusable JW/parity/BK mappings, exact integer additive symmetries, simultaneous charge sectors, zero-charge qudit spectators, and guarded restricted Hamiltonian spaces.
- [Feature incubator](feature-incubator.md): living ledger for deferred or immature feature ideas, design fragments, unresolved questions, promotion criteria, and links to ideas that later enter formal phase specifications.
- [Implementation status](implementation-status.md): durable progress, verification evidence, decisions, blockers, and the next milestone for long-running agents.
- [Local benchmarking](benchmarking.md): microbenchmarks, integration benchmarks, local result history, and manual regression comparison.
- [Phase 1 review notes](phase-1-review-notes.md): implementation gaps, performance findings, and owner decisions awaiting review.
- [Phase 1 acceptance review, 2026-08-01](phase-1-acceptance-review-2026-08-01.md): archived initial acceptance findings, blocker remediation, and final local verification evidence.
- [Phase 2 acceptance review, 2026-08-02](phase-2-acceptance-review-2026-08-02.md): archived review findings, owner scope decisions, row-sign blocker, remediation handoff, and final acceptance evidence.
- [Phase 3 implementation review, 2026-08-02](phase-3-review-2026-08-02.md): archived review findings, focused remediation outcome, verification evidence, and explicitly deferred performance work.
- [Phase 4 implementation and performance review, 2026-08-02](phase-4-review-2026-08-02.md): focused acceptance audit of correctness evidence, SPPS scaling, memory guards, public estimator metadata, and representative performance.
- [Release process](releasing.md): the separation between continuous integration, artifact builds, GitHub Releases, and PyPI publication.

Add new design documents here with descriptive lowercase names. Update this index when adding, replacing, or retiring a document, and record the document status near its title when it is a proposal rather than an implemented contract.
