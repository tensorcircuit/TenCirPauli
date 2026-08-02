# Vibe Design Index

TenCirPauli is an experimental, vibe-coded project. Working specifications, architecture decisions, implementation plans, acceptance gates, and operational design notes live in this directory so that the repository root remains a stable open-source project entry point.

## Documents

- [Architecture and roadmap](architecture.md): scope, module boundaries, algorithms, differentiation strategy, risks, benchmarks, and go/no-go gates.
- [Core semantics](semantics.md): Pauli representation, phase, ordering, coefficients, canonicalization, matrix conventions, and owner decisions that must be frozen before autonomous implementation.
- [P0 reference vectors](reference-vectors.md): the independent NumPy dense oracle, fixed regression vectors, random seed, dtype, and tolerances.
- [Phase 1 implementation specification](phase-1-spec.md): bounded milestones, required deliverables, non-goals, acceptance gates, and completion checklist for the first implementation goal.
- [Phase 2 symmetry/sector Spike](phase-2-spec.md): background, concrete Python API, Z2 analysis and tapering, explicit U1 sector reduction, implementation slices, and end-to-end acceptance semantics.
- [Phase 3 Rust-native propagation specification](phase-3-spec.md): frozen propagation semantics, GateTape and Python API, real PTM contract, dynamic Rust implementation plan, performance requirements, benchmarks, milestones, and handoff boundaries.
- [Phase 4 frozen-support reverse and SPPS specification](phase-4-spec.md): frozen deterministic sparse-trace reverse semantics, SPPS sampling/PAD contracts, public APIs, checkpointing, seeded parallelism, benchmarks, milestones, and integration boundaries.
- [Phase 5 arbitrary-width packed U1 specification](phase-5-spec.md): multiword occupation representation, bounded restricted indices, combinatorial rank/lookup, aggregated leakage validation, MVP/CSR construction, wide-system tests, benchmarks, and implementation slices.
- [Multiple-observable Pauli propagation specification](multi-observable-propagation-spec.md): proposed batched deterministic propagation for multiple observables sharing one GateTape, with shared schedules, operator-level parallelism, Clifford-frame optimization, deterministic aggregation, and acceptance benchmarks.
- [Implementation status](implementation-status.md): durable progress, verification evidence, decisions, blockers, and the next milestone for long-running agents.
- [Local benchmarking](benchmarking.md): microbenchmarks, integration benchmarks, local result history, and manual regression comparison.
- [Phase 1 review notes](phase-1-review-notes.md): implementation gaps, performance findings, and owner decisions awaiting review.
- [Phase 1 acceptance review, 2026-08-01](phase-1-acceptance-review-2026-08-01.md): archived initial acceptance findings, blocker remediation, and final local verification evidence.
- [Phase 2 acceptance review, 2026-08-02](phase-2-acceptance-review-2026-08-02.md): archived review findings, owner scope decisions, row-sign blocker, remediation handoff, and final acceptance evidence.
- [Phase 3 implementation review, 2026-08-02](phase-3-review-2026-08-02.md): archived review findings, focused remediation outcome, verification evidence, and explicitly deferred performance work.
- [Phase 4 implementation and performance review, 2026-08-02](phase-4-review-2026-08-02.md): focused acceptance audit of correctness evidence, SPPS scaling, memory guards, public estimator metadata, and representative performance.
- [Release process](releasing.md): the separation between continuous integration, artifact builds, GitHub Releases, and PyPI publication.

Add new design documents here with descriptive lowercase names. Update this index when adding, replacing, or retiring a document, and record the document status near its title when it is a proposal rather than an implemented contract.
