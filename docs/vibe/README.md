# Vibe Design Index

TenCirPauli is an experimental, vibe-coded project. Working specifications, architecture decisions, implementation plans, acceptance gates, and operational design notes live in this directory so that the repository root remains a stable open-source project entry point.

## Documents

- [Architecture and roadmap](architecture.md): scope, module boundaries, algorithms, differentiation strategy, risks, benchmarks, and go/no-go gates.
- [Core semantics](semantics.md): Pauli representation, phase, ordering, coefficients, canonicalization, matrix conventions, and owner decisions that must be frozen before autonomous implementation.
- [P0 reference vectors](reference-vectors.md): the independent NumPy dense oracle, fixed regression vectors, random seed, dtype, and tolerances.
- [Phase 1 implementation specification](phase-1-spec.md): bounded milestones, required deliverables, non-goals, acceptance gates, and completion checklist for the first implementation goal.
- [Implementation status](implementation-status.md): durable progress, verification evidence, decisions, blockers, and the next milestone for long-running agents.
- [Local benchmarking](benchmarking.md): microbenchmarks, integration benchmarks, local result history, and manual regression comparison.
- [Phase 1 review notes](phase-1-review-notes.md): implementation gaps, performance findings, and owner decisions awaiting review.
- [Release process](releasing.md): the separation between continuous integration, artifact builds, GitHub Releases, and PyPI publication.

Add new design documents here with descriptive lowercase names. Update this index when adding, replacing, or retiring a document, and record the document status near its title when it is a proposal rather than an implemented contract.
