# Phase 7 third-round acceptance review

Review date: 2026-08-03

Reviewed range: `2b0f0dc..c1f1375`, principally remediation commit `b5bcc0c` and the clean benchmark handoff recorded by `c1f1375`.

Scope: final triage of the concerns in `phase-7-second-round-review-2026-08-03.md` after implementation review, targeted adversarial probes, the complete local quality gate, and owner disposition of performance-evidence scope. No implementation, test, benchmark, specification, or status source file was changed during this review; only this archived report and the `docs/vibe/README.md` index were updated.

## Verdict

The Phase 7 computational core has no reproduced release-blocking algebra, mapping, finite-target, or dimension-overflow defect. C1, M1, M2, M4, and N1 from the second-round review are accepted as addressed. Two acceptance items remain: one missing independent Holstein/spin-boson correctness differential, and one actual public-plan metadata defect. Phase 7 should remain under acceptance review until these two items are fixed and verified.

## Closure note (2026-08-04)

The subsequent remediation and Phase 8 contract pass closed the two remaining Phase 7 items. This archived review records the earlier checkpoint; current status is acceptance-closed for the 0.2 release.

## Compliance checklist

| Second-round concern | Result | Disposition |
| --- | --- | --- |
| C1 raw/mapped operand order | PASS | Eager mapping preserves operand order; both-order, nested, commutator, adjoint, and graded tensor paths pass. |
| M1 useful CAR expansion guard | PASS | The false exponential preflight rejection is removed; canonical, inversion-only, nilpotent, contraction-heavy, and bounded-guard regressions pass. The documented same-machine three-variant A/B is accepted as sufficient evidence; retaining a standalone reproduction driver is not a Phase 7 blocker. |
| M2 checked direct-Weyl dimensions | PASS | Plan dimensions use checked Python-integer products, are cached, and reject `3**50` before exponent-array allocation. |
| M3 correctness/property matrix | FAIL | The focused matrix is substantially complete, but still lacks an independent Holstein or spin-boson differential. |
| M4 release benchmark handoff | PASS | The clean `phase7-second-round-remediation-v5-20260803` record is accepted. Finer provenance for contribution/thread counters is informational and not a Phase 7 blocker. |
| M5 plan metadata, typing, and docstrings | FAIL | Typing, docstrings, schema fields, and direct-Weyl operation metadata are improved, but structured native basis metadata remains semantically wrong. |
| N1 dead finite-transition code and stale labels | PASS | Duplicate runtime transition code was removed and benchmark labels were corrected. |
| Repository quality gate | PASS | Formatting, Clippy, Ruff, strict mypy, release build/install, 31 Rust tests, 230 Python tests, and 165 benchmark-smoke cases pass. |

## CRITICAL

No remaining CRITICAL issue was found.

## MAJOR

### M1. Add an independent Holstein or spin-boson correctness differential

Locations: `tests/test_structured_algebra.py:111-777` and `benchmarks/python/test_structured_algebra_benchmark.py:552-605`; contract: `phase-7-spec.md:440-455,511-515`.

The current Holstein benchmark compares repeated applications of the same native plan. It verifies replay and performance but cannot detect a shared Jordan-Wigner, mixed-radix, projected-boson-boundary, or coefficient error. Add a small one- or two-site Holstein or spin-boson fixture whose reference matrix is assembled independently from explicit fermion matrices, projected boson ladder matrices, Pauli matrices where applicable, and Kronecker products. Compare the independent reference with dense, COO, CSR, and native MVP results.

### M2. Correct structured-plan basis and domain-count metadata

Locations: `python/tencirpauli/hamiltonian.py:95-178` and `python/tencirpauli/structured.py:2527-2544,2632-2652`; contract: `phase-7-spec.md:395-399,426-438`.

Boson, native-Weyl, and mixed-radix hybrid `NativeMVPPlan` values currently inherit `basis_ordering="qubit0_msb_matrix"`, although their actual basis is the ordered `OperatorSpace` mixed-radix basis. The direct-Weyl backend factory also stores the qudit-site count in Pauli-specific `nqubits` and `word_count` fields. These values do not currently alter numerical execution because the executors dispatch through `plan_kind` and `local_dimensions`, but they are incorrect public API metadata and can mislead downstream plan consumers.

Define stable basis-ordering labels for structured native plans, pass the correct label during compilation, and keep Pauli-only count fields neutral or explicitly inapplicable for direct-Weyl plans. Add exact metadata assertions for Pauli, mapped fermion, boson, hybrid, native Weyl, and backend Weyl plans.

## MINOR

No remaining MINOR issue was found.

## OBSERVATIONS

- The original raw-left/mapped-right C1 reproducer now has zero dense error. Broader nested early-mapping and graded tensor-product probes also had zero error.
- Random CAR word differentials, long canonical/inversion-only words, finite-boson boundary checks, direct-Weyl NumPy/JAX execution, and checked dimension overflow pass.
- M1 performance reproduction packaging and M4 fine-grained benchmark-counter provenance are explicitly accepted as non-blocking owner decisions for this phase.

## RECOMMENDED IMPROVEMENTS

1. Add the independent Holstein/spin-boson dense, COO, CSR, and native-MVP differential.
2. Correct plan basis/domain metadata and add exact public metadata regressions.
3. Rerun the focused Phase 7 suite, `python scripts/check.py --benchmark smoke`, the structured example, and the clean Phase 7 benchmark record after the two changes.

## Validation performed

- `conda run -p .conda pytest -q tests/test_structured_algebra.py tests/test_hamiltonian.py` — 56 passed.
- `conda run -p .conda python scripts/check.py --benchmark smoke` — passed; 31 Rust tests, 230 Python tests, and 165 selected benchmark-smoke cases.
- `conda run -p .conda python examples/structured_algebra.py` — passed and printed `structured targets agree`.
- Independent read-only probes covered the original C1 case, 320 nested early-mapping variants, 48 graded raw/mapped tensor cases, 700 random CAR words, long CAR fast paths, direct-Weyl dimension overflow, and public plan metadata.
