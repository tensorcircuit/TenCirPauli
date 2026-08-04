# Research-driven examples

These examples exercise complete scientific workflows and are intentionally manual. They are not imported as tests and are not added to the continuous-integration execution allowlist; the normal quality checks still format and lint their Python source.

Each study has its own directory. Keep the first version small: one executable script per implementation, a README describing the question and conventions, and independent reference code only when it is needed. Large scans, figures, and machine-specific outputs stay local.

- [`fermi_hubbard_4x4/`](fermi_hubbard_4x4/): matrix-vector products for a half-filled, spin-balanced open-boundary Fermi–Hubbard model, with a TenCirPauli restricted MVP script and a QuSpin `quantum_LinearOperator` companion.
