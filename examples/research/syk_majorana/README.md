# SYK Majorana MVP study

This study builds a real, quartic Sachdev–Ye–Kitaev Hamiltonian directly in the Majorana algebra, maps it to qubits in one batched operation, and computes its lowest eigenvalue with SciPy's Lanczos solver. Every Lanczos matrix-vector product uses the reusable `MajoranaOperator.compile("native_mvp")` plan, so the example does not materialize a dense Hamiltonian.

The default case uses `N = 2 * n_modes = 24` Majorana generators and a 4096-dimensional qubit state space:

```bash
conda run -p .conda python examples/research/syk_majorana/run_tencirpauli.py
```

Use a larger or smaller Majorana system explicitly, and choose a different fermion-to-qubit mapping when comparing setup costs:

```bash
conda run -p .conda python examples/research/syk_majorana/run_tencirpauli.py --n-modes 12 --mapping bravyi_kitaev
```

QuSpin is intentionally kept in a separate environment. Run the matched workload with the same `--n-modes`, `--coupling`, and `--seed`:

```bash
conda run -n quspin python examples/research/syk_majorana/run_quspin.py --n-modes 10
conda run -n quspin python examples/research/syk_majorana/run_quspin.py --n-modes 12
```

The TenCirPauli and QuSpin outputs use the same full Fock-space dimension `2**n_modes`, random couplings, and SciPy `eigsh` interface. TenCirPauli reports Majorana canonicalization and mapped native-MVP setup; QuSpin reports basis construction and `+/-` operator-string construction. Compare `mvp_seconds_median` for steady matrix-vector products and `eigsh_seconds` for the complete ground-state solve. The two processes should be run separately because their environments and dependency stacks are intentionally different.

The JSON result separates random-coupling generation, Majorana canonicalization, mapped native-MVP plan construction, first and steady MVP application, and the ground-state solve. The SYK script uses `storage="lazy"` by default; pass `--storage eager` to retain the grouped X-mask diagonals for a repeated-MVP comparison. `plan_storage` and `plan_strategy` are both reported, and `plan_strategy="term_direct"` describes the MVP kernel rather than the storage mode. `setup_end_to_end_seconds` covers workload generation through the first MVP, while `ground_state_end_to_end_seconds` covers the same setup plus `eigsh`; these are the matched end-to-end fields across the two environments. It also reports the residual `||H|psi> - E|psi>||` and the retained plan metadata. The convention is `H = i^(q/2) sum J_abcd chi_a chi_b chi_c chi_d` with `q=4`, `chi_a^2=1`, and `Var(J_abcd) = 3! J^2 / N^3`.
