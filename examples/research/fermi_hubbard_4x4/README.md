# Fermi–Hubbard restricted MVP study

This study compares two independent matrix-vector-product implementations for the open-boundary spinful Fermi–Hubbard Hamiltonian on a rectangular lattice. The selected sector is half filling with `N_up = N_down`, which is a fixed-particle-number, `S_z=0` sector; it is not a projection onto the total-spin singlet subspace.

The TenCirPauli script uses raw fermion terms, two exact additive-charge constraints, and the explicitly selected `storage="lazy"` restricted MVP. The QuSpin companion uses `quantum_LinearOperator`, which applies the operator through the basis on every matvec without retaining a sparse transition graph. The scripts intentionally build the model independently so that the comparison is not a shared-construction test.

The default run is a 4x3 lattice because the current eager restricted plan is a sparse transition table and is too large for 4x4. Use `--preflight` for the 4x4 target. An actual 4x4 MVP requires `--allow-large` and a high-memory machine; it is a manual experiment, not a CI or commit check.

Run a small correctness and MVP smoke comparison with separate environments:

```bash
conda run -p .conda python examples/research/fermi_hubbard_4x4/run_tencirpauli.py --rows 2 --cols 3 --eigsh
conda run -n quspin python examples/research/fermi_hubbard_4x4/run_quspin.py --rows 2 --cols 3 --eigsh
```

Run the practical 4x3 MVP comparison:

```bash
conda run -p .conda python examples/research/fermi_hubbard_4x4/run_tencirpauli.py --rows 4 --cols 3
conda run -n quspin python examples/research/fermi_hubbard_4x4/run_quspin.py --rows 4 --cols 3
```

Inspect the 4x4 resource boundary without allocating the sector state or transition graph:

```bash
conda run -p .conda python examples/research/fermi_hubbard_4x4/run_tencirpauli.py --rows 4 --cols 4 --preflight
conda run -n quspin python examples/research/fermi_hubbard_4x4/run_quspin.py --rows 4 --cols 4 --preflight
```

The JSON result reports sector dimension, plan construction time, MVP time, and an optional lowest-eigenvalue residual. Results are printed only; benchmark records and output files remain local.
