# Fermion-to-qubit mapping comparison

This study compares the Jordan–Wigner, parity, and Bravyi–Kitaev occupation mappings implemented by TenCirPauli with the corresponding OpenFermion transforms on the same deterministic spinful workloads. It is intended as a reproducible example research workflow rather than a CI performance gate.

The mode ordering is interleaved by site, `(site 0, up), (site 0, down), (site 1, up), (site 1, down), ...`. The `hubbard` workload contains nearest-neighbour hopping, on-site interaction, and chemical-potential terms. The `all_to_all` workload replaces nearest-neighbour hopping with all-pairs hopping and adds all-pairs density–density terms, so the input size grows quadratically with the number of modes. The `dense_quartic` workload contains all disjoint two-body creation/annihilation channels and is useful for reaching a longer OpenFermion runtime at moderate mode counts. Both implementations receive the same ordered list of fermion factors and coefficients.

For small cases, the comparison checks the complete canonical Pauli word/coefficient dictionaries. `correct` is true only when both the word sets and all complex coefficients agree to `1e-10`; term counts alone are not treated as a correctness check. For larger cases, both runners emit a stable digest, term count, and maximum Pauli weight without serializing every term; the digest uses a `1e-12` zero tolerance to match OpenFermion's numerical compression. The existing TenCirPauli mapping tests additionally compare mapped dense matrices with independently encoded Fock-space matrices.

The timing fields separate input construction, mapping-plan construction, mapping of a pre-built operator, mapping through a reusable TenCirPauli plan, and end-to-end construction plus mapping. The main cross-library comparison is `mapping_seconds_median` and `end_to_end_seconds_median`. Imports, process startup, and JSON serialization are outside the timed region. Run the two environments as separate processes because OpenFermion is an optional comparison dependency, not a TenCirPauli runtime dependency.

OpenFermion must be installed in a separate comparison environment. The TenCirPauli project environment is used for the native implementation, while the OpenFermion interpreter is supplied explicitly when running the combined comparison.

Run one implementation directly on the small local workload:

```bash
conda run -p .conda python examples/research/fermion_mapping/run_tencirpauli.py --n-modes 8 --repetitions 7
conda run -n YOUR_OPENFERMION_ENV python examples/research/fermion_mapping/run_openfermion.py --n-modes 8 --repetitions 7
```

Run a longer dense-quartic workload directly. Start with 16 modes and increase the size only after checking available memory; the runners emit summaries by default and do not build a dense matrix:

```bash
conda run -p .conda python examples/research/fermion_mapping/run_tencirpauli.py --workload dense_quartic --n-modes 16 --repetitions 3
conda run -n YOUR_OPENFERMION_ENV python examples/research/fermion_mapping/run_openfermion.py --workload dense_quartic --n-modes 16 --repetitions 3
```

Run the cross-environment correctness and timing comparison:

```bash
export OPENFERMION_PYTHON=/path/to/openfermion/bin/python
python examples/research/fermion_mapping/compare.py --workload dense_quartic --n-modes 8,16 --repetitions 3 --openfermion-python "$OPENFERMION_PYTHON"
```

Do not keep increasing the mode count after the OpenFermion term dictionary or mapped output becomes the memory bottleneck. This study is a boundary and scaling example, not a request to force arbitrarily large systems into a single-process Python reference implementation. Use `--emit-terms` only for small cases where a human-readable termwise audit is useful.

If the environments are installed elsewhere, pass their Python executables explicitly with `--tencirpauli-python` and `--openfermion-python`. Use release-mode native builds and keep the machine, thread settings, Python versions, and repetition count fixed when comparing performance. The scripts print JSON only and do not write machine-specific result files into the repository.
