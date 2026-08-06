# BCH convergence study

This manual study evaluates the fixed-order Baker–Campbell–Hausdorff series in the Pauli algebra. It uses the existing `PauliOperator.commutator`, `add`, and `scale` primitives through fourth order, compares the native result with an independent pure-Python dictionary recurrence, and optionally compares `exp(Z_k)` with the dense reference `exp(A) @ exp(B)` for small systems.

Run the default correctness and timing case with the project environment. The default Pauli workload is `nqubits=8`, `terms=44`; its independent Python Dict BCH recurrence takes about one second on the reference machine. Use `--no-reference` when measuring native-vs-Dict algebra and MVP preparation without the dense exponential oracle:

```bash
conda run -p .conda python examples/research/bch_convergence/run_tencirpauli.py --no-reference
conda run -p .conda python examples/research/bch_convergence/run_python_dict.py
```

Run a wider algebra-only case to expose packed-word and term-aggregation scaling. Dense matrix exponentials are intentionally disabled for this case:

```bash
conda run -p .conda python examples/research/bch_convergence/run_tencirpauli.py --nqubits 16 --terms 32 --no-reference
conda run -p .conda python examples/research/bch_convergence/run_python_dict.py --nqubits 16 --terms 32
```

The native script reports the native operator build, nested commutator, BCH assembly, plain string/weight export, explicit Python term materialization, independent-reference, and dense-reference times separately. It also reports matched native/Python construction-plus-algebra end-to-end times and their speedup. The dense reference is a small-system numerical oracle, not the primary performance baseline; the pure-Python dictionary recurrence is the matched algebra baseline. Pauli algebra is native-backed by default, and the script uses `to_dict()` for its main correctness comparison before separately timing the explicit `.terms` boundary. Outputs are printed only and are not stored as repository benchmark records.

Run the structured-family cases as well:

```bash
conda run -p .conda python examples/research/bch_convergence/run_structured.py --family both
```

Use `--family fermion` or `--family boson` to run one case. These cases compare native CAR/CCR BCH values with independent pure-Python dictionary recurrences. The JSON reports native and Python construction, algebra, plain-export, explicit typed-term materialization, and matched end-to-end times; the end-to-end comparison includes canonical construction plus the fourth-order BCH result in a plain dictionary representation. They are research evidence, not CI performance gates.
