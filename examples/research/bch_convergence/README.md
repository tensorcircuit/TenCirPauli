# BCH convergence study

This manual study evaluates the fixed-order Baker–Campbell–Hausdorff series in the Pauli algebra. It uses the existing `PauliOperator.commutator`, `add`, and `scale` primitives through fourth order, compares the native result with an independent pure-Python dictionary recurrence, and optionally compares `exp(Z_k)` with the dense reference `exp(A) @ exp(B)` for small systems.

Run the default correctness and timing case with the project environment:

```bash
conda run -p .conda python examples/research/bch_convergence/run_tencirpauli.py
conda run -p .conda python examples/research/bch_convergence/run_python_dict.py
```

Run a wider algebra-only case to expose packed-word and term-aggregation scaling. Dense matrix exponentials are intentionally disabled for this case:

```bash
conda run -p .conda python examples/research/bch_convergence/run_tencirpauli.py --nqubits 16 --terms 32 --no-reference
conda run -p .conda python examples/research/bch_convergence/run_python_dict.py --nqubits 16 --terms 32
```

The native script reports the operator build, nested commutator, BCH assembly, plain string/weight export, explicit Python term materialization, independent-reference, and dense-reference times separately. The dense reference is a small-system numerical oracle, not the primary performance baseline; the pure-Python dictionary recurrence is the matched algebra baseline. Pauli algebra is native-backed by default, and the script uses `to_dict()` for its main correctness comparison before separately timing the explicit `.terms` boundary. Outputs are printed only and are not stored as repository benchmark records.

Run the structured-family cases as well:

```bash
conda run -p .conda python examples/research/bch_convergence/run_structured.py --family fermion
conda run -p .conda python examples/research/bch_convergence/run_structured.py --family boson
```

These cases compare native CAR/CCR BCH values with independent pure-Python dictionary recurrences. Their timing fields distinguish native algebra, plain export, explicit typed-term materialization, and the Python reference; they are research evidence, not CI performance gates.
