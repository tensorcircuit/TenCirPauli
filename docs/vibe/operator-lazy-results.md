# Default lazy operator results

Status: implemented for Pauli, Fermion, Boson, Qudit, Hybrid, and Majorana operators. The public object remains the concrete operator family; its private storage is a Rust-owned native handle, and `.terms` is the explicit typed-word materialization boundary.

The default algebra path does not construct display words for intermediate results. `term_count`, family-specific `to_dict()`, native algebra, mapping kernels, and finite compilation consume the cached canonical arrays where the corresponding native kernel supports the operation. There is no public `deferred()` entry point and no public handler type.

| Family | Default private storage | Native-default operations | Known Python materialization fallback |
| --- | --- | --- | --- |
| Pauli | Private Rust packed operator handle | Construction, add, scale, multiply, commutator, anticommutator, adjoint, Hermiticity, matrix/MVP targets | `.terms` and APIs whose contract explicitly needs typed Pauli words |
| Fermion | `NativeFermionOperatorHandle` | CAR construction, add, scale, multiply, commutator, adjoint, Jordan-Wigner, and plain export | Charge analysis and APIs explicitly requesting typed words |
| Boson | `NativeBosonOperatorHandle` | CCR construction, add, scale, multiply, commutator, adjoint, and plain export | Charge analysis and APIs explicitly requesting typed words |
| Qudit | `NativeHybridOperatorHandle` with Weyl triples | Weyl construction, add, scale, multiply, commutator, adjoint, finite native compilation, and plain export | Tensor products remain an intentional materialized fallback; APIs explicitly requesting typed words |
| Hybrid | `NativeHybridOperatorHandle` | Raw hybrid construction, add, scale, multiply, commutator, adjoint, Jordan-Wigner input, finite compilation, and plain export | Tensor products remain an intentional materialized fallback; mixed raw/mapped fermion ambiguity and APIs explicitly requesting typed words |
| Majorana | `NativeMajoranaOperatorHandle` | Construction, add, scale, multiply, commutator, adjoint, Majorana-to-fermion conversion, and plain export | Internal `_from_canonical` compatibility paths and APIs explicitly requesting typed terms |

The fallback list is intentional and is not a performance claim: those paths either need word-level semantic inspection or combine representations that the current native kernels do not yet accept. In particular, a canonical term containing both raw and mapped fermion factors is rejected rather than assigned an arbitrary multiplication order. The list is kept visible so future profiling can distinguish a genuine native path from Python materialization.

The structured BCH study is in [`examples/research/bch_convergence/run_structured.py`](../../examples/research/bch_convergence/run_structured.py). It validates Fermion CAR and Boson CCR results against independent Python dictionary recurrences and reports native algebra, plain export, typed-term materialization, and reference timings separately.
