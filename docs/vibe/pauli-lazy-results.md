# Native-backed lazy operator results

Status: implemented for `PauliOperator`; the family-wide storage and fallback matrix is documented in [`operator-lazy-results.md`](operator-lazy-results.md).

## Public boundary

The public intermediate result is always `PauliOperator`. A private PyO3 `NativePauliOperatorHandle` may back that object, but the handle is not a user-facing algebra type. Native-backed `PauliOperator` instances retain `nqubits`, `term_count`, and the private Rust operator without constructing Python `PauliTerm` or `PauliWord` objects.

PauliOperator constructors and algebra results are native-backed by default. Every intermediate result is a public `PauliOperator` shell; there is no separate public lazy-result or handler type. The private PyO3 handle is an implementation detail owned by the shell.

## Materialization and plain export

The `terms` property is the explicit Python-object materialization boundary. Its first access creates and caches the canonical `tuple[PauliTerm, ...]`; `term_count`, native algebra, dense/COO/CSR/MVP targets, and `repr` do not require that tuple. `PauliOperator.to_dict()` and the native-backed path's plain exporter return a deterministic `{pauli_string: coefficient}` mapping without creating `PauliTerm` or `PauliWord` objects.

The plain mapping is the preferred diagnostic and serialization path for large sparse results when callers need words and weights but do not need the object-rich term API.

## Extension to other operator families

Fermion, Boson, Qudit, Hybrid, and Majorana now use the same public boundary with family-specific canonical arrays rather than one shared symbolic-engine handle. Their native coverage and the deliberately retained Python fallbacks are tracked in [`operator-lazy-results.md`](operator-lazy-results.md).
