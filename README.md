# TenCirPauli

TenCirPauli is an experimental Rust-native Pauli algebra and propagation library with a first-class Python API. It is developed independently from TensorCircuit while keeping compatible Pauli, gate, qubit-ordering, and Hamiltonian conventions through an optional integration layer.

The initial scope includes bit-packed Pauli words, Pauli operator algebra, commuting measurement groups, Hamiltonian matrix/MVP plans, symmetry analysis, and dynamic Pauli-weight-truncated propagation. The implementation does not reproduce TensorCircuit's fixed-buffer top-k sparse propagation.

## Architecture

```text
tencir-pauli-core        Pure Rust algorithms; no Python or TensorCircuit dependency
        │
        ▼
tencirpauli-native       PyO3 binding compiled as tencirpauli._native
        │
        ▼
tencirpauli              The only public Python package and PyPI distribution
        └── integrations.tensorcircuit   Optional adapter
```

## Development status

The project is in bootstrap stage. The current implementation provides a minimal end-to-end `PauliWord` path for validating the Rust core, PyO3 extension, Python wrapper, packaging, and tests. See the [vibe design index](docs/vibe/README.md) for the planned architecture, release process, and acceptance gates.

## Local development

Create an isolated environment containing Python, Rust/Cargo, and maturin, then build the mixed project:

```bash
conda create -n tencirpauli-dev python=3.11 pip
conda activate tencirpauli-dev
conda install -c conda-forge rust maturin pytest
maturin develop --release
pytest
```

Run the repository quality checks with:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
black --check python tests benchmarks scripts
ruff check python tests benchmarks scripts
mypy
```

The complete local pre-commit workflow is `python scripts/check.py`; add `--fix` before staging to apply rustfmt and Black. Enable the tracked hook once per clone with `git config core.hooksPath .githooks`. The hook runs all format, lint, type, test, and local benchmark recording steps before Git accepts a commit.

Project-specific local paths belong in the ignored `AGENTS.local.md`, not in committed files.

## Python example

```python
from tencirpauli import PauliWord

x0 = PauliWord(nqubits=2, x_words=(0b01,), z_words=(0,))
z0 = PauliWord(nqubits=2, x_words=(0,), z_words=(0b01,))

assert x0.weight == 1
assert not x0.commutes_with(z0)
```

## License

Apache License 2.0.
