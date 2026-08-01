# Contributing to TenCirPauli

TenCirPauli accepts focused changes that preserve the separation between the pure Rust core, the PyO3 boundary, the public Python API, and optional framework integrations.

## Development checks

Run the following before submitting a substantial change:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
black --check python tests benchmarks scripts
ruff check python tests benchmarks scripts
mypy
maturin develop --release
pytest
```

Install the local tooling with `python -m pip install -e ".[quality,benchmark]"`. Rust formatting and linting use rustfmt and Clippy from the selected Rust toolchain. Python uses Black for deterministic formatting, Ruff for fast linting/import checks, and strict mypy for the public package; Pylint is intentionally not duplicated unless a concrete uncovered rule justifies it.

Run `python scripts/check.py --fix` before staging when formatting may be needed. The tracked pre-commit hook runs the same workflow in check mode, followed by Rust/Python tests and a full local benchmark record. Enable it once per clone with `git config core.hooksPath .githooks`; it never commits benchmark results.

New algebraic rules require property tests or differential tests against a small dense reference. Performance claims require a reproducible benchmark that separates construction, first execution, steady execution, memory, and numerical error.

Local performance history is managed separately from CI. Install the `benchmark` Python extra, then use `python benchmarks/run.py record`, `python benchmarks/run.py list`, and `python benchmarks/run.py compare <baseline-label>`. See `benchmarks/README.md`; benchmark results under `.benchmarks/` are machine-specific and must not be committed.

Do not add machine-specific paths, Conda environment names, credentials, or developer workarounds to tracked files. Put local setup information in the ignored `AGENTS.local.md`.
