# Local Benchmarks

TenCirPauli benchmarks are intentionally local and informational. They do not run as CI performance gates. The committed benchmark code defines stable workloads, while machine-specific measurements live under the ignored `.benchmarks/` directory.

## Setup

Install the benchmark extra in the active development environment, then make sure Rust, Cargo, and maturin come from the same selected toolchain:

```bash
python -m pip install -e ".[benchmark]"
```

## Record a run

Record both the Rust Criterion microbenchmarks and Python/PyO3 integration benchmarks:

```bash
python benchmarks/run.py record
```

The generated label combines UTC time, the current Git commit, and a `dirty` suffix when the worktree is not clean. Use `--label NAME` to choose a stable name and `--suite rust` or `--suite python` to run only one layer.

The runner stores Criterion baselines, pytest-benchmark JSON, and a metadata manifest containing the commit, dirty state, platform, and tool versions. It deliberately omits usernames, hostnames, and absolute project paths.

The repository pre-commit hook runs `python scripts/check.py`, which records a full benchmark only after formatting, linting, typing, and correctness tests pass. Use `python scripts/check.py --benchmark smoke` for a fast manual harness check or `--benchmark skip` when benchmarking is intentionally handled separately; the installed hook always uses `record`.

## Compare with an earlier run

List available local records and compare the current checkout with one baseline:

```bash
python benchmarks/run.py list
python benchmarks/run.py compare <baseline-label>
```

Criterion prints statistical change estimates and writes its HTML report below `.benchmarks/rust-target/criterion/`. Pytest-benchmark prints the current-versus-baseline table using the JSON stored below `.benchmarks/python/`.

## Measurement discipline

- Compare runs on the same machine, power mode, toolchain, feature set, and thread configuration.
- Close competing CPU-heavy processes and allow the machine to reach a stable thermal state.
- Treat a single small delta as noise. Repeat suspicious results and inspect confidence intervals, absolute time, throughput, and memory together.
- Keep benchmark names and workload semantics stable. Add a new case instead of silently changing an old baseline.
- Validate outputs outside the timed region and use release builds for all performance claims.
- Do not commit `.benchmarks/`; archive it separately if long-term machine history matters.
