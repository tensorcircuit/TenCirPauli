"""Run the complete local quality, test, and benchmark workflow."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_MODES = ("record", "smoke", "skip")


def command_environment() -> Dict[str, str]:
    """Put tools installed beside the active Python first on PATH."""
    environment = os.environ.copy()
    python_bin = str(Path(sys.executable).parent)
    environment["PATH"] = python_bin + os.pathsep + environment.get("PATH", "")
    environment["CONDA_PREFIX"] = sys.prefix
    environment.pop("VIRTUAL_ENV", None)
    return environment


def run(command: List[str], env: Optional[Dict[str, str]] = None) -> None:
    """Run one check from the repository root and fail immediately on error."""
    print(f"\n$ {shlex.join(command)}", flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env=env or command_environment(),
        check=True,
    )


def run_formatters(fix: bool) -> None:
    """Apply formatters or verify that committed sources are formatted."""
    if fix:
        run(["cargo", "fmt", "--all"])
        run(["black", "python", "tests", "benchmarks", "scripts", "examples"])
    else:
        run(["cargo", "fmt", "--all", "--", "--check"])
        run(
            [
                "black",
                "--check",
                "python",
                "tests",
                "benchmarks",
                "scripts",
                "examples",
            ]
        )


def run_quality_checks() -> None:
    """Run Rust and Python linting and static type analysis."""
    run(
        [
            "cargo",
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ]
    )
    run(["ruff", "check", "python", "tests", "benchmarks", "scripts", "examples"])
    run(["mypy"])
    run(["git", "diff", "--check"])


def run_tests() -> None:
    """Run the complete Rust and Python correctness suites."""
    run(["cargo", "test", "--locked", "--workspace"])
    run(["maturin", "develop", "--release", "--locked"])
    run([sys.executable, "-m", "pytest"])
    run([sys.executable, "scripts/run_doctest.py"])


def run_benchmarks(mode: str) -> None:
    """Record full local benchmarks or only smoke-test their harnesses."""
    if mode == "skip":
        return
    if mode == "record":
        run([sys.executable, "benchmarks/run.py", "record"])
        return
    for bench in ("pauli_word", "symmetry", "propagation"):
        run(
            [
                "cargo",
                "bench",
                "--locked",
                "-p",
                "tencir-pauli-core",
                "--bench",
                bench,
                "--",
                "--test",
            ]
        )
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "benchmarks/python",
            "--benchmark-disable",
            "-m",
            "not performance_large",
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse local check options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply rustfmt and Black before running checks",
    )
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARK_MODES,
        default="record",
        help="record full local results by default; use smoke or skip when needed",
    )
    return parser.parse_args()


def main() -> int:
    """Run formatting, linting, tests, and the selected benchmark mode."""
    arguments = parse_args()
    try:
        run_formatters(arguments.fix)
        run_quality_checks()
        run_tests()
        run_benchmarks(arguments.benchmark)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"\nlocal checks failed: {error}", file=sys.stderr)
        return 1
    print("\nAll local checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
