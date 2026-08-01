"""Record and compare local Rust and Python benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / ".benchmarks"
RUST_TARGET = RESULTS_ROOT / "rust-target"
PYTHON_STORAGE = RESULTS_ROOT / "python"
RUNS_ROOT = RESULTS_ROOT / "runs"
SUITES = ("all", "rust", "python")


def command_environment() -> Dict[str, str]:
    """Bind subprocesses to the environment that owns the active Python."""
    environment = os.environ.copy()
    python_bin = str(Path(sys.executable).parent)
    environment["PATH"] = python_bin + os.pathsep + environment.get("PATH", "")
    environment["CONDA_PREFIX"] = sys.prefix
    environment.pop("VIRTUAL_ENV", None)
    return environment


def run(command: List[str], env: Optional[Dict[str, str]] = None) -> None:
    """Run a command from the repository root and show it first."""
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env=env or command_environment(),
        check=True,
    )


def capture(command: List[str]) -> str:
    """Capture a best-effort single-line command result."""
    try:
        return subprocess.check_output(
            command, cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def git_commit() -> str:
    """Return the current commit or a marker for a new repository."""
    value = capture(["git", "rev-parse", "HEAD"])
    return value if value != "unavailable" else "no-commit"


def git_dirty() -> bool:
    """Return whether tracked or untracked project files differ from HEAD."""
    return bool(capture(["git", "status", "--porcelain", "--untracked-files=normal"]))


def default_label() -> str:
    """Build a unique, human-readable run label."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = git_commit()
    short_commit = commit[:12] if commit != "no-commit" else commit
    suffix = "-dirty" if git_dirty() else ""
    return f"{timestamp}_{short_commit}{suffix}"


def validate_label(label: str) -> str:
    """Reject labels that are unsafe as file and Criterion baseline names."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError("labels may contain only letters, numbers, '.', '_', and '-'")
    return label


def tool_version(command: List[str]) -> str:
    """Return the first line of a tool version."""
    return capture(command).splitlines()[0]


def metadata(label: str, suite: str) -> Dict[str, object]:
    """Collect reproducibility metadata without usernames, paths, or hostnames."""
    return {
        "schema_version": 1,
        "label": label,
        "suite": suite,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "tools": {
            "python": platform.python_version(),
            "rustc": tool_version(["rustc", "--version"]),
            "cargo": tool_version(["cargo", "--version"]),
            "maturin": tool_version(["maturin", "--version"]),
        },
        "status": "running",
    }


def write_manifest(data: Dict[str, object]) -> None:
    """Persist one local benchmark-run manifest."""
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    path = RUNS_ROOT / f"{data['label']}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def rust_environment() -> Dict[str, str]:
    """Keep Criterion build products and baselines outside Cargo's normal target."""
    environment = command_environment()
    environment["CARGO_TARGET_DIR"] = str(RUST_TARGET)
    return environment


def record_rust(label: str) -> None:
    """Record a named Criterion baseline."""
    run(
        [
            "cargo",
            "bench",
            "--locked",
            "-p",
            "tencir-pauli-core",
            "--bench",
            "pauli_word",
            "--",
            "--save-baseline",
            label,
        ],
        env=rust_environment(),
    )


def python_storage_uri() -> str:
    """Return pytest-benchmark's local file-storage URI."""
    PYTHON_STORAGE.mkdir(parents=True, exist_ok=True)
    return "file://.benchmarks/python"


def build_python_extension() -> None:
    """Install the current native extension in release mode before timing it."""
    run(["maturin", "develop", "--release", "--locked"])


def record_python(label: str) -> None:
    """Record a named pytest-benchmark run."""
    build_python_extension()
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "benchmarks/python",
            "--benchmark-only",
            f"--benchmark-storage={python_storage_uri()}",
            f"--benchmark-save={label}",
            "--benchmark-sort=name",
        ]
    )


def record(label: str, suite: str) -> None:
    """Record selected suites and their reproducibility metadata."""
    manifest_path = RUNS_ROOT / f"{label}.json"
    if manifest_path.exists():
        raise ValueError(f"benchmark label already exists: {label}")

    data = metadata(label, suite)
    write_manifest(data)
    try:
        if suite in ("all", "rust"):
            record_rust(label)
        if suite in ("all", "python"):
            record_python(label)
    except (OSError, subprocess.CalledProcessError):
        data["status"] = "failed"
        write_manifest(data)
        raise
    data["status"] = "complete"
    write_manifest(data)
    print(f"Recorded benchmark run: {label}")


def find_python_result(label: str) -> str:
    """Resolve a saved label to pytest-benchmark's generated result ID."""
    matches = sorted(
        PYTHON_STORAGE.glob(f"**/*_{label}.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise ValueError(f"no Python benchmark result found for label: {label}")
    return matches[-1].stem


def compare_rust(label: str) -> None:
    """Run Criterion and compare current measurements with a named baseline."""
    run(
        [
            "cargo",
            "bench",
            "--locked",
            "-p",
            "tencir-pauli-core",
            "--bench",
            "pauli_word",
            "--",
            "--baseline",
            label,
        ],
        env=rust_environment(),
    )


def compare_python(label: str) -> None:
    """Run Python benchmarks and compare them with a saved result."""
    build_python_extension()
    result_id = find_python_result(label)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "benchmarks/python",
            "--benchmark-only",
            f"--benchmark-storage={python_storage_uri()}",
            f"--benchmark-compare={result_id}",
            "--benchmark-sort=name",
        ]
    )


def compare(label: str, suite: Optional[str]) -> None:
    """Compare current code with a previously recorded local run."""
    manifest_path = RUNS_ROOT / f"{label}.json"
    if not manifest_path.exists():
        raise ValueError(f"unknown benchmark label: {label}")
    saved_data = json.loads(manifest_path.read_text())
    selected_suite = suite or str(saved_data["suite"])
    if selected_suite in ("all", "rust"):
        compare_rust(label)
    if selected_suite in ("all", "python"):
        compare_python(label)


def list_runs() -> None:
    """Print locally recorded benchmark manifests."""
    if not RUNS_ROOT.exists():
        print("No local benchmark runs recorded.")
        return
    for path in sorted(RUNS_ROOT.glob("*.json")):
        data = json.loads(path.read_text())
        print(
            f"{data['label']}  {data['status']}  "
            f"commit={str(data['git_commit'])[:12]}  dirty={data['git_dirty']}"
        )


def parse_args() -> argparse.Namespace:
    """Parse the local benchmark command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="record a new local run")
    record_parser.add_argument("--label", help="optional unique result label")
    record_parser.add_argument("--suite", choices=SUITES, default="all")

    compare_parser = subparsers.add_parser(
        "compare", help="compare current code with a saved run"
    )
    compare_parser.add_argument("baseline", help="saved run label")
    compare_parser.add_argument(
        "--suite", choices=SUITES, help="defaults to the suite saved in the baseline"
    )

    subparsers.add_parser("list", help="list saved local runs")
    return parser.parse_args()


def main() -> int:
    """Run the selected benchmark operation."""
    arguments = parse_args()
    try:
        if arguments.command == "record":
            label = validate_label(arguments.label or default_label())
            record(label, arguments.suite)
        elif arguments.command == "compare":
            compare(validate_label(arguments.baseline), arguments.suite)
        else:
            list_runs()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"benchmark command failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
