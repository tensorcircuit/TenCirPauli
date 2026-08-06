"""Run the two fermion-mapping implementations and compare their outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

from common import MAPPING_NAMES, parse_mode_list


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]


def _default_python(relative_path: str, fallback: str) -> str:
    candidate = _REPO_ROOT / relative_path
    return str(candidate) if candidate.exists() else fallback


def _run_script(python: str, script: Path, args: Iterable[str]) -> Dict[str, Any]:
    environment = os.environ.copy()
    result = subprocess.run(
        [python, str(script), *args],
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{script} failed with exit code {result.returncode}:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{script} produced no JSON output")
    return json.loads(lines[-1])


def _record_map(records: Iterable[Dict[str, Any]]) -> Dict[str, complex]:
    return {
        str(record["word"]): complex(float(record["real"]), float(record["imag"]))
        for record in records
    }


def _compare_terms(
    left: Iterable[Dict[str, Any]], right: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    left_map = _record_map(left)
    right_map = _record_map(right)
    missing_from_right = sorted(set(left_map) - set(right_map))
    missing_from_left = sorted(set(right_map) - set(left_map))
    shared = set(left_map) & set(right_map)
    max_error = max(
        (abs(left_map[word] - right_map[word]) for word in shared), default=0.0
    )
    return {
        "mode": "termwise",
        "correct": not missing_from_right
        and not missing_from_left
        and max_error <= 1e-10,
        "max_coefficient_error": max_error,
        "missing_from_openfermion": missing_from_right,
        "missing_from_tencirpauli": missing_from_left,
    }


def _compare_outputs(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    if "terms" in left and "terms" in right:
        return _compare_terms(left["terms"], right["terms"])
    return {
        "mode": "digest",
        "correct": (
            left["term_count"] == right["term_count"]
            and left["max_weight"] == right["max_weight"]
            and left["term_digest"] == right["term_digest"]
        ),
        "max_coefficient_error": None,
        "missing_from_openfermion": [],
        "missing_from_tencirpauli": [],
    }


def _without_terms(result: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in result.items() if key != "terms"}


def _compare_one(
    tencirpauli_python: str,
    openfermion_python: str,
    n_modes: int,
    workload: str,
    repetitions: int,
    emit_terms: bool,
) -> Dict[str, Any]:
    script_args = [
        "--n-modes",
        str(n_modes),
        "--workload",
        workload,
        "--repetitions",
        str(repetitions),
    ]
    if emit_terms:
        script_args.append("--emit-terms")
    tencirpauli = _run_script(
        tencirpauli_python, _HERE / "run_tencirpauli.py", script_args
    )
    openfermion = _run_script(
        openfermion_python, _HERE / "run_openfermion.py", script_args
    )
    tencirpauli_by_name = {
        result["mapping"]: result for result in tencirpauli["results"]
    }
    openfermion_by_name = {
        result["mapping"]: result for result in openfermion["results"]
    }
    comparisons = []
    for mapping in MAPPING_NAMES:
        left = tencirpauli_by_name[mapping]
        right = openfermion_by_name[mapping]
        mapping_speedup = (
            right["mapping_seconds_median"] / left["mapping_seconds_median"]
        )
        end_to_end_speedup = (
            right["end_to_end_seconds_median"] / left["end_to_end_seconds_median"]
        )
        comparisons.append(
            {
                "mapping": mapping,
                "correctness": _compare_outputs(left, right),
                "speedup_openfermion_over_tencirpauli": mapping_speedup,
                "end_to_end_speedup_openfermion_over_tencirpauli": end_to_end_speedup,
                "tencirpauli": _without_terms(left),
                "openfermion": _without_terms(right),
            }
        )
    return {
        "n_modes": n_modes,
        "workload": workload,
        "repetitions": repetitions,
        "tencirpauli_python": tencirpauli_python,
        "openfermion_python": openfermion_python,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-modes", default="4,8,12")
    parser.add_argument("--workload", default="hubbard")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--termwise-max-modes",
        type=int,
        default=12,
        help="emit and compare full Pauli terms up to this size",
    )
    parser.add_argument(
        "--tencirpauli-python",
        default=_default_python(".conda/bin/python", sys.executable),
    )
    parser.add_argument(
        "--openfermion-python",
        default=os.environ.get("OPENFERMION_PYTHON", sys.executable),
    )
    args = parser.parse_args()
    comparisons = [
        _compare_one(
            args.tencirpauli_python,
            args.openfermion_python,
            n_modes,
            args.workload,
            args.repetitions,
            n_modes <= args.termwise_max_modes,
        )
        for n_modes in parse_mode_list(args.n_modes)
    ]
    print(json.dumps({"comparisons": comparisons}, sort_keys=True))


if __name__ == "__main__":
    main()
