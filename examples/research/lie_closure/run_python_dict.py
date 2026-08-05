"""Manual pure-Python dict baseline for the Lie-closure study."""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict

from common import closure, from_terms, generator_terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("su2", "su4", "sum2", "chain"), default="chain"
    )
    parser.add_argument("--mode", choices=("word", "sum"), default="word")
    parser.add_argument("--nqubits", type=int, default=4)
    parser.add_argument("--max-dimension", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nqubits < 1 or args.max_dimension < 1 or args.tolerance <= 0.0:
        raise SystemExit("nqubits, max-dimension, and tolerance must be positive")
    if args.case == "sum2" and args.mode == "word":
        raise SystemExit("sum2 requires --mode sum")

    specification, effective_nqubits = generator_terms(args.case, args.nqubits)
    generators = [from_terms(terms) for terms in specification]
    start = time.perf_counter()
    basis, candidate_count, complete, jacobi = closure(
        generators,
        args.mode,
        args.max_dimension,
        args.tolerance,
    )
    elapsed = time.perf_counter() - start
    output: Dict[str, object] = {
        "backend": "python_dict",
        "case": args.case,
        "mode": args.mode,
        "nqubits": effective_nqubits,
        "generator_count": len(generators),
        "closure_dimension": len(basis),
        "ambient_su_dimension": (4**effective_nqubits) - 1,
        "complete": complete,
        "candidate_brackets": candidate_count,
        "jacobi_residual": jacobi,
        "timings_seconds": {"closure": elapsed},
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
