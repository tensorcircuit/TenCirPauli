"""Manual pure-Python dict baseline for the BCH research study."""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict

from common import bch_series, from_terms, make_terms, scaled_terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", type=int, default=8)
    parser.add_argument("--terms", type=int, default=44)
    parser.add_argument("--t", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_a = make_terms(args.nqubits, args.terms, args.seed)
    base_b = make_terms(args.nqubits, args.terms, args.seed + 1)
    raw_a = scaled_terms(base_a, -1.0j * args.t)
    raw_b = scaled_terms(base_b, -1.0j * args.t)

    end_to_end_start = time.perf_counter()
    build_start = time.perf_counter()
    operator_a = from_terms(raw_a)
    operator_b = from_terms(raw_b)
    build_seconds = time.perf_counter() - build_start
    algebra_start = time.perf_counter()
    truncations = bch_series(operator_a, operator_b)
    algebra_seconds = time.perf_counter() - algebra_start
    end_to_end_seconds = time.perf_counter() - end_to_end_start
    output: Dict[str, object] = {
        "backend": "python_dict",
        "nqubits": args.nqubits,
        "input_terms_per_generator": args.terms,
        "t": args.t,
        "orders": [
            {"order": index, "canonical_terms": len(operator)}
            for index, operator in enumerate(truncations, start=1)
        ],
        "timings_seconds": {
            "python_build": build_seconds,
            "python_algebra": algebra_seconds,
            "python_end_to_end": end_to_end_seconds,
        },
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
