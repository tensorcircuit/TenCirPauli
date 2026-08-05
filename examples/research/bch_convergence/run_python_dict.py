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
    parser.add_argument("--terms", type=int, default=16)
    parser.add_argument("--t", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_a = make_terms(args.nqubits, args.terms, args.seed)
    base_b = make_terms(args.nqubits, args.terms, args.seed + 1)
    operator_a = from_terms(scaled_terms(base_a, -1.0j * args.t))
    operator_b = from_terms(scaled_terms(base_b, -1.0j * args.t))

    start = time.perf_counter()
    truncations = bch_series(operator_a, operator_b)
    algebra_seconds = time.perf_counter() - start
    output: Dict[str, object] = {
        "backend": "python_dict",
        "nqubits": args.nqubits,
        "input_terms_per_generator": args.terms,
        "t": args.t,
        "orders": [
            {"order": index, "canonical_terms": len(operator)}
            for index, operator in enumerate(truncations, start=1)
        ],
        "timings_seconds": {"algebra": algebra_seconds},
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
