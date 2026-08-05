"""Manual BCH convergence and native Pauli-algebra timing study."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from common import bch_series as dict_bch_series
from common import from_terms, make_terms, scaled_terms
from scipy.linalg import expm

import tencirpauli as tcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", type=int, default=8)
    parser.add_argument("--terms", type=int, default=16)
    parser.add_argument("--t", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--reference-qubits", type=int, default=8)
    parser.add_argument("--no-reference", action="store_true")
    return parser.parse_args()


def tcp_operator(
    nqubits: int, terms: Iterable[Tuple[Tuple[int, ...], complex]]
) -> tcp.PauliOperator:
    """Construct one native operator from the shared raw workload."""
    return tcp.PauliOperator.from_terms(nqubits, tuple(terms))


def native_bch(
    operator_a: tcp.PauliOperator,
    operator_b: tcp.PauliOperator,
    max_bytes: int,
) -> Tuple[Tuple[tcp.PauliOperator, ...], Dict[str, Any]]:
    """Build BCH truncations and time the default native-backed algebra path."""
    total_start = time.perf_counter()
    commutator_start = time.perf_counter()
    ab = operator_a.commutator(operator_b, max_bytes=max_bytes)
    commutator_seconds = time.perf_counter() - commutator_start

    assembly_seconds: List[float] = []
    nested_seconds: List[float] = []

    start = time.perf_counter()
    order_one = operator_a.add(operator_b, max_bytes=max_bytes)
    assembly_seconds.append(time.perf_counter() - start)

    start = time.perf_counter()
    order_two = order_one.add(ab.scale(0.5), max_bytes=max_bytes)
    assembly_seconds.append(time.perf_counter() - start)

    start = time.perf_counter()
    aa = operator_a.commutator(ab, max_bytes=max_bytes)
    bb = operator_b.commutator(ab.scale(-1.0), max_bytes=max_bytes)
    nested_seconds.append(time.perf_counter() - start)
    start = time.perf_counter()
    order_three = order_two.add(
        aa.add(bb, max_bytes=max_bytes).scale(1.0 / 12.0),
        max_bytes=max_bytes,
    )
    assembly_seconds.append(time.perf_counter() - start)

    start = time.perf_counter()
    fourth = operator_b.commutator(aa, max_bytes=max_bytes)
    nested_seconds.append(time.perf_counter() - start)
    start = time.perf_counter()
    order_four = order_three.add(fourth.scale(-1.0 / 24.0), max_bytes=max_bytes)
    assembly_seconds.append(time.perf_counter() - start)

    return (order_one, order_two, order_three, order_four), {
        "commutator_seconds": commutator_seconds,
        "nested_commutator_seconds": nested_seconds,
        "assembly_seconds": assembly_seconds,
        "algebra_seconds": time.perf_counter() - total_start,
    }


def compare_plain_with_dict(
    actual_values: Sequence[Dict[str, complex]],
    expected: Sequence[Dict[Tuple[int, ...], complex]],
) -> Tuple[float, int]:
    """Check plain string/weight output against the independent dict recurrence.

    Native canonicalization removes exact zeros, while the independent Python
    recurrence can retain round-off residuals after a different summation order.
    """
    largest_error = 0.0
    largest_support_difference = 0
    comparison_tolerance = 1.0e-13
    for actual, reference in zip(actual_values, expected):
        reference_strings = {
            "".join("IXYZ"[code] for code in word): coefficient
            for word, coefficient in reference.items()
        }
        actual_support = {
            word
            for word, coefficient in actual.items()
            if abs(coefficient) > comparison_tolerance
        }
        reference_support = {
            word
            for word, coefficient in reference_strings.items()
            if abs(coefficient) > comparison_tolerance
        }
        largest_support_difference = max(
            largest_support_difference, len(actual_support ^ reference_support)
        )
        error = max(
            (
                abs(actual.get(word, 0.0j) - reference_strings.get(word, 0.0j))
                for word in set(actual) | set(reference_strings)
            ),
            default=0.0,
        )
        largest_error = max(largest_error, float(error))
    if largest_error > 1.0e-10:
        raise AssertionError(f"native and dict BCH values differ by {largest_error}")
    return largest_error, largest_support_difference


def dense_errors(
    operator_a: tcp.PauliOperator,
    operator_b: tcp.PauliOperator,
    truncations: Sequence[tcp.PauliOperator],
    max_bytes: int,
) -> Tuple[List[float], float]:
    """Compare exponentials against an independent dense matrix reference."""
    start = time.perf_counter()
    exact = expm(operator_a.dense(max_bytes=max_bytes)) @ expm(
        operator_b.dense(max_bytes=max_bytes)
    )
    errors: List[float] = []
    denominator = float(np.linalg.norm(exact, ord="fro"))
    for operator in truncations:
        approximation = expm(operator.dense(max_bytes=max_bytes))
        errors.append(
            float(np.linalg.norm(approximation - exact, ord="fro") / denominator)
        )
    return errors, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if args.nqubits < 1 or args.terms < 1 or args.reference_qubits < 1:
        raise SystemExit("nqubits, terms, and reference-qubits must be positive")
    if args.nqubits > args.reference_qubits and not args.no_reference:
        raise SystemExit("use --no-reference for workloads wider than reference-qubits")

    base_a = make_terms(args.nqubits, args.terms, args.seed)
    base_b = make_terms(args.nqubits, args.terms, args.seed + 1)
    scaled_a = scaled_terms(base_a, -1.0j * args.t)
    scaled_b = scaled_terms(base_b, -1.0j * args.t)

    build_start = time.perf_counter()
    operator_a = tcp_operator(args.nqubits, scaled_a)
    operator_b = tcp_operator(args.nqubits, scaled_b)
    build_seconds = time.perf_counter() - build_start

    reference_start = time.perf_counter()
    dict_a = from_terms(scaled_a)
    dict_b = from_terms(scaled_b)
    dict_reference = dict_bch_series(dict_a, dict_b)
    reference_seconds = time.perf_counter() - reference_start

    truncations, timings = native_bch(operator_a, operator_b, args.max_bytes)
    plain_start = time.perf_counter()
    plain = tuple(operator.to_dict() for operator in truncations)
    plain_seconds = time.perf_counter() - plain_start
    term_start = time.perf_counter()
    tuple(tuple(operator.terms) for operator in truncations)
    term_materialization_seconds = time.perf_counter() - term_start
    dict_error, support_difference = compare_plain_with_dict(plain, dict_reference)

    errors: List[float] = []
    dense_seconds = 0.0
    if not args.no_reference:
        errors, dense_seconds = dense_errors(
            operator_a, operator_b, truncations, args.max_bytes
        )

    output: Dict[str, object] = {
        "backend": "tencirpauli",
        "nqubits": args.nqubits,
        "input_terms_per_generator": args.terms,
        "t": args.t,
        "orders": [
            {
                "order": index,
                "canonical_terms": operator.term_count,
            }
            for index, operator in enumerate(truncations, start=1)
        ],
        "timings_seconds": {
            "operator_build": build_seconds,
            "independent_dict_reference": reference_seconds,
            "plain_export": plain_seconds,
            "python_term_materialization": term_materialization_seconds,
            "native_algebra": timings,
            "dense_reference": dense_seconds,
        },
        "checks": {
            "dict_max_coefficient_error": dict_error,
            "dict_max_support_symmetric_difference": support_difference,
            "dict_support_comparison_tolerance": 1.0e-13,
            "dense_relative_frobenius_errors": errors,
        },
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
