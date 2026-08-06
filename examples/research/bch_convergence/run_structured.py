"""Manual BCH timing and correctness study for native fermion/boson algebra."""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Dict, Iterable, List, Tuple

import tencirpauli as tcp


FermionFactor = Tuple[int, str]
FermionKey = Tuple[FermionFactor, ...]
BosonFactor = Tuple[int, str]
BosonKey = Tuple[Tuple[int, int, int], ...]
RawTerm = Tuple[Tuple[Tuple[int, str], ...], complex]
SparseOperator = Dict[object, complex]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=("fermion", "boson", "both"), default="both"
    )
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--terms", type=int, default=8)
    parser.add_argument("--t", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def add_to(result: SparseOperator, key: object, value: complex) -> None:
    result[key] = result.get(key, 0.0j) + value


def _fermion_reduce(word: FermionKey) -> SparseOperator:
    for index in range(len(word) - 1):
        left_mode, left_action = word[index]
        right_mode, right_action = word[index + 1]
        if left_action == right_action == "create":
            if left_mode == right_mode:
                return {}
            if left_mode > right_mode:
                swapped = (
                    word[:index]
                    + word[index + 1 : index + 2]
                    + word[index : index + 1]
                    + word[index + 2 :]
                )
                return {key: -value for key, value in _fermion_reduce(swapped).items()}
        elif left_action == right_action == "annihilate":
            if left_mode == right_mode:
                return {}
            if left_mode < right_mode:
                swapped = (
                    word[:index]
                    + word[index + 1 : index + 2]
                    + word[index : index + 1]
                    + word[index + 2 :]
                )
                return {key: -value for key, value in _fermion_reduce(swapped).items()}
        elif left_action == "annihilate" and right_action == "create":
            result: SparseOperator = {}
            if left_mode == right_mode:
                result = _fermion_reduce(word[:index] + word[index + 2 :])
            swapped = (
                word[:index]
                + word[index + 1 : index + 2]
                + word[index : index + 1]
                + word[index + 2 :]
            )
            for key, value in _fermion_reduce(swapped).items():
                add_to(result, key, -value)
            return result
    return {word: 1.0 + 0j}


def fermion_from_terms(terms: Iterable[RawTerm]) -> SparseOperator:
    result: SparseOperator = {}
    for word, coefficient in terms:
        for key, value in _fermion_reduce(word).items():
            add_to(result, key, coefficient * value)
    return {key: value for key, value in result.items() if value != 0.0j}


def _boson_reduce(word: Tuple[BosonFactor, ...]) -> SparseOperator:
    for index in range(len(word) - 1):
        left_mode, left_action = word[index]
        right_mode, right_action = word[index + 1]
        if left_mode > right_mode:
            swapped = (
                word[:index]
                + word[index + 1 : index + 2]
                + word[index : index + 1]
                + word[index + 2 :]
            )
            return _boson_reduce(swapped)
        if (
            left_mode == right_mode
            and left_action == "annihilate"
            and right_action == "create"
        ):
            result = _boson_reduce(word[:index] + word[index + 2 :])
            swapped = (
                word[:index]
                + word[index + 1 : index + 2]
                + word[index : index + 1]
                + word[index + 2 :]
            )
            for key, value in _boson_reduce(swapped).items():
                add_to(result, key, value)
            return result
    blocks: Dict[int, List[int]] = {}
    for mode, action in word:
        counts = blocks.setdefault(mode, [0, 0])
        counts[0 if action == "create" else 1] += 1
    return {
        tuple(
            (mode, counts[0], counts[1])
            for mode, counts in sorted(blocks.items())
            if counts[0] or counts[1]
        ): 1.0
        + 0j
    }


def boson_from_terms(terms: Iterable[RawTerm]) -> SparseOperator:
    result: SparseOperator = {}
    for word, coefficient in terms:
        for key, value in _boson_reduce(word).items():
            add_to(result, key, coefficient * value)
    return {key: value for key, value in result.items() if value != 0.0j}


def add(left: SparseOperator, right: SparseOperator) -> SparseOperator:
    result = dict(left)
    for key, value in right.items():
        add_to(result, key, value)
    return {key: value for key, value in result.items() if value != 0.0j}


def scale(operator: SparseOperator, coefficient: complex) -> SparseOperator:
    return {
        key: coefficient * value
        for key, value in operator.items()
        if coefficient * value != 0.0j
    }


def multiply_fermion(left: SparseOperator, right: SparseOperator) -> SparseOperator:
    result: SparseOperator = {}
    for left_key, left_value in left.items():
        for right_key, right_value in right.items():
            for key, phase in _fermion_reduce(left_key + right_key).items():
                add_to(result, key, left_value * right_value * phase)
    return {key: value for key, value in result.items() if value != 0.0j}


def multiply_boson(left: SparseOperator, right: SparseOperator) -> SparseOperator:
    result: SparseOperator = {}
    for left_key, left_value in left.items():
        left_factors = tuple(
            (mode, "create") for mode, create, _ in left_key for _ in range(create)
        ) + tuple(
            (mode, "annihilate")
            for mode, _, annihilate in left_key
            for _ in range(annihilate)
        )
        for right_key, right_value in right.items():
            right_factors = tuple(
                (mode, "create") for mode, create, _ in right_key for _ in range(create)
            ) + tuple(
                (mode, "annihilate")
                for mode, _, annihilate in right_key
                for _ in range(annihilate)
            )
            for key, phase in _boson_reduce(left_factors + right_factors).items():
                add_to(result, key, left_value * right_value * phase)
    return {key: value for key, value in result.items() if value != 0.0j}


def commutator(
    left: SparseOperator, right: SparseOperator, family: str
) -> SparseOperator:
    multiply = multiply_fermion if family == "fermion" else multiply_boson
    return add(multiply(left, right), scale(multiply(right, left), -1.0))


def bch_series(
    left: SparseOperator, right: SparseOperator, family: str
) -> Tuple[SparseOperator, ...]:
    ab = commutator(left, right, family)
    order_one = add(left, right)
    order_two = add(order_one, scale(ab, 0.5))
    aa = commutator(left, ab, family)
    bb = commutator(right, scale(ab, -1.0), family)
    order_three = add(order_two, scale(add(aa, bb), 1.0 / 12.0))
    fourth = commutator(right, aa, family)
    return (
        order_one,
        order_two,
        order_three,
        add(order_three, scale(fourth, -1.0 / 24.0)),
    )


def make_terms(modes: int, count: int, seed: int) -> Tuple[RawTerm, ...]:
    rng = random.Random(seed)
    result: List[RawTerm] = []
    for index in range(count):
        degree = 2 + index % 2
        factors = tuple(
            (rng.randrange(modes), "create" if rng.randrange(2) == 0 else "annihilate")
            for _ in range(degree)
        )
        result.append((factors, complex(0.2 + 0.03 * (index + 1), 0.01 * (index % 3))))
    return tuple(result)


def native_operator(family: str, modes: int, terms: Tuple[RawTerm, ...]) -> object:
    if family == "fermion":
        return tcp.FermionOperator.from_terms(modes, terms)
    return tcp.BosonOperator.from_terms(modes, terms)


def bch_series_native(left: object, right: object) -> Tuple[object, ...]:
    ab = left.commutator(right)
    order_one = left.add(right)
    order_two = order_one.add(ab.scale(0.5))
    aa = left.commutator(ab)
    bb = right.commutator(ab.scale(-1.0))
    order_three = order_two.add(aa.add(bb).scale(1.0 / 12.0))
    fourth = right.commutator(aa)
    return order_one, order_two, order_three, order_three.add(fourth.scale(-1.0 / 24.0))


def run_family(args: argparse.Namespace, family: str) -> Dict[str, object]:
    """Run one structured-family BCH case with matched end-to-end timings."""
    raw_a = make_terms(args.modes, args.terms, args.seed)
    raw_b = make_terms(args.modes, args.terms, args.seed + 1)
    scaled_a = tuple((word, -1.0j * args.t * value) for word, value in raw_a)
    scaled_b = tuple((word, -1.0j * args.t * value) for word, value in raw_b)
    builder = fermion_from_terms if family == "fermion" else boson_from_terms

    native_end_to_end_start = time.perf_counter()
    operator_a = native_operator(family, args.modes, scaled_a)
    operator_b = native_operator(family, args.modes, scaled_b)
    native_build_seconds = time.perf_counter() - native_end_to_end_start

    start = time.perf_counter()
    native_truncations = bch_series_native(operator_a, operator_b)
    native_seconds = time.perf_counter() - start
    start = time.perf_counter()
    native_plain = tuple(operator.to_dict() for operator in native_truncations)
    plain_seconds = time.perf_counter() - start
    native_end_to_end_seconds = time.perf_counter() - native_end_to_end_start
    start = time.perf_counter()
    tuple(tuple(operator.terms) for operator in native_truncations)
    terms_seconds = time.perf_counter() - start

    python_end_to_end_start = time.perf_counter()
    reference_a, reference_b = builder(scaled_a), builder(scaled_b)
    python_build_seconds = time.perf_counter() - python_end_to_end_start
    start = time.perf_counter()
    reference_truncations = bch_series(reference_a, reference_b, family)
    python_seconds = time.perf_counter() - start
    python_end_to_end_seconds = time.perf_counter() - python_end_to_end_start
    errors = [
        max(
            (
                abs(actual.get(key, 0.0j) - expected.get(key, 0.0j))
                for key in set(actual) | set(expected)
            ),
            default=0.0,
        )
        for actual, expected in zip(native_plain, reference_truncations)
    ]
    if max(errors, default=0.0) > 1.0e-10:
        raise AssertionError(f"native and Python BCH values differ by {max(errors)}")
    return {
        "backend": "tencirpauli",
        "family": family,
        "modes": args.modes,
        "input_terms_per_generator": args.terms,
        "orders": [operator.term_count for operator in native_truncations],
        "timings_seconds": {
            "native_build": native_build_seconds,
            "native_algebra": native_seconds,
            "native_plain_export": plain_seconds,
            "native_end_to_end": native_end_to_end_seconds,
            "python_build": python_build_seconds,
            "python_algebra": python_seconds,
            "python_end_to_end": python_end_to_end_seconds,
            "native_term_materialization": terms_seconds,
        },
        "checks": {
            "max_coefficient_error": max(errors, default=0.0),
            "native_algebra_speedup": (
                python_seconds / native_seconds if native_seconds else float("inf")
            ),
            "native_end_to_end_speedup": (
                python_end_to_end_seconds / native_end_to_end_seconds
                if native_end_to_end_seconds
                else float("inf")
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.modes < 1 or args.terms < 1:
        raise SystemExit("modes and terms must be positive")
    families = ("fermion", "boson") if args.family == "both" else (args.family,)
    results = []
    for family in families:
        args_for_family = argparse.Namespace(**{**vars(args), "family": family})
        results.append(run_family(args_for_family, family))
    output: object = (
        results[0]
        if len(results) == 1
        else {
            "backend": "tencirpauli",
            "family": "both",
            "results": results,
        }
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
