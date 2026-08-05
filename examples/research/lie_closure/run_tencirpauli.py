"""Manual native Pauli Lie-closure dimension and timing study."""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Sequence, Tuple

from common import (
    GeneratorSpec,
    SparseOperator,
    add,
    generator_terms,
    independent,
    operator_norm,
)
from common import (
    closure as dict_closure,
)
from common import (
    commutator as dict_commutator,
)

import tencirpauli as tcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("su2", "su4", "sum2", "chain"), default="chain"
    )
    parser.add_argument("--mode", choices=("word", "sum"), default="word")
    parser.add_argument("--nqubits", type=int, default=4)
    parser.add_argument("--max-dimension", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--max-bytes", type=int, default=512 * 1024**2)
    return parser.parse_args()


def to_dict(operator: tcp.PauliOperator) -> SparseOperator:
    """Convert one native result to deterministic string-keyed coordinates."""
    return operator.to_dict()


def make_generators(
    specification: GeneratorSpec, nqubits: int
) -> List[tcp.PauliOperator]:
    """Construct anti-Hermitian native generators."""
    return [tcp.PauliOperator.from_terms(nqubits, terms) for terms in specification]


def native_closure(
    generators: Sequence[tcp.PauliOperator],
    mode: str,
    max_dimension: int,
    tolerance: float,
    max_bytes: int,
) -> Tuple[List[tcp.PauliOperator], int, bool, float, float]:
    """Compute a deterministic bounded closure through coarse native calls."""
    start = time.perf_counter()
    basis: List[tcp.PauliOperator] = []
    basis_dicts: List[SparseOperator] = []
    words: set[str] = set()
    for generator in generators:
        generator_dict = to_dict(generator)
        if mode == "word":
            word = next(iter(generator_dict))
            if word in words:
                continue
            words.add(word)
            basis.append(generator)
            basis_dicts.append(generator_dict)
        elif independent(basis_dicts, generator_dict, tolerance):
            basis.append(generator)
            basis_dicts.append(generator_dict)
        if len(basis) >= max_dimension:
            return basis, 0, False, 0.0, time.perf_counter() - start

    candidate_count = 0
    index = 0
    while index < len(basis):
        left = basis[index]
        for right in tuple(basis):
            candidate_count += 1
            candidate = left.commutator(right, max_bytes=max_bytes)
            candidate_dict = to_dict(candidate)
            if mode == "word":
                if not candidate_dict:
                    continue
                if len(candidate_dict) != 1:
                    raise AssertionError("word closure produced a non-word bracket")
                word = next(iter(candidate_dict))
                if word in words:
                    continue
                words.add(word)
                basis.append(candidate)
                basis_dicts.append(candidate_dict)
            elif independent(basis_dicts, candidate_dict, tolerance):
                basis.append(candidate)
                basis_dicts.append(candidate_dict)
            else:
                continue
            if len(basis) >= max_dimension:
                return basis, candidate_count, False, 0.0, time.perf_counter() - start
        index += 1

    jacobi = 0.0
    if len(generators) >= 3:
        first, second, third = (to_dict(generator) for generator in generators[:3])
        jacobi_operator = add(
            add(
                dict_commutator(first, dict_commutator(second, third)),
                dict_commutator(second, dict_commutator(third, first)),
            ),
            dict_commutator(third, dict_commutator(first, second)),
        )
        jacobi = operator_norm(jacobi_operator)
    return basis, candidate_count, True, jacobi, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if args.nqubits < 1 or args.max_dimension < 1 or args.tolerance <= 0.0:
        raise SystemExit("nqubits, max-dimension, and tolerance must be positive")
    if args.case == "sum2" and args.mode == "word":
        raise SystemExit("sum2 requires --mode sum")

    specification, effective_nqubits = generator_terms(args.case, args.nqubits)
    build_start = time.perf_counter()
    generators = make_generators(specification, effective_nqubits)
    build_seconds = time.perf_counter() - build_start
    native_basis, candidate_count, complete, jacobi, closure_seconds = native_closure(
        generators,
        args.mode,
        args.max_dimension,
        args.tolerance,
        args.max_bytes,
    )

    dict_generators = [
        {word: coefficient for word, coefficient in terms} for terms in specification
    ]
    dict_basis, dict_candidates, dict_complete, dict_jacobi = dict_closure(
        dict_generators,
        args.mode,
        args.max_dimension,
        args.tolerance,
    )
    native_words = {word for operator in native_basis for word in to_dict(operator)}
    dict_words = {word for operator in dict_basis for word in operator}
    if args.mode == "word" and native_words != dict_words:
        raise AssertionError("native and dict closures have different word supports")
    if len(native_basis) != len(dict_basis) or complete != dict_complete:
        raise AssertionError("native and dict closure dimensions differ")

    output: Dict[str, object] = {
        "backend": "tencirpauli",
        "case": args.case,
        "mode": args.mode,
        "nqubits": effective_nqubits,
        "generator_count": len(generators),
        "initial_term_counts": [generator.term_count for generator in generators],
        "closure_dimension": len(native_basis),
        "ambient_su_dimension": (4**effective_nqubits) - 1,
        "complete": complete,
        "candidate_brackets": candidate_count,
        "dict_candidate_brackets": dict_candidates,
        "dict_closure_dimension": len(dict_basis),
        "jacobi_residual": jacobi,
        "dict_jacobi_residual": dict_jacobi,
        "timings_seconds": {
            "native_generator_build": build_seconds,
            "native_closure": closure_seconds,
        },
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
