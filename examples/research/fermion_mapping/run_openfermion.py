"""Compare OpenFermion fermion-to-qubit mappings on deterministic workloads."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Callable, Dict, Iterable, Tuple

from common import MAPPING_NAMES, build_terms, pauli_summary
from openfermion.ops import FermionOperator
from openfermion.transforms import (
    binary_code_transform,
    bravyi_kitaev,
    jordan_wigner,
    parity_code,
)


def _median_seconds(function: Callable[[], object], repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _build_operator(n_modes: int, workload: str) -> FermionOperator:
    operator = FermionOperator()
    for factors, coefficient in build_terms(n_modes, workload):
        operator += FermionOperator(
            tuple((mode, action == "create") for mode, action in factors),
            coefficient,
        )
    return operator


def _map_operator(operator: FermionOperator, mapping: str, n_modes: int):
    if mapping == "jordan_wigner":
        return jordan_wigner(operator)
    if mapping == "parity":
        return binary_code_transform(operator, parity_code(n_modes))
    if mapping == "bravyi_kitaev":
        return bravyi_kitaev(operator, n_qubits=n_modes)
    raise ValueError(f"unsupported mapping: {mapping}")


def _pauli_items(operator, n_modes: int) -> Iterable[Tuple[str, complex]]:
    for word, coefficient in operator.terms.items():
        characters = ["I"] * n_modes
        for index, character in word:
            characters[index] = character
        yield "".join(characters), complex(coefficient)


def _measure_mapping(
    n_modes: int,
    workload: str,
    mapping: str,
    repetitions: int,
    emit_terms: bool,
) -> Dict[str, object]:
    operator = _build_operator(n_modes, workload)
    _map_operator(operator, mapping, n_modes)

    construction_seconds = _median_seconds(
        lambda: _build_operator(n_modes, workload), repetitions
    )
    mapping_seconds = _median_seconds(
        lambda: _map_operator(operator, mapping, n_modes), repetitions
    )
    end_to_end_seconds = _median_seconds(
        lambda: _map_operator(_build_operator(n_modes, workload), mapping, n_modes),
        repetitions,
    )

    mapped = _map_operator(operator, mapping, n_modes)
    summary = pauli_summary(_pauli_items(mapped, n_modes), emit_terms=emit_terms)
    return {
        "mapping": mapping,
        "construction_seconds_median": construction_seconds,
        "mapping_seconds_median": mapping_seconds,
        "end_to_end_seconds_median": end_to_end_seconds,
        **summary,
    }


def run(
    n_modes: int, workload: str, repetitions: int, emit_terms: bool = False
) -> Dict[str, object]:
    if isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be positive")
    results = [
        _measure_mapping(n_modes, workload, mapping, repetitions, emit_terms)
        for mapping in MAPPING_NAMES
    ]
    return {
        "implementation": "openfermion",
        "python": sys.executable,
        "package": "openfermion",
        "n_modes": n_modes,
        "workload": workload,
        "input_term_count": len(build_terms(n_modes, workload)),
        "repetitions": repetitions,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-modes", type=int, default=8)
    parser.add_argument("--workload", default="hubbard")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--emit-terms", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.n_modes, args.workload, args.repetitions, args.emit_terms),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
