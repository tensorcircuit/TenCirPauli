"""Manual SYK ground-state study using QuSpin's fermion Hamiltonian."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import resource
import sys
import time
from typing import Any

import numpy as np
from quspin.basis import spinless_fermion_basis_1d
from quspin.operators import hamiltonian
from scipy.sparse.linalg import LinearOperator, eigsh


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on supported local platforms."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def make_syk_terms(
    n_modes: int, coupling: float, seed: int
) -> tuple[tuple[tuple[int, ...], float], ...]:
    """Return the shared deterministic quartic SYK coupling workload."""
    n_majorana = 2 * n_modes
    scale = math.sqrt(6.0) * coupling / (n_majorana**1.5)
    generator = np.random.default_rng(seed)
    return tuple(
        (indices, float(generator.normal(loc=0.0, scale=scale)))
        for indices in itertools.combinations(range(n_majorana), 4)
    )


def random_state(dimension: int, seed: int) -> np.ndarray[Any, Any]:
    """Return a deterministic normalized complex starting vector."""
    generator = np.random.default_rng(seed)
    state = generator.normal(size=dimension) + 1j * generator.normal(size=dimension)
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)


def quspin_static(
    terms: tuple[tuple[tuple[int, ...], float], ...],
) -> list[list[object]]:
    """Expand Majorana products into QuSpin ``+``/``-`` operator strings.

    The even Majorana generator is ``c† + c`` and the odd generator is
    ``i(c† - c)``, matching TenCirPauli's Majorana convention. The factor order
    and branch phases follow the same product convention as the native batch
    expansion.
    """
    grouped: dict[str, list[list[object]]] = {}
    for indices, coefficient in terms:
        for mask in range(1 << len(indices)):
            operators: list[str] = []
            modes: list[int] = []
            # q=4 uses i^(q/2)=-1 in the shared SYK convention.
            branch_coefficient = -complex(coefficient)
            for position, majorana_index in enumerate(indices):
                create = (mask & (1 << (len(indices) - position - 1))) == 0
                operators.append("+" if create else "-")
                modes.append(majorana_index // 2)
                if majorana_index % 2:
                    branch_coefficient *= 1j if create else -1j
            grouped.setdefault("".join(operators), []).append(
                [branch_coefficient, *modes]
            )
    return [
        [operator_string, entries]
        for operator_string, entries in sorted(grouped.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-modes", type=int, default=12)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--state-seed", type=int, default=20260807)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--ncv", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_modes < 2:
        raise SystemExit("n-modes must be at least 2")
    if not math.isfinite(args.coupling) or args.coupling <= 0.0:
        raise SystemExit("coupling must be finite and positive")
    if args.repeats < 1:
        raise SystemExit("repeats must be positive")

    end_to_end_start = time.perf_counter()
    generation_start = time.perf_counter()
    raw_terms = make_syk_terms(args.n_modes, args.coupling, args.seed)
    generation_seconds = time.perf_counter() - generation_start

    basis_start = time.perf_counter()
    basis = spinless_fermion_basis_1d(args.n_modes)
    basis_seconds = time.perf_counter() - basis_start
    build_start = time.perf_counter()
    static = quspin_static(raw_terms)
    operator = hamiltonian(
        static,
        [],
        basis=basis,
        dtype=np.complex128,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
    build_seconds = time.perf_counter() - build_start

    state = random_state(operator.Ns, args.state_seed)
    first_start = time.perf_counter()
    result = operator.dot(state)
    first_mvp_seconds = time.perf_counter() - first_start
    setup_end_to_end_seconds = time.perf_counter() - end_to_end_start
    mvp_seconds = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = operator.dot(state)
        mvp_seconds.append(time.perf_counter() - start)

    ncv = min(max(args.ncv, 3), operator.Ns - 1)
    linear = LinearOperator(
        (operator.Ns, operator.Ns), matvec=operator.dot, dtype=np.complex128
    )
    eigsh_start = time.perf_counter()
    values, vectors = eigsh(
        linear,
        k=1,
        which="SA",
        v0=state,
        tol=args.tol,
        maxiter=args.maxiter,
        ncv=ncv,
    )
    eigsh_seconds = time.perf_counter() - eigsh_start
    ground_state = vectors[:, 0]
    energy = float(values[0].real)
    residual = np.linalg.norm(operator.dot(ground_state) - energy * ground_state)
    ground_state_end_to_end_seconds = time.perf_counter() - end_to_end_start

    output: dict[str, object] = {
        "library": "quspin",
        "model": "SYK_q4",
        "n_modes": args.n_modes,
        "n_majorana": 2 * args.n_modes,
        "coupling": args.coupling,
        "seed": args.seed,
        "majorana_term_count": len(raw_terms),
        "quspin_static_contributions": sum(len(entries) for _, entries in static),
        "quspin_operator_strings": len(static),
        "dimension": operator.Ns,
        "state_bytes": state.nbytes,
        "generation_seconds": generation_seconds,
        "basis_seconds": basis_seconds,
        "quspin_build_seconds": build_seconds,
        "setup_end_to_end_seconds": setup_end_to_end_seconds,
        "first_mvp_seconds": first_mvp_seconds,
        "mvp_seconds": mvp_seconds,
        "mvp_seconds_median": float(np.median(mvp_seconds)),
        "eigsh_seconds": eigsh_seconds,
        "ground_state_end_to_end_seconds": ground_state_end_to_end_seconds,
        "ground_energy": energy,
        "ground_residual": float(residual),
        "output_norm": float(np.linalg.norm(result)),
        "peak_rss_bytes": peak_rss_bytes(),
        "thread_environment": {
            name: os.environ.get(name, "unset")
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
