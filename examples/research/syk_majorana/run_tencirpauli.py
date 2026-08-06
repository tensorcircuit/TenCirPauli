"""Manual SYK ground-state study using a Majorana native MVP plan."""

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
from scipy.sparse.linalg import LinearOperator, eigsh

import tencirpauli as tcp


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on supported local platforms."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def make_syk_terms(
    n_modes: int, coupling: float, seed: int
) -> tuple[tuple[tuple[int, ...], float], ...]:
    """Return independent quartic SYK couplings in Majorana-generator order."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-modes",
        type=int,
        default=12,
        help="number of complex fermion modes, giving N=2*n_modes Majoranas",
    )
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--state-seed", type=int, default=20260807)
    parser.add_argument(
        "--mapping",
        choices=("jordan_wigner", "parity", "bravyi_kitaev"),
        default="jordan_wigner",
    )
    parser.add_argument("--storage", choices=("lazy", "eager"), default="lazy")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--ncv", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=512 * 1024**2)
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

    build_start = time.perf_counter()
    operator = tcp.MajoranaOperator.from_terms(
        args.n_modes, ((indices, -coefficient) for indices, coefficient in raw_terms)
    )
    build_seconds = time.perf_counter() - build_start

    plan_start = time.perf_counter()
    plan = operator.compile(
        "native_mvp",
        storage=args.storage,
        mapping=args.mapping,
        max_bytes=args.max_bytes,
    )
    plan_seconds = time.perf_counter() - plan_start

    state = random_state(plan.dimension, args.state_seed)
    first_start = time.perf_counter()
    result = plan.apply(state, max_bytes=args.max_bytes)
    first_mvp_seconds = time.perf_counter() - first_start
    setup_end_to_end_seconds = time.perf_counter() - end_to_end_start
    mvp_seconds = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = plan.apply(state, max_bytes=args.max_bytes)
        mvp_seconds.append(time.perf_counter() - start)

    ncv = min(max(args.ncv, 3), plan.dimension - 1)
    linear = LinearOperator(
        (plan.dimension, plan.dimension),
        matvec=lambda vector: plan.apply(vector, max_bytes=args.max_bytes),
        dtype=np.complex128,
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
    residual = np.linalg.norm(
        plan.apply(ground_state, max_bytes=args.max_bytes) - energy * ground_state
    )
    ground_state_end_to_end_seconds = time.perf_counter() - end_to_end_start

    output: dict[str, object] = {
        "library": "tencirpauli",
        "model": "SYK_q4",
        "n_modes": args.n_modes,
        "n_majorana": 2 * args.n_modes,
        "coupling": args.coupling,
        "seed": args.seed,
        "mapping": args.mapping,
        "majorana_term_count": operator.term_count,
        "mapped_term_count": plan.term_count,
        "dimension": plan.dimension,
        "plan_storage": plan.storage,
        "plan_strategy": plan.strategy,
        "plan_estimated_bytes": plan.estimated_bytes,
        "state_bytes": state.nbytes,
        "generation_seconds": generation_seconds,
        "majorana_build_seconds": build_seconds,
        "native_mvp_plan_seconds": plan_seconds,
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
                "RAYON_NUM_THREADS",
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
