"""Manual TenCirPauli Fermi-Hubbard restricted-MVP study."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

import tencirpauli as tcp


def lattice_bonds(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    bonds: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            site = row * cols + col
            if col + 1 < cols:
                bonds.append((site, site + 1))
            if row + 1 < rows:
                bonds.append((site, site + cols))
    return tuple(bonds)


def model_info(rows: int, cols: int) -> dict[str, int]:
    sites = rows * cols
    if sites % 2:
        raise ValueError("the half-filled S_z=0 study requires an even site count")
    particles_per_spin = sites // 2
    configurations = math.comb(sites, particles_per_spin)
    dimension = configurations * configurations
    bonds = lattice_bonds(rows, cols)
    hopping_transitions = (
        4 * len(bonds) * math.comb(sites - 2, particles_per_spin - 1) * configurations
    )
    transition_count = dimension + hopping_transitions
    return {
        "sites": sites,
        "bonds": len(bonds),
        "particles_per_spin": particles_per_spin,
        "dimension": dimension,
        "transition_count_lower_bound": transition_count,
        "transition_bytes_lower_bound": transition_count * 32,
    }


def hubbard_terms(
    rows: int, cols: int, hopping: float, interaction: float
) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    sites = rows * cols
    terms: list[tuple[tuple[tuple[int, str], ...], complex]] = []
    for site in range(sites):
        up = site
        down = sites + site
        terms.append(
            (
                (
                    (up, "create"),
                    (up, "annihilate"),
                    (down, "create"),
                    (down, "annihilate"),
                ),
                complex(interaction),
            )
        )
    for left, right in lattice_bonds(rows, cols):
        for spin in (0, 1):
            left_mode = left + spin * sites
            right_mode = right + spin * sites
            terms.append(
                (((left_mode, "create"), (right_mode, "annihilate")), -hopping)
            )
            terms.append(
                (((right_mode, "create"), (left_mode, "annihilate")), -hopping)
            )
    return terms


def build_sector(sites: int) -> tcp.ChargeSector:
    space = tcp.OperatorSpace(fermions=2 * sites)
    total = tcp.AdditiveCharge(
        space,
        name="particle_number",
        fermions={mode: 1 for mode in range(2 * sites)},
    )
    spin_balance = tcp.AdditiveCharge(
        space,
        name="two_Sz",
        fermions={mode: (1 if mode < sites else -1) for mode in range(2 * sites)},
    )
    return tcp.ChargeSector(
        ((total, sites), (spin_balance, 0)),
        max_bytes=16 * 1024**3,
    )


def random_state(dimension: int, seed: int) -> np.ndarray[Any, Any]:
    generator = np.random.default_rng(seed)
    state = generator.normal(size=dimension) + 1j * generator.normal(size=dimension)
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--storage", choices=("eager", "lazy"), default="lazy")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--eigsh", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--ncv", type=int, default=20)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--allow-large", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = model_info(args.rows, args.cols)
    if args.preflight:
        print(json.dumps({"library": "tencirpauli", **info}, sort_keys=True))
        return
    if info["dimension"] > 2_000_000 and not args.allow_large:
        raise SystemExit(
            "refusing a large run without --allow-large; use --preflight first"
        )
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    operator = tcp.FermionOperator.from_terms(
        2 * info["sites"], hubbard_terms(args.rows, args.cols, args.t, args.u)
    )
    sector_start = time.perf_counter()
    sector = build_sector(info["sites"])
    sector_seconds = time.perf_counter() - sector_start
    build_start = time.perf_counter()
    restricted = operator.restrict_charge(sector, storage=args.storage)
    plan = restricted.mvp_plan()
    build_seconds = time.perf_counter() - build_start
    state = random_state(restricted.dimension, args.seed)
    mvp_seconds: list[float] = []
    result = state
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = restricted.apply(state)
        mvp_seconds.append(time.perf_counter() - start)

    output: dict[str, object] = {
        "library": "tencirpauli",
        "storage": args.storage,
        "rows": args.rows,
        "cols": args.cols,
        "t": args.t,
        "u": args.u,
        "dimension": restricted.dimension,
        "term_count": plan.term_count,
        "plan_estimated_bytes": plan.estimated_bytes,
        "sector_seconds": sector_seconds,
        "plan_build_seconds": build_seconds,
        "mvp_seconds": mvp_seconds,
        "mvp_seconds_median": float(np.median(mvp_seconds)),
        "output_norm": float(np.linalg.norm(result)),
    }
    if args.eigsh:
        linear = LinearOperator(
            (restricted.dimension, restricted.dimension),
            matvec=lambda vector: restricted.apply(vector),
            dtype=np.complex128,
        )
        start = time.perf_counter()
        values, vectors = eigsh(
            linear,
            k=1,
            which="SA",
            v0=state,
            tol=args.tol,
            maxiter=args.maxiter,
            ncv=args.ncv,
        )
        eigsh_seconds = time.perf_counter() - start
        ground_state = vectors[:, 0]
        energy = float(values[0].real)
        residual = np.linalg.norm(
            restricted.apply(ground_state) - energy * ground_state
        )
        output.update(
            {
                "eigsh_seconds": eigsh_seconds,
                "ground_energy": energy,
                "ground_residual": float(residual),
            }
        )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
