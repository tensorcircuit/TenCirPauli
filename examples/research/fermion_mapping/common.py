"""Shared deterministic workloads for the fermion-mapping comparison."""

from __future__ import annotations

import hashlib
import itertools
from typing import Iterable, List, Tuple


FermionFactor = Tuple[int, str]
FermionTerm = Tuple[Tuple[FermionFactor, ...], complex]
MAPPING_NAMES = ("jordan_wigner", "parity", "bravyi_kitaev")


def build_terms(n_modes: int, workload: str = "hubbard") -> List[FermionTerm]:
    """Build a reproducible local or all-to-all spin-interleaved workload."""
    if isinstance(n_modes, bool) or not isinstance(n_modes, int) or n_modes < 2:
        raise ValueError("n_modes must be an integer greater than or equal to 2")
    if n_modes % 2:
        raise ValueError("the Hubbard workload requires an even n_modes")
    if workload not in {"hubbard", "all_to_all", "dense_quartic"}:
        raise ValueError("workload must be 'hubbard', 'all_to_all', or 'dense_quartic'")

    hopping = 0.5
    interaction = 1.3
    chemical_potential = -0.2
    n_sites = n_modes // 2
    terms: List[FermionTerm] = []

    if workload == "dense_quartic":
        for quartet in itertools.combinations(range(n_modes), 4):
            for created in itertools.combinations(quartet, 2):
                annihilated = tuple(mode for mode in quartet if mode not in created)
                if created[0] > annihilated[0]:
                    continue
                coefficient = 0.01 * (1 + sum(quartet) % 17)
                terms.extend(
                    (
                        (
                            tuple((mode, "create") for mode in created)
                            + tuple(
                                (mode, "annihilate") for mode in reversed(annihilated)
                            ),
                            coefficient,
                        ),
                        (
                            tuple((mode, "create") for mode in annihilated)
                            + tuple((mode, "annihilate") for mode in reversed(created)),
                            coefficient,
                        ),
                    )
                )
        return terms

    for mode in range(n_modes):
        terms.append((((mode, "create"), (mode, "annihilate")), chemical_potential))

    for site in range(n_sites):
        up = 2 * site
        down = up + 1
        terms.append(
            (
                (
                    (up, "create"),
                    (up, "annihilate"),
                    (down, "create"),
                    (down, "annihilate"),
                ),
                interaction,
            )
        )

    for spin in (0, 1):
        site_pairs = ((site, site + 1) for site in range(n_sites - 1))
        if workload == "all_to_all":
            site_pairs = (
                (left_site, right_site)
                for left_site in range(n_sites)
                for right_site in range(left_site + 1, n_sites)
            )
        for left_site, right_site in site_pairs:
            left = 2 * left_site + spin
            right = 2 * right_site + spin
            terms.extend(
                (
                    (((left, "create"), (right, "annihilate")), -hopping),
                    (((right, "create"), (left, "annihilate")), -hopping),
                )
            )

    if workload == "all_to_all":
        for left in range(n_modes):
            for right in range(left + 1, n_modes):
                terms.append(
                    (
                        (
                            (left, "create"),
                            (left, "annihilate"),
                            (right, "create"),
                            (right, "annihilate"),
                        ),
                        interaction,
                    )
                )
    return terms


def parse_mode_list(value: str) -> Tuple[int, ...]:
    """Parse a comma-separated list of even mode counts."""
    modes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not modes:
        raise ValueError("n_modes must contain at least one mode count")
    return modes


def pauli_summary(
    terms: Iterable[Tuple[str, complex]], emit_terms: bool = False
) -> dict[str, object]:
    """Return bounded metadata and an optional term listing.

    OpenFermion compresses coefficients below its numerical tolerance. Apply
    the same comparison tolerance here so cancellation-heavy workloads do not
    differ only because one implementation retains round-off residues.
    """
    zero_tolerance = 1e-12
    normalized = sorted(
        (word, complex(value))
        for word, value in terms
        if abs(complex(value)) > zero_tolerance
    )
    digest = hashlib.sha256()
    records: List[dict[str, object]] = []
    max_weight = 0
    for word, value in normalized:
        max_weight = max(max_weight, sum(character != "I" for character in word))
        rounded_real = round(value.real, 10)
        rounded_imag = round(value.imag, 10)
        digest.update(f"{word}\0{rounded_real.hex()}\0{rounded_imag.hex()}\n".encode())
        if emit_terms:
            records.append(
                {
                    "word": word,
                    "real": float(value.real),
                    "imag": float(value.imag),
                }
            )
    result: dict[str, object] = {
        "term_count": len(normalized),
        "max_weight": max_weight,
        "term_digest": digest.hexdigest(),
        "zero_tolerance": zero_tolerance,
        "terms_emitted": emit_terms,
    }
    if emit_terms:
        result["terms"] = records
    return result
