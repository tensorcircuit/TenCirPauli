"""Small executable Phase 7.5 Majorana/mapping/charge example."""

from __future__ import annotations

import numpy as np

import tencirpauli as tcp


def main() -> None:
    majorana = tcp.MajoranaOperator.from_terms(2, [((0, 1), 0.5), ((2, 3), -0.5j)])
    mapping = tcp.FermionQubitMapping.bravyi_kitaev(2)
    mapped = majorana.map_fermions(mapping)

    space = tcp.OperatorSpace(fermions=2)
    charge = tcp.AdditiveCharge(space, name="particle_number", fermions={0: 1, 1: 1})
    hopping = space.fermion.create(0) * space.fermion.annihilate(
        1
    ) + space.fermion.create(1) * space.fermion.annihilate(0)
    sector = charge.sector(1)
    restricted = hopping.restrict_charge(sector)
    state = np.asarray([1.0 + 0j, 0.0 + 0j])

    print("Majorana terms:", majorana.term_count)
    print("Mapped qubits:", mapped.nqubits)
    print("Sector dimension:", sector.dimension)
    print("Restricted action:", restricted.apply(state))


if __name__ == "__main__":
    main()
