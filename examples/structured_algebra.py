"""Small executable example for the Phase 7 structured algebra API."""

from __future__ import annotations

import numpy as np

import tencirpauli as tcp


def main() -> None:
    space = tcp.OperatorSpace(bosons=1)
    boson_hamiltonian = (
        space.boson.create(0) * space.boson.annihilate(0)
        + 0.25 * space.boson.create(0)
        + 0.25 * space.boson.annihilate(0)
    )
    cutoffs = {0: 3}
    dense = boson_hamiltonian.compile("dense", boson_cutoffs=cutoffs)
    native_plan = boson_hamiltonian.compile("native_mvp", boson_cutoffs=cutoffs)
    state = np.arange(native_plan.dimension, dtype=np.complex128)
    np.testing.assert_allclose(native_plan(state), dense @ state)

    qudit = tcp.QuditWeylOperator.from_terms(
        3, [(((0, 1, 2),), 0.75 - 0.2j)], n_sites=1
    )
    backend_plan = qudit.compile("backend_mvp")
    backend_state = np.arange(3, dtype=np.complex128)
    np.testing.assert_allclose(
        tcp.backend_mvp(backend_plan)(backend_state),
        qudit.compile("dense") @ backend_state,
    )
    print("structured targets agree")


if __name__ == "__main__":
    main()
