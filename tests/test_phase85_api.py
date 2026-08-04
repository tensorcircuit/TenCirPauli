"""Phase 8.5 explicit native-storage and reusable-execution contracts."""

from __future__ import annotations

import numpy as np
import pytest

import tencirpauli as tcp


def test_pauli_storage_is_fixed_and_eager_budget_is_strict() -> None:
    operator = tcp.PauliOperator.from_terms(3, [("XZI", 1.0), ("IXX", 0.5)])
    lazy = operator.compile("native_mvp")
    assert lazy.storage == "lazy"
    assert lazy.strategy == "term_direct"
    with pytest.raises(MemoryError):
        operator.compile("native_mvp", storage="eager", max_bytes=1)
    eager = operator.compile("native_mvp", storage="eager", max_bytes=None)
    assert eager.storage == "eager"
    assert eager.strategy == "x_mask_diagonal"
    assert lazy.storage == "lazy"


def test_restricted_facade_materialization_is_cached_without_upgrading_lazy_plan() -> (
    None
):
    space = tcp.OperatorSpace(fermions=3)
    number = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1, 2: 1})
    operator = space.fermion.create(0) * space.fermion.annihilate(1)
    restricted = operator.restrict_charge(number.sector(1))
    lazy_plan = restricted.mvp_plan()
    before = restricted.estimated_bytes
    first = restricted.csr()
    after = restricted.estimated_bytes
    assert lazy_plan.storage == "lazy"
    assert after > before
    assert restricted.mvp_plan() is lazy_plan
    assert restricted.mvp_plan(storage="eager") is restricted.mvp_plan(storage="eager")
    assert first.data.flags.writeable is False


@pytest.mark.parametrize("factory", ["pauli", "structured", "charge", "u1"])
def test_cpu_native_apply_into_is_strict_and_matches_apply(factory: str) -> None:
    if factory == "pauli":
        plan = tcp.PauliOperator.from_terms(2, [("XY", 1.0)]).compile("native_mvp")
    elif factory == "structured":
        space = tcp.OperatorSpace(bosons=1)
        plan = space.boson.create(0).compile("native_mvp", boson_cutoffs={0: 2})
    elif factory == "charge":
        space = tcp.OperatorSpace(fermions=2)
        charge = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1})
        operator = space.fermion.create(0) * space.fermion.annihilate(1)
        plan = operator.restrict_charge(charge.sector(1)).mvp_plan()
    else:
        operator = tcp.PauliOperator.from_terms(2, [("ZI", 1.0)])
        plan = operator.restrict_charge(tcp.U1Sector(2, 1)).mvp_plan()
    state = np.arange(plan.dimension, dtype=np.complex128)
    output = np.full(plan.dimension, 7.0 + 2.0j, dtype=np.complex128)
    assert plan.apply_into(state, output, max_bytes=None) is None
    np.testing.assert_allclose(output, plan.apply(state, max_bytes=None))
    with pytest.raises(ValueError, match="overlap"):
        plan.apply_into(state, state, max_bytes=None)
    with pytest.raises(TypeError, match="complex128"):
        plan.apply_into(state.astype(np.complex64), output, max_bytes=None)


def test_u1_restriction_alias_is_deprecated_and_unified_entry_point_is_supported() -> (
    None
):
    operator = tcp.PauliOperator.from_terms(3, [("ZII", 1.0)])
    sector = tcp.U1Sector(3, 1)
    with pytest.warns(DeprecationWarning, match="restrict_charge"):
        compatibility = operator.restrict_u1(sector)
    unified = operator.restrict_charge(sector)
    assert compatibility.mvp_plan().storage == "lazy"
    assert unified.mvp_plan().storage == "lazy"
    np.testing.assert_allclose(
        compatibility.apply(np.ones(3, dtype=np.complex128)),
        unified.apply(np.ones(3, dtype=np.complex128)),
    )
