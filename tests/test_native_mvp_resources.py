"""Native storage and reusable-execution contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

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
    eager = restricted.mvp_plan(storage="eager")
    assert first.data.flags.writeable is False
    assert eager.indptr.flags.writeable is False
    assert eager.columns.flags.writeable is False
    assert eager.coefficients.flags.writeable is False
    assert "indptr" not in type(eager).__slots__
    assert "columns" not in type(eager).__slots__
    assert "coefficients" not in type(eager).__slots__


def test_restricted_facade_concurrent_first_materialization_shares_cache() -> None:
    space = tcp.OperatorSpace(fermions=8)
    number = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(8)})
    operator = space.fermion.create(0) * space.fermion.annihilate(1)
    restricted = operator.restrict_charge(number.sector(4))
    before = restricted.estimated_bytes
    with ThreadPoolExecutor(max_workers=4) as executor:
        matrices = list(
            executor.map(lambda _: restricted.csr(max_bytes=None), range(4))
        )
    assert all(
        matrix.shape == (restricted.dimension, restricted.dimension)
        for matrix in matrices
    )
    assert restricted.estimated_bytes > before
    assert restricted._eager_plan is not None


def test_restricted_materialization_preflights_before_cache_and_allows_retry() -> None:
    space = tcp.OperatorSpace(fermions=10)
    number = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(10)})
    operator = space.fermion.create(0) * space.fermion.annihilate(1)
    restricted = operator.restrict_charge(number.sector(5))
    before = restricted.estimated_bytes
    with pytest.raises(MemoryError):
        restricted.dense(max_bytes=100_000)
    assert restricted.estimated_bytes == before
    dense = restricted.dense(max_bytes=None)
    assert dense.shape == (restricted.dimension, restricted.dimension)


@pytest.mark.parametrize("target", ["coo", "csr"])
def test_sparse_materialization_failure_does_not_publish_cache(target: str) -> None:
    space = tcp.OperatorSpace(fermions=6)
    number = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(6)})
    operator = space.fermion.create(0) * space.fermion.annihilate(1)
    restricted = operator.restrict_charge(number.sector(3))
    reference = operator.restrict_charge(number.sector(3)).mvp_plan(storage="eager")
    target_bytes = (
        reference.transition_count * 32
        if target == "coo"
        else (restricted.dimension + 1) * 8 + reference.transition_count * 24
    )
    before = restricted.estimated_bytes
    limit = before + reference.estimated_bytes + target_bytes - 1
    with pytest.raises(MemoryError):
        getattr(restricted, target)(max_bytes=limit)
    assert restricted.estimated_bytes == before
    matrix = getattr(restricted, target)(max_bytes=None)
    assert matrix.shape == (restricted.dimension, restricted.dimension)


def test_spinful_cache_budget_falls_back_without_exceeding_plan_budget() -> None:
    sites = 16
    space = tcp.OperatorSpace(fermions=2 * sites)
    total = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(2 * sites)})
    balance = tcp.AdditiveCharge(
        space,
        fermions={index: (1 if index < sites else -1) for index in range(2 * sites)},
    )
    sector = tcp.ChargeSector(((total, sites), (balance, 0)))
    operator = tcp.FermionOperator.from_terms(
        2 * sites, [(((0, "create"), (1, "annihilate")), 1.0)]
    )
    bounded = operator.restrict_charge(sector, max_bytes=350_000)
    plan = bounded.mvp_plan()
    assert plan.estimated_bytes <= 350_000


def test_large_eager_charge_parallel_rows_match_serial_rows() -> None:
    nfermions = 16
    space = tcp.OperatorSpace(fermions=nfermions, bosons=1)
    charge = tcp.AdditiveCharge(
        space, fermions={index: 1 for index in range(nfermions)}
    )
    operator: Any = None
    for left in range(nfermions):
        for right in range(left + 1, nfermions):
            term = space.fermion.create(left) * space.fermion.annihilate(right)
            term = term + space.fermion.create(right) * space.fermion.annihilate(left)
            operator = term if operator is None else operator + term
    restricted = operator.restrict_charge(
        tcp.ChargeSector(((charge, nfermions // 2),), boson_cutoffs={0: 0})
    )
    plan = restricted.mvp_plan(storage="eager")
    assert plan.transition_count > 1 << 19
    state = np.arange(plan.dimension, dtype=np.complex128)
    serial = plan._native_plan.apply_with_parallelism(state, 2**63 - 1, False)
    parallel = plan._native_plan.apply_with_parallelism(state, 2**63 - 1, True)
    np.testing.assert_array_equal(serial, parallel)


def test_cached_eager_plan_still_obeys_call_budget_and_exposes_csr_storage() -> None:
    space = tcp.OperatorSpace(fermions=3)
    number = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1, 2: 1})
    operator = space.fermion.create(0) * space.fermion.annihilate(1)
    restricted = operator.restrict_charge(number.sector(1))
    eager = restricted.mvp_plan(storage="eager", max_bytes=None)
    assert eager.indptr.shape == (restricted.dimension + 1,)
    with pytest.raises(MemoryError):
        restricted.mvp_plan(storage="eager", max_bytes=0)
    with pytest.raises(MemoryError):
        restricted.csr(max_bytes=0)
    assert restricted.mvp_plan(storage="eager", max_bytes=None) is eager


def test_native_allocating_mvp_returns_one_owned_output() -> None:
    plan = tcp.PauliOperator.from_terms(4, [("XXII", 1.0)]).compile("native_mvp")
    result = plan.apply(np.ones(plan.dimension, dtype=np.complex128))
    assert result.flags.c_contiguous
    assert result.base is not None


def test_apply_into_overwrite_overlap_and_concurrent_reuse() -> None:
    plan = tcp.PauliOperator.from_terms(3, [("XZI", 1.0)]).compile("native_mvp")
    state = np.arange(plan.dimension, dtype=np.complex128)
    original = state.copy()
    first = np.full(plan.dimension, 9.0 + 3.0j, dtype=np.complex128)
    second = np.full(plan.dimension, -4.0 + 2.0j, dtype=np.complex128)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda output: plan.apply_into(state, output, max_bytes=0),
                (first, second),
            )
        )
    expected = plan.apply(state)
    np.testing.assert_array_equal(state, original)
    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected)

    readonly = np.empty_like(state)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        plan.apply_into(state, readonly)
    noncontiguous = np.empty(plan.dimension * 2, dtype=np.complex128)[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        plan.apply_into(state, noncontiguous)
    overlapping = np.empty(plan.dimension + 1, dtype=np.complex128)
    with pytest.raises(ValueError, match="overlap"):
        plan.apply_into(overlapping[:-1], overlapping[1:])


def test_u1_lazy_apply_into_accounts_for_native_scratch() -> None:
    operator = tcp.PauliOperator.from_terms(
        8, [("XX" + "I" * 6, 1.0), ("YY" + "I" * 6, 1.0)]
    )
    restricted = operator.restrict_charge(tcp.U1Sector(8, 1))
    with pytest.raises(MemoryError):
        restricted.mvp_plan(max_bytes=0)
    plan = restricted.mvp_plan()
    state = np.ones(plan.dimension, dtype=np.complex128)
    output = np.empty_like(state)
    with pytest.raises(MemoryError):
        plan.apply_into(state, output, max_bytes=0)


def test_u1_materialization_preflights_before_publishing_eager_cache() -> None:
    operator = tcp.PauliOperator.from_terms(8, [("Z" + "I" * 7, 1.0)])
    restricted = operator.restrict_charge(tcp.U1Sector(8, 1))
    before = restricted.estimated_bytes
    with pytest.raises(MemoryError):
        restricted.dense(max_bytes=700)
    assert restricted.estimated_bytes == before
    dense = restricted.dense(max_bytes=None)
    assert dense.shape == (restricted.dimension, restricted.dimension)


def test_structured_eager_retains_a_real_bounded_cache() -> None:
    space = tcp.OperatorSpace(bosons=2)
    operator = space.boson.create(0) * space.boson.annihilate(1)
    lazy = operator.compile("native_mvp", storage="lazy", boson_cutoffs={0: 3, 1: 3})
    eager = operator.compile("native_mvp", storage="eager", boson_cutoffs={0: 3, 1: 3})
    assert eager.storage == "eager"
    assert eager.estimated_bytes == lazy.estimated_bytes
    assert eager.strategy == lazy.strategy
    state = np.arange(lazy.dimension, dtype=np.complex128)
    np.testing.assert_allclose(lazy.apply(state), eager.apply(state))


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
        plan = operator.restrict_charge(charge.sector(1)).mvp_plan(storage="eager")
    else:
        operator = tcp.PauliOperator.from_terms(2, [("ZI", 1.0)])
        plan = operator.restrict_charge(tcp.U1Sector(2, 1)).mvp_plan(storage="eager")
    state = np.arange(plan.dimension, dtype=np.complex128)
    output = np.full(plan.dimension, 7.0 + 2.0j, dtype=np.complex128)
    assert plan.apply_into(state, output, max_bytes=None) is None
    np.testing.assert_allclose(output, plan.apply(state, max_bytes=None))
    if factory in {"pauli", "charge", "u1"}:
        for buffer in (output, np.empty_like(output)):
            buffer.fill(9.0 + 4.0j)
            plan.apply_into(state, buffer, max_bytes=0)
            np.testing.assert_allclose(buffer, plan.apply(state, max_bytes=None))
    with pytest.raises(ValueError, match="overlap"):
        plan.apply_into(state, state, max_bytes=None)
    wrong_shape = np.full(plan.dimension + 1, 9.0 + 4.0j, dtype=np.complex128)
    with pytest.raises(ValueError, match="shape"):
        plan.apply_into(state, wrong_shape, max_bytes=None)
    np.testing.assert_array_equal(wrong_shape, np.full_like(wrong_shape, 9.0 + 4.0j))
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
