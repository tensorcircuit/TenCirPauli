"""SciPy LinearOperator interoperability tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh, expm_multiply

import tencirpauli as tcp


def _fermion_charge_plan(storage: str):
    space = tcp.OperatorSpace(fermions=2)
    charge = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1})
    operator = tcp.FermionOperator.from_terms(
        2,
        (
            (((0, "create"), (1, "annihilate")), 1.0),
            (((1, "create"), (0, "annihilate")), 1.0),
        ),
    )
    return operator.restrict_charge(charge.sector(1), storage=storage).mvp_plan(
        storage=storage
    )


def test_native_plan_linear_operator_matches_apply_and_matmat() -> None:
    operator = tcp.PauliOperator.from_terms(2, (("XX", 0.5), ("ZI", -1.25j)))
    plan = operator.native_mvp_plan()
    linear = plan.to_scipy_linear_operator()
    rng = np.random.default_rng(20260807)
    vector = rng.normal(size=4) + 1j * rng.normal(size=4)
    np.testing.assert_allclose(linear.matvec(vector), plan.apply(vector))
    np.testing.assert_allclose(
        linear.matvec(vector[:, None]), plan.apply(vector)[:, None]
    )
    columns = np.column_stack((vector, 2.0 * vector))
    np.testing.assert_allclose(
        linear.matmat(columns),
        np.column_stack((plan.apply(vector), plan.apply(2.0 * vector))),
    )
    assert linear.shape == (4, 4)
    assert linear.dtype == np.dtype(np.complex128)
    with pytest.raises(ValueError, match=r"dimension mismatch|shape"):
        linear.matvec(np.ones(3))
    with pytest.raises(ValueError, match=r"dimension mismatch|shape"):
        linear.matvec(np.ones((3, 1)))
    with pytest.raises(ValueError, match=r"dimension mismatch|shape"):
        linear.matvec(np.ones((1, 4)))
    with pytest.raises(ValueError, match=r"dimension mismatch|shape"):
        linear.matvec(np.ones((4, 1, 1)))


def test_pauli_convenience_linear_operator_compiles_and_reuses_native_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = tcp.PauliOperator.from_terms(2, (("XX", 1.0), ("IZ", 0.25)))
    vector = np.random.default_rng(20260806).normal(
        size=4
    ) + 1j * np.random.default_rng(7).normal(size=4)
    expected = operator.native_mvp_plan().apply(vector)
    calls = 0
    original = tcp.PauliOperator.native_mvp_plan

    def counted(self: tcp.PauliOperator, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(self, **kwargs)

    monkeypatch.setattr(tcp.PauliOperator, "native_mvp_plan", counted)
    linear = operator.to_scipy_linear_operator()
    np.testing.assert_allclose(linear @ vector, expected)
    np.testing.assert_allclose(linear @ (2.0 * vector), 2.0 * expected)
    assert calls == 1
    with pytest.raises(TypeError):
        operator.to_scipy_linear_operator(mapping="jordan_wigner")  # type: ignore[call-arg]
    np.testing.assert_allclose(
        operator.native_mvp_plan().to_scipy_linear_operator(max_bytes=1).matvec(vector),
        expected,
    )


def test_backend_plan_does_not_advertise_native_scipy_interop() -> None:
    operator = tcp.PauliOperator.from_terms(2, (("XX", 0.5), ("ZI", -0.25)))
    backend_plan = operator.backend_mvp_plan()
    assert not hasattr(backend_plan, "to_scipy_linear_operator")


@pytest.mark.parametrize(
    "storage, plan_type", [("lazy", "ChargeLazyMvpPlan"), ("eager", "ChargeMvpPlan")]
)
def test_charge_plan_linear_operator_supports_both_storage_strategies(
    storage: str, plan_type: str
) -> None:
    plan = _fermion_charge_plan(storage)
    assert type(plan).__name__ == plan_type
    linear = plan.to_scipy_linear_operator()
    vector = np.array([1.0 + 2.0j, -0.5 + 0.25j])
    np.testing.assert_allclose(linear.matvec(vector), plan.apply(vector))
    np.testing.assert_allclose(linear.rmatvec(vector), plan.apply(vector))


def test_u1_plan_linear_operator_and_eigsh_match_dense_reference() -> None:
    operator = tcp.PauliOperator.from_terms(
        4,
        (
            ("ZIII", 0.5),
            ("IZII", -0.5),
            ("XXII", 0.25),
            ("YYII", 0.25),
        ),
    )
    restricted = operator.restrict_charge(tcp.U1Sector(4, 2))
    plan = restricted.mvp_plan()
    linear = plan.to_scipy_linear_operator()
    vector = np.arange(6, dtype=np.float64) + 1j * np.arange(6, dtype=np.float64)
    np.testing.assert_allclose(linear.matvec(vector), plan.apply(vector))
    dense = restricted.dense()
    np.testing.assert_allclose(linear.matvec(vector), dense @ vector)
    values, _ = eigsh(linear, k=1, which="SA")
    np.testing.assert_allclose(values[0], np.linalg.eigvalsh(dense)[0], atol=1.0e-12)


def _adjoint_case(kind: str, storage: str) -> tuple[Any, np.ndarray]:
    kwargs: dict[str, Any] = {}
    if kind == "pauli":
        operator = tcp.PauliOperator.from_terms(2, [("XY", 0.3 + 0.7j), ("ZI", -0.2j)])
    elif kind == "weyl":
        operator = tcp.QuditWeylOperator.from_terms(
            3, [(((0, 1, 2),), 0.3 + 0.7j), (((1, 2, 1),), -0.2j)], n_sites=2
        )
    elif kind == "structured_qubit":
        space = tcp.OperatorSpace(qubits=2)
        operator = (0.3 + 0.7j) * space.qubit.y(0) + space.qubit.x(1)
    elif kind in {"fermion", "jordan_wigner", "parity", "bravyi_kitaev"}:
        space = tcp.OperatorSpace(fermions=3)
        operator = (0.3 + 0.7j) * space.fermion.create(0) * space.fermion.annihilate(2)
        if kind != "fermion":
            operator = operator.map_fermions(kind)
    elif kind == "boson":
        space = tcp.OperatorSpace(bosons=1)
        operator = (0.3 + 0.7j) * space.boson.create(0)
        kwargs["boson_cutoffs"] = {0: 3}
    elif kind == "hybrid":
        space = tcp.OperatorSpace(fermions=1, bosons=1, qubits=1, qudits=(3,))
        operator = (
            (0.3 + 0.7j)
            * space.fermion.create(0)
            * space.boson.annihilate(0)
            * space.qubit.y(0)
            * space.qudit.weyl(0, 1, 2)
        )
        kwargs["boson_cutoffs"] = {0: 1}
    elif kind == "u1":
        operator = tcp.PauliOperator.from_terms(
            3, [("XXI", 0.3 + 0.7j), ("YYI", 0.3 + 0.7j), ("ZII", -0.2j)]
        )
        restricted = operator.restrict_charge(tcp.U1Sector(3, 1), storage=storage)
        return restricted.mvp_plan(storage=storage), restricted.dense()
    else:
        if kind == "charge_pauli":
            space = tcp.OperatorSpace(qubits=3)
            operator = tcp.PauliOperator.from_terms(
                3, [("XXI", 0.3 + 0.7j), ("YYI", 0.3 + 0.7j), ("ZII", -0.2j)]
            )
            charge = tcp.AdditiveCharge(space, qubits={i: (0, 1) for i in range(3)})
            sector = charge.sector(1)
        elif kind == "charge_fermion":
            space = tcp.OperatorSpace(fermions=3)
            operator = (
                (0.3 + 0.7j) * space.fermion.create(0) * space.fermion.annihilate(2)
            )
            charge = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1, 2: 1})
            sector = charge.sector(1)
        elif kind == "charge_boson":
            space = tcp.OperatorSpace(bosons=2)
            operator = (0.3 + 0.7j) * space.boson.create(0) * space.boson.annihilate(1)
            charge = tcp.AdditiveCharge(space, bosons={0: 1, 1: 1})
            sector = tcp.ChargeSector(((charge, 2),), boson_cutoffs={0: 2, 1: 2})
        elif kind == "charge_factor":
            space = tcp.OperatorSpace(fermions=2, qubits=2)
            operator = (
                (0.3 + 0.7j)
                * space.fermion.create(0)
                * space.fermion.annihilate(1)
                * space.qubit.y(0)
            )
            charge = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1})
            sector = charge.sector(1)
        elif kind == "charge_generic":
            space = tcp.OperatorSpace(fermions=1, bosons=1)
            operator = (
                (0.3 + 0.7j) * space.fermion.create(0) * space.boson.annihilate(0)
            )
            charge = tcp.AdditiveCharge(space, fermions={0: 1}, bosons={0: 1})
            sector = tcp.ChargeSector(((charge, 2),), boson_cutoffs={0: 2})
        else:
            raise AssertionError(kind)
        restricted = operator.restrict_charge(sector, storage=storage)
        return restricted.mvp_plan(storage=storage), restricted.dense()
    return operator.compile("native_mvp", storage=storage, **kwargs), operator.compile(
        "dense", **kwargs
    )


@pytest.mark.parametrize("storage", ["lazy", "eager"])
@pytest.mark.parametrize(
    "kind",
    [
        "pauli",
        "weyl",
        "structured_qubit",
        "fermion",
        "jordan_wigner",
        "parity",
        "bravyi_kitaev",
        "boson",
        "hybrid",
        "u1",
        "charge_fermion",
        "charge_boson",
        "charge_factor",
        "charge_generic",
        "charge_pauli",
    ],
)
def test_native_adjoint_matches_dense(kind: str, storage: str) -> None:
    plan, dense = _adjoint_case(kind, storage)
    rng = np.random.default_rng(19)
    u, v = rng.normal(size=(2, plan.dimension)) + 1j * rng.normal(
        size=(2, plan.dimension)
    )
    np.testing.assert_allclose(plan.apply_adjoint(v), dense.conj().T @ v, atol=1e-12)
    np.testing.assert_allclose(
        np.vdot(u, plan.apply(v)), np.vdot(plan.apply_adjoint(u), v), atol=1e-12
    )
    columns = np.column_stack((u, v))
    np.testing.assert_allclose(
        plan.to_scipy_linear_operator().H @ columns,
        dense.conj().T @ columns,
        atol=1e-12,
    )
    with pytest.raises(MemoryError):
        plan._execution.apply_adjoint(v, 0)


@pytest.mark.parametrize("kind", ["pauli", "u1", "charge_generic"])
def test_scipy_exponential_of_nonhermitian_plan_matches_dense(kind: str) -> None:
    plan, dense = _adjoint_case(kind, "lazy")
    rng = np.random.default_rng(19)
    state = rng.normal(size=plan.dimension) + 1j * rng.normal(size=plan.dimension)
    np.testing.assert_allclose(
        expm_multiply(
            -0.2j * plan.to_scipy_linear_operator(),
            state,
            traceA=-0.2j * np.trace(dense),
        ),
        expm(-0.2j * dense) @ state,
        atol=1e-12,
    )


def test_scipy_exponential_of_hermitian_operator_at_multiple_times() -> None:
    operator = tcp.PauliOperator.from_terms(2, [("XI", 1.0), ("ZI", 0.2)])
    linear = operator.to_scipy_linear_operator()
    state = np.array([1, 0, 0, 0], dtype=np.complex128)
    generator = -0.2j * operator.dense()
    actual = expm_multiply(-0.2j * linear, state, start=0, stop=1, num=3, traceA=0.0)
    expected = np.stack([expm(t * generator) @ state for t in [0, 0.5, 1]])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_parallel_fermion_adjoint_matches_mapped_pauli_reference() -> None:
    operator = tcp.FermionOperator.from_terms(
        14,
        [(((i, "create"), (i + 1, "annihilate")), 0.3 + 0.07j * i) for i in range(13)],
    )
    plan = operator.compile("native_mvp")
    state = np.arange(plan.dimension, dtype=np.complex128) + 0.5j
    reference = operator.map_fermions().adjoint().native_mvp_plan()
    np.testing.assert_allclose(
        plan.apply_adjoint(state), reference.apply(state), atol=1e-10
    )
