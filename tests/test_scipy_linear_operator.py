"""SciPy LinearOperator interoperability tests."""

from __future__ import annotations

import numpy as np
import pytest

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
    with pytest.raises(MemoryError):
        operator.native_mvp_plan().to_scipy_linear_operator(max_bytes=1).matvec(vector)


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
    assert linear.rmatvec is not None
    with pytest.raises(NotImplementedError):
        linear.rmatvec(vector)


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
    from scipy.sparse.linalg import eigsh

    values, _ = eigsh(linear, k=1, which="SA")
    np.testing.assert_allclose(values[0], np.linalg.eigvalsh(dense)[0], atol=1.0e-12)
