"""P4 Hamiltonian target differential and allocation-guard tests."""

from __future__ import annotations

import numpy as np
import pytest
from reference import dense_operator

from tencirpauli import BackendMVPPlan, PauliOperator


def make_operator(nqubits: int) -> PauliOperator:
    structures = [
        tuple((index + qubit) % 4 for qubit in range(nqubits))
        for index in range(min(10, 4**nqubits))
    ]
    terms = tuple(
        (structure, complex(index + 1, -index / 3))
        for index, structure in enumerate(structures)
    )
    return PauliOperator.from_terms(nqubits, terms)


def reconstruct_coo(target: object) -> np.ndarray:
    assert hasattr(target, "row")
    assert hasattr(target, "column")
    assert hasattr(target, "data")
    assert hasattr(target, "shape")
    result = np.zeros(target.shape, dtype=np.complex128)  # type: ignore[attr-defined]
    result[target.row, target.column] = target.data  # type: ignore[attr-defined]
    return result


def reconstruct_csr(target: object) -> np.ndarray:
    assert hasattr(target, "indptr")
    assert hasattr(target, "indices")
    assert hasattr(target, "data")
    assert hasattr(target, "shape")
    result = np.zeros(target.shape, dtype=np.complex128)  # type: ignore[attr-defined]
    for row in range(target.shape[0]):  # type: ignore[attr-defined]
        start = target.indptr[row]  # type: ignore[attr-defined]
        end = target.indptr[row + 1]  # type: ignore[attr-defined]
        result[row, target.indices[start:end]] = target.data[start:end]  # type: ignore[attr-defined]
    return result


@pytest.mark.parametrize("nqubits", (0, 1, 2, 5))
def test_dense_coo_csr_and_mvp_match_independent_numpy_reference(nqubits: int) -> None:
    operator = make_operator(nqubits)
    structures = tuple(term.word.to_codes() for term in operator.terms)
    coefficients = tuple(term.coefficient for term in operator.terms)
    expected = dense_operator(nqubits, structures, coefficients)
    dense = operator.dense()
    np.testing.assert_allclose(dense, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        reconstruct_coo(operator.coo()), expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        reconstruct_csr(operator.csr()), expected, rtol=1e-12, atol=1e-12
    )
    state = np.arange(1 << nqubits, dtype=np.float64) + 1j * np.arange(1 << nqubits)
    np.testing.assert_allclose(
        operator.mvp(state), expected @ state, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("code", (1, 2, 3))
def test_first_and_last_qubit_matrix_ordering(code: int) -> None:
    first = PauliOperator.from_terms(3, ((tuple([code, 0, 0]), 1.0),)).dense()
    last = PauliOperator.from_terms(3, ((tuple([0, 0, code]), 1.0),)).dense()
    np.testing.assert_allclose(first, dense_operator(3, ((code, 0, 0),), (1.0,)))
    np.testing.assert_allclose(last, dense_operator(3, ((0, 0, code),), (1.0,)))
    assert not np.array_equal(first, last)


def test_backend_plan_has_versioned_arrays_and_independent_numpy_executor() -> None:
    operator = make_operator(3)
    plan = operator.backend_mvp_plan()
    assert isinstance(plan, BackendMVPPlan)
    assert plan.schema_version == 1
    assert plan.ordering == "qubit0_msb_matrix"
    assert plan.x_words.dtype == np.uint64
    assert plan.z_words.dtype == np.uint64
    assert plan.coefficients.dtype == np.complex128
    state = np.random.default_rng(20260801).normal(size=8) + 1j * np.random.default_rng(
        7
    ).normal(size=8)
    np.testing.assert_allclose(
        plan.apply(state), operator.dense() @ state, rtol=1e-12, atol=1e-12
    )


def test_empty_identity_shape_invalid_state_and_allocation_guards() -> None:
    zero = PauliOperator.empty(0)
    np.testing.assert_array_equal(zero.dense(), np.zeros((1, 1), dtype=np.complex128))
    identity = PauliOperator.from_terms(0, (((), 1.0),))
    np.testing.assert_array_equal(
        identity.dense(), np.ones((1, 1), dtype=np.complex128)
    )
    with pytest.raises(ValueError, match="expected structure length"):
        PauliOperator.empty(2).mvp(np.ones(3, dtype=np.complex128))
    with pytest.raises(MemoryError, match="requested"):
        make_operator(10).dense(max_bytes=1)
    with pytest.raises(OverflowError, match="dimension"):
        PauliOperator.empty(64).dense()


def test_compile_target_dispatch_is_explicit() -> None:
    operator = PauliOperator.from_terms(1, (((1,), 2.0),))
    np.testing.assert_array_equal(operator.compile("dense"), operator.dense())
    plan = operator.compile("backend_mvp")
    assert isinstance(plan, BackendMVPPlan)
    with pytest.raises(ValueError, match="target"):
        operator.compile("unknown")
