"""P2 PauliOperator canonicalization and algebra differential tests."""

from __future__ import annotations

import numpy as np
import pytest
from reference import dense_operator

import tencirpauli.pauli as pauli_module
from tencirpauli import PauliOperator, PauliPhase, PauliWord


def as_dense(operator: PauliOperator) -> np.ndarray:
    return dense_operator(
        operator.nqubits,
        (term.word.to_codes() for term in operator.terms),
        tuple(term.coefficient for term in operator.terms),
    )


def test_canonical_order_duplicate_aggregation_and_exact_cancellation() -> None:
    operator = PauliOperator.from_terms(
        2,
        (("ZX", 1.0), ("II", 2.0), ("ZX", -0.25j), ("II", -2.0)),
    )
    assert tuple(term.word.to_string() for term in operator.terms) == ("ZX",)
    assert operator.terms[0].coefficient == 1.0 - 0.25j
    assert PauliOperator.empty(2).terms == ()


def test_batch_canonicalization_retains_mapping_and_exact_zero_key() -> None:
    result = PauliOperator.canonicalize_batch(
        2,
        (("ZX", 1.0), ("II", 2.0), ("ZX", -0.25j), ("II", -2.0)),
    )
    assert result.canonical_structures == ((0, 0), (3, 1))
    assert result.coefficients == (0j, 1.0 - 0.25j)
    assert result.input_to_canonical == (1, 0, 1, 0)
    assert result.phase_multipliers == (PauliPhase.PLUS_ONE,) * 4


def test_random_operator_algebra_matches_numpy_dense_reference() -> None:
    rng = np.random.default_rng(20260801)
    for nqubits in range(4):
        structures_left = rng.integers(0, 4, size=(6, nqubits)).tolist()
        structures_right = rng.integers(0, 4, size=(5, nqubits)).tolist()
        coefficients_left = rng.normal(size=6) + 1j * rng.normal(size=6)
        coefficients_right = rng.normal(size=5) + 1j * rng.normal(size=5)
        left = PauliOperator.from_terms(
            nqubits, zip(structures_left, coefficients_left.tolist())
        )
        right = PauliOperator.from_terms(
            nqubits, zip(structures_right, coefficients_right.tolist())
        )
        left_dense = dense_operator(nqubits, structures_left, coefficients_left)
        right_dense = dense_operator(nqubits, structures_right, coefficients_right)
        np.testing.assert_allclose(as_dense(left), left_dense, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(as_dense(right), right_dense, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            as_dense(left + right), left_dense + right_dense, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            as_dense(left.multiply(right)),
            left_dense @ right_dense,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            as_dense(left.commutator(right)),
            left_dense @ right_dense - right_dense @ left_dense,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            as_dense(left.anticommutator(right)),
            left_dense @ right_dense + right_dense @ left_dense,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            as_dense(left.adjoint()), left_dense.conj().T, rtol=1e-12, atol=1e-12
        )


def test_complex_scaling_and_hermiticity_validation() -> None:
    operator = PauliOperator.from_terms(1, (("X", 2.0), ("Y", -1.5j)))
    np.testing.assert_allclose(
        as_dense(operator.scale(0.5 + 0.25j)), (0.5 + 0.25j) * as_dense(operator)
    )
    assert operator.is_hermitian() is False
    hermitian = PauliOperator.from_terms(2, (("XX", 1.0), ("IZ", -2.0)))
    assert hermitian.is_hermitian()
    with pytest.raises(ValueError, match="finite"):
        operator.scale(complex(float("nan"), 0.0))
    with pytest.raises(ValueError, match="tolerance"):
        hermitian.is_hermitian(-1.0)


def test_scaling_preserves_zero_free_finite_coefficients() -> None:
    operator = PauliOperator.from_terms(1, (("X", 2.0), ("Z", 1e-300)))

    assert operator.scale(0.0).terms == ()
    assert PauliOperator.from_terms(1, (("Z", 1e-300),)).scale(1e-300).terms == ()
    with pytest.raises(ValueError, match="finite"):
        operator.scale(1e308)
    with pytest.raises(ValueError, match="finite"):
        PauliOperator.from_terms(1, (("X", 1e308), ("X", 1e308)))


def test_operator_input_and_output_qubit_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="expected 2 qubits"):
        PauliOperator.from_terms(2, (("X", 1.0),))
    with pytest.raises(ValueError, match="incompatible qubit counts"):
        PauliOperator.empty(1).add(PauliOperator.empty(2))
    with pytest.raises(ValueError, match="finite"):
        PauliOperator.from_terms(1, ((PauliWord.from_string("X"), float("inf")),))


def test_operator_construction_uses_one_native_batch_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_word = PauliWord.from_string("ZX")
    original_canonicalize = pauli_module._native.pauli_canonicalize
    canonicalize_calls = 0

    def count_canonicalize(*args: object) -> object:
        nonlocal canonicalize_calls
        canonicalize_calls += 1
        return original_canonicalize(*args)

    def reject_per_term_call(*args: object) -> None:
        del args
        raise AssertionError("operator construction made a per-term native call")

    monkeypatch.setattr(pauli_module._native, "pauli_canonicalize", count_canonicalize)
    for name in ("pauli_from_codes", "pauli_codes", "pauli_batch_from_codes"):
        monkeypatch.setattr(pauli_module._native, name, reject_per_term_call)

    operator = PauliOperator.from_terms(
        2,
        (("xi", 1.0), ((2, 3), 2.0), (packed_word, 3.0)),
    )

    assert canonicalize_calls == 1
    assert operator._arrays() == (
        ((1, 0), (2, 3), (3, 1)),
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0),
    )


def test_operator_arrays_are_cached_and_do_not_affect_public_value_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = PauliOperator.from_terms(2, (("ZX", 1.0 - 0.25j), ("II", -2.0)))
    equal_operator = PauliOperator.from_terms(
        2, (((3, 1), 1.0 - 0.25j), ((0, 0), -2.0))
    )
    first = operator._arrays()
    second = operator._arrays()

    assert all(left is right for left, right in zip(first, second))
    assert first == (
        ((0, 0), (3, 1)),
        (-2.0, 1.0),
        (0.0, -0.25),
    )
    assert operator == equal_operator
    assert "_canonical_structures" not in repr(operator)

    def reject_code_conversion(*args: object) -> None:
        del args
        raise AssertionError("cached arrays performed native code conversion")

    monkeypatch.setattr(pauli_module._native, "pauli_codes", reject_code_conversion)
    assert operator._arrays() == first


def test_code_array_construction_uses_contiguous_native_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pauli_module._native.pauli_canonicalize_array
    calls = 0

    def count_array_call(*args: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args)

    def reject_nested_sequence_call(*args: object) -> None:
        del args
        raise AssertionError(
            "code-array construction used the nested-sequence boundary"
        )

    monkeypatch.setattr(
        pauli_module._native, "pauli_canonicalize_array", count_array_call
    )
    monkeypatch.setattr(
        pauli_module._native, "pauli_canonicalize", reject_nested_sequence_call
    )
    operator = PauliOperator.from_terms(
        3,
        ((((1, 2, 3)), 1.0), (((0, 0, 0)), 2.0), (((1, 2, 3)), -0.5)),
    )

    assert calls == 1
    assert tuple(
        (term.word.to_string(), term.coefficient) for term in operator.terms
    ) == (
        ("III", 2.0 + 0.0j),
        ("XYZ", 0.5 + 0.0j),
    )


def test_explicit_code_array_api_preserves_mapping_and_validation() -> None:
    structures = np.array([[3, 1], [0, 0], [3, 1]], dtype=np.int16)
    coefficients = np.array([1.0, 2.0, -0.25j], dtype=np.complex128)
    operator = PauliOperator.from_code_arrays(structures, coefficients)
    result = PauliOperator.canonicalize_code_arrays(structures, coefficients)
    array_result = PauliOperator.canonicalize_code_arrays_numpy(
        structures, coefficients
    )

    assert tuple(
        (term.word.to_string(), term.coefficient) for term in operator.terms
    ) == (
        ("II", 2.0 + 0.0j),
        ("ZX", 1.0 - 0.25j),
    )
    assert result.input_to_canonical == (1, 0, 1)
    np.testing.assert_array_equal(array_result.canonical_structures, [[0, 0], [3, 1]])
    np.testing.assert_array_equal(array_result.coefficients, [2.0, 1.0 - 0.25j])
    np.testing.assert_array_equal(array_result.input_to_canonical, [1, 0, 1])
    assert not array_result.canonical_structures.flags.writeable
    with pytest.raises(ValueError, match=r"integers in 0\.\.3"):
        PauliOperator.from_code_arrays([[4]], [1.0])
    with pytest.raises(ValueError, match="one value per structure"):
        PauliOperator.from_code_arrays([[1], [2]], [1.0])
