"""P1 public PauliWord differential and boundary tests."""

from __future__ import annotations

import numpy as np
import pytest
from reference import codes_to_dense, commutes, multiply_codes, support

from tencirpauli import PauliPhase, PauliWord


def test_code_string_and_packed_round_trip() -> None:
    for value in ("", "I", "XYZ", "ZIIX", "Y" * 65):
        word = PauliWord.from_string(value)
        assert word.to_string() == value
        assert word.to_codes() == tuple("IXYZ".index(char) for char in value)
        assert PauliWord.from_codes(word.to_codes()) == word


def test_full_single_qubit_phase_vectors() -> None:
    expected = {
        ("X", "Y"): ("Z", PauliPhase.PLUS_I),
        ("Y", "X"): ("Z", PauliPhase.MINUS_I),
        ("Y", "Z"): ("X", PauliPhase.PLUS_I),
        ("Z", "Y"): ("X", PauliPhase.MINUS_I),
    }
    for (left, right), (result, phase) in expected.items():
        product = PauliWord.from_string(left).multiply(PauliWord.from_string(right))
        assert product.word.to_string() == result
        assert product.phase is phase


def test_random_words_match_independent_dense_reference() -> None:
    rng = np.random.default_rng(20260801)
    for nqubits in range(7):
        for _ in range(30):
            left = tuple(int(code) for code in rng.integers(0, 4, size=nqubits))
            right = tuple(int(code) for code in rng.integers(0, 4, size=nqubits))
            left_word = PauliWord.from_codes(left)
            right_word = PauliWord.from_codes(right)
            result_codes, phase = multiply_codes(left, right)
            result = left_word.multiply(right_word)
            assert result.word.to_codes() == result_codes
            assert result.phase.value_complex == phase
            np.testing.assert_allclose(
                result.phase.value_complex * codes_to_dense(result_codes),
                codes_to_dense(left) @ codes_to_dense(right),
                rtol=0.0,
                atol=0.0,
            )
            assert left_word.support == support(left)
            assert left_word.weight == len(support(left))
            assert left_word.commutes_with(right_word) is commutes(left, right)
            assert left_word.symplectic_inner_product(right_word) in (0, 1)


def test_batch_conversion_uses_one_public_call_shape() -> None:
    structures = ((1, 0, 3), (2, 2, 0), (0, 0, 0))
    words = PauliWord.batch_from_codes(3, structures)
    assert tuple(word.to_codes() for word in words) == structures
    assert words[0].x_words == (1,)
    assert words[0].z_words == (4,)


def test_invalid_code_shape_and_incompatible_operands_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="codes"):
        PauliWord.from_codes((4,))
    with pytest.raises(ValueError, match="structure length"):
        PauliWord.batch_from_codes(2, ((1,),))
    with pytest.raises(ValueError, match="incompatible qubit counts"):
        PauliWord.from_string("X").commutes_with(PauliWord.from_string("XX"))
    with pytest.raises(ValueError, match="expected 2 packed words"):
        PauliWord(65, (0,), (0,)).to_codes()
