import pytest

from tencirpauli import PauliWord


def test_weight_and_unused_bits() -> None:
    word = PauliWord(nqubits=2, x_words=(2**64 - 1,), z_words=(0,))
    assert word.weight == 2


def test_commutation() -> None:
    x0 = PauliWord(nqubits=2, x_words=(0b01,), z_words=(0,))
    z0 = PauliWord(nqubits=2, x_words=(0,), z_words=(0b01,))
    xx = PauliWord(nqubits=2, x_words=(0b11,), z_words=(0,))
    zz = PauliWord(nqubits=2, x_words=(0,), z_words=(0b11,))

    assert not x0.commutes_with(z0)
    assert xx.commutes_with(zz)


def test_incompatible_word_length_fails() -> None:
    word = PauliWord(nqubits=65, x_words=(0,), z_words=(0,))
    with pytest.raises(ValueError, match="expected 2 packed words"):
        _ = word.weight
