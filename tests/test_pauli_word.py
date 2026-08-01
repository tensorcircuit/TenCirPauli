import pytest

from tencirpauli import PauliWord


def test_weight_and_unused_bits() -> None:
    word = PauliWord(nqubits=2, x_words=(2**64 - 1,), z_words=(0,))
    assert word.x_words == (0b11,)
    assert word.weight == 2


def test_commutation() -> None:
    x0 = PauliWord(nqubits=2, x_words=(0b01,), z_words=(0,))
    z0 = PauliWord(nqubits=2, x_words=(0,), z_words=(0b01,))
    xx = PauliWord(nqubits=2, x_words=(0b11,), z_words=(0,))
    zz = PauliWord(nqubits=2, x_words=(0,), z_words=(0b11,))

    assert not x0.commutes_with(z0)
    assert xx.commutes_with(zz)


def test_constructor_canonicalizes_unused_tail_bits() -> None:
    identity = PauliWord(nqubits=1, x_words=(0,), z_words=(0,))
    dirty_identity = PauliWord(nqubits=1, x_words=(2,), z_words=(2**64 - 2,))

    assert dirty_identity.x_words == (0,)
    assert dirty_identity.z_words == (0,)
    assert dirty_identity == identity
    assert hash(dirty_identity) == hash(identity)
    assert not dirty_identity < identity
    assert not identity < dirty_identity


def test_constructor_masks_only_the_final_packed_word() -> None:
    word = PauliWord(
        nqubits=65,
        x_words=(2**64 - 1, 2**64 - 1),
        z_words=(0, 2**64 - 1),
    )

    assert word.x_words == (2**64 - 1, 1)
    assert word.z_words == (0, 1)


@pytest.mark.parametrize(
    ("nqubits", "x_words", "z_words", "expected_count"),
    (
        (0, (0,), (), 0),
        (1, (), (0,), 1),
        (65, (0,), (0, 0), 2),
        (65, (0, 0), (0,), 2),
    ),
)
def test_incompatible_word_length_fails_at_construction(
    nqubits: int,
    x_words: tuple[int, ...],
    z_words: tuple[int, ...],
    expected_count: int,
) -> None:
    with pytest.raises(ValueError, match=rf"expected {expected_count} packed words"):
        PauliWord(nqubits=nqubits, x_words=x_words, z_words=z_words)


def test_zero_qubit_word_has_canonical_empty_storage() -> None:
    word = PauliWord(nqubits=0, x_words=(), z_words=())

    assert word == PauliWord.from_string("")
    assert word.x_words == ()
    assert word.z_words == ()
