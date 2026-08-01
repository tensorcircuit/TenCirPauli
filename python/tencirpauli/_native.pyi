from typing import Sequence

__version__: str

def pauli_weight(
    nqubits: int, x_words: Sequence[int], z_words: Sequence[int]
) -> int: ...
def pauli_commutes(
    nqubits: int,
    x_words_left: Sequence[int],
    z_words_left: Sequence[int],
    x_words_right: Sequence[int],
    z_words_right: Sequence[int],
) -> bool: ...
