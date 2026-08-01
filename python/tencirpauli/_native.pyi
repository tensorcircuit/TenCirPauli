from typing import Sequence

__version__: str

def pauli_weight(
    nqubits: int, x_words: Sequence[int], z_words: Sequence[int]
) -> int: ...
def pauli_support(
    nqubits: int, x_words: Sequence[int], z_words: Sequence[int]
) -> Sequence[int]: ...
def pauli_codes(
    nqubits: int, x_words: Sequence[int], z_words: Sequence[int]
) -> Sequence[int]: ...
def pauli_from_codes(
    nqubits: int, codes: Sequence[int]
) -> tuple[Sequence[int], Sequence[int]]: ...
def pauli_batch_from_codes(
    nqubits: int, structures: Sequence[Sequence[int]]
) -> tuple[int, Sequence[int], Sequence[int]]: ...
def pauli_multiply(
    nqubits: int, left_codes: Sequence[int], right_codes: Sequence[int]
) -> tuple[Sequence[int], int]: ...
def pauli_symplectic_inner_product(
    nqubits: int,
    x_words_left: Sequence[int],
    z_words_left: Sequence[int],
    x_words_right: Sequence[int],
    z_words_right: Sequence[int],
) -> int: ...
def pauli_commutes(
    nqubits: int,
    x_words_left: Sequence[int],
    z_words_left: Sequence[int],
    x_words_right: Sequence[int],
    z_words_right: Sequence[int],
) -> bool: ...
