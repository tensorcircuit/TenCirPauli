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
def pauli_canonicalize(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
) -> tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]]: ...
def pauli_operator_binary(
    nqubits: int,
    left: tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]],
    right: tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]],
    operation: int,
) -> tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]]: ...
def pauli_operator_scale(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    scalar_re: float,
    scalar_im: float,
) -> tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]]: ...
def pauli_operator_adjoint(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
) -> tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]]: ...
def pauli_operator_is_hermitian(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    tolerance: float,
) -> bool: ...
def pauli_group(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    mode: int,
    algorithm: int,
) -> Sequence[Sequence[int]]: ...
def pauli_compatibility_matrix(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    mode: int,
    max_entries: int,
) -> Sequence[bool]: ...
def pauli_incompatibility_edges(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    mode: int,
    max_edges: int,
) -> Sequence[tuple[int, int]]: ...
def pauli_commutes(
    nqubits: int,
    x_words_left: Sequence[int],
    z_words_left: Sequence[int],
    x_words_right: Sequence[int],
    z_words_right: Sequence[int],
) -> bool: ...
