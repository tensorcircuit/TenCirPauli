from typing import Sequence

__version__: str

class NativeMvpPlan:
    @property
    def nqubits(self) -> int: ...
    @property
    def term_count(self) -> int: ...
    @property
    def strategy(self) -> str: ...
    def apply(
        self,
        state: object,
        max_bytes: int,
    ) -> object: ...

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
def pauli_canonicalize_batch(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
) -> tuple[
    Sequence[Sequence[int]],
    Sequence[float],
    Sequence[float],
    Sequence[int],
    Sequence[int],
]: ...
def pauli_canonicalize_array(
    nqubits: int,
    structures: object,
    coefficients: object,
) -> tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]]: ...
def pauli_canonicalize_batch_array(
    nqubits: int,
    structures: object,
    coefficients: object,
) -> tuple[
    Sequence[Sequence[int]],
    Sequence[float],
    Sequence[float],
    Sequence[int],
    Sequence[int],
]: ...
def pauli_canonicalize_batch_numpy(
    nqubits: int,
    structures: object,
    coefficients: object,
) -> tuple[int, object, object, object, object]: ...
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
    max_entries: int,
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
def pauli_dense(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[int, Sequence[float], Sequence[float]]: ...
def pauli_dense_array(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[int, object]: ...
def pauli_coo(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[int, Sequence[int], Sequence[int], Sequence[float], Sequence[float]]: ...
def pauli_coo_array(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[int, object, object, object]: ...
def pauli_csr(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[int, Sequence[int], Sequence[int], Sequence[float], Sequence[float]]: ...
def pauli_csr_array(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[int, object, object, object]: ...
def pauli_mvp_array(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    state: object,
    max_bytes: int,
) -> object: ...
def pauli_mvp_plan(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> NativeMvpPlan: ...
def pauli_backend_plan(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[
    int, int, int, Sequence[int], Sequence[int], Sequence[float], Sequence[float]
]: ...
def pauli_commutes(
    nqubits: int,
    x_words_left: Sequence[int],
    z_words_left: Sequence[int],
    x_words_right: Sequence[int],
    z_words_right: Sequence[int],
) -> bool: ...
