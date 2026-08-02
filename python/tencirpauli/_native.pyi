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

class NativeZ2TaperingPlan:
    @property
    def nqubits_before(self) -> int: ...
    @property
    def nqubits_after(self) -> int: ...
    @property
    def generators(self) -> Sequence[Sequence[int]]: ...
    @property
    def sector(self) -> Sequence[int]: ...
    @property
    def removed_qubits(self) -> Sequence[int]: ...
    @property
    def clifford_operations(self) -> Sequence[tuple[int, int, int]]: ...
    def transform_operator(
        self,
        nqubits: int,
        structures: object,
        coefficients_re: object,
        coefficients_im: object,
    ) -> tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]]: ...

class NativeU1RestrictedOperator:
    @property
    def nqubits(self) -> int: ...
    @property
    def particle_number(self) -> int: ...
    @property
    def dimension(self) -> int: ...
    def apply(self, state: object, max_bytes: int) -> object: ...
    def mvp_plan(self, max_bytes: int) -> NativeU1MvpPlan: ...
    def dense(self, max_bytes: int) -> tuple[int, object]: ...
    def coo(self, max_bytes: int) -> tuple[int, object, object, object]: ...
    def csr(self, max_bytes: int) -> tuple[int, object, object, object]: ...

class NativeU1MvpPlan:
    @property
    def nqubits(self) -> int: ...
    @property
    def particle_number(self) -> int: ...
    @property
    def dimension(self) -> int: ...
    def apply(self, state: object, max_bytes: int) -> object: ...

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
def pauli_find_z2_symmetries(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[Sequence[Sequence[int]], int]: ...
def pauli_z2_tapering_plan(
    nqubits: int,
    generators: Sequence[Sequence[int]],
    sector: Sequence[int],
) -> NativeZ2TaperingPlan: ...
def pauli_restrict_u1(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    particle_number: int,
    max_bytes: int,
) -> NativeU1RestrictedOperator: ...
def u1_basis_words(
    nqubits: int,
    particle_number: int,
    max_bytes: int,
) -> tuple[int, object]: ...
