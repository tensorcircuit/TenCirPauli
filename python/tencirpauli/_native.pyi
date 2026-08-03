from typing import Sequence

__version__: str

class StructuredMvpPlan:
    @property
    def dimension(self) -> int: ...
    @property
    def estimated_bytes(self) -> int: ...
    def apply(self, state: object, max_bytes: int) -> object: ...

class NativeMappingPlan:
    @property
    def n_modes(self) -> int: ...
    @property
    def encoding(self) -> Sequence[Sequence[int]]: ...
    @property
    def inverse_encoding(self) -> Sequence[Sequence[int]]: ...
    @property
    def cnot_operations(self) -> Sequence[tuple[int, int]]: ...
    @property
    def estimated_bytes(self) -> int: ...
    def transform(
        self,
        structures: object,
        coefficients_re: Sequence[float],
        coefficients_im: Sequence[float],
        max_bytes: int,
    ) -> tuple[object, Sequence[float], Sequence[float]]: ...

def mapping_plan(mapping: str, n_modes: int, max_bytes: int) -> NativeMappingPlan: ...

class NativeChargeSectorPlan:
    @property
    def dimension(self) -> int: ...
    @property
    def estimated_bytes(self) -> int: ...
    def rank(self, occupations: Sequence[int]) -> int: ...
    def unrank(self, index: int) -> Sequence[int]: ...
    def basis_states(self, max_bytes: int) -> object: ...

def charge_sector_plan(
    local_dimensions: Sequence[int],
    contributions: object,
    target: Sequence[int],
    max_bytes: int,
) -> NativeChargeSectorPlan: ...
def structured_dense(
    local_dimensions: Sequence[int],
    operations: object,
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[int, object]: ...
def structured_sparse(
    local_dimensions: Sequence[int],
    operations: object,
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[int, Sequence[int], Sequence[int], Sequence[float], Sequence[float]]: ...
def structured_sparse_plan(
    local_dimensions: Sequence[int],
    operations: object,
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> StructuredMvpPlan: ...
def structured_fermion_canonicalize(
    n_modes: int,
    factors: object,
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[
    Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[float], Sequence[float]
]: ...
def structured_fermion_multiply(
    n_modes: int,
    left: tuple[object, object, Sequence[float], Sequence[float]],
    right: tuple[object, object, Sequence[float], Sequence[float]],
    max_bytes: int,
) -> tuple[
    Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[float], Sequence[float]
]: ...
def structured_fermion_jordan_wigner(
    n_modes: int,
    creation: object,
    annihilation: object,
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]]: ...
def structured_boson_canonicalize(
    n_modes: int,
    factors: object,
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[object, Sequence[float], Sequence[float]]: ...
def structured_boson_multiply(
    n_modes: int,
    left: tuple[object, Sequence[float], Sequence[float]],
    right: tuple[object, Sequence[float], Sequence[float]],
    max_bytes: int,
) -> tuple[object, Sequence[float], Sequence[float]]: ...
def structured_hybrid_multiply(
    n_modes: int,
    n_bosons: int,
    n_qubits: int,
    n_qudit_sites: int,
    qudit_dimension: int,
    left: object,
    right: object,
    max_bytes: int,
) -> tuple[
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    Sequence[float],
    Sequence[float],
]: ...
def structured_hybrid_canonicalize(
    n_modes: int,
    n_bosons: int,
    n_qubits: int,
    n_qudit_sites: int,
    qudit_dimension: int,
    input: object,
    max_bytes: int,
) -> tuple[
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    Sequence[float],
    Sequence[float],
]: ...
def structured_hybrid_jordan_wigner(
    n_modes: int,
    n_bosons: int,
    n_qubits: int,
    n_qudit_sites: int,
    qudit_dimension: int,
    input: object,
    max_bytes: int,
) -> tuple[
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    Sequence[float],
    Sequence[float],
]: ...
def charge_mvp_apply(
    dimension: int,
    rows: object,
    columns: object,
    coefficients: object,
    state: object,
    max_bytes: int,
) -> object: ...
def charge_compile_transitions(
    dimension: int,
    basis: object,
    local_dimensions: Sequence[int],
    fermion_positions: Sequence[int],
    boson_positions: Sequence[int],
    qubit_positions: Sequence[int],
    qudit_positions: Sequence[int],
    fermion_creation: object,
    fermion_annihilation: object,
    boson_blocks: object,
    qubit_codes: object,
    mapped_present: Sequence[bool],
    mapped_codes: object,
    qudit_present: Sequence[bool],
    qudit_triples: object,
    coefficients: object,
    qudit_dimension: int,
    max_bytes: int,
) -> tuple[Sequence[int], Sequence[int], Sequence[float], Sequence[float]]: ...
def majorana_canonicalize(
    n_modes: int,
    indices: object,
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[object, Sequence[float], Sequence[float]]: ...
def majorana_multiply(
    n_modes: int,
    left_indices: object,
    left_coefficients_re: Sequence[float],
    left_coefficients_im: Sequence[float],
    right_indices: object,
    right_coefficients_re: Sequence[float],
    right_coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[object, Sequence[float], Sequence[float]]: ...
def majorana_to_fermion(
    n_modes: int,
    indices: object,
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> tuple[
    Sequence[Sequence[int]],
    Sequence[Sequence[int]],
    Sequence[float],
    Sequence[float],
]: ...

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

class NativeU1CircuitPlan:
    @property
    def nqubits(self) -> int: ...
    @property
    def particle_number(self) -> int: ...
    @property
    def dimension(self) -> int: ...
    @property
    def nparameters(self) -> int: ...
    @property
    def gate_count(self) -> int: ...
    def run(self, initial_state: object, parameters: object) -> object: ...
    def run_cached(
        self, initial_state: object, parameters: object
    ) -> NativeU1FinalState: ...
    def probability(self, initial_state: object, parameters: object) -> object: ...
    def to_dense(self, initial_state: object, parameters: object) -> object: ...
    def probability_full(self, initial_state: object, parameters: object) -> object: ...
    def expectation(
        self,
        initial_state: object,
        structures: object,
        coefficients_re: object,
        coefficients_im: object,
        parameters: object,
    ) -> tuple[float, float]: ...
    def value_and_grad(
        self,
        initial_state: object,
        structures: object,
        coefficients_re: object,
        coefficients_im: object,
        parameters: object,
    ) -> tuple[float, object]: ...

class NativeU1FinalState:
    def state_array(self) -> object: ...
    def probability(self) -> object: ...
    def to_dense(self) -> object: ...
    def probability_full(self) -> object: ...
    def expectation(
        self,
        structures: object,
        coefficients_re: object,
        coefficients_im: object,
    ) -> tuple[float, float]: ...
    def value_and_grad(
        self,
        structures: object,
        coefficients_re: object,
        coefficients_im: object,
    ) -> tuple[float, object]: ...

class NativePropagationEngine:
    @property
    def nqubits(self) -> int: ...
    @property
    def nparameters(self) -> int: ...
    @property
    def gate_count(self) -> int: ...
    @property
    def max_weight(self) -> int | None: ...
    @property
    def is_exact(self) -> bool: ...
    def expectation(self, parameters: object) -> float: ...
    def value_and_grad(
        self, parameters: object, checkpoint_interval: int | None = None
    ) -> tuple[float, object]: ...
    def propagate_operator(
        self, parameters: object
    ) -> tuple[Sequence[Sequence[int]], Sequence[float], Sequence[float]]: ...
    def profile(
        self, parameters: object
    ) -> tuple[float, int, int, int, int, Sequence[int], float]: ...

class NativePropagationBatch:
    @property
    def nqubits(self) -> int: ...
    @property
    def nparameters(self) -> int: ...
    @property
    def gate_count(self) -> int: ...
    @property
    def observable_count(self) -> int: ...
    @property
    def max_weight(self) -> int | None: ...
    def expectations(self, parameters: object) -> object: ...
    def values_and_gradients(
        self, parameters: object, checkpoint_interval: int | None = None
    ) -> tuple[object, object]: ...

class NativeSPPSEngine:
    @property
    def nqubits(self) -> int: ...
    @property
    def nparameters(self) -> int: ...
    @property
    def gate_count(self) -> int: ...
    @property
    def observable_terms(self) -> int: ...
    @property
    def smoothing(self) -> float: ...
    def expectation(
        self, parameters: object, samples_per_term: int, seed: int
    ) -> tuple[float, float, int, Sequence[int], int, int]: ...
    def value_and_grad(
        self, parameters: object, samples_per_term: int, seed: int
    ) -> tuple[
        float,
        object,
        float,
        int,
        Sequence[int],
        int,
        int,
        float | None,
        Sequence[float] | None,
        bool | None,
    ]: ...
    def value_and_grad_adaptive(
        self,
        parameters: object,
        initial_samples_per_term: int,
        max_samples_per_term: int,
        gradient_tolerance: float,
        seed: int,
    ) -> tuple[
        float,
        object,
        float,
        int,
        Sequence[int],
        int,
        int,
        float | None,
        Sequence[float] | None,
        bool | None,
    ]: ...

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
) -> tuple[int, int, object]: ...
def u1_circuit_plan(
    nqubits: int,
    particle_number: int,
    schema_version: int,
    nparameters: int,
    expression_nodes: object,
    gates: object,
    max_bytes: int,
) -> NativeU1CircuitPlan: ...
def pauli_propagation_engine(
    nqubits: int,
    operations: Sequence[tuple[int, int, int, int, float, Sequence[float]]],
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    state_kind: int,
    state_bits: Sequence[int],
    state_values: Sequence[float],
    max_weight: int | None = ...,
    max_bytes: int | None = ...,
) -> NativePropagationEngine: ...
def pauli_propagation_batch(
    nqubits: int,
    operations: Sequence[tuple[int, int, int, int, float, Sequence[float]]],
    observable_offsets: Sequence[int],
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    state_kind: int,
    state_bits: Sequence[int],
    state_values: Sequence[float],
    max_weight: int | None = None,
    max_bytes: int | None = None,
) -> NativePropagationBatch: ...
def pauli_spps_engine(
    nqubits: int,
    operations: Sequence[tuple[int, int, int, int, float, Sequence[float]]],
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    state_kind: int,
    state_bits: Sequence[int],
    state_values: Sequence[float],
    smoothing: float = 0.01,
    max_bytes: int | None = None,
) -> NativeSPPSEngine: ...
