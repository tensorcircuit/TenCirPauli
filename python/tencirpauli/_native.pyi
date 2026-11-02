from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

Complex128Array: TypeAlias = NDArray[np.complex128]
Float64Array: TypeAlias = NDArray[np.float64]
UInt64Array: TypeAlias = NDArray[np.uint64]
UIntpArray: TypeAlias = NDArray[np.uintp]
UInt32Array: TypeAlias = NDArray[np.uint32]
UInt8Array: TypeAlias = NDArray[np.uint8]
Int8Array: TypeAlias = NDArray[np.int8]
RealOrComplexArray: TypeAlias = Float64Array | Complex128Array

HybridCanonicalInput = tuple[
    Sequence[Sequence[tuple[int, int]]],
    Sequence[Sequence[tuple[int, int]]],
    Sequence[Sequence[int]],
    Sequence[bool],
    Sequence[Sequence[tuple[int, int, int]]],
    Sequence[float],
    Sequence[float],
]
HybridJordanWignerInput = tuple[
    Sequence[bool],
    Sequence[Sequence[int]],
    Sequence[Sequence[int]],
    Sequence[bool],
    Sequence[Sequence[tuple[int, int, int]]],
    Sequence[Sequence[int]],
    Sequence[bool],
    Sequence[Sequence[int]],
    Sequence[bool],
    Sequence[Sequence[tuple[int, int, int]]],
    Sequence[float],
    Sequence[float],
]

__version__: str

class StructuredMvpPlan:
    def apply_adjoint(
        self, state: Complex128Array, max_bytes: int
    ) -> Complex128Array: ...
    @property
    def dimension(self) -> int: ...
    @property
    def estimated_bytes(self) -> int: ...
    @property
    def estimated_construction_peak_bytes(self) -> int: ...
    @property
    def source_term_count(self) -> int: ...
    @property
    def schedule_count(self) -> int: ...
    @property
    def execution_family(self) -> str: ...
    @property
    def native_variant(self) -> str: ...
    @property
    def retained_format(self) -> str: ...
    @property
    def transition_count(self) -> int | None: ...
    def debug_counters(self) -> tuple[int, int, int, int]: ...
    def debug_metadata(self) -> tuple[int, int, int]: ...
    def debug_fermion_mask_phase_profile(
        self, samples: int
    ) -> tuple[int, int, int, int, int, int, int, int] | None: ...
    def apply(self, state: Complex128Array, max_bytes: int) -> Complex128Array: ...
    def apply_into(
        self, state: Complex128Array, output: Complex128Array, max_bytes: int
    ) -> None: ...

class StructuredEagerMvpPlan:
    def apply_adjoint(
        self, state: Complex128Array, max_bytes: int
    ) -> Complex128Array: ...
    @property
    def dimension(self) -> int: ...
    @property
    def estimated_bytes(self) -> int: ...
    @property
    def estimated_construction_peak_bytes(self) -> int: ...
    @property
    def source_term_count(self) -> int: ...
    @property
    def schedule_count(self) -> int: ...
    @property
    def execution_family(self) -> str: ...
    @property
    def native_variant(self) -> str: ...
    @property
    def retained_format(self) -> str: ...
    @property
    def transition_count(self) -> int: ...
    def csr(
        self, max_bytes: int
    ) -> tuple[UInt64Array, UInt64Array, Complex128Array]: ...
    def apply(self, state: Complex128Array, max_bytes: int) -> Complex128Array: ...
    def apply_into(
        self, state: Complex128Array, output: Complex128Array, max_bytes: int
    ) -> None: ...

class NativePauliOperatorHandle:
    @property
    def nqubits(self) -> int: ...
    @property
    def term_count(self) -> int: ...
    def distinct_x_mask_count(self) -> int: ...
    def add(
        self, other: "NativePauliOperatorHandle", max_bytes: int
    ) -> "NativePauliOperatorHandle": ...
    def scale(
        self, scalar_re: float, scalar_im: float
    ) -> "NativePauliOperatorHandle": ...
    def multiply(
        self, other: "NativePauliOperatorHandle", max_bytes: int
    ) -> "NativePauliOperatorHandle": ...
    def commutator(
        self, other: "NativePauliOperatorHandle", max_bytes: int
    ) -> "NativePauliOperatorHandle": ...
    def anticommutator(
        self, other: "NativePauliOperatorHandle", max_bytes: int
    ) -> "NativePauliOperatorHandle": ...
    def materialize(
        self,
    ) -> tuple[int, int, UInt64Array, UInt64Array, Complex128Array]: ...
    def materialize_arrays(
        self,
    ) -> tuple[int, int, UInt8Array, Complex128Array]: ...
    def materialize_strings(self) -> tuple[Sequence[str], Complex128Array]: ...
    def adjoint(self) -> "NativePauliOperatorHandle": ...
    def is_hermitian(self, tolerance: float) -> bool: ...
    def termwise_conserves_charge(
        self, qubit_levels: Sequence[tuple[float, float]]
    ) -> bool: ...
    def content_eq(self, other: "NativePauliOperatorHandle") -> bool: ...
    def content_hash(self) -> int: ...

class NativeFermionOperatorHandle:
    @property
    def n_modes(self) -> int: ...
    @property
    def term_count(self) -> int: ...
    def add(
        self, other: "NativeFermionOperatorHandle", max_bytes: int
    ) -> "NativeFermionOperatorHandle": ...
    def scale(
        self, scalar_re: float, scalar_im: float
    ) -> "NativeFermionOperatorHandle": ...
    def multiply(
        self, other: "NativeFermionOperatorHandle", max_bytes: int
    ) -> "NativeFermionOperatorHandle": ...
    def adjoint(self, max_bytes: int) -> "NativeFermionOperatorHandle": ...
    def jordan_wigner(self, max_bytes: int) -> "NativePauliOperatorHandle": ...
    def commutator(
        self, other: "NativeFermionOperatorHandle", max_bytes: int
    ) -> "NativeFermionOperatorHandle": ...
    def anticommutator(
        self, other: "NativeFermionOperatorHandle", max_bytes: int
    ) -> "NativeFermionOperatorHandle": ...
    def to_majorana(self, max_bytes: int) -> "NativeMajoranaOperatorHandle": ...
    def to_hybrid(self) -> "NativeHybridOperatorHandle": ...
    def is_hermitian(self, tolerance: float) -> bool: ...
    def content_eq(self, other: "NativeFermionOperatorHandle") -> bool: ...
    def content_hash(self) -> int: ...
    def materialize(
        self,
    ) -> tuple[
        int, UInt32Array, UIntpArray, UInt32Array, UIntpArray, Complex128Array
    ]: ...

class NativeBosonOperatorHandle:
    @property
    def n_modes(self) -> int: ...
    @property
    def term_count(self) -> int: ...
    def add(
        self, other: "NativeBosonOperatorHandle", max_bytes: int
    ) -> "NativeBosonOperatorHandle": ...
    def scale(
        self, scalar_re: float, scalar_im: float
    ) -> "NativeBosonOperatorHandle": ...
    def multiply(
        self, other: "NativeBosonOperatorHandle", max_bytes: int
    ) -> "NativeBosonOperatorHandle": ...
    def adjoint(self) -> "NativeBosonOperatorHandle": ...
    def commutator(
        self, other: "NativeBosonOperatorHandle", max_bytes: int
    ) -> "NativeBosonOperatorHandle": ...
    def anticommutator(
        self, other: "NativeBosonOperatorHandle", max_bytes: int
    ) -> "NativeBosonOperatorHandle": ...
    def to_hybrid(self) -> "NativeHybridOperatorHandle": ...
    def is_hermitian(self, tolerance: float) -> bool: ...
    def content_eq(self, other: "NativeBosonOperatorHandle") -> bool: ...
    def content_hash(self) -> int: ...
    def materialize(
        self,
    ) -> tuple[int, UInt32Array, UIntpArray, Complex128Array]: ...

class NativeHybridOperatorHandle:
    @property
    def term_count(self) -> int: ...
    def has_raw_fermions(self) -> bool: ...
    def has_mapped_fermions(self) -> bool: ...
    def has_mixed_fermion_roles(self) -> bool: ...
    def add(
        self, other: "NativeHybridOperatorHandle", max_bytes: int
    ) -> "NativeHybridOperatorHandle": ...
    def scale(
        self, scalar_re: float, scalar_im: float
    ) -> "NativeHybridOperatorHandle": ...
    def multiply(
        self, other: "NativeHybridOperatorHandle", max_bytes: int
    ) -> "NativeHybridOperatorHandle": ...
    def jordan_wigner(self, max_bytes: int) -> "NativeHybridOperatorHandle": ...
    def adjoint(self) -> "NativeHybridOperatorHandle": ...
    def commutator(
        self, other: "NativeHybridOperatorHandle", max_bytes: int
    ) -> "NativeHybridOperatorHandle": ...
    def anticommutator(
        self, other: "NativeHybridOperatorHandle", max_bytes: int
    ) -> "NativeHybridOperatorHandle": ...
    def to_pauli(
        self, axes: Sequence[tuple[int, int]], max_bytes: int
    ) -> "NativePauliOperatorHandle": ...
    def to_fermion(self) -> "NativeFermionOperatorHandle": ...
    def to_boson(self) -> "NativeBosonOperatorHandle": ...
    def embed(
        self,
        target_n_modes: int,
        target_n_bosons: int,
        target_nqubits: int,
        target_n_qudit_sites: int,
        target_qudit_dimension: int,
        fermion_map: Sequence[int],
        boson_map: Sequence[int],
        qubit_map: Sequence[int],
        qudit_map: Sequence[int],
        max_bytes: int,
    ) -> "NativeHybridOperatorHandle": ...
    def analyze_charge(
        self,
        fermion_weights: Sequence[float],
        boson_weights: Sequence[float],
        qubit_levels: Sequence[tuple[float, float]],
        qudit_weights: Sequence[float],
        max_bytes: int,
    ) -> tuple[bool, int]: ...
    def termwise_conserves_charge(
        self,
        fermion_weights: Sequence[float],
        boson_weights: Sequence[float],
        qubit_levels: Sequence[tuple[float, float]],
        qudit_weights: Sequence[float],
    ) -> bool: ...
    def materialize(
        self,
    ) -> tuple[
        int,
        int,
        tuple[
            UInt8Array,
            UInt32Array,
            UIntpArray,
            UInt32Array,
            UIntpArray,
            UInt32Array,
            UIntpArray,
        ],
        tuple[UInt8Array, UInt8Array, UInt32Array, UIntpArray, Complex128Array],
    ]: ...
    def direct_weyl_flat(
        self, max_bytes: int
    ) -> tuple[UInt32Array, UInt32Array, Complex128Array]: ...
    def is_hermitian(self, tolerance: float) -> bool: ...
    def content_eq(self, other: "NativeHybridOperatorHandle") -> bool: ...
    def content_hash(self) -> int: ...

class NativeMajoranaOperatorHandle:
    @property
    def n_modes(self) -> int: ...
    @property
    def term_count(self) -> int: ...
    def add(
        self, other: "NativeMajoranaOperatorHandle", max_bytes: int
    ) -> "NativeMajoranaOperatorHandle": ...
    def scale(
        self, scalar_re: float, scalar_im: float
    ) -> "NativeMajoranaOperatorHandle": ...
    def multiply(
        self, other: "NativeMajoranaOperatorHandle", max_bytes: int
    ) -> "NativeMajoranaOperatorHandle": ...
    def adjoint(self) -> "NativeMajoranaOperatorHandle": ...
    def commutator(
        self, other: "NativeMajoranaOperatorHandle", max_bytes: int
    ) -> "NativeMajoranaOperatorHandle": ...
    def anticommutator(
        self, other: "NativeMajoranaOperatorHandle", max_bytes: int
    ) -> "NativeMajoranaOperatorHandle": ...
    def materialize(
        self,
    ) -> tuple[int, UInt64Array, UIntpArray, Complex128Array]: ...
    def to_fermion(self, max_bytes: int) -> "NativeFermionOperatorHandle": ...
    def is_hermitian(self, tolerance: float) -> bool: ...
    def content_eq(self, other: "NativeMajoranaOperatorHandle") -> bool: ...
    def content_hash(self) -> int: ...

class NativeMappingPlan:
    @property
    def n_modes(self) -> int: ...
    def encoding_flat(self) -> Sequence[int]: ...
    def inverse_encoding_flat(self) -> Sequence[int]: ...
    @property
    def cnot_operations(self) -> Sequence[tuple[int, int]]: ...
    @property
    def cnot_count(self) -> int: ...
    @property
    def estimated_bytes(self) -> int: ...
    def encode_occupation(self, occupation: Sequence[int]) -> Sequence[int]: ...
    def transform_pauli_handle(
        self,
        handle: "NativePauliOperatorHandle",
        max_bytes: int,
    ) -> "NativePauliOperatorHandle": ...
    def transform_pauli_handle_prefix(
        self,
        handle: "NativePauliOperatorHandle",
        prefix_length: int,
        max_bytes: int,
    ) -> "NativePauliOperatorHandle": ...
    def transform_majorana_handle(
        self,
        handle: "NativeMajoranaOperatorHandle",
        max_bytes: int,
    ) -> "NativePauliOperatorHandle": ...
    def transform_hybrid_handle(
        self,
        handle: "NativeHybridOperatorHandle",
        max_bytes: int,
    ) -> "NativeHybridOperatorHandle": ...

class NativeQwcGroupingHandle:
    def reconstruct(
        self, group_index: int, bitstrings: Int8Array
    ) -> tuple[int, int, Int8Array]: ...

def mapping_plan(mapping: str, n_modes: int, max_bytes: int) -> NativeMappingPlan: ...

class NativeChargeSectorPlan:
    @property
    def dimension(self) -> int: ...
    @property
    def estimated_bytes(self) -> int: ...
    def rank(self, occupations: Sequence[int]) -> int: ...
    def unrank(self, index: int) -> Sequence[int]: ...
    def basis_states(self, max_bytes: int) -> UInt64Array: ...
    def compile_mvp_pauli_handle(
        self,
        handle: NativePauliOperatorHandle,
        termwise_conserved: bool,
        max_bytes: int,
    ) -> NativeChargeMvpPlan: ...
    def compile_mvp_hybrid_handle(
        self,
        handle: NativeHybridOperatorHandle,
        termwise_conserved: bool,
        max_bytes: int,
    ) -> NativeChargeMvpPlan: ...
    def compile_eager_pauli_handle(
        self,
        handle: NativePauliOperatorHandle,
        termwise_conserved: bool,
        max_bytes: int,
    ) -> NativeChargeEagerMvpPlan: ...
    def compile_eager_hybrid_handle(
        self,
        handle: NativeHybridOperatorHandle,
        termwise_conserved: bool,
        max_bytes: int,
    ) -> NativeChargeEagerMvpPlan: ...

class NativeChargeMvpPlan:
    def apply_adjoint(
        self, state: Complex128Array, max_bytes: int
    ) -> Complex128Array: ...
    @property
    def dimension(self) -> int: ...
    @property
    def estimated_bytes(self) -> int: ...
    @property
    def execution_family(self) -> str: ...
    @property
    def native_variant(self) -> str: ...
    @property
    def schedule_count(self) -> int: ...
    def debug_counters(self) -> tuple[int, ...] | None: ...
    def apply(self, state: Complex128Array, max_bytes: int) -> Complex128Array: ...
    def apply_into(
        self, state: Complex128Array, output: Complex128Array, max_bytes: int
    ) -> None: ...
    def compile_eager(self, max_bytes: int) -> NativeChargeEagerMvpPlan: ...
    def compile_eager_with_chunk_capacity(
        self, max_bytes: int, chunk_capacity: int
    ) -> NativeChargeEagerMvpPlan: ...

class NativeChargeEagerMvpPlan:
    def apply_adjoint(
        self, state: Complex128Array, max_bytes: int
    ) -> Complex128Array: ...
    @property
    def dimension(self) -> int: ...
    @property
    def transition_count(self) -> int: ...
    @property
    def estimated_bytes(self) -> int: ...
    @property
    def estimated_construction_peak_bytes(self) -> int: ...
    @property
    def schedule_count(self) -> int: ...
    @property
    def native_variant(self) -> str: ...
    def construction_profile(
        self,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int]: ...
    def csr(
        self, max_bytes: int
    ) -> tuple[UInt64Array, UInt64Array, Complex128Array]: ...
    def coo(
        self, max_bytes: int
    ) -> tuple[UInt64Array, UInt64Array, Complex128Array]: ...
    def dense(self, max_bytes: int) -> Complex128Array: ...
    def apply(self, state: Complex128Array, max_bytes: int) -> Complex128Array: ...
    def apply_with_parallelism(
        self, state: Complex128Array, max_bytes: int, parallel: bool
    ) -> Complex128Array: ...
    def apply_into(
        self, state: Complex128Array, output: Complex128Array, max_bytes: int
    ) -> None: ...

def charge_sector_plan_compact(
    local_dimensions: Sequence[int],
    axis_kinds: Sequence[int],
    axis_indices: Sequence[int],
    fermion_weights: Sequence[Sequence[int]],
    boson_weights: Sequence[Sequence[int]],
    qubit_levels: Sequence[Sequence[tuple[int, int]]],
    qudit_weights: Sequence[Sequence[int]],
    target: Sequence[int],
    max_bytes: int,
) -> NativeChargeSectorPlan: ...
def pauli_analyze_charge_handle(
    handle: NativePauliOperatorHandle,
    qubit_levels: Sequence[tuple[float, float]],
    max_bytes: int,
) -> tuple[bool, int]: ...
def structured_dense_handle(
    handle: NativeHybridOperatorHandle,
    local_dimensions: Sequence[int],
    axes: Sequence[tuple[int, int]],
    max_bytes: int,
) -> tuple[int, Complex128Array]: ...
def structured_sparse_handle(
    handle: NativeHybridOperatorHandle,
    local_dimensions: Sequence[int],
    axes: Sequence[tuple[int, int]],
    max_bytes: int,
) -> tuple[int, UInt64Array, UInt64Array, Complex128Array]: ...
def structured_csr_handle(
    handle: NativeHybridOperatorHandle,
    local_dimensions: Sequence[int],
    axes: Sequence[tuple[int, int]],
    max_bytes: int,
) -> tuple[int, UInt64Array, UInt64Array, Complex128Array]: ...
def structured_sparse_plan_handle(
    handle: NativeHybridOperatorHandle,
    local_dimensions: Sequence[int],
    axes: Sequence[tuple[int, int]],
    max_bytes: int,
) -> StructuredMvpPlan: ...
def structured_eager_plan_handle(
    handle: NativeHybridOperatorHandle,
    local_dimensions: Sequence[int],
    axes: Sequence[tuple[int, int]],
    max_bytes: int,
) -> StructuredEagerMvpPlan: ...
def structured_fermion_canonicalize(
    n_modes: int,
    factors: Sequence[Sequence[tuple[int, int]]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> NativeFermionOperatorHandle: ...
def structured_fermion_integrals(
    one_body: RealOrComplexArray,
    two_body: RealOrComplexArray,
    constant_re: float,
    constant_im: float,
    max_bytes: int,
) -> NativeFermionOperatorHandle: ...
def structured_fermion_integral_blocks(
    n_spatial: int,
    one_alpha: RealOrComplexArray,
    one_beta: RealOrComplexArray,
    eri_aa: RealOrComplexArray,
    eri_ab: RealOrComplexArray,
    eri_ba: RealOrComplexArray,
    eri_bb: RealOrComplexArray,
    ordering: int,
    constant_re: float,
    constant_im: float,
    max_bytes: int,
) -> NativeFermionOperatorHandle: ...
def structured_boson_canonicalize(
    n_modes: int,
    factors: Sequence[Sequence[tuple[int, int]]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> NativeBosonOperatorHandle: ...
def structured_hybrid_canonicalize(
    n_modes: int,
    n_bosons: int,
    nqubits: int,
    n_qudit_sites: int,
    qudit_dimension: int,
    input: HybridCanonicalInput,
    max_bytes: int,
) -> NativeHybridOperatorHandle: ...
def structured_hybrid_jordan_wigner(
    n_modes: int,
    n_bosons: int,
    nqubits: int,
    n_qudit_sites: int,
    qudit_dimension: int,
    input: HybridJordanWignerInput,
    max_bytes: int,
) -> NativeHybridOperatorHandle: ...
def majorana_canonicalize(
    n_modes: int,
    indices: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> NativeMajoranaOperatorHandle: ...

class NativeMvpPlan:
    def apply_adjoint(
        self, state: Complex128Array, max_bytes: int
    ) -> Complex128Array: ...
    @property
    def nqubits(self) -> int: ...
    @property
    def term_count(self) -> int: ...
    @property
    def transition_count(self) -> int | None: ...
    @property
    def native_variant(self) -> str: ...
    def apply_into(
        self, state: Complex128Array, output: Complex128Array, max_bytes: int
    ) -> None: ...
    def apply(
        self,
        state: Complex128Array,
        max_bytes: int,
    ) -> Complex128Array: ...

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
    def transform_operator_handle(
        self, operator: NativePauliOperatorHandle
    ) -> NativePauliOperatorHandle: ...

class NativeU1RestrictedOperator:
    @property
    def nqubits(self) -> int: ...
    @property
    def particle_number(self) -> int: ...
    @property
    def dimension(self) -> int: ...
    @property
    def transition_count(self) -> int: ...
    def apply(self, state: Complex128Array, max_bytes: int) -> Complex128Array: ...
    def apply_into(
        self, state: Complex128Array, output: Complex128Array, max_bytes: int
    ) -> None: ...
    def mvp_plan(self, max_bytes: int) -> NativeU1MvpPlan: ...
    def dense(self, max_bytes: int) -> tuple[int, Complex128Array]: ...
    def coo(
        self, max_bytes: int
    ) -> tuple[int, UInt64Array, UInt64Array, Complex128Array]: ...
    def csr(
        self, max_bytes: int
    ) -> tuple[int, UInt64Array, UInt64Array, Complex128Array]: ...

class NativeU1MvpPlan:
    def apply_adjoint(
        self, state: Complex128Array, max_bytes: int
    ) -> Complex128Array: ...
    @property
    def nqubits(self) -> int: ...
    @property
    def particle_number(self) -> int: ...
    @property
    def dimension(self) -> int: ...
    @property
    def transition_count(self) -> int: ...
    @property
    def native_variant(self) -> str: ...
    def apply(self, state: Complex128Array, max_bytes: int) -> Complex128Array: ...
    def apply_into(
        self, state: Complex128Array, output: Complex128Array, max_bytes: int
    ) -> None: ...

class NativeU1LazyMvpPlan:
    def apply_adjoint(
        self, state: Complex128Array, max_bytes: int
    ) -> Complex128Array: ...
    @property
    def nqubits(self) -> int: ...
    @property
    def particle_number(self) -> int: ...
    @property
    def dimension(self) -> int: ...
    @property
    def estimated_bytes(self) -> int: ...
    @property
    def native_variant(self) -> str: ...
    def apply(self, state: Complex128Array, max_bytes: int) -> Complex128Array: ...
    def apply_into(
        self, state: Complex128Array, output: Complex128Array, max_bytes: int
    ) -> None: ...

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
    def run(
        self, initial_state: Complex128Array, parameters: object
    ) -> Complex128Array: ...
    def run_cached(
        self, initial_state: Complex128Array, parameters: object
    ) -> NativeU1FinalState: ...
    def probability(
        self, initial_state: Complex128Array, parameters: object
    ) -> Float64Array: ...
    def to_dense(
        self, initial_state: Complex128Array, parameters: object
    ) -> Complex128Array: ...
    def probability_full(
        self, initial_state: Complex128Array, parameters: object
    ) -> Float64Array: ...
    def expectation_handle(
        self,
        initial_state: Complex128Array,
        observable: NativePauliOperatorHandle,
        parameters: object,
    ) -> tuple[float, float]: ...
    def value_and_grad_handle(
        self,
        initial_state: Complex128Array,
        observable: NativePauliOperatorHandle,
        parameters: object,
    ) -> tuple[float, Float64Array]: ...

class NativeU1FinalState:
    def state_array(self) -> Complex128Array: ...
    def probability(self) -> Float64Array: ...
    def to_dense(self) -> Complex128Array: ...
    def probability_full(self) -> Float64Array: ...
    def expectation_handle(
        self, observable: NativePauliOperatorHandle
    ) -> tuple[float, float]: ...
    def value_and_grad_handle(
        self, observable: NativePauliOperatorHandle
    ) -> tuple[float, Float64Array]: ...

class NativePropagationEngine:
    @property
    def nqubits(self) -> int: ...
    @property
    def nparameters(self) -> int: ...
    @property
    def gate_count(self) -> int: ...
    @property
    def term_count(self) -> int: ...
    @property
    def max_weight(self) -> int | None: ...
    @property
    def is_exact(self) -> bool: ...
    @property
    def is_hermitian_observable(self) -> bool: ...
    def expectation(self, parameters: object) -> float: ...
    def value_and_grad(
        self, parameters: object, checkpoint_interval: int | None = None
    ) -> tuple[float, Float64Array]: ...
    def propagate_operator(self, parameters: object) -> NativePauliOperatorHandle: ...
    def profile(
        self, parameters: object
    ) -> tuple[float, int, int, int, int, Sequence[int], float]: ...

class NativeGateTape:
    @property
    def nqubits(self) -> int: ...
    @property
    def nparameters(self) -> int: ...
    @property
    def gate_count(self) -> int: ...

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
    def expectations(self, parameters: object) -> Float64Array: ...
    def values_and_gradients(
        self, parameters: object, checkpoint_interval: int | None = None
    ) -> tuple[Float64Array, Float64Array]: ...

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
        Float64Array,
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
        Float64Array,
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
    nqubits: int,
    x_words_left: Sequence[int],
    z_words_left: Sequence[int],
    x_words_right: Sequence[int],
    z_words_right: Sequence[int],
) -> tuple[Sequence[int], Sequence[int], int]: ...
def pauli_symplectic_inner_product(
    nqubits: int,
    x_words_left: Sequence[int],
    z_words_left: Sequence[int],
    x_words_right: Sequence[int],
    z_words_right: Sequence[int],
) -> int: ...
def pauli_canonicalize_batch_numpy(
    nqubits: int,
    structures: UInt8Array,
    coefficients: Complex128Array,
) -> tuple[int, UInt8Array, Complex128Array, UIntpArray, UInt8Array]: ...
def pauli_operator_native(
    nqubits: int,
    structures: Sequence[Sequence[int]],
    coefficients_re: Sequence[float],
    coefficients_im: Sequence[float],
    max_bytes: int,
) -> NativePauliOperatorHandle: ...
def pauli_operator_native_array(
    nqubits: int, structures: UInt8Array, coefficients: Complex128Array, max_bytes: int
) -> NativePauliOperatorHandle: ...
def pauli_group_handle(
    handle: NativePauliOperatorHandle,
    mode: int,
    algorithm: int,
    max_entries: int,
) -> tuple[
    Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[Sequence[Sequence[int]]]
]: ...
def pauli_qwc_group_handle(
    handle: NativePauliOperatorHandle,
    algorithm: int,
    max_entries: int,
) -> tuple[
    Sequence[Sequence[int]],
    Sequence[Sequence[int]],
    NativeQwcGroupingHandle,
]: ...
def pauli_compatibility_matrix_handle(
    handle: NativePauliOperatorHandle, mode: int, max_entries: int
) -> Sequence[bool]: ...
def pauli_incompatibility_edges_handle(
    handle: NativePauliOperatorHandle, mode: int, max_edges: int
) -> Sequence[tuple[int, int]]: ...
def pauli_dense_handle(
    handle: NativePauliOperatorHandle, max_bytes: int
) -> tuple[int, Complex128Array]: ...
def pauli_coo_handle(
    handle: NativePauliOperatorHandle, max_bytes: int
) -> tuple[int, UInt64Array, UInt64Array, Complex128Array]: ...
def pauli_csr_handle(
    handle: NativePauliOperatorHandle, max_bytes: int
) -> tuple[int, UInt64Array, UInt64Array, Complex128Array]: ...
def pauli_mvp_handle(
    handle: NativePauliOperatorHandle, state: Complex128Array, max_bytes: int
) -> Complex128Array: ...
def pauli_mvp_plan_handle(
    handle: NativePauliOperatorHandle,
    max_bytes: int,
    storage: str = ...,
) -> NativeMvpPlan: ...
def pauli_backend_plan_handle(
    handle: NativePauliOperatorHandle, max_bytes: int
) -> tuple[
    int,
    int,
    int,
    UInt64Array,
    UInt64Array,
    Complex128Array,
]: ...
def pauli_commutes(
    nqubits: int,
    x_words_left: Sequence[int],
    z_words_left: Sequence[int],
    x_words_right: Sequence[int],
    z_words_right: Sequence[int],
) -> bool: ...
def pauli_find_z2_symmetries_handle(
    operator: NativePauliOperatorHandle, max_bytes: int
) -> tuple[Sequence[Sequence[int]], int]: ...
def pauli_z2_tapering_plan(
    nqubits: int,
    generators: Sequence[Sequence[int]],
    sector: Sequence[int],
) -> NativeZ2TaperingPlan: ...
def pauli_restrict_u1_handle(
    handle: NativePauliOperatorHandle, particle_number: int, max_bytes: int
) -> NativeU1RestrictedOperator: ...
def pauli_restrict_u1_lazy_handle(
    handle: NativePauliOperatorHandle, particle_number: int, max_bytes: int
) -> NativeU1LazyMvpPlan: ...
def u1_basis_words(
    nqubits: int,
    particle_number: int,
    max_bytes: int,
) -> tuple[int, int, UInt64Array]: ...
def u1_circuit_plan(
    nqubits: int,
    particle_number: int,
    schema_version: int,
    nparameters: int,
    angles: object,
    gates: object,
    max_bytes: int,
) -> NativeU1CircuitPlan: ...
def pauli_gate_tape(
    nqubits: int,
    operations: Sequence[tuple[int, int, int, int, float, Sequence[float]]],
    max_bytes: int | None = None,
) -> NativeGateTape: ...
def pauli_propagation_engine_tape(
    tape: NativeGateTape,
    observable: NativePauliOperatorHandle,
    state_kind: int,
    state_bits: Sequence[int],
    state_values: Sequence[float],
    max_weight: int | None = None,
    max_bytes: int | None = None,
) -> NativePropagationEngine: ...
def pauli_propagation_batch_handles_tape(
    tape: NativeGateTape,
    observables: Sequence[NativePauliOperatorHandle],
    state_kind: int,
    state_bits: Sequence[int],
    state_values: Sequence[float],
    max_weight: int | None = None,
    max_bytes: int | None = None,
) -> NativePropagationBatch: ...
def pauli_spps_engine_tape(
    tape: NativeGateTape,
    observable: NativePauliOperatorHandle,
    state_kind: int,
    state_bits: Sequence[int],
    state_values: Sequence[float],
    smoothing: float = 0.01,
    max_bytes: int | None = None,
) -> NativeSPPSEngine: ...

class NativeSnapshotsHandle:
    @property
    def protocol(self) -> str: ...
    @property
    def nqubits(self) -> int: ...
    @property
    def nsettings(self) -> int: ...
    @property
    def repeats(self) -> int: ...
    @property
    def shots(self) -> int: ...
    def pauli_arrays(self) -> tuple[UInt8Array, UInt8Array]: ...
    def clifford_arrays(
        self,
    ) -> tuple[
        UInt64Array,
        UInt64Array,
        Int8Array,
        UInt8Array,
        int,
        int,
    ]: ...
    def expectation(
        self,
        observable: NativePauliOperatorHandle,
        blocks: int,
        return_blocks: bool,
    ) -> list[float]: ...
    def estimate_many(
        self,
        observables: Sequence[NativePauliOperatorHandle],
        blocks: int,
        return_blocks: bool,
    ) -> list[float]: ...
    def energy(
        self,
        hamiltonian: NativePauliOperatorHandle,
        blocks: int,
        return_blocks: bool,
    ) -> list[float]: ...
    def rdm(
        self, keep: Sequence[int], max_bytes: int
    ) -> tuple[int, Complex128Array]: ...
    def renyi_entropy(
        self, keep: Sequence[int], blocks: int, method: int, max_bytes: int
    ) -> float: ...
    def energy_variance(
        self, hamiltonian: NativePauliOperatorHandle, blocks: int, max_bytes: int
    ) -> float: ...
    def fidelity(
        self, target: NativeStabilizerStateHandle, blocks: int, return_blocks: bool
    ) -> list[float]: ...
    def fidelity_statevector(
        self, target: Complex128Array, blocks: int, return_blocks: bool
    ) -> list[float]: ...

class NativeStabilizerStateHandle:
    @property
    def nqubits(self) -> int: ...
    def tableau(
        self,
    ) -> tuple[
        UInt64Array,
        UInt64Array,
        Int8Array,
    ]: ...

class NativeStabilizerCodeHandle:
    @property
    def nqubits(self) -> int: ...
    @property
    def nchecks(self) -> int: ...
    @property
    def rank(self) -> int: ...
    @property
    def nlogical(self) -> int: ...
    @property
    def nrelations(self) -> int: ...
    @property
    def word_count(self) -> int: ...
    def check_arrays(self) -> tuple[UInt64Array, UInt64Array, Int8Array]: ...
    def check_weights(self) -> UIntpArray: ...
    def qubit_degrees(self) -> UIntpArray: ...
    def check_relations_unpacked(self) -> UInt8Array: ...
    def check_relations_packed(self) -> UInt64Array: ...
    def logical_operator_arrays(
        self,
    ) -> tuple[UInt64Array, UInt64Array, UInt64Array, UInt64Array]: ...
    def search_logical_witness(
        self,
        sector: int,
        max_trials: int,
        sketch_rows: int,
        stop_below_weight: int | None,
        seed: int,
        max_bytes: int,
    ) -> tuple[
        tuple[UInt64Array, UInt64Array, UInt64Array, UInt64Array],
        tuple[
            tuple[int, int, int, str, int, int, int, int | None, int, bool],
            tuple[int, int, int, int, int],
        ],
    ]: ...
    def syndrome_one(
        self, x_words: Sequence[int], z_words: Sequence[int]
    ) -> UInt8Array: ...
    def classify_one(
        self, x_words: Sequence[int], z_words: Sequence[int]
    ) -> tuple[UInt8Array, int, UInt8Array | None, UInt8Array | None]: ...
    def syndrome_many_words_unpacked(
        self, x_words: UInt64Array, z_words: UInt64Array, max_bytes: int
    ) -> UInt8Array: ...
    def syndrome_many_words_packed(
        self, x_words: UInt64Array, z_words: UInt64Array, max_bytes: int
    ) -> UInt64Array: ...
    def syndrome_many_codes_unpacked(
        self, codes: UInt8Array, max_bytes: int
    ) -> UInt8Array: ...
    def syndrome_many_codes_packed(
        self, codes: UInt8Array, max_bytes: int
    ) -> UInt64Array: ...
    def classify_many_words_unpacked(
        self, x_words: UInt64Array, z_words: UInt64Array, max_bytes: int
    ) -> tuple[UInt8Array, UInt8Array, UIntpArray, UInt8Array, UInt8Array]: ...
    def classify_many_words_packed(
        self, x_words: UInt64Array, z_words: UInt64Array, max_bytes: int
    ) -> tuple[UInt64Array, UInt8Array, UIntpArray, UInt8Array, UInt8Array]: ...
    def classify_many_codes_unpacked(
        self, codes: UInt8Array, max_bytes: int
    ) -> tuple[UInt8Array, UInt8Array, UIntpArray, UInt8Array, UInt8Array]: ...
    def classify_many_codes_packed(
        self, codes: UInt8Array, max_bytes: int
    ) -> tuple[UInt64Array, UInt8Array, UIntpArray, UInt8Array, UInt8Array]: ...
    def error_set_words(
        self, x_words: UInt64Array, z_words: UInt64Array, max_bytes: int
    ) -> tuple[bool, tuple[int, int] | None, UInt8Array | None, UInt8Array | None]: ...
    def error_set_codes(
        self, codes: UInt8Array, max_bytes: int
    ) -> tuple[bool, tuple[int, int] | None, UInt8Array | None, UInt8Array | None]: ...
    def corrections_words_unpacked(
        self,
        error_x: UInt64Array,
        error_z: UInt64Array,
        correction_x: UInt64Array,
        correction_z: UInt64Array,
        max_bytes: int,
    ) -> tuple[UInt8Array, UInt8Array, UIntpArray, UInt8Array, UInt8Array]: ...
    def corrections_words_packed(
        self,
        error_x: UInt64Array,
        error_z: UInt64Array,
        correction_x: UInt64Array,
        correction_z: UInt64Array,
        max_bytes: int,
    ) -> tuple[UInt64Array, UInt8Array, UIntpArray, UInt8Array, UInt8Array]: ...
    def corrections_codes_unpacked(
        self, errors: UInt8Array, corrections: UInt8Array, max_bytes: int
    ) -> tuple[UInt8Array, UInt8Array, UIntpArray, UInt8Array, UInt8Array]: ...
    def corrections_codes_packed(
        self, errors: UInt8Array, corrections: UInt8Array, max_bytes: int
    ) -> tuple[UInt64Array, UInt8Array, UIntpArray, UInt8Array, UInt8Array]: ...

def shadows_from_pauli_measurements(
    bases: UInt8Array,
    outcomes: UInt8Array,
    max_bytes: int,
) -> NativeSnapshotsHandle: ...
def shadows_from_clifford_measurements(
    frame_x_words: UInt64Array,
    frame_z_words: UInt64Array,
    frame_signs: Int8Array,
    outcomes: UInt8Array,
    nqubits: int,
    frame_version: int,
    max_bytes: int,
) -> NativeSnapshotsHandle: ...
def shadows_sample_statevector(
    state: Complex128Array,
    nsettings: int,
    protocol_code: int,
    repeats: int,
    seed: int,
    max_bytes: int,
) -> NativeSnapshotsHandle: ...
def shadows_sample_stabilizer(
    state: NativeStabilizerStateHandle,
    nsettings: int,
    protocol_code: int,
    repeats: int,
    seed: int,
    max_bytes: int,
) -> NativeSnapshotsHandle: ...
def stabilizer_zero_state(nqubits: int) -> NativeStabilizerStateHandle: ...
def stabilizer_from_tableau(
    x_words: UInt64Array,
    z_words: UInt64Array,
    signs: Int8Array,
) -> NativeStabilizerStateHandle: ...
def stabilizer_from_tape(tape: NativeGateTape) -> NativeStabilizerStateHandle: ...
def stabilizer_code_from_packed(
    nqubits: int,
    x_words: UInt64Array,
    z_words: UInt64Array,
    signs: Int8Array,
    max_bytes: int,
) -> NativeStabilizerCodeHandle: ...
def stabilizer_code_from_symplectic(
    x: UInt8Array,
    z: UInt8Array,
    signs: Int8Array,
    max_bytes: int,
) -> NativeStabilizerCodeHandle: ...
def stabilizer_code_from_css(
    x_checks: UInt8Array,
    z_checks: UInt8Array,
    x_signs: Int8Array,
    z_signs: Int8Array,
    max_bytes: int,
) -> NativeStabilizerCodeHandle: ...
