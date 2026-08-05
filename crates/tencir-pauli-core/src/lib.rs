//! Pure Rust Pauli algebra and deterministic structural utilities.

mod charge;
mod charge_sector;
mod circuit_ir;
mod error;
mod gate;
mod grouping;
mod hamiltonian;
mod majorana;
mod mapping;
mod operator;
mod propagation;
mod scalar;
mod sector;
mod spps;
mod structured;
mod symmetry;
mod u1_circuit;
mod word;

pub use charge::{
    apply_charge_csr_into, apply_charge_mvp_from_plan, apply_charge_mvp_from_plan_into,
    apply_charge_mvp_from_prepared_plan, apply_charge_mvp_from_prepared_plan_into,
    build_fast_fermion_mvp_plan, compile_charge_transitions, compile_charge_transitions_from_plan,
    compile_charge_transitions_from_prepared_plan, estimate_charge_transition_terms_bytes,
    prepare_charge_transition_plan_layout, ChargeTransitionLayout, ChargeTransitionPlanLayout,
    ChargeTransitionResult, ChargeTransitionTerm, FastFermionMvpPlan,
    PreparedChargeTransitionPlanLayout,
};
pub use charge_sector::{
    build_charge_sector_plan, build_compact_charge_sector_plan, ChargeSectorPlan,
};
pub use circuit_ir::{CircuitGate, CircuitProgram, ParameterExprNode, CIRCUIT_SCHEMA_VERSION};
pub use error::PauliError;
pub use gate::{Clifford1, Clifford2, GateOperation, ParameterRef, RotationAxis};
pub use grouping::{
    compatibility_matrix, group_words, group_words_bounded, incompatibility_edges,
    GroupingAlgorithm, GroupingMode, DEFAULT_MAX_GROUPING_ENTRIES,
};
pub use hamiltonian::{BackendMvpPlan, CooMatrix, CsrMatrix, MvpPlan, MvpStrategy};
pub use majorana::{
    canonicalize_majorana_terms, fermion_to_majorana_terms, majorana_to_fermion_terms,
    multiply_majorana_terms, MajoranaBatch, MajoranaCanonicalResult,
};
pub use mapping::{build_mapping_plan, MappingPlan};
pub use operator::{Canonicalization, PauliOperator, PauliTerm};
pub use propagation::{
    ProductState, PropagationBatch, PropagationBatchValueAndGradient, PropagationEngine,
    PropagationResult, PropagationStats, PropagationValueAndGradient,
};
pub use scalar::Complex64;
pub use sector::{
    PackedU1Basis, U1CooMatrix, U1CsrMatrix, U1LazyMvpPlan, U1MvpPlan, U1RestrictedOperator,
    U1Sector,
};
pub use spps::{SPPSEngine, SPPSEstimate, SPPSValueEstimate};
pub use structured::{
    canonicalize_boson_terms, canonicalize_fermion_terms, canonicalize_hybrid_terms,
    jordan_wigner_hybrid_terms, jordan_wigner_terms, multiply_boson_terms, multiply_fermion_terms,
    multiply_hybrid_terms, structured_dense_matrix, structured_mvp_plan, structured_sparse_matrix,
    BosonCanonicalResult, FermionBatch, FermionCanonicalResult, HybridBatch, HybridCanonicalResult,
    HybridLayout, HybridRawBatch, StructuredMvpPlan, StructuredOperation, StructuredSparseResult,
};
pub use symmetry::{find_z2_symmetries, CliffordOperation, Z2SymmetryAnalysis, Z2TaperingPlan};
pub use u1_circuit::U1CircuitPlan;
pub use word::{packed_word_count, PauliPhase, PauliWord};

#[cfg(test)]
mod tests;
