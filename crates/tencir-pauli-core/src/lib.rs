//! Pure Rust Pauli algebra and deterministic structural utilities.

mod circuit_ir;
mod error;
mod gate;
mod grouping;
mod hamiltonian;
mod operator;
mod propagation;
mod scalar;
mod sector;
mod spps;
mod structured;
mod symmetry;
mod u1_circuit;
mod word;

pub use circuit_ir::{CircuitGate, CircuitProgram, ParameterExprNode, CIRCUIT_SCHEMA_VERSION};
pub use error::PauliError;
pub use gate::{Clifford1, Clifford2, GateOperation, ParameterRef, RotationAxis};
pub use grouping::{
    compatibility_matrix, group_words, group_words_bounded, incompatibility_edges,
    GroupingAlgorithm, GroupingMode, DEFAULT_MAX_GROUPING_ENTRIES,
};
pub use hamiltonian::{BackendMvpPlan, CooMatrix, CsrMatrix, MvpPlan, MvpStrategy};
pub use operator::{Canonicalization, PauliOperator, PauliTerm};
pub use propagation::{
    ProductState, PropagationBatch, PropagationBatchValueAndGradient, PropagationEngine,
    PropagationResult, PropagationStats, PropagationValueAndGradient,
};
pub use scalar::Complex64;
pub use sector::{
    PackedU1Basis, U1CooMatrix, U1CsrMatrix, U1MvpPlan, U1RestrictedOperator, U1Sector,
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
