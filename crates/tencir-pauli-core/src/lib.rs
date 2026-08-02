//! Pure Rust Pauli algebra and deterministic structural utilities.

mod error;
mod gate;
mod grouping;
mod hamiltonian;
mod operator;
mod propagation;
mod scalar;
mod sector;
mod symmetry;
mod word;

pub use error::PauliError;
pub use gate::{Clifford1, Clifford2, GateOperation, ParameterRef, RotationAxis};
pub use grouping::{
    compatibility_matrix, group_words, group_words_bounded, incompatibility_edges,
    GroupingAlgorithm, GroupingMode, DEFAULT_MAX_GROUPING_ENTRIES,
};
pub use hamiltonian::{BackendMvpPlan, CooMatrix, CsrMatrix, MvpPlan, MvpStrategy};
pub use operator::{Canonicalization, PauliOperator, PauliTerm};
pub use propagation::{ProductState, PropagationEngine, PropagationResult, PropagationStats};
pub use scalar::Complex64;
pub use sector::{U1CooMatrix, U1CsrMatrix, U1MvpPlan, U1RestrictedOperator, U1Sector};
pub use symmetry::{find_z2_symmetries, CliffordOperation, Z2SymmetryAnalysis, Z2TaperingPlan};
pub use word::{packed_word_count, PauliPhase, PauliWord};

#[cfg(test)]
mod tests;
