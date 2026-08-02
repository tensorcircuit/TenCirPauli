//! Pure Rust Pauli algebra and deterministic structural utilities.

mod error;
mod grouping;
mod hamiltonian;
mod operator;
mod scalar;
mod sector;
mod symmetry;
mod word;

pub use error::PauliError;
pub use grouping::{
    compatibility_matrix, group_words, group_words_bounded, incompatibility_edges,
    GroupingAlgorithm, GroupingMode, DEFAULT_MAX_GROUPING_ENTRIES,
};
pub use hamiltonian::{BackendMvpPlan, CooMatrix, CsrMatrix, MvpPlan, MvpStrategy};
pub use operator::{Canonicalization, PauliOperator, PauliTerm};
pub use scalar::Complex64;
pub use sector::{U1CooMatrix, U1CsrMatrix, U1MvpPlan, U1RestrictedOperator, U1Sector};
pub use symmetry::{find_z2_symmetries, CliffordOperation, Z2SymmetryAnalysis, Z2TaperingPlan};
pub use word::{packed_word_count, PauliPhase, PauliWord};

#[cfg(test)]
mod tests;
