use std::fmt;

/// Typed errors shared by the Rust core and the Python exception mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PauliError {
    /// A code was not one of I/X/Y/Z.
    InvalidCode { code: u8, index: usize },
    /// Packed arrays do not have the required number of words.
    InvalidWordLength { expected: usize, actual: usize },
    /// Two operands use different qubit counts.
    IncompatibleQubitCounts { left: usize, right: usize },
    /// A structure has the wrong number of sites.
    InvalidStructureLength { expected: usize, actual: usize },
    /// A non-finite coefficient was supplied.
    NonFiniteCoefficient { index: usize },
    /// A supposedly canonical term sequence is not strictly ordered and nonzero.
    NonCanonicalTerms { index: usize },
    /// A requested dimension cannot be represented or allocated.
    Overflow { context: &'static str },
    /// A requested allocation exceeds an explicit limit.
    MemoryLimit { requested: u128, limit: u128 },
    /// A requested sector value is invalid.
    InvalidSector { context: &'static str },
    /// A Pauli operator does not preserve the requested sector.
    SectorLeakage { input: usize, output: usize },
    /// An observable is incompatible with the selected symmetry sector.
    IncompatibleSymmetry,
    /// A requested index or bitstring is outside the sector domain.
    InvalidIndex { context: &'static str },
    /// A Clifford operation is malformed.
    InvalidClifford { context: &'static str },
}

impl fmt::Display for PauliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCode { code, index } => {
                write!(
                    formatter,
                    "invalid Pauli code {code} at index {index}; expected 0..3"
                )
            }
            Self::InvalidWordLength { expected, actual } => {
                write!(formatter, "expected {expected} packed words, got {actual}")
            }
            Self::IncompatibleQubitCounts { left, right } => {
                write!(formatter, "incompatible qubit counts: {left} and {right}")
            }
            Self::InvalidStructureLength { expected, actual } => {
                write!(
                    formatter,
                    "expected structure length {expected}, got {actual}"
                )
            }
            Self::NonFiniteCoefficient { index } => {
                write!(formatter, "coefficient at index {index} is not finite")
            }
            Self::NonCanonicalTerms { index } => {
                write!(
                    formatter,
                    "term sequence is not canonical at index {index}; expected strictly ordered nonzero terms"
                )
            }
            Self::Overflow { context } => write!(formatter, "integer overflow while {context}"),
            Self::MemoryLimit { requested, limit } => {
                write!(
                    formatter,
                    "requested {requested} bytes exceeds memory limit {limit}"
                )
            }
            Self::InvalidSector { context } => write!(formatter, "invalid sector: {context}"),
            Self::SectorLeakage { input, output } => write!(
                formatter,
                "U(1) sector leakage from basis state {input} to {output}"
            ),
            Self::IncompatibleSymmetry => {
                write!(
                    formatter,
                    "operator does not commute with the selected symmetry"
                )
            }
            Self::InvalidIndex { context } => write!(formatter, "invalid sector index: {context}"),
            Self::InvalidClifford { context } => {
                write!(formatter, "invalid Clifford plan: {context}")
            }
        }
    }
}

impl std::error::Error for PauliError {}
