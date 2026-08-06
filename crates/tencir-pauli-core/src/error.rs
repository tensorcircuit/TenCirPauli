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
    /// Molecular integral entries violate the supported Hermitian-pair contract.
    NonHermitianIntegral { context: &'static str },
    /// A supposedly canonical term sequence is not strictly ordered and nonzero.
    NonCanonicalTerms { index: usize },
    /// A requested dimension cannot be represented or allocated.
    Overflow { context: &'static str },
    /// A requested allocation exceeds an explicit limit.
    MemoryLimit { requested: u128, limit: u128 },
    /// A requested sector value is invalid.
    InvalidSector { context: &'static str },
    /// A Pauli operator does not preserve the requested sector after aggregation.
    SectorLeakage {
        source_index: u64,
        expected: usize,
        actual: usize,
    },
    /// An observable is incompatible with the selected symmetry sector.
    IncompatibleSymmetry,
    /// A requested index or bitstring is outside the sector domain.
    InvalidIndex { context: &'static str },
    /// A Clifford operation is malformed.
    InvalidClifford { context: &'static str },
    /// A gate wire is outside the tape's qubit range.
    InvalidWire { wire: usize, nqubits: usize },
    /// A two-qubit gate was given the same wire twice.
    DuplicateWire,
    /// A runtime parameter vector has the wrong length.
    InvalidParameterLength { expected: usize, actual: usize },
    /// A runtime gate parameter is not finite.
    NonFiniteParameter { index: usize },
    /// A Pauli-weight cutoff is invalid.
    InvalidMaxWeight,
    /// An observable requested a physical expectation but is not Hermitian.
    NonHermitianExpectation,
    /// A custom PTM has an invalid matrix shape.
    InvalidPtmShape { expected: usize, actual: usize },
    /// A custom PTM entry is not finite.
    NonFinitePtm { index: usize },
    /// A reverse checkpoint interval is invalid.
    InvalidCheckpointInterval,
    /// An SPPS smoothing parameter is invalid.
    InvalidSppsSmoothing,
    /// An SPPS sample budget or tolerance is invalid.
    InvalidSppsBudget { context: &'static str },
    /// An operation is not supported by SPPS.
    UnsupportedSppsGate,
    /// A common circuit IR is malformed.
    InvalidCircuit { context: &'static str },
    /// A circuit gate is outside the implemented schema.
    UnsupportedCircuitGate { context: &'static str },
}

impl fmt::Display for PauliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCode { code, index } => {
                write!(
                    formatter,
                    "invalid Pauli code {code} at index {index}; expected 0..=3"
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
            Self::NonHermitianIntegral { context } => {
                write!(formatter, "{context}")
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
            Self::SectorLeakage {
                source_index,
                expected,
                actual,
            } => write!(
                formatter,
                "U(1) sector leakage from restricted source index {source_index}: expected particle number {expected}, got destination weight {actual}"
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
            Self::InvalidWire { wire, nqubits } => {
                write!(formatter, "wire {wire} is outside 0..{nqubits}")
            }
            Self::DuplicateWire => write!(formatter, "two-qubit gate wires must differ"),
            Self::InvalidParameterLength { expected, actual } => write!(
                formatter,
                "expected parameter vector of length {expected}, got {actual}"
            ),
            Self::NonFiniteParameter { index } => {
                write!(formatter, "parameter at index {index} is not finite")
            }
            Self::InvalidMaxWeight => {
                write!(formatter, "max_weight must be a non-negative integer")
            }
            Self::NonHermitianExpectation => {
                write!(formatter, "expectation requires a Hermitian observable")
            }
            Self::InvalidPtmShape { expected, actual } => {
                write!(
                    formatter,
                    "expected PTM with {expected} entries, got {actual}"
                )
            }
            Self::NonFinitePtm { index } => {
                write!(formatter, "PTM entry at index {index} is not finite")
            }
            Self::InvalidCheckpointInterval => {
                write!(
                    formatter,
                    "checkpoint_interval must be a positive integer or None"
                )
            }
            Self::InvalidSppsSmoothing => {
                write!(formatter, "SPPS smoothing must be a finite positive float")
            }
            Self::InvalidSppsBudget { context } => write!(formatter, "invalid SPPS {context}"),
            Self::UnsupportedSppsGate => {
                write!(
                    formatter,
                    "SPPS supports Clifford and Pauli rotation gates only"
                )
            }
            Self::InvalidCircuit { context } => write!(formatter, "invalid circuit: {context}"),
            Self::UnsupportedCircuitGate { context } => {
                write!(formatter, "unsupported circuit gate: {context}")
            }
        }
    }
}

impl std::error::Error for PauliError {}
