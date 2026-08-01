//! Pure Rust Pauli algebra primitives for TenCirPauli.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Errors produced while constructing or combining Pauli words.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PauliError {
    /// The packed word count does not match the declared number of qubits.
    InvalidWordCount {
        /// Number of words required for `nqubits`.
        expected: usize,
        /// Number of X words supplied by the caller.
        x_words: usize,
        /// Number of Z words supplied by the caller.
        z_words: usize,
    },
    /// Two Pauli words describe different Hilbert spaces.
    IncompatibleQubitCounts {
        /// Number of qubits in the left operand.
        left: usize,
        /// Number of qubits in the right operand.
        right: usize,
    },
}

impl Display for PauliError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWordCount {
                expected,
                x_words,
                z_words,
            } => write!(
                formatter,
                "expected {expected} packed words, received {x_words} X words and {z_words} Z words"
            ),
            Self::IncompatibleQubitCounts { left, right } => {
                write!(formatter, "incompatible qubit counts: {left} and {right}")
            }
        }
    }
}

impl Error for PauliError {}

/// A phase-free Pauli word in binary symplectic representation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PauliWord {
    nqubits: usize,
    x_words: Vec<u64>,
    z_words: Vec<u64>,
}

impl PauliWord {
    /// Constructs a canonical word and clears unused bits above `nqubits`.
    pub fn from_words(
        nqubits: usize,
        mut x_words: Vec<u64>,
        mut z_words: Vec<u64>,
    ) -> Result<Self, PauliError> {
        let expected = nqubits.div_ceil(64);
        if x_words.len() != expected || z_words.len() != expected {
            return Err(PauliError::InvalidWordCount {
                expected,
                x_words: x_words.len(),
                z_words: z_words.len(),
            });
        }

        if let Some(mask) = final_word_mask(nqubits) {
            let last = expected - 1;
            x_words[last] &= mask;
            z_words[last] &= mask;
        }

        Ok(Self {
            nqubits,
            x_words,
            z_words,
        })
    }

    /// Returns the number of qubits represented by the word.
    pub fn nqubits(&self) -> usize {
        self.nqubits
    }

    /// Returns the packed X masks.
    pub fn x_words(&self) -> &[u64] {
        &self.x_words
    }

    /// Returns the packed Z masks.
    pub fn z_words(&self) -> &[u64] {
        &self.z_words
    }

    /// Returns the number of non-identity sites.
    pub fn weight(&self) -> u32 {
        self.x_words
            .iter()
            .zip(&self.z_words)
            .map(|(x_word, z_word)| (x_word | z_word).count_ones())
            .sum()
    }

    /// Returns whether this word commutes with `other`.
    pub fn commutes_with(&self, other: &Self) -> Result<bool, PauliError> {
        if self.nqubits != other.nqubits {
            return Err(PauliError::IncompatibleQubitCounts {
                left: self.nqubits,
                right: other.nqubits,
            });
        }

        let parity = self
            .x_words
            .iter()
            .zip(&self.z_words)
            .zip(&other.x_words)
            .zip(&other.z_words)
            .fold(
                0_u32,
                |accumulator, (((x_left, z_left), x_right), z_right)| {
                    accumulator
                        ^ ((x_left & z_right).count_ones() & 1)
                        ^ ((z_left & x_right).count_ones() & 1)
                },
            );
        Ok(parity == 0)
    }
}

fn final_word_mask(nqubits: usize) -> Option<u64> {
    let remainder = nqubits % 64;
    if remainder == 0 {
        None
    } else {
        Some((1_u64 << remainder) - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::PauliWord;

    #[test]
    fn computes_weight_and_canonicalizes_unused_bits() {
        let word = PauliWord::from_words(2, vec![u64::MAX], vec![0]).unwrap();
        assert_eq!(word.x_words(), &[0b11]);
        assert_eq!(word.weight(), 2);
    }

    #[test]
    fn checks_symplectic_commutation() {
        let x0 = PauliWord::from_words(2, vec![0b01], vec![0]).unwrap();
        let z0 = PauliWord::from_words(2, vec![0], vec![0b01]).unwrap();
        let xx = PauliWord::from_words(2, vec![0b11], vec![0]).unwrap();
        let zz = PauliWord::from_words(2, vec![0], vec![0b11]).unwrap();

        assert!(!x0.commutes_with(&z0).unwrap());
        assert!(xx.commutes_with(&zz).unwrap());
    }
}
