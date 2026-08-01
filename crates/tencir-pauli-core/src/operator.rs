use std::cmp::Ordering;

use rustc_hash::FxHashMap;

use crate::error::PauliError;
use crate::scalar::{is_exact_zero, Complex64};
use crate::word::{PauliPhase, PauliWord};

/// A coefficient-bearing canonical Pauli term.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliTerm {
    /// Phase-free Pauli structure.
    pub word: PauliWord,
    /// Complex128-compatible coefficient.
    pub coefficient: Complex64,
}

/// Result of deterministic batch canonicalization before static zero removal.
pub struct Canonicalization {
    /// Canonical keys and their aggregated coefficients, including exact zeros.
    pub terms: Vec<PauliTerm>,
    /// Input-term index to canonical-key index mapping.
    pub input_to_canonical: Vec<usize>,
    /// Exact phase carried by each phase-free code-array input term.
    pub phase_multipliers: Vec<PauliPhase>,
}

/// A deterministic, canonical Pauli operator.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliOperator {
    pub(crate) nqubits: usize,
    pub(crate) terms: Vec<PauliTerm>,
}

fn canonicalize(
    nqubits: usize,
    structures: &[Vec<u8>],
    coefficients: &[Complex64],
) -> Result<Canonicalization, PauliError> {
    if structures.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: structures.len(),
            actual: coefficients.len(),
        });
    }
    let mut aggregate = FxHashMap::<PauliWord, Vec<(usize, Complex64)>>::with_capacity_and_hasher(
        structures.len(),
        Default::default(),
    );
    for (index, (structure, &coefficient)) in structures.iter().zip(coefficients).enumerate() {
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index });
        }
        let word = PauliWord::from_codes(nqubits, structure)?;
        aggregate
            .entry(word)
            .or_default()
            .push((index, coefficient));
    }
    let mut ordered = aggregate.into_iter().collect::<Vec<_>>();
    ordered.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let mut terms = Vec::with_capacity(ordered.len());
    let mut input_to_canonical = vec![0_usize; structures.len()];
    for (canonical_index, (word, mut contributions)) in ordered.into_iter().enumerate() {
        let first_input_index = contributions[0].0;
        for (input_index, _) in &contributions {
            input_to_canonical[*input_index] = canonical_index;
        }
        // Sort duplicate contributions by their IEEE bit patterns so
        // aggregation is independent of input order while retaining the
        // exact-zero policy. Input indices are a final deterministic tie-break.
        contributions
            .sort_by_key(|(index, value)| (value.re.to_bits(), value.im.to_bits(), *index));
        let coefficient = contributions
            .into_iter()
            .map(|(_, value)| value)
            .fold(Complex64::default(), |sum, value| sum + value);
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient {
                index: first_input_index,
            });
        }
        terms.push(PauliTerm { word, coefficient });
    }
    Ok(Canonicalization {
        terms,
        input_to_canonical,
        phase_multipliers: vec![PauliPhase::PlusOne; structures.len()],
    })
}

impl PauliOperator {
    /// Construct an operator, aggregate duplicate words, sort by code tuple,
    /// and remove only exact-zero coefficients.
    pub fn from_terms(
        nqubits: usize,
        structures: &[Vec<u8>],
        coefficients: &[Complex64],
    ) -> Result<Self, PauliError> {
        if structures.len() != coefficients.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: structures.len(),
                actual: coefficients.len(),
            });
        }
        let mut aggregate = FxHashMap::<PauliWord, Vec<Complex64>>::with_capacity_and_hasher(
            structures.len(),
            Default::default(),
        );
        for (index, (structure, &coefficient)) in structures.iter().zip(coefficients).enumerate() {
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index });
            }
            let word = PauliWord::from_codes(nqubits, structure)?;
            aggregate.entry(word).or_default().push(coefficient);
        }
        let mut ordered = aggregate.into_iter().collect::<Vec<_>>();
        ordered.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let mut terms = Vec::with_capacity(ordered.len());
        for (canonical_index, (word, mut values)) in ordered.into_iter().enumerate() {
            // Sort duplicate contributions by their IEEE bit patterns so
            // aggregation is independent of input order while retaining the
            // exact-zero policy.
            values.sort_by_key(|value| (value.re.to_bits(), value.im.to_bits()));
            let coefficient = values
                .into_iter()
                .fold(Complex64::default(), |sum, value| sum + value);
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient {
                    index: canonical_index,
                });
            }
            if !is_exact_zero(coefficient) {
                terms.push(PauliTerm { word, coefficient });
            }
        }
        Ok(Self { nqubits, terms })
    }

    /// Rebuild an already canonical static operator without sorting or reduction.
    pub fn from_canonical_terms(
        nqubits: usize,
        structures: &[Vec<u8>],
        coefficients: &[Complex64],
    ) -> Result<Self, PauliError> {
        if structures.len() != coefficients.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: structures.len(),
                actual: coefficients.len(),
            });
        }
        let mut terms = Vec::with_capacity(structures.len());
        for (index, (structure, &coefficient)) in structures.iter().zip(coefficients).enumerate() {
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index });
            }
            if is_exact_zero(coefficient) {
                return Err(PauliError::NonCanonicalTerms { index });
            }
            let word = PauliWord::from_codes(nqubits, structure)?;
            if terms
                .last()
                .is_some_and(|previous: &PauliTerm| previous.word >= word)
            {
                return Err(PauliError::NonCanonicalTerms { index });
            }
            terms.push(PauliTerm { word, coefficient });
        }
        Ok(Self { nqubits, terms })
    }

    /// Canonicalize a batch while retaining input mapping and exact zeros.
    pub fn canonicalize(
        nqubits: usize,
        structures: &[Vec<u8>],
        coefficients: &[Complex64],
    ) -> Result<Canonicalization, PauliError> {
        canonicalize(nqubits, structures, coefficients)
    }

    /// Construct an empty operator on `nqubits`.
    pub fn empty(nqubits: usize) -> Self {
        Self {
            nqubits,
            terms: Vec::new(),
        }
    }

    /// Return the qubit count.
    pub fn nqubits(&self) -> usize {
        self.nqubits
    }

    /// Return canonical terms.
    pub fn terms(&self) -> &[PauliTerm] {
        &self.terms
    }

    /// Add two operators by merging their canonical term streams.
    pub fn add(&self, other: &Self) -> Result<Self, PauliError> {
        self.ensure_compatible(other)?;
        let mut terms = Vec::with_capacity(self.terms.len() + other.terms.len());
        let mut left = 0;
        let mut right = 0;
        while left < self.terms.len() && right < other.terms.len() {
            match self.terms[left].word.cmp(&other.terms[right].word) {
                Ordering::Less => {
                    terms.push(self.terms[left].clone());
                    left += 1;
                }
                Ordering::Greater => {
                    terms.push(other.terms[right].clone());
                    right += 1;
                }
                Ordering::Equal => {
                    let coefficient = self.terms[left].coefficient + other.terms[right].coefficient;
                    if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                        return Err(PauliError::NonFiniteCoefficient { index: terms.len() });
                    }
                    if !is_exact_zero(coefficient) {
                        terms.push(PauliTerm {
                            word: self.terms[left].word.clone(),
                            coefficient,
                        });
                    }
                    left += 1;
                    right += 1;
                }
            }
        }
        terms.extend_from_slice(&self.terms[left..]);
        terms.extend_from_slice(&other.terms[right..]);
        Ok(Self {
            nqubits: self.nqubits,
            terms,
        })
    }

    /// Scale all coefficients by a finite complex scalar.
    pub fn scale(&self, scalar: Complex64) -> Result<Self, PauliError> {
        if !scalar.re.is_finite() || !scalar.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index: 0 });
        }
        if is_exact_zero(scalar) {
            return Ok(Self::empty(self.nqubits));
        }
        let mut terms = Vec::with_capacity(self.terms.len());
        for (index, term) in self.terms.iter().enumerate() {
            let coefficient = term.coefficient * scalar;
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index });
            }
            if !is_exact_zero(coefficient) {
                terms.push(PauliTerm {
                    word: term.word.clone(),
                    coefficient,
                });
            }
        }
        Ok(Self {
            nqubits: self.nqubits,
            terms,
        })
    }

    /// Multiply two canonical operators with exact Pauli phases.
    pub fn multiply(&self, other: &Self) -> Result<Self, PauliError> {
        self.ensure_compatible(other)?;
        self.terms
            .len()
            .checked_mul(other.terms.len())
            .ok_or(PauliError::Overflow {
                context: "estimating operator product terms",
            })?;
        let mut aggregate = FxHashMap::<PauliWord, Vec<Complex64>>::default();
        let mut product_index = 0;
        for left in &self.terms {
            for right in &other.terms {
                let (word, phase) = left.word.multiply(&right.word)?;
                let coefficient = left.coefficient * right.coefficient * phase.as_complex();
                if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                    return Err(PauliError::NonFiniteCoefficient {
                        index: product_index,
                    });
                }
                aggregate.entry(word).or_default().push(coefficient);
                product_index += 1;
            }
        }
        let mut ordered = aggregate.into_iter().collect::<Vec<_>>();
        ordered.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let mut terms = Vec::with_capacity(ordered.len());
        for (canonical_index, (word, mut values)) in ordered.into_iter().enumerate() {
            values.sort_by_key(|value| (value.re.to_bits(), value.im.to_bits()));
            let coefficient = values
                .into_iter()
                .fold(Complex64::default(), |sum, value| sum + value);
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient {
                    index: canonical_index,
                });
            }
            if !is_exact_zero(coefficient) {
                terms.push(PauliTerm { word, coefficient });
            }
        }
        Ok(Self {
            nqubits: self.nqubits,
            terms,
        })
    }

    /// Compute `[self, other]`.
    pub fn commutator(&self, other: &Self) -> Result<Self, PauliError> {
        self.multiply(other)?
            .add(&other.multiply(self)?.scale(Complex64::new(-1.0, 0.0))?)
    }

    /// Compute `{self, other}`.
    pub fn anticommutator(&self, other: &Self) -> Result<Self, PauliError> {
        self.multiply(other)?.add(&other.multiply(self)?)
    }

    /// Return the adjoint operator.
    pub fn adjoint(&self) -> Self {
        Self {
            nqubits: self.nqubits,
            terms: self
                .terms
                .iter()
                .map(|term| PauliTerm {
                    word: term.word.adjoint(),
                    coefficient: term.coefficient.conj(),
                })
                .collect(),
        }
    }

    /// Check exact Hermiticity or Hermiticity within an explicit tolerance.
    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }
        self.terms.iter().all(|term| {
            let difference = term.coefficient - term.coefficient.conj();
            difference.norm_sqr() <= tolerance * tolerance
        })
    }

    fn ensure_compatible(&self, other: &Self) -> Result<(), PauliError> {
        if self.nqubits != other.nqubits {
            return Err(PauliError::IncompatibleQubitCounts {
                left: self.nqubits,
                right: other.nqubits,
            });
        }
        Ok(())
    }
}
