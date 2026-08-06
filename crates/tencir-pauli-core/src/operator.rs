use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use rustc_hash::FxHashMap;

use crate::error::PauliError;
use crate::scalar::{hash_complex, is_exact_zero, Complex64};
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
        terms.push(PauliTerm { word, coefficient });
    }
    Ok(Canonicalization {
        terms,
        input_to_canonical,
        phase_multipliers: vec![PauliPhase::PlusOne; structures.len()],
    })
}

fn check_operator_bytes(
    entries: u128,
    nqubits: usize,
    max_bytes: u128,
    context: &'static str,
) -> Result<(), PauliError> {
    let word_count = nqubits
        .checked_add(63)
        .ok_or(PauliError::Overflow { context })?
        / 64;
    let bytes_per_entry = (word_count as u128)
        .checked_mul(16)
        .and_then(|value| value.checked_add(16))
        .ok_or(PauliError::Overflow { context })?;
    let requested = entries
        .max(1)
        .checked_mul(bytes_per_entry)
        .ok_or(PauliError::Overflow { context })?;
    if requested > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    Ok(())
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
        for (word, mut values) in ordered {
            // Sort duplicate contributions by their IEEE bit patterns so
            // aggregation is independent of input order while retaining the
            // exact-zero policy.
            values.sort_by_key(|value| (value.re.to_bits(), value.im.to_bits()));
            let coefficient = values
                .into_iter()
                .fold(Complex64::default(), |sum, value| sum + value);
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
        self.add_with_limit(other, u128::MAX)
    }

    /// Add two operators with a checked output/workspace estimate.
    pub fn add_with_limit(&self, other: &Self, max_bytes: u128) -> Result<Self, PauliError> {
        self.ensure_compatible(other)?;
        check_operator_bytes(
            (self.terms.len() as u128)
                .checked_add(other.terms.len() as u128)
                .ok_or(PauliError::Overflow {
                    context: "estimating Pauli operator addition",
                })?,
            self.nqubits,
            max_bytes,
            "estimating Pauli operator addition",
        )?;
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
        for term in &self.terms {
            let coefficient = term.coefficient * scalar;
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
        self.multiply_with_limit(other, u128::MAX)
    }

    /// Multiply two operators with a checked product workspace estimate.
    pub fn multiply_with_limit(&self, other: &Self, max_bytes: u128) -> Result<Self, PauliError> {
        self.ensure_compatible(other)?;
        let pair_count =
            self.terms
                .len()
                .checked_mul(other.terms.len())
                .ok_or(PauliError::Overflow {
                    context: "estimating operator product terms",
                })?;
        check_operator_bytes(
            pair_count as u128,
            self.nqubits,
            max_bytes,
            "estimating Pauli operator product",
        )?;
        let mut aggregate = FxHashMap::<PauliWord, Vec<Complex64>>::default();
        for left in &self.terms {
            for right in &other.terms {
                let (word, phase) = left.word.multiply(&right.word)?;
                let coefficient = left.coefficient * right.coefficient * phase.as_complex();
                aggregate.entry(word).or_default().push(coefficient);
            }
        }
        let mut ordered = aggregate.into_iter().collect::<Vec<_>>();
        ordered.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let mut terms = Vec::with_capacity(ordered.len());
        for (word, mut values) in ordered {
            values.sort_by_key(|value| (value.re.to_bits(), value.im.to_bits()));
            let coefficient = values
                .into_iter()
                .fold(Complex64::default(), |sum, value| sum + value);
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
        self.commutator_with_limit(other, u128::MAX)
    }

    /// Compute `[self, other]` with checked intermediate estimates.
    pub fn commutator_with_limit(&self, other: &Self, max_bytes: u128) -> Result<Self, PauliError> {
        self.multiply_with_limit(other, max_bytes)?.add_with_limit(
            &other
                .multiply_with_limit(self, max_bytes)?
                .scale(Complex64::new(-1.0, 0.0))?,
            max_bytes,
        )
    }

    /// Compute `{self, other}`.
    pub fn anticommutator(&self, other: &Self) -> Result<Self, PauliError> {
        self.anticommutator_with_limit(other, u128::MAX)
    }

    /// Compute `{self, other}` with checked intermediate estimates.
    pub fn anticommutator_with_limit(
        &self,
        other: &Self,
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        self.multiply_with_limit(other, max_bytes)?
            .add_with_limit(&other.multiply_with_limit(self, max_bytes)?, max_bytes)
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
            difference.re.hypot(difference.im) <= tolerance
        })
    }

    /// Analyze a diagonal qubit charge using deterministic native selection rules.
    pub fn analyze_charge(
        &self,
        qubit_levels: &[(f64, f64)],
        max_bytes: u128,
    ) -> Result<(bool, usize), PauliError> {
        if qubit_levels.len() != self.nqubits {
            return Err(PauliError::InvalidStructureLength {
                expected: self.nqubits,
                actual: qubit_levels.len(),
            });
        }
        check_operator_bytes(
            self.terms.len() as u128,
            self.nqubits,
            max_bytes,
            "estimating additive-charge analysis",
        )?;
        let mut aggregate = FxHashMap::<PauliWord, Vec<Complex64>>::default();
        for term in &self.terms {
            let codes = term.word.codes();
            for (index, code) in codes.into_iter().enumerate() {
                if code != 1 && code != 2 {
                    continue;
                }
                let difference = qubit_levels[index].0 - qubit_levels[index].1;
                if difference == 0.0 {
                    continue;
                }
                let mut changed = term.word.codes();
                changed[index] = if code == 1 { 2 } else { 1 };
                let word = PauliWord::from_codes(self.nqubits, &changed)?;
                let scale = if code == 1 { -difference } else { difference };
                aggregate
                    .entry(word)
                    .or_default()
                    .push(term.coefficient * Complex64::new(0.0, scale));
            }
        }
        let nonzero = aggregate
            .into_values()
            .filter_map(|mut values| {
                values.sort_by_key(|value| (value.re.to_bits(), value.im.to_bits()));
                let value = values
                    .into_iter()
                    .fold(Complex64::default(), |sum, value| sum + value);
                (value.re != 0.0 || value.im != 0.0).then_some(())
            })
            .count();
        Ok((nonzero == 0, nonzero))
    }

    /// Return whether every canonical term preserves a diagonal qubit charge.
    pub fn termwise_conserves_charge(&self, qubit_levels: &[(f64, f64)]) -> bool {
        if qubit_levels.len() != self.nqubits {
            return false;
        }
        self.terms.iter().all(|term| {
            term.word
                .codes()
                .into_iter()
                .enumerate()
                .all(|(index, code)| {
                    code == 0 || code == 3 || qubit_levels[index].0 == qubit_levels[index].1
                })
        })
    }

    pub fn content_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.nqubits.hash(&mut hasher);
        for term in &self.terms {
            term.word.hash(&mut hasher);
            hash_complex(term.coefficient, &mut hasher);
        }
        hasher.finish()
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
