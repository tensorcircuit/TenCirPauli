//! Fixed-Hamming-weight U(1) sectors and restricted Pauli operators.

use std::{mem::size_of, sync::Arc};

use rayon::prelude::*;

use crate::error::PauliError;
use crate::operator::{PauliOperator, PauliTerm};
use crate::scalar::{is_exact_zero, Complex64};

/// A fixed-particle-number computational-basis sector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U1Sector {
    nqubits: usize,
    particle_number: usize,
}

/// A matrix-free plan over a fixed-Hamming-weight basis.
#[derive(Clone, Debug, PartialEq)]
pub struct U1MvpPlan {
    sector: U1Sector,
    indptr: Arc<[usize]>,
    columns: Arc<[usize]>,
    values: Arc<[Complex64]>,
}

/// A restricted operator whose setup has already validated sector preservation.
#[derive(Clone, Debug, PartialEq)]
pub struct U1RestrictedOperator {
    plan: U1MvpPlan,
}

/// CSR arrays for a restricted operator.
#[derive(Clone, Debug, PartialEq)]
pub struct U1CsrMatrix {
    pub dimension: usize,
    pub indptr: Vec<u64>,
    pub columns: Vec<u64>,
    pub values: Vec<Complex64>,
}

/// COO arrays for a restricted operator.
#[derive(Clone, Debug, PartialEq)]
pub struct U1CooMatrix {
    pub dimension: usize,
    pub rows: Vec<u64>,
    pub columns: Vec<u64>,
    pub values: Vec<Complex64>,
}

impl U1Sector {
    pub fn new(nqubits: usize, particle_number: usize) -> Result<Self, PauliError> {
        if particle_number > nqubits {
            return Err(PauliError::InvalidSector {
                context: "particle_number must be between 0 and nqubits",
            });
        }
        choose(nqubits, particle_number)?;
        Ok(Self {
            nqubits,
            particle_number,
        })
    }

    pub fn nqubits(&self) -> usize {
        self.nqubits
    }

    pub fn particle_number(&self) -> usize {
        self.particle_number
    }

    pub fn dimension(&self) -> Result<usize, PauliError> {
        choose(self.nqubits, self.particle_number)
    }

    /// Rank a TensorCircuit-order computational basis integer.
    pub fn rank(&self, bitstring: usize) -> Result<usize, PauliError> {
        ensure_integer_width(self.nqubits)?;
        self.rank_native(bitstring)
    }

    fn rank_native(&self, bitstring: usize) -> Result<usize, PauliError> {
        if self.nqubits < usize::BITS as usize && bitstring >= (1_usize << self.nqubits) {
            return Err(PauliError::InvalidIndex {
                context: "bitstring is outside the computational basis",
            });
        }
        if bitstring.count_ones() as usize != self.particle_number {
            return Err(PauliError::InvalidIndex {
                context: "bitstring has the wrong Hamming weight",
            });
        }
        let mut rank = 0_usize;
        let mut ones = 0_usize;
        for position in 0..self.nqubits {
            let bit = (bitstring >> (self.nqubits - 1 - position)) & 1;
            if bit == 1 {
                let remaining = self.nqubits - position - 1;
                let needed = self.particle_number - ones;
                rank =
                    rank.checked_add(choose(remaining, needed)?)
                        .ok_or(PauliError::Overflow {
                            context: "ranking U1 basis state",
                        })?;
                ones += 1;
            }
        }
        Ok(rank)
    }

    /// Unrank in ascending TensorCircuit computational-basis integer order.
    pub fn unrank(&self, index: usize) -> Result<usize, PauliError> {
        ensure_integer_width(self.nqubits)?;
        self.unrank_native(index)
    }

    fn unrank_native(&self, mut index: usize) -> Result<usize, PauliError> {
        let dimension = self.dimension()?;
        if index >= dimension {
            return Err(PauliError::InvalidIndex {
                context: "restricted basis index is out of range",
            });
        }
        let mut value = 0_usize;
        let mut remaining_ones = self.particle_number;
        for position in 0..self.nqubits {
            let remaining_sites = self.nqubits - position - 1;
            let zero_count = choose(remaining_sites, remaining_ones)?;
            if index >= zero_count {
                index -= zero_count;
                if remaining_ones == 0 {
                    return Err(PauliError::InvalidIndex {
                        context: "invalid combinatorial sector index",
                    });
                }
                value |= 1_usize << (self.nqubits - 1 - position);
                remaining_ones -= 1;
            }
        }
        Ok(value)
    }

    pub fn basis_words(&self, max_bytes: u128) -> Result<Vec<usize>, PauliError> {
        ensure_integer_width(self.nqubits)?;
        let dimension = self.dimension()?;
        let bytes = (dimension as u128)
            .checked_mul(size_of::<usize>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 basis memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        (0..dimension).map(|index| self.unrank(index)).collect()
    }
}

impl U1RestrictedOperator {
    pub fn new(
        operator: &PauliOperator,
        sector: U1Sector,
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        if operator.nqubits() != sector.nqubits {
            return Err(PauliError::IncompatibleQubitCounts {
                left: operator.nqubits(),
                right: sector.nqubits,
            });
        }
        ensure_native_width(sector.nqubits)?;
        let dimension = sector.dimension()?;
        let upper_bound =
            dimension
                .checked_mul(operator.terms().len())
                .ok_or(PauliError::Overflow {
                    context: "estimating U1 restricted transitions",
                })?;
        let estimated = operator
            .terms()
            .len()
            .checked_mul(size_of::<U1Term>())
            .and_then(|bytes| {
                dimension
                    .checked_add(1)
                    .and_then(|pointers| pointers.checked_mul(size_of::<usize>()))
                    .and_then(|pointers| bytes.checked_add(pointers))
            })
            .and_then(|bytes| {
                upper_bound
                    .checked_mul(size_of::<(usize, Complex64)>())
                    .and_then(|entries| bytes.checked_add(entries))
            })
            .ok_or(PauliError::Overflow {
                context: "estimating U1 restricted plan memory",
            })?;
        check_allocation(estimated as u128, max_bytes)?;

        let terms = operator
            .terms()
            .iter()
            .map(precompute_term)
            .collect::<Vec<_>>();
        let mut row_counts = vec![0_usize; dimension];
        let mut raw = Vec::with_capacity(terms.len());
        let mut aggregate = Vec::with_capacity(terms.len());
        for source_index in 0..dimension {
            let source = sector.unrank_native(source_index)?;
            aggregate_source(source, &terms, sector, &mut raw, &mut aggregate)?;
            for &(destination, _) in &aggregate {
                row_counts[destination] =
                    row_counts[destination]
                        .checked_add(1)
                        .ok_or(PauliError::Overflow {
                            context: "counting U1 restricted transitions",
                        })?;
            }
        }

        let mut indptr = vec![0_usize; dimension + 1];
        for row in 0..dimension {
            indptr[row + 1] =
                indptr[row]
                    .checked_add(row_counts[row])
                    .ok_or(PauliError::Overflow {
                        context: "sizing U1 restricted transitions",
                    })?;
        }
        let entry_count = indptr[dimension];
        let mut columns = vec![0_usize; entry_count];
        let mut values = vec![Complex64::default(); entry_count];
        let mut next = indptr[..dimension].to_vec();
        for source_index in 0..dimension {
            let source = sector.unrank_native(source_index)?;
            aggregate_source(source, &terms, sector, &mut raw, &mut aggregate)?;
            for &(destination, value) in &aggregate {
                let position = next[destination];
                columns[position] = source_index;
                values[position] = value;
                next[destination] += 1;
            }
        }

        Ok(Self {
            plan: U1MvpPlan {
                sector,
                indptr: Arc::from(indptr.into_boxed_slice()),
                columns: Arc::from(columns.into_boxed_slice()),
                values: Arc::from(values.into_boxed_slice()),
            },
        })
    }

    pub fn sector(&self) -> U1Sector {
        self.plan.sector
    }

    pub fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        self.plan.apply(state, max_bytes)
    }

    pub fn mvp_plan(&self, max_bytes: u128) -> Result<U1MvpPlan, PauliError> {
        self.plan.clone_with_budget(max_bytes)
    }

    pub fn dense(&self, max_bytes: u128) -> Result<(usize, Vec<Complex64>), PauliError> {
        let dimension = self.dimension();
        let entries = dimension
            .checked_mul(dimension)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 dense matrix entries",
            })?;
        let bytes = (entries as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 dense matrix memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        let mut values = vec![Complex64::default(); entries];
        for destination in 0..dimension {
            let start = self.plan.indptr[destination];
            let stop = self.plan.indptr[destination + 1];
            for index in start..stop {
                values[destination * dimension + self.plan.columns[index]] +=
                    self.plan.values[index];
            }
        }
        Ok((dimension, values))
    }

    pub fn coo(&self, max_bytes: u128) -> Result<U1CooMatrix, PauliError> {
        let dimension = self.dimension();
        let entry_count = self.plan.values.len();
        let output_bytes = (entry_count as u128)
            .checked_mul((2 * size_of::<u64>() + size_of::<Complex64>()) as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 COO output memory",
            })?;
        let mut rows = Vec::with_capacity(entry_count);
        let mut columns = Vec::with_capacity(entry_count);
        let mut values = Vec::with_capacity(entry_count);
        check_allocation(output_bytes, max_bytes)?;
        for destination in 0..dimension {
            let start = self.plan.indptr[destination];
            let stop = self.plan.indptr[destination + 1];
            for index in start..stop {
                rows.push(destination as u64);
                columns.push(self.plan.columns[index] as u64);
                values.push(self.plan.values[index]);
            }
        }
        Ok(U1CooMatrix {
            dimension,
            rows,
            columns,
            values,
        })
    }

    pub fn csr(&self, max_bytes: u128) -> Result<U1CsrMatrix, PauliError> {
        self.plan.csr(max_bytes)
    }
}

impl U1MvpPlan {
    pub fn sector(&self) -> U1Sector {
        self.sector
    }

    pub fn dimension(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        let dimension = self.dimension();
        if state.len() != dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: dimension,
                actual: state.len(),
            });
        }
        let bytes = (dimension as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 MVP output memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        let mut output = vec![Complex64::default(); dimension];
        self.apply_into(state, &mut output)?;
        Ok(output)
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        if state.len() != self.dimension() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension(),
                actual: state.len(),
            });
        }
        if output.len() != self.dimension() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension(),
                actual: output.len(),
            });
        }
        output.fill(Complex64::default());
        self.indptr
            .par_windows(2)
            .zip(output.par_iter_mut())
            .for_each(|(window, result)| {
                let mut value = Complex64::default();
                for index in window[0]..window[1] {
                    value += self.values[index] * state[self.columns[index]];
                }
                *result = value;
            });
        Ok(())
    }

    fn clone_with_budget(&self, max_bytes: u128) -> Result<Self, PauliError> {
        let bytes = (self.indptr.len() as u128)
            .checked_mul(size_of::<usize>() as u128)
            .and_then(|pointers| {
                (self.columns.len() as u128)
                    .checked_mul(size_of::<usize>() as u128)
                    .and_then(|columns| pointers.checked_add(columns))
            })
            .and_then(|bytes| {
                (self.values.len() as u128)
                    .checked_mul(size_of::<Complex64>() as u128)
                    .and_then(|values| bytes.checked_add(values))
            })
            .ok_or(PauliError::Overflow {
                context: "estimating U1 MVP plan memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        Ok(self.clone())
    }

    fn csr(&self, max_bytes: u128) -> Result<U1CsrMatrix, PauliError> {
        let dimension = self.dimension();
        let entries = self.values.len();
        let bytes = (dimension as u128 + 1)
            .checked_mul(size_of::<u64>() as u128)
            .and_then(|pointers| {
                (entries as u128)
                    .checked_mul((size_of::<u64>() + size_of::<Complex64>()) as u128)
                    .and_then(|values| pointers.checked_add(values))
            })
            .ok_or(PauliError::Overflow {
                context: "estimating U1 CSR memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        Ok(U1CsrMatrix {
            dimension,
            indptr: self.indptr.iter().map(|value| *value as u64).collect(),
            columns: self.columns.iter().map(|value| *value as u64).collect(),
            values: self.values.to_vec(),
        })
    }
}

#[derive(Clone, Copy)]
struct U1Term {
    x_mask: usize,
    z_mask: usize,
    coefficient: Complex64,
}

fn precompute_term(term: &PauliTerm) -> U1Term {
    let word = &term.word;
    let nqubits = word.nqubits();
    let mut x_mask = 0_usize;
    let mut z_mask = 0_usize;
    let mut y_count = 0_u32;
    for qubit in 0..nqubits {
        let mask = 1_u64 << (qubit % 64);
        let x = word.x_words()[qubit / 64] & mask != 0;
        let z = word.z_words()[qubit / 64] & mask != 0;
        let mask = 1_usize << (nqubits - 1 - qubit);
        if x {
            x_mask |= mask;
        }
        if z {
            z_mask |= mask;
        }
        if x && z {
            y_count += 1;
        }
    }
    let y_phase = match y_count % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    };
    U1Term {
        x_mask,
        z_mask,
        coefficient: term.coefficient * y_phase,
    }
}

fn aggregate_source(
    source: usize,
    terms: &[U1Term],
    sector: U1Sector,
    raw: &mut Vec<(usize, Complex64)>,
    aggregate: &mut Vec<(usize, Complex64)>,
) -> Result<(), PauliError> {
    raw.clear();
    aggregate.clear();
    raw.extend(terms.iter().map(|term| {
        let destination = source ^ term.x_mask;
        let sign = if (term.z_mask & source).count_ones() & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        (destination, term.coefficient * sign)
    }));
    raw.sort_unstable_by_key(|(destination, _)| *destination);
    let mut index = 0;
    while index < raw.len() {
        let destination = raw[index].0;
        let mut value = raw[index].1;
        index += 1;
        while index < raw.len() && raw[index].0 == destination {
            value += raw[index].1;
            index += 1;
        }
        if is_exact_zero(value) {
            continue;
        }
        if destination.count_ones() as usize != sector.particle_number {
            return Err(PauliError::SectorLeakage {
                input: source,
                output: destination,
            });
        }
        aggregate.push((sector.rank_native(destination)?, value));
    }
    Ok(())
}

fn choose(n: usize, k: usize) -> Result<usize, PauliError> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut value = 1_u128;
    for index in 1..=k {
        value = value
            .checked_mul((n - k + index) as u128)
            .and_then(|product| product.checked_div(index as u128))
            .ok_or(PauliError::Overflow {
                context: "computing U1 sector dimension",
            })?;
    }
    usize::try_from(value).map_err(|_| PauliError::Overflow {
        context: "converting U1 sector dimension",
    })
}

fn ensure_integer_width(nqubits: usize) -> Result<(), PauliError> {
    if nqubits >= usize::BITS as usize {
        return Err(PauliError::Overflow {
            context: "representing a computational basis integer",
        });
    }
    Ok(())
}

fn ensure_native_width(nqubits: usize) -> Result<(), PauliError> {
    if nqubits >= usize::BITS as usize {
        return Err(PauliError::Overflow {
            context: "using native U1 restriction with a single usize basis index",
        });
    }
    Ok(())
}

fn check_allocation(requested: u128, limit: u128) -> Result<(), PauliError> {
    if requested > limit {
        return Err(PauliError::MemoryLimit { requested, limit });
    }
    Ok(())
}
