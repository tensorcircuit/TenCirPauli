//! Generic finite charge-sector transition compilation.

use std::collections::BTreeMap;

use crate::charge_sector::ChargeSectorPlan;
use crate::{Complex64, PauliError};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

// Release profiling keeps the serial gather for medium graphs (where Rayon
// scheduling dominates) and activates disjoint-row parallelism only for the
// large all-to-all graphs that showed a material end-to-end benefit.
const CSR_PARALLEL_TRANSITION_THRESHOLD: usize = 1 << 19;

/// One canonical structured term in a finite charge-sector layout.
#[derive(Clone, Debug)]
pub struct ChargeTransitionTerm {
    pub fermion_creation: Vec<u32>,
    pub fermion_annihilation: Vec<u32>,
    pub boson_blocks: Vec<(u32, u32, u32)>,
    pub qubit_codes: Vec<u8>,
    pub mapped_present: bool,
    pub mapped_codes: Vec<u8>,
    pub qudit_present: bool,
    pub qudit_triples: Vec<(u32, u32, u32)>,
    pub coefficient: Complex64,
}

/// Estimate the logical native storage for a batch of term descriptors.
///
/// Vec headers are counted together with their logical payloads; allocator
/// capacity and Python object overhead remain outside this best-effort policy.
pub fn estimate_charge_transition_terms_bytes(
    terms: &[ChargeTransitionTerm],
) -> Result<u128, PauliError> {
    terms.iter().try_fold(0_u128, |total, term| {
        let payload = (term.fermion_creation.len() as u128)
            .checked_mul(std::mem::size_of::<u32>() as u128)
            .and_then(|value| {
                value.checked_add(
                    (term.fermion_annihilation.len() as u128)
                        .checked_mul(std::mem::size_of::<u32>() as u128)?,
                )
            })
            .and_then(|value| {
                value.checked_add(
                    (term.boson_blocks.len() as u128).checked_mul(std::mem::size_of::<(
                        u32,
                        u32,
                        u32,
                    )>()
                        as u128)?,
                )
            })
            .and_then(|value| value.checked_add(term.qubit_codes.len() as u128))
            .and_then(|value| value.checked_add(term.mapped_codes.len() as u128))
            .and_then(|value| {
                value.checked_add(
                    (term.qudit_triples.len() as u128).checked_mul(std::mem::size_of::<(
                        u32,
                        u32,
                        u32,
                    )>()
                        as u128)?,
                )
            })
            .ok_or(PauliError::Overflow {
                context: "estimating charge transition descriptors",
            })?;
        total
            .checked_add(std::mem::size_of::<ChargeTransitionTerm>() as u128)
            .and_then(|value| value.checked_add(payload))
            .ok_or(PauliError::Overflow {
                context: "estimating charge transition descriptors",
            })
    })
}

/// Deterministic restricted transition arrays.
pub type ChargeTransitionResult = (Vec<u64>, Vec<u64>, Vec<Complex64>);

/// Apply a validated destination-major CSR graph without allocating
/// state-sized worker buffers. Parallel rows have deterministic per-row
/// accumulation order because each worker owns one complete output row.
pub fn apply_charge_csr_into(
    indptr: &[usize],
    columns: &[usize],
    values: &[Complex64],
    state: &[Complex64],
    output: &mut [Complex64],
    parallel: bool,
) -> Result<(), PauliError> {
    if indptr.len()
        != output.len().checked_add(1).ok_or(PauliError::Overflow {
            context: "validating charge CSR output length",
        })?
        || columns.len() != values.len()
        || indptr.last().copied() != Some(columns.len())
        || columns.iter().any(|&column| column >= state.len())
    {
        return Err(PauliError::InvalidSector {
            context: "invalid charge CSR execution graph",
        });
    }
    output.fill(Complex64::default());
    let use_parallel = parallel && values.len() >= CSR_PARALLEL_TRANSITION_THRESHOLD;
    if use_parallel {
        indptr
            .par_windows(2)
            .zip(output.par_iter_mut())
            .for_each(|(window, result)| {
                let mut value = Complex64::default();
                for index in window[0]..window[1] {
                    value += values[index] * state[columns[index]];
                }
                *result = value;
            });
    } else {
        for (window, result) in indptr.windows(2).zip(output.iter_mut()) {
            let mut value = Complex64::default();
            for index in window[0]..window[1] {
                value += values[index] * state[columns[index]];
            }
            *result = value;
        }
    }
    Ok(())
}

/// Immutable layout and memory policy for one restricted compilation.
pub struct ChargeTransitionLayout<'a> {
    pub dimension: usize,
    pub basis: &'a [u64],
    pub local_dimensions: &'a [u64],
    pub fermion_positions: &'a [u64],
    pub boson_positions: &'a [u64],
    pub qubit_positions: &'a [u64],
    pub qudit_positions: &'a [u64],
    pub qudit_dimension: u64,
    pub max_bytes: u128,
}

/// Layout for direct restricted compilation against a reusable charge plan.
pub struct ChargeTransitionPlanLayout<'a> {
    pub dimension: usize,
    pub local_dimensions: &'a [u64],
    pub fermion_positions: &'a [u64],
    pub boson_positions: &'a [u64],
    pub qubit_positions: &'a [u64],
    pub qudit_positions: &'a [u64],
    pub qudit_dimension: u64,
    pub max_bytes: u128,
}

/// Owned charge-sector layout validated once for repeated native execution.
pub struct PreparedChargeTransitionPlanLayout {
    dimension: usize,
    local_dimensions: Vec<u64>,
    fermion_positions: Vec<usize>,
    boson_positions: Vec<usize>,
    qubit_positions: Vec<usize>,
    qudit_positions: Vec<usize>,
    qudit_dimension: u64,
    estimated_bytes: u128,
}

impl PreparedChargeTransitionPlanLayout {
    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }
}

fn invalid_sector() -> PauliError {
    PauliError::InvalidSector {
        context: "invalid charge-sector transition input",
    }
}

fn check_bytes(entries: usize, bytes_per_entry: usize, limit: u128) -> Result<(), PauliError> {
    let requested = (entries as u128)
        .checked_mul(bytes_per_entry as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector transition storage",
        })?;
    if requested > limit {
        return Err(PauliError::MemoryLimit { requested, limit });
    }
    Ok(())
}

fn positions(values: &[u64], axis_count: usize, limit: u128) -> Result<Vec<usize>, PauliError> {
    check_bytes(values.len(), std::mem::size_of::<usize>(), limit)?;
    let mut result = Vec::with_capacity(values.len());
    for &value in values {
        let position = usize::try_from(value).map_err(|_| invalid_sector())?;
        if position >= axis_count {
            return Err(invalid_sector());
        }
        result.push(position);
    }
    if result.iter().collect::<FxHashSet<_>>().len() != result.len() {
        return Err(invalid_sector());
    }
    Ok(result)
}

fn validate_qudit_term(
    term: &ChargeTransitionTerm,
    index: usize,
    site_count: usize,
    dimension: u64,
) -> Result<(), PauliError> {
    if (!term.qudit_present && !term.qudit_triples.is_empty())
        || term
            .qudit_triples
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
        || term.qudit_triples.iter().any(|&(site, a, b)| {
            usize::try_from(site).map_or(true, |site| site >= site_count)
                || u64::from(a) >= dimension
                || u64::from(b) >= dimension
        })
    {
        return Err(PauliError::NonCanonicalTerms { index });
    }
    Ok(())
}

/// Convert and validate immutable charge layout and term metadata once.
#[allow(clippy::too_many_arguments)]
pub fn prepare_charge_transition_plan_layout(
    plan: &ChargeSectorPlan,
    dimension: usize,
    local_dimensions: Vec<u64>,
    fermion_positions: &[u64],
    boson_positions: &[u64],
    qubit_positions: &[u64],
    qudit_positions: &[u64],
    qudit_dimension: u64,
    terms: &[ChargeTransitionTerm],
    max_bytes: u128,
) -> Result<PreparedChargeTransitionPlanLayout, PauliError> {
    let axis_count = local_dimensions.len();
    if dimension != plan.dimension()
        || plan.local_dimensions().len() != axis_count
        || plan
            .local_dimensions()
            .iter()
            .zip(&local_dimensions)
            .any(|(&left, &right)| u64::try_from(left).ok() != Some(right))
        || local_dimensions.contains(&0)
    {
        return Err(invalid_sector());
    }
    let retained_positions = fermion_positions
        .len()
        .checked_add(boson_positions.len())
        .and_then(|value| value.checked_add(qubit_positions.len()))
        .and_then(|value| value.checked_add(qudit_positions.len()))
        .ok_or(PauliError::Overflow {
            context: "estimating prepared charge layout storage",
        })?;
    let estimated_bytes = (local_dimensions.len() as u128)
        .checked_mul(std::mem::size_of::<u64>() as u128)
        .and_then(|local_bytes| {
            (retained_positions as u128)
                .checked_mul(std::mem::size_of::<usize>() as u128)
                .and_then(|position_bytes| local_bytes.checked_add(position_bytes))
        })
        .ok_or(PauliError::Overflow {
            context: "estimating prepared charge layout storage",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    let fermion_positions = positions(fermion_positions, axis_count, max_bytes)?;
    let boson_positions = positions(boson_positions, axis_count, max_bytes)?;
    let qubit_positions = positions(qubit_positions, axis_count, max_bytes)?;
    let qudit_positions = positions(qudit_positions, axis_count, max_bytes)?;
    for (index, term) in terms.iter().enumerate() {
        if term.qubit_codes.len() != qubit_positions.len()
            || term.mapped_codes.len() != fermion_positions.len()
            || (term.mapped_present
                && (!term.fermion_creation.is_empty() || !term.fermion_annihilation.is_empty()))
        {
            return Err(invalid_sector());
        }
        validate_qudit_term(term, index, qudit_positions.len(), qudit_dimension)?;
        if !term.coefficient.re.is_finite() || !term.coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index });
        }
    }
    Ok(PreparedChargeTransitionPlanLayout {
        dimension,
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        qudit_positions,
        qudit_dimension,
        estimated_bytes,
    })
}

fn apply_phase(value: &mut Complex64, phase: Complex64) {
    *value *= phase;
}

fn apply_pauli(
    occupations: &mut [u64],
    codes: &[u8],
    positions: &[usize],
    coefficient: &mut Complex64,
) -> Result<(), PauliError> {
    if codes.len() != positions.len() {
        return Err(invalid_sector());
    }
    for (&code, &position) in codes.iter().zip(positions) {
        let bit = occupations[position];
        if bit > 1 {
            return Err(invalid_sector());
        }
        match code {
            0 => {}
            1 => occupations[position] = 1 - bit,
            2 => {
                occupations[position] = 1 - bit;
                apply_phase(
                    coefficient,
                    if bit == 0 {
                        Complex64::new(0.0, 1.0)
                    } else {
                        Complex64::new(0.0, -1.0)
                    },
                );
            }
            3 => {
                if bit == 1 {
                    *coefficient = -*coefficient;
                }
            }
            _ => return Err(invalid_sector()),
        }
    }
    Ok(())
}

fn apply_fermions(
    occupations: &mut [u64],
    creation: &[u32],
    annihilation: &[u32],
    positions: &[usize],
    coefficient: &mut Complex64,
) -> Result<bool, PauliError> {
    let packed_parity = positions.len() <= 128;
    let mut packed = [0_u64; 2];
    if packed_parity {
        for mode in 0..positions.len() {
            if occupations[positions[mode]] != 0 {
                packed[mode / 64] |= 1_u64 << (mode % 64);
            }
        }
    }
    let mut apply = |mode: u32, create: bool| -> Result<bool, PauliError> {
        let mode = usize::try_from(mode).map_err(|_| invalid_sector())?;
        if mode >= positions.len() {
            return Err(invalid_sector());
        }
        let position = positions[mode];
        let occupied = occupations[position];
        if occupied > 1 {
            return Err(invalid_sector());
        }
        let parity = if packed_parity {
            if mode < 64 {
                (packed[0] & if mode == 0 { 0 } else { (1_u64 << mode) - 1 }).count_ones() & 1
            } else {
                (packed[0].count_ones() + (packed[1] & ((1_u64 << (mode - 64)) - 1)).count_ones())
                    & 1
            }
        } else {
            ((0..mode)
                .map(|lower| occupations[positions[lower]])
                .sum::<u64>()
                & 1) as u32
        };
        if parity != 0 {
            *coefficient = -*coefficient;
        }
        if create {
            if occupied != 0 {
                return Ok(false);
            }
            occupations[position] = 1;
            if packed_parity {
                packed[mode / 64] |= 1_u64 << (mode % 64);
            }
        } else {
            if occupied == 0 {
                return Ok(false);
            }
            occupations[position] = 0;
            if packed_parity {
                packed[mode / 64] &= !(1_u64 << (mode % 64));
            }
        }
        Ok(true)
    };

    for &mode in annihilation.iter().rev() {
        if !apply(mode, false)? {
            return Ok(false);
        }
    }
    for &mode in creation.iter().rev() {
        if !apply(mode, true)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn apply_bosons(
    occupations: &mut [u64],
    blocks: &[(u32, u32, u32)],
    positions: &[usize],
    local_dimensions: &[u64],
    coefficient: &mut Complex64,
) -> Result<bool, PauliError> {
    for &(mode, creation, annihilation) in blocks.iter().rev() {
        let mode = usize::try_from(mode).map_err(|_| invalid_sector())?;
        if mode >= positions.len() {
            return Err(invalid_sector());
        }
        let position = positions[mode];
        for _ in 0..annihilation {
            let occupation = occupations[position];
            if occupation == 0 {
                return Ok(false);
            }
            *coefficient *= (occupation as f64).sqrt();
            occupations[position] = occupation - 1;
        }
        for _ in 0..creation {
            let occupation = occupations[position];
            if occupation
                .checked_add(1)
                .map(|value| value >= local_dimensions[position])
                .unwrap_or(true)
            {
                return Ok(false);
            }
            *coefficient *= ((occupation + 1) as f64).sqrt();
            occupations[position] = occupation + 1;
        }
    }
    Ok(true)
}

fn apply_qudits(
    occupations: &mut [u64],
    triples: &[(u32, u32, u32)],
    positions: &[usize],
    dimension: u64,
    coefficient: &mut Complex64,
) -> Result<(), PauliError> {
    if dimension == 0 {
        return Err(invalid_sector());
    }
    for &(site, a, b) in triples {
        let site = usize::try_from(site).map_err(|_| invalid_sector())?;
        if site >= positions.len() || u64::from(a) >= dimension || u64::from(b) >= dimension {
            return Err(invalid_sector());
        }
        let position = positions[site];
        let input = occupations[position];
        let angle = 2.0 * std::f64::consts::PI * ((u128::from(b) * u128::from(input)) as f64)
            / (dimension as f64);
        apply_phase(coefficient, Complex64::from_polar(1.0, angle));
        occupations[position] = (input + u64::from(a)) % dimension;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_charge_term(
    source: &[u64],
    destination: &mut [u64],
    term: &ChargeTransitionTerm,
    local_dimensions: &[u64],
    fermion_positions: &[usize],
    boson_positions: &[usize],
    qubit_positions: &[usize],
    qudit_positions: &[usize],
    qudit_dimension: u64,
) -> Result<Option<Complex64>, PauliError> {
    destination.copy_from_slice(source);
    let mut value = term.coefficient;
    if !apply_fermions(
        destination,
        &term.fermion_creation,
        &term.fermion_annihilation,
        fermion_positions,
        &mut value,
    )? {
        return Ok(None);
    }
    if !apply_bosons(
        destination,
        &term.boson_blocks,
        boson_positions,
        local_dimensions,
        &mut value,
    )? {
        return Ok(None);
    }
    apply_pauli(destination, &term.qubit_codes, qubit_positions, &mut value)?;
    if term.mapped_present {
        apply_pauli(
            destination,
            &term.mapped_codes,
            fermion_positions,
            &mut value,
        )?;
    }
    if term.qudit_present {
        apply_qudits(
            destination,
            &term.qudit_triples,
            qudit_positions,
            qudit_dimension,
            &mut value,
        )?;
    }
    if value.re == 0.0 && value.im == 0.0 {
        return Ok(None);
    }
    Ok(Some(value))
}

/// Return ``n choose k`` without overflowing the intermediate product when
/// the final result still fits in ``u128``.
fn binomial(n: usize, k: usize) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result = 1_u128;
    for index in 1..=k {
        let mut numerator = u128::from((n - k + index) as u64);
        let mut denominator = u128::from(index as u64);
        let common = gcd(numerator, denominator);
        numerator /= common;
        denominator /= common;
        let common = gcd(result, denominator);
        result /= common;
        denominator /= common;
        if denominator != 1 {
            return None;
        }
        result = result.checked_mul(numerator)?;
    }
    Some(result)
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn unrank_combination(length: usize, particles: usize, mut rank: u128) -> Option<u128> {
    let mut mask = 0_u128;
    let mut remaining = particles;
    for position in 0..length {
        if remaining == 0 {
            break;
        }
        let zero_count = binomial(length - position - 1, remaining)?;
        if rank < zero_count {
            continue;
        }
        rank -= zero_count;
        mask |= 1_u128 << position;
        remaining -= 1;
    }
    if remaining == 0 && rank == 0 {
        Some(mask)
    } else {
        None
    }
}

fn rank_combination(length: usize, particles: usize, mask: u128) -> Option<u128> {
    if mask.count_ones() as usize != particles || (length < 128 && mask >> length != 0) {
        return None;
    }
    let mut rank = 0_u128;
    let mut remaining = particles;
    for position in 0..length {
        if mask & (1_u128 << position) != 0 {
            rank = rank.checked_add(binomial(length - position - 1, remaining)?)?;
            remaining -= 1;
        }
    }
    (remaining == 0).then_some(rank)
}

struct FastFermionSectorIndex {
    sites: usize,
    particles: usize,
    combination_count: usize,
    combination_masks: Option<Vec<u128>>,
    rank_table: Option<Vec<u32>>,
}

const FAST_COMBINATION_CACHE_LIMIT: u128 = 64 * 1024 * 1024;
const FAST_RANK_CACHE_LIMIT: u128 = 4 * 1024 * 1024;

impl FastFermionSectorIndex {
    #[allow(clippy::too_many_arguments)]
    fn new(
        dimension: usize,
        local_dimensions: &[u64],
        fermion_positions: &[usize],
        boson_positions: &[usize],
        qubit_positions: &[usize],
        qudit_positions: &[usize],
        particles: usize,
        scratch_budget: u128,
    ) -> Option<Self> {
        if fermion_positions.len() % 2 != 0
            || !boson_positions.is_empty()
            || !qubit_positions.is_empty()
            || !qudit_positions.is_empty()
            || local_dimensions.iter().any(|&value| value != 2)
            || fermion_positions
                .iter()
                .enumerate()
                .any(|(mode, &position)| mode != position)
        {
            return None;
        }
        let sites = fermion_positions.len() / 2;
        if sites == 0 || sites > 64 || particles > sites {
            return None;
        }
        let combination_count = usize::try_from(binomial(sites, particles)?).ok()?;
        let expected_dimension = combination_count.checked_mul(combination_count)?;
        if dimension != expected_dimension {
            return None;
        }

        // These tables are optional. The direct combinatorial routines remain
        // the bounded-memory fallback for larger sectors.
        let mut remaining_budget = scratch_budget;
        let combination_masks = if (combination_count as u128)
            .checked_mul(std::mem::size_of::<u128>() as u128)?
            <= FAST_COMBINATION_CACHE_LIMIT
            && (combination_count as u128).checked_mul(16)? <= remaining_budget
        {
            let mut masks = Vec::with_capacity(combination_count);
            for rank in 0..combination_count {
                masks.push(unrank_combination(sites, particles, rank as u128)?);
            }
            remaining_budget -= (combination_count as u128) * 16;
            Some(masks)
        } else {
            None
        };
        let rank_table = if sites < usize::BITS as usize
            && sites <= 20
            && combination_count <= u32::MAX as usize
        {
            let table_len = 1_usize << sites;
            let table_bytes = (table_len as u128).checked_mul(4)?;
            if table_bytes <= FAST_RANK_CACHE_LIMIT && table_bytes <= remaining_budget {
                let mut table = vec![u32::MAX; table_len];
                for rank in 0..combination_count {
                    let mask = if let Some(masks) = &combination_masks {
                        masks[rank]
                    } else {
                        unrank_combination(sites, particles, rank as u128)?
                    };
                    table[mask as usize] = rank as u32;
                }
                Some(table)
            } else {
                None
            }
        } else {
            None
        };
        Some(Self {
            sites,
            particles,
            combination_count,
            combination_masks,
            rank_table,
        })
    }

    fn unrank(&self, rank: usize) -> Option<u128> {
        if rank >= self.combination_count {
            return None;
        }
        self.combination_masks
            .as_ref()
            .map(|masks| masks[rank])
            .or_else(|| unrank_combination(self.sites, self.particles, rank as u128))
    }

    fn rank(&self, mask: u128) -> Option<usize> {
        if let Some(table) = &self.rank_table {
            let index = usize::try_from(mask).ok()?;
            let rank = *table.get(index)?;
            (rank != u32::MAX).then_some(rank as usize)
        } else {
            usize::try_from(rank_combination(self.sites, self.particles, mask)?).ok()
        }
    }

    fn estimated_bytes(&self) -> u128 {
        let combinations = self
            .combination_masks
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<u128>());
        let ranks = self
            .rank_table
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<u32>());
        (combinations + ranks) as u128
    }
}

/// Cached specialized execution for a balanced pure spinful fermion sector.
///
/// The index and validated term descriptors are constructed once and shared by
/// all applications. The kernel writes directly into caller-owned output and
/// never allocates a state-sized temporary.
pub struct FastFermionMvpPlan {
    index: FastFermionSectorIndex,
    terms: Vec<FastFermionTerm>,
    estimated_bytes: u128,
}

#[derive(Clone, Debug)]
enum FastFermionTerm {
    Diagonal {
        modes: Box<[u8]>,
        coefficient: Complex64,
    },
    Hopping {
        creation: u8,
        annihilation: u8,
        coefficient: Complex64,
    },
    Generic {
        creation: Box<[u8]>,
        annihilation: Box<[u8]>,
        coefficient: Complex64,
    },
}

impl FastFermionMvpPlan {
    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        if state.len() != output.len() {
            return Err(invalid_sector());
        }
        let combinations = self.index.combination_count;
        if state.len()
            != combinations
                .checked_mul(combinations)
                .ok_or(PauliError::Overflow {
                    context: "sizing fast fermion MVP",
                })?
        {
            return Err(invalid_sector());
        }
        let mode_count = self.index.sites * 2;
        let group_mask = (1_u128 << self.index.sites) - 1;
        output.fill(Complex64::default());
        for (column, state_value) in state.iter().enumerate() {
            let up_rank = column / combinations;
            let down_rank = column % combinations;
            let source = self
                .index
                .unrank(up_rank)
                .ok_or(PauliError::InvalidSector {
                    context: "fast fermion sector unrank failed",
                })?
                | (self
                    .index
                    .unrank(down_rank)
                    .ok_or(PauliError::InvalidSector {
                        context: "fast fermion sector unrank failed",
                    })?
                    << self.index.sites);
            for term in &self.terms {
                let Some((destination, value)) = apply_fast_fermion_term(source, term, mode_count)?
                else {
                    continue;
                };
                let row = if destination == source {
                    column
                } else {
                    let destination_up = destination & group_mask;
                    let destination_down = (destination >> self.index.sites) & group_mask;
                    let up_index = if destination_up == (source & group_mask) {
                        up_rank
                    } else {
                        self.index
                            .rank(destination_up)
                            .ok_or(PauliError::InvalidSector {
                                context: "fast fermion sector rank failed",
                            })?
                    };
                    let down_index = if destination_down == (source >> self.index.sites) {
                        down_rank
                    } else {
                        self.index
                            .rank(destination_down)
                            .ok_or(PauliError::InvalidSector {
                                context: "fast fermion sector rank failed",
                            })?
                    };
                    up_index * combinations + down_index
                };
                output[row] += value * *state_value;
            }
        }
        Ok(())
    }
}

/// Build the cached spinful fast path when the native descriptors are eligible.
#[allow(clippy::too_many_arguments)]
pub fn build_fast_fermion_mvp_plan(
    plan: &ChargeSectorPlan,
    local_dimensions: &[u64],
    fermion_positions: &[u64],
    boson_positions: &[u64],
    qubit_positions: &[u64],
    qudit_positions: &[u64],
    terms: &[ChargeTransitionTerm],
    particles: Option<usize>,
    max_bytes: u128,
) -> Result<Option<FastFermionMvpPlan>, PauliError> {
    let Some(particles) = particles else {
        return Ok(None);
    };
    if terms.iter().any(|term| {
        !term.boson_blocks.is_empty()
            || !term.qubit_codes.is_empty()
            || term.mapped_present
            // The Python descriptor keeps a fixed-width zero code row for
            // terms without a mapped-fermion Pauli.  It is semantically
            // absent unless the corresponding presence bit is set.
            || (term.mapped_present && !term.mapped_codes.is_empty())
            || !term.qudit_triples.is_empty()
            || term.qudit_present
    }) {
        return Ok(None);
    }
    let fermion_positions = positions(fermion_positions, local_dimensions.len(), max_bytes)?;
    let boson_positions = positions(boson_positions, local_dimensions.len(), max_bytes)?;
    let qubit_positions = positions(qubit_positions, local_dimensions.len(), max_bytes)?;
    let qudit_positions = positions(qudit_positions, local_dimensions.len(), max_bytes)?;
    let Some(index) = FastFermionSectorIndex::new(
        plan.dimension(),
        local_dimensions,
        &fermion_positions,
        &boson_positions,
        &qubit_positions,
        &qudit_positions,
        particles,
        max_bytes,
    ) else {
        return Ok(None);
    };
    let mode_count = index.sites * 2;
    let mut compact_terms = Vec::with_capacity(terms.len());
    for (index, term) in terms.iter().enumerate() {
        if !term.coefficient.re.is_finite() || !term.coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index });
        }
        if term
            .fermion_creation
            .iter()
            .chain(&term.fermion_annihilation)
            .any(|&mode| usize::try_from(mode).map_or(true, |mode| mode >= mode_count))
        {
            return Err(PauliError::InvalidSector {
                context: "fast fermion term mode is outside the sector layout",
            });
        }
        let to_compact = |modes: &[u32]| -> Result<Box<[u8]>, PauliError> {
            modes
                .iter()
                .map(|&mode| {
                    u8::try_from(mode).map_err(|_| PauliError::InvalidSector {
                        context: "fast fermion term mode does not fit compact descriptor",
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Vec::into_boxed_slice)
        };
        if !term.fermion_creation.is_empty() && term.fermion_creation == term.fermion_annihilation {
            compact_terms.push(FastFermionTerm::Diagonal {
                modes: to_compact(&term.fermion_creation)?,
                coefficient: term.coefficient,
            });
        } else if term.fermion_creation.len() == 1 && term.fermion_annihilation.len() == 1 {
            compact_terms.push(FastFermionTerm::Hopping {
                creation: u8::try_from(term.fermion_creation[0]).map_err(|_| {
                    PauliError::InvalidSector {
                        context: "fast fermion hopping mode does not fit descriptor",
                    }
                })?,
                annihilation: u8::try_from(term.fermion_annihilation[0]).map_err(|_| {
                    PauliError::InvalidSector {
                        context: "fast fermion hopping mode does not fit descriptor",
                    }
                })?,
                coefficient: term.coefficient,
            });
        } else {
            compact_terms.push(FastFermionTerm::Generic {
                creation: to_compact(&term.fermion_creation)?,
                annihilation: to_compact(&term.fermion_annihilation)?,
                coefficient: term.coefficient,
            });
        }
    }
    let descriptor_bytes = compact_terms.iter().try_fold(0_u128, |total, term| {
        let payload = match term {
            FastFermionTerm::Diagonal { modes, .. } => modes.len(),
            FastFermionTerm::Hopping { .. } => 0,
            FastFermionTerm::Generic {
                creation,
                annihilation,
                ..
            } => creation.len() + annihilation.len(),
        };
        total
            .checked_add(std::mem::size_of::<FastFermionTerm>() as u128)
            .and_then(|value| value.checked_add(payload as u128))
            .ok_or(PauliError::Overflow {
                context: "estimating fast fermion descriptors",
            })
    })?;
    let estimated_bytes = index
        .estimated_bytes()
        .checked_add(descriptor_bytes)
        .ok_or(PauliError::Overflow {
            context: "estimating fast fermion plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    Ok(Some(FastFermionMvpPlan {
        index,
        terms: compact_terms,
        estimated_bytes,
    }))
}

fn fast_fermion_lower_mask(mode: usize) -> u128 {
    if mode == 0 {
        0
    } else {
        (1_u128 << mode) - 1
    }
}

fn apply_fast_fermion_term(
    source: u128,
    term: &FastFermionTerm,
    mode_count: usize,
) -> Result<Option<(u128, Complex64)>, PauliError> {
    let (creation, annihilation, mut value) = match term {
        FastFermionTerm::Diagonal { modes, coefficient } => {
            if modes.iter().any(|&mode| source & (1_u128 << mode) == 0) {
                return Ok(None);
            }
            return Ok(Some((source, *coefficient)));
        }
        FastFermionTerm::Hopping {
            creation,
            annihilation,
            coefficient,
        } => (
            std::slice::from_ref(creation),
            std::slice::from_ref(annihilation),
            *coefficient,
        ),
        FastFermionTerm::Generic {
            creation,
            annihilation,
            coefficient,
        } => (creation.as_ref(), annihilation.as_ref(), *coefficient),
    };
    let mut destination = source;
    let mut apply = |raw_mode: u8, create: bool| -> Result<bool, PauliError> {
        let mode = usize::from(raw_mode);
        if mode >= mode_count {
            return Err(invalid_sector());
        }
        let occupied = destination & (1_u128 << mode) != 0;
        if (destination & fast_fermion_lower_mask(mode)).count_ones() & 1 != 0 {
            value = -value;
        }
        if create {
            if occupied {
                return Ok(false);
            }
            destination |= 1_u128 << mode;
        } else {
            if !occupied {
                return Ok(false);
            }
            destination &= !(1_u128 << mode);
        }
        Ok(true)
    };
    for &mode in annihilation.iter().rev() {
        if !apply(mode, false)? {
            return Ok(None);
        }
    }
    for &mode in creation.iter().rev() {
        if !apply(mode, true)? {
            return Ok(None);
        }
    }
    if value.re == 0.0 && value.im == 0.0 {
        return Ok(None);
    }
    Ok(Some((destination, value)))
}

/// Compile a structured operator directly in a finite selected basis.
///
/// Terms that reach the same destination are aggregated before the
/// destination is checked against the selected basis. This is required for
/// exact cancellation such as XX + YY in a fixed-particle-number sector.
pub fn compile_charge_transitions(
    layout: ChargeTransitionLayout<'_>,
    terms: &[ChargeTransitionTerm],
) -> Result<ChargeTransitionResult, PauliError> {
    let ChargeTransitionLayout {
        dimension,
        basis,
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        qudit_positions,
        qudit_dimension,
        max_bytes,
    } = layout;
    let axis_count = local_dimensions.len();
    let expected_basis = dimension
        .checked_mul(axis_count)
        .ok_or(PauliError::Overflow {
            context: "sizing charge-sector basis",
        })?;
    if basis.len() != expected_basis {
        return Err(invalid_sector());
    }
    check_bytes(
        dimension,
        axis_count
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|value| value.checked_add(32))
            .ok_or(PauliError::Overflow {
                context: "estimating charge-sector basis workspace",
            })?,
        max_bytes,
    )?;
    for &local_dimension in local_dimensions {
        if local_dimension == 0 {
            return Err(invalid_sector());
        }
    }
    let fermion_positions = positions(fermion_positions, axis_count, max_bytes)?;
    let boson_positions = positions(boson_positions, axis_count, max_bytes)?;
    let qubit_positions = positions(qubit_positions, axis_count, max_bytes)?;
    let qudit_positions = positions(qudit_positions, axis_count, max_bytes)?;
    for row in basis.chunks_exact(axis_count.max(1)) {
        for (index, &value) in row.iter().take(axis_count).enumerate() {
            if value >= local_dimensions[index] {
                return Err(invalid_sector());
            }
        }
    }
    for (index, term) in terms.iter().enumerate() {
        if term.qubit_codes.len() != qubit_positions.len()
            || term.mapped_codes.len() != fermion_positions.len()
            || (term.mapped_present
                && (!term.fermion_creation.is_empty() || !term.fermion_annihilation.is_empty()))
        {
            return Err(invalid_sector());
        }
        validate_qudit_term(term, index, qudit_positions.len(), qudit_dimension)?;
        if !term.coefficient.re.is_finite() || !term.coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index });
        }
    }

    let mut basis_index: FxHashMap<Vec<u64>, u64> =
        FxHashMap::with_capacity_and_hasher(dimension, Default::default());
    for row_index in 0..dimension {
        let start = row_index
            .checked_mul(axis_count)
            .ok_or(PauliError::Overflow {
                context: "indexing charge-sector basis",
            })?;
        let key = basis[start..start + axis_count].to_vec();
        let index = u64::try_from(row_index).map_err(|_| PauliError::Overflow {
            context: "indexing charge-sector transitions",
        })?;
        if basis_index.insert(key, index).is_some() {
            return Err(invalid_sector());
        }
    }

    let mut transitions: BTreeMap<(u64, u64), Complex64> = BTreeMap::new();
    for column in 0..dimension {
        let start = column.checked_mul(axis_count).ok_or(PauliError::Overflow {
            context: "indexing charge-sector basis",
        })?;
        let source = &basis[start..start + axis_count];
        let mut destinations: BTreeMap<Vec<u64>, Complex64> = BTreeMap::new();
        for term in terms {
            let mut destination = source.to_vec();
            let mut value = term.coefficient;
            if !apply_fermions(
                &mut destination,
                &term.fermion_creation,
                &term.fermion_annihilation,
                &fermion_positions,
                &mut value,
            )? {
                continue;
            }
            if !apply_bosons(
                &mut destination,
                &term.boson_blocks,
                &boson_positions,
                local_dimensions,
                &mut value,
            )? {
                continue;
            }
            apply_pauli(
                &mut destination,
                &term.qubit_codes,
                &qubit_positions,
                &mut value,
            )?;
            if term.mapped_present {
                apply_pauli(
                    &mut destination,
                    &term.mapped_codes,
                    &fermion_positions,
                    &mut value,
                )?;
            }
            if term.qudit_present {
                apply_qudits(
                    &mut destination,
                    &term.qudit_triples,
                    &qudit_positions,
                    qudit_dimension,
                    &mut value,
                )?;
            }
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let entry = destinations
                .entry(destination)
                .or_insert(Complex64::new(0.0, 0.0));
            *entry += value;
            check_bytes(destinations.len(), 64, max_bytes)?;
        }
        for (destination, value) in destinations {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let row = basis_index
                .get(&destination)
                .ok_or(PauliError::InvalidSector {
                    context: "operator leaks outside the selected charge sector",
                })?;
            let column_index = u64::try_from(column).map_err(|_| PauliError::Overflow {
                context: "indexing charge-sector transitions",
            })?;
            let entry = transitions
                .entry((*row, column_index))
                .or_insert(Complex64::new(0.0, 0.0));
            *entry += value;
            check_bytes(transitions.len(), 40, max_bytes)?;
        }
    }

    let mut rows = Vec::with_capacity(transitions.len());
    let mut columns = Vec::with_capacity(transitions.len());
    let mut coefficients = Vec::with_capacity(transitions.len());
    for ((row, column), value) in transitions {
        if value.re != 0.0 || value.im != 0.0 {
            rows.push(row);
            columns.push(column);
            coefficients.push(value);
        }
    }
    Ok((rows, columns, coefficients))
}

/// Compile transitions directly against a reusable rank/unrank plan.
///
/// Source and destination occupation buffers are reused for every term. A
/// destination occupation vector is used as the aggregate key so the path
/// does not encode the full Cartesian Hilbert space into one integer; this is
/// essential for wide layouts whose selected sector still fits platform
/// indices. Destination aggregation still happens before sector membership is
/// checked.
pub fn compile_charge_transitions_from_prepared_plan(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    max_bytes: u128,
) -> Result<ChargeTransitionResult, PauliError> {
    let dimension = layout.dimension;
    let local_dimensions = &layout.local_dimensions;
    let fermion_positions = &layout.fermion_positions;
    let boson_positions = &layout.boson_positions;
    let qubit_positions = &layout.qubit_positions;
    let qudit_positions = &layout.qudit_positions;
    let qudit_dimension = layout.qudit_dimension;
    let axis_count = local_dimensions.len();
    if dimension != plan.dimension() {
        return Err(invalid_sector());
    }
    let scratch_bytes = axis_count
        .checked_mul(std::mem::size_of::<u64>())
        .and_then(|value| value.checked_add(terms.len().checked_mul(64)?))
        .and_then(|value| value.checked_add(32))
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector transition workspace",
        })?;
    check_bytes(1, scratch_bytes, max_bytes)?;

    let mut source = vec![0_u64; axis_count];
    let mut destination = vec![0_u64; axis_count];
    let mut remaining = vec![0_i128; plan.constraint_count()];
    let mut candidate_remaining = vec![0_i128; plan.constraint_count()];
    let mut transitions: FxHashMap<(u64, u64), Complex64> = FxHashMap::default();
    let mut destinations: FxHashMap<Vec<u64>, Complex64> =
        FxHashMap::with_capacity_and_hasher(terms.len(), Default::default());
    let destination_entry_bytes = 64usize
        .checked_add(axis_count.checked_mul(std::mem::size_of::<u64>()).ok_or(
            PauliError::Overflow {
                context: "estimating charge-sector destination storage",
            },
        )?)
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector destination storage",
        })?;
    for column in 0..dimension {
        plan.unrank_into_with_scratch(
            u64::try_from(column).map_err(|_| PauliError::Overflow {
                context: "indexing charge-sector transitions",
            })?,
            &mut source,
            &mut remaining,
            &mut candidate_remaining,
        )?;
        destinations.clear();
        for term in terms {
            destination.copy_from_slice(&source);
            let mut value = term.coefficient;
            if !apply_fermions(
                &mut destination,
                &term.fermion_creation,
                &term.fermion_annihilation,
                fermion_positions,
                &mut value,
            )? {
                continue;
            }
            if !apply_bosons(
                &mut destination,
                &term.boson_blocks,
                boson_positions,
                local_dimensions,
                &mut value,
            )? {
                continue;
            }
            apply_pauli(
                &mut destination,
                &term.qubit_codes,
                qubit_positions,
                &mut value,
            )?;
            if term.mapped_present {
                apply_pauli(
                    &mut destination,
                    &term.mapped_codes,
                    fermion_positions,
                    &mut value,
                )?;
            }
            if term.qudit_present {
                apply_qudits(
                    &mut destination,
                    &term.qudit_triples,
                    qudit_positions,
                    qudit_dimension,
                    &mut value,
                )?;
            }
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            *destinations
                .entry(destination.clone())
                .or_insert(Complex64::new(0.0, 0.0)) += value;
            check_bytes(destinations.len(), destination_entry_bytes, max_bytes)?;
        }
        for (destination, value) in destinations.drain() {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let row = match plan.rank_into(&destination, &mut remaining, &mut candidate_remaining) {
                Ok(row) => row,
                Err(PauliError::InvalidSector { .. }) => {
                    return Err(PauliError::InvalidSector {
                        context: "operator leaks outside the selected charge sector",
                    });
                }
                Err(error) => return Err(error),
            };
            let column_index = u64::try_from(column).map_err(|_| PauliError::Overflow {
                context: "indexing charge-sector transitions",
            })?;
            *transitions
                .entry((row, column_index))
                .or_insert(Complex64::new(0.0, 0.0)) += value;
            check_bytes(transitions.len(), 40, max_bytes)?;
        }
    }
    let mut entries: Vec<_> = transitions
        .into_iter()
        .filter(|(_, value)| value.re != 0.0 || value.im != 0.0)
        .collect();
    entries.sort_unstable_by_key(|(key, _)| *key);
    let mut rows = Vec::with_capacity(entries.len());
    let mut columns = Vec::with_capacity(entries.len());
    let mut coefficients = Vec::with_capacity(entries.len());
    for ((row, column), value) in entries {
        rows.push(row);
        columns.push(column);
        coefficients.push(value);
    }
    Ok((rows, columns, coefficients))
}

/// Compile transitions from a borrowed layout, preserving the original
/// one-shot API while routing reusable native handles through the prepared
/// counterpart.
pub fn compile_charge_transitions_from_plan(
    plan: &ChargeSectorPlan,
    layout: ChargeTransitionPlanLayout<'_>,
    terms: &[ChargeTransitionTerm],
) -> Result<ChargeTransitionResult, PauliError> {
    let prepared = prepare_charge_transition_plan_layout(
        plan,
        layout.dimension,
        layout.local_dimensions.to_vec(),
        layout.fermion_positions,
        layout.boson_positions,
        layout.qubit_positions,
        layout.qudit_positions,
        layout.qudit_dimension,
        terms,
        layout.max_bytes,
    )?;
    let remaining = layout
        .max_bytes
        .checked_sub(prepared.estimated_bytes())
        .ok_or(PauliError::MemoryLimit {
            requested: prepared.estimated_bytes(),
            limit: layout.max_bytes,
        })?;
    compile_charge_transitions_from_prepared_plan(plan, &prepared, terms, remaining)
}

/// Apply a structured operator directly against a reusable charge-sector plan.
///
/// This is the explicitly lazy counterpart to
/// [`compile_charge_transitions_from_plan`]. It keeps only the input/output
/// vectors and one source-column destination aggregate in memory; it does not
/// retain the full restricted transition graph. Destination terms are
/// aggregated before sector membership is checked, preserving the same exact
/// cancellation semantics as eager compilation.
pub fn apply_charge_mvp_from_prepared_plan(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    state: &[Complex64],
    termwise_conserved: bool,
    fast_fermion_plan: Option<&FastFermionMvpPlan>,
    max_bytes: u128,
) -> Result<Vec<Complex64>, PauliError> {
    let dimension = layout.dimension;
    let output_bytes = dimension
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or(PauliError::Overflow {
            context: "estimating charge MVP output",
        })?;
    if output_bytes as u128 > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: output_bytes as u128,
            limit: max_bytes,
        });
    }
    let mut output = vec![Complex64::new(0.0, 0.0); dimension];
    apply_charge_mvp_from_prepared_plan_into(
        plan,
        layout,
        terms,
        state,
        &mut output,
        termwise_conserved,
        fast_fermion_plan,
        max_bytes,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn apply_charge_mvp_from_prepared_plan_into(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    state: &[Complex64],
    output: &mut [Complex64],
    termwise_conserved: bool,
    fast_fermion_plan: Option<&FastFermionMvpPlan>,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let dimension = layout.dimension;
    let local_dimensions = &layout.local_dimensions;
    let fermion_positions = &layout.fermion_positions;
    let boson_positions = &layout.boson_positions;
    let qubit_positions = &layout.qubit_positions;
    let qudit_positions = &layout.qudit_positions;
    let qudit_dimension = layout.qudit_dimension;
    if dimension != plan.dimension() || state.len() != dimension || output.len() != dimension {
        return Err(invalid_sector());
    }

    // The cached spinful kernel has no execution-time major allocation. Run it
    // before the generic workspace preflight so apply_into(max_bytes=0) remains
    // valid for a caller-owned output buffer.
    if termwise_conserved {
        if let Some(fast_plan) = fast_fermion_plan {
            fast_plan.apply_into(state, output)?;
            return Ok(());
        }
    }

    let output_bytes = dimension
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or(PauliError::Overflow {
            context: "estimating charge MVP output",
        })?;
    if output_bytes as u128 > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: output_bytes as u128,
            limit: max_bytes,
        });
    }
    let scratch_limit = max_bytes - output_bytes as u128;
    let axis_count = local_dimensions.len();
    let scratch_bytes = axis_count
        .checked_mul(std::mem::size_of::<u64>())
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(terms.len().checked_mul(64)?))
        .and_then(|value| value.checked_add(32))
        .ok_or(PauliError::Overflow {
            context: "estimating charge MVP workspace",
        })?;
    check_bytes(1, scratch_bytes, scratch_limit)?;

    let destination_entry_bytes = 64usize
        .checked_add(axis_count.checked_mul(std::mem::size_of::<u64>()).ok_or(
            PauliError::Overflow {
                context: "estimating charge MVP destination storage",
            },
        )?)
        .ok_or(PauliError::Overflow {
            context: "estimating charge MVP destination storage",
        })?;
    output.fill(Complex64::new(0.0, 0.0));
    let mut source = vec![0_u64; axis_count];
    let mut destination = vec![0_u64; axis_count];
    let mut remaining = vec![0_i128; plan.constraint_count()];
    let mut candidate_remaining = vec![0_i128; plan.constraint_count()];
    let mut destinations: FxHashMap<Vec<u64>, Complex64> =
        FxHashMap::with_capacity_and_hasher(terms.len(), Default::default());

    for (column, state_value) in state.iter().enumerate() {
        plan.unrank_into_with_scratch(
            u64::try_from(column).map_err(|_| PauliError::Overflow {
                context: "indexing charge-sector MVP source",
            })?,
            &mut source,
            &mut remaining,
            &mut candidate_remaining,
        )?;
        destinations.clear();
        if termwise_conserved {
            for term in terms {
                let Some(value) = apply_charge_term(
                    &source,
                    &mut destination,
                    term,
                    local_dimensions,
                    fermion_positions,
                    boson_positions,
                    qubit_positions,
                    qudit_positions,
                    qudit_dimension,
                )?
                else {
                    continue;
                };
                let row = if destination == source {
                    u64::try_from(column).map_err(|_| PauliError::Overflow {
                        context: "indexing charge-sector MVP destination",
                    })?
                } else {
                    plan.rank_into(&destination, &mut remaining, &mut candidate_remaining)?
                };
                let row = usize::try_from(row).map_err(|_| PauliError::Overflow {
                    context: "converting charge-sector MVP destination",
                })?;
                output[row] += value * *state_value;
            }
        } else {
            for term in terms {
                let Some(value) = apply_charge_term(
                    &source,
                    &mut destination,
                    term,
                    local_dimensions,
                    fermion_positions,
                    boson_positions,
                    qubit_positions,
                    qudit_positions,
                    qudit_dimension,
                )?
                else {
                    continue;
                };
                *destinations
                    .entry(destination.clone())
                    .or_insert(Complex64::new(0.0, 0.0)) += value;
                check_bytes(destinations.len(), destination_entry_bytes, scratch_limit)?;
            }
        }
        for (destination, value) in destinations.drain() {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let row = match plan.rank_into(&destination, &mut remaining, &mut candidate_remaining) {
                Ok(row) => usize::try_from(row).map_err(|_| PauliError::Overflow {
                    context: "converting charge-sector MVP destination",
                })?,
                Err(PauliError::InvalidSector { .. }) => {
                    return Err(PauliError::InvalidSector {
                        context: "operator leaks outside the selected charge sector",
                    });
                }
                Err(error) => return Err(error),
            };
            output[row] += value * *state_value;
        }
    }
    Ok(())
}

/// Apply through the original borrowed-layout API.
pub fn apply_charge_mvp_from_plan(
    plan: &ChargeSectorPlan,
    layout: ChargeTransitionPlanLayout<'_>,
    terms: &[ChargeTransitionTerm],
    state: &[Complex64],
    termwise_conserved: bool,
    fast_fermion_plan: Option<&FastFermionMvpPlan>,
) -> Result<Vec<Complex64>, PauliError> {
    let prepared = prepare_charge_transition_plan_layout(
        plan,
        layout.dimension,
        layout.local_dimensions.to_vec(),
        layout.fermion_positions,
        layout.boson_positions,
        layout.qubit_positions,
        layout.qudit_positions,
        layout.qudit_dimension,
        terms,
        layout.max_bytes,
    )?;
    let remaining = layout
        .max_bytes
        .checked_sub(prepared.estimated_bytes())
        .ok_or(PauliError::MemoryLimit {
            requested: prepared.estimated_bytes(),
            limit: layout.max_bytes,
        })?;
    apply_charge_mvp_from_prepared_plan(
        plan,
        &prepared,
        terms,
        state,
        termwise_conserved,
        fast_fermion_plan,
        remaining,
    )
}

/// Apply into caller-owned output through the original borrowed-layout API.
pub fn apply_charge_mvp_from_plan_into(
    plan: &ChargeSectorPlan,
    layout: ChargeTransitionPlanLayout<'_>,
    terms: &[ChargeTransitionTerm],
    state: &[Complex64],
    output: &mut [Complex64],
    termwise_conserved: bool,
    fast_fermion_plan: Option<&FastFermionMvpPlan>,
) -> Result<(), PauliError> {
    let prepared = prepare_charge_transition_plan_layout(
        plan,
        layout.dimension,
        layout.local_dimensions.to_vec(),
        layout.fermion_positions,
        layout.boson_positions,
        layout.qubit_positions,
        layout.qudit_positions,
        layout.qudit_dimension,
        terms,
        layout.max_bytes,
    )?;
    let remaining = layout
        .max_bytes
        .checked_sub(prepared.estimated_bytes())
        .ok_or(PauliError::MemoryLimit {
            requested: prepared.estimated_bytes(),
            limit: layout.max_bytes,
        })?;
    apply_charge_mvp_from_prepared_plan_into(
        plan,
        &prepared,
        terms,
        state,
        output,
        termwise_conserved,
        fast_fermion_plan,
        remaining,
    )
}
