//! Anytime search for concrete non-trivial logical Pauli witnesses.

use std::cmp::Ordering;
use std::mem::size_of;
use std::sync::OnceLock;
use std::time::Instant;

use rayon::prelude::*;

use crate::error::PauliError;
use crate::gf2::{get_bit, null_space, packed_count, rank, set_bit, xor_assign, RrefBasis};
use crate::random::RandomStream;
use crate::stabilizer_code::{words_from_bits, StabilizerCode};

/// The seeded algorithm identifier for the first logical-witness search path.
pub const LOGICAL_SEARCH_ALGORITHM: &str = "randomized_information_set_v2";

const LOGICAL_SEARCH_DOMAIN: u64 = 0x6c6f_6769_6361_6c15;
const GENERAL_DIRECTION: u64 = 0;
const X_DIRECTION: u64 = 1;
const Z_DIRECTION: u64 = 2;
const TRIAL_CHUNK_SIZE: usize = 256;

const PAULI_CODE_FROM_PAIR: [u8; 4] = [0, 1, 3, 2];

// These are the six invertible binary 2-by-2 maps, represented by their
// action on the public I/X/Y/Z code. Every map preserves the non-identity
// set and therefore preserves physical Pauli weight.
const LOCAL_MAP_CODES: [[u8; 4]; 6] = [
    [0, 1, 2, 3],
    [0, 3, 2, 1],
    [0, 2, 1, 3],
    [0, 1, 3, 2],
    [0, 3, 1, 2],
    [0, 2, 3, 1],
];
const LOCAL_MAP_INVERSE: [usize; 6] = [0, 1, 2, 3, 5, 4];

/// The requested search sector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LogicalSearchSector {
    Auto = 0,
    X = 1,
    Z = 2,
    General = 3,
}

/// The physical support type of a returned logical witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WitnessSector {
    X = 0,
    Z = 1,
    Mixed = 2,
}

/// A validated phase-free physical Pauli in the non-trivial logical quotient.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalWitness {
    pub x_words: Vec<u64>,
    pub z_words: Vec<u64>,
    pub logical_x_words: Vec<u64>,
    pub logical_z_words: Vec<u64>,
    pub weight: usize,
    pub sector: WitnessSector,
}

/// Result metadata for an anytime logical-witness search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalWitnessSearchResult {
    pub best: LogicalWitness,
    pub best_weight: usize,
    pub sector: LogicalSearchSector,
    pub algorithm: &'static str,
    pub sketch_rows: usize,
    pub max_trials: usize,
    pub trials_run: usize,
    pub stop_below_weight: Option<usize>,
    pub seed: u64,
    pub early_stopped: bool,
    pub initial_incumbent_weight: usize,
    pub initialization_candidates: usize,
    pub resolved_sketch_rows: usize,
    pub setup_nanos: u64,
    pub initialization_nanos: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct PhysicalRow {
    x_words: Vec<u64>,
    z_words: Vec<u64>,
}

#[derive(Clone, Debug)]
struct TaggedBasis {
    nqubits: usize,
    word_count: usize,
    tag_words: usize,
    dimension: usize,
    x_words: Vec<u64>,
    z_words: Vec<u64>,
    tags: Vec<u64>,
}

impl TaggedBasis {
    fn from_rows(
        nqubits: usize,
        stabilizers: &[PhysicalRow],
        logical_rows: &[PhysicalRow],
    ) -> Result<Self, PauliError> {
        let word_count = (nqubits / 64) + usize::from(nqubits % 64 != 0);
        let logical_dimension = logical_rows.len();
        let tag_words = packed_count(logical_dimension);
        let dimension =
            stabilizers
                .len()
                .checked_add(logical_dimension)
                .ok_or(PauliError::Overflow {
                    context: "building logical-search basis dimension",
                })?;
        let physical_capacity = checked_product(
            dimension,
            word_count,
            "allocating logical-search basis physical words",
        )?;
        let tag_capacity = checked_product(
            dimension,
            tag_words,
            "allocating logical-search basis quotient tags",
        )?;
        let mut x_words = Vec::with_capacity(physical_capacity);
        let mut z_words = Vec::with_capacity(physical_capacity);
        let mut tags = vec![0_u64; tag_capacity];
        for row in stabilizers.iter().chain(logical_rows) {
            if row.x_words.len() != word_count || row.z_words.len() != word_count {
                return Err(PauliError::InvalidStabilizer {
                    context: "logical-search basis row width is inconsistent",
                });
            }
            x_words.extend_from_slice(&row.x_words);
            z_words.extend_from_slice(&row.z_words);
        }
        for logical in 0..logical_dimension {
            set_bit(
                &mut tags[(stabilizers.len() + logical) * tag_words
                    ..(stabilizers.len() + logical + 1) * tag_words],
                logical,
                true,
            );
        }
        Ok(Self {
            nqubits,
            word_count,
            tag_words,
            dimension,
            x_words,
            z_words,
            tags,
        })
    }

    fn row_x(&self, row: usize) -> &[u64] {
        &self.x_words[row * self.word_count..(row + 1) * self.word_count]
    }

    fn row_z(&self, row: usize) -> &[u64] {
        &self.z_words[row * self.word_count..(row + 1) * self.word_count]
    }

    fn row_tag(&self, row: usize) -> &[u64] {
        &self.tags[row * self.tag_words..(row + 1) * self.tag_words]
    }

    fn estimated_bytes(&self) -> Result<u128, PauliError> {
        checked_sum(&[
            checked_bytes(
                checked_product(
                    self.dimension,
                    self.word_count,
                    "estimating logical-search basis X words",
                )?,
                size_of::<u64>(),
                "estimating logical-search basis X words",
            )?,
            checked_bytes(
                checked_product(
                    self.dimension,
                    self.word_count,
                    "estimating logical-search basis Z words",
                )?,
                size_of::<u64>(),
                "estimating logical-search basis Z words",
            )?,
            checked_bytes(
                checked_product(
                    self.dimension,
                    self.tag_words,
                    "estimating logical-search basis tags",
                )?,
                size_of::<u64>(),
                "estimating logical-search basis tags",
            )?,
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchRepresentation {
    General,
    CssX,
    CssZ,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedSearch {
    sector: WitnessSector,
    direction: u64,
    basis: TaggedBasis,
    logical_rows: Vec<PhysicalRow>,
    representation: SearchRepresentation,
}

#[derive(Clone, Debug)]
pub(crate) struct CssAnalysis {
    canonical: Vec<PhysicalRow>,
    pure_x_rank: usize,
    pure_z_rank: usize,
}

#[derive(Debug)]
pub(crate) struct LogicalSearchCaches {
    general: OnceLock<Result<PreparedSearch, PauliError>>,
    css_analysis: OnceLock<Result<CssAnalysis, PauliError>>,
    css_x: OnceLock<Result<PreparedSearch, PauliError>>,
    css_z: OnceLock<Result<PreparedSearch, PauliError>>,
}

impl LogicalSearchCaches {
    pub(crate) fn new() -> Self {
        Self {
            general: OnceLock::new(),
            css_analysis: OnceLock::new(),
            css_x: OnceLock::new(),
            css_z: OnceLock::new(),
        }
    }

    pub(crate) fn general(&self) -> &OnceLock<Result<PreparedSearch, PauliError>> {
        &self.general
    }

    pub(crate) fn css_analysis(&self) -> &OnceLock<Result<CssAnalysis, PauliError>> {
        &self.css_analysis
    }

    pub(crate) fn css_x(&self) -> &OnceLock<Result<PreparedSearch, PauliError>> {
        &self.css_x
    }

    pub(crate) fn css_z(&self) -> &OnceLock<Result<PreparedSearch, PauliError>> {
        &self.css_z
    }
}

impl CssAnalysis {
    fn estimated_bytes(&self, word_count: usize) -> Result<u128, PauliError> {
        checked_bytes(
            checked_product(
                checked_product(
                    self.canonical.len(),
                    word_count,
                    "estimating cached CSS canonical rows",
                )?,
                2,
                "estimating cached CSS canonical rows",
            )?,
            size_of::<u64>(),
            "estimating cached CSS canonical rows",
        )
    }
}

impl PreparedSearch {
    fn resolved_rows(&self, requested: usize) -> usize {
        requested.min(self.basis.dimension)
    }

    fn estimated_bytes(&self) -> Result<u128, PauliError> {
        let logical_bytes = checked_bytes(
            checked_product(
                self.logical_rows.len(),
                self.basis.word_count,
                "estimating logical-search initialization rows",
            )?
            .checked_mul(2)
            .ok_or(PauliError::Overflow {
                context: "estimating logical-search initialization rows",
            })?,
            size_of::<u64>(),
            "estimating logical-search initialization rows",
        )?;
        self.basis
            .estimated_bytes()?
            .checked_add(logical_bytes)
            .ok_or(PauliError::Overflow {
                context: "estimating logical-search prepared basis",
            })
    }
}

#[derive(Clone, Debug)]
struct BestBuffer {
    valid: bool,
    x_words: Vec<u64>,
    z_words: Vec<u64>,
    logical_x_words: Vec<u64>,
    logical_z_words: Vec<u64>,
    weight: usize,
    sector: WitnessSector,
}

impl BestBuffer {
    fn new(word_count: usize, logical_word_count: usize) -> Self {
        Self {
            valid: false,
            x_words: vec![0; word_count],
            z_words: vec![0; word_count],
            logical_x_words: vec![0; logical_word_count],
            logical_z_words: vec![0; logical_word_count],
            weight: 0,
            sector: WitnessSector::Mixed,
        }
    }

    fn clear(&mut self) {
        self.valid = false;
    }

    #[allow(clippy::too_many_arguments)]
    fn consider_tagged(
        &mut self,
        code: &StabilizerCode,
        sector: WitnessSector,
        x_words: &[u64],
        z_words: &[u64],
        tag: &[u64],
        coordinates_x: &mut [u64],
        coordinates_z: &mut [u64],
    ) {
        // The tagged basis is a basis of the centralizer, and its logical
        // rows have independent quotient tags.  A nonzero tag therefore
        // proves that this row is a non-trivial logical before the final
        // result validation; avoid rescanning every original check for every
        // candidate in the trial hot path.
        if tag.iter().all(|&word| word == 0) {
            return;
        }
        let weight = physical_weight(x_words, z_words);
        if weight == 0
            || (self.valid
                && !candidate_is_better(
                    weight,
                    x_words,
                    z_words,
                    self.weight,
                    &self.x_words,
                    &self.z_words,
                    code.nqubits(),
                ))
        {
            return;
        }
        code.logical_coordinates_packed_words(x_words, z_words, coordinates_x, coordinates_z);
        if coordinates_x.iter().all(|&word| word == 0)
            && coordinates_z.iter().all(|&word| word == 0)
        {
            return;
        }
        self.valid = true;
        self.x_words.copy_from_slice(x_words);
        self.z_words.copy_from_slice(z_words);
        self.logical_x_words.copy_from_slice(coordinates_x);
        self.logical_z_words.copy_from_slice(coordinates_z);
        self.weight = weight;
        self.sector = resolved_witness_sector(sector, x_words, z_words);
    }

    fn consider_known_logical(
        &mut self,
        code: &StabilizerCode,
        sector: WitnessSector,
        x_words: &[u64],
        z_words: &[u64],
        coordinates_x: &mut [u64],
        coordinates_z: &mut [u64],
    ) -> bool {
        let weight = physical_weight(x_words, z_words);
        if weight == 0
            || (self.valid
                && !candidate_is_better(
                    weight,
                    x_words,
                    z_words,
                    self.weight,
                    &self.x_words,
                    &self.z_words,
                    code.nqubits(),
                ))
        {
            return false;
        }
        code.logical_coordinates_packed_words(x_words, z_words, coordinates_x, coordinates_z);
        if coordinates_x.iter().all(|&word| word == 0)
            && coordinates_z.iter().all(|&word| word == 0)
        {
            return false;
        }
        self.valid = true;
        self.x_words.copy_from_slice(x_words);
        self.z_words.copy_from_slice(z_words);
        self.logical_x_words.copy_from_slice(coordinates_x);
        self.logical_z_words.copy_from_slice(coordinates_z);
        self.weight = weight;
        self.sector = resolved_witness_sector(sector, x_words, z_words);
        true
    }

    fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.x_words.len(), other.x_words.len());
        debug_assert_eq!(self.z_words.len(), other.z_words.len());
        debug_assert_eq!(self.logical_x_words.len(), other.logical_x_words.len());
        debug_assert_eq!(self.logical_z_words.len(), other.logical_z_words.len());
        self.valid = other.valid;
        if other.valid {
            self.x_words.copy_from_slice(&other.x_words);
            self.z_words.copy_from_slice(&other.z_words);
            self.logical_x_words.copy_from_slice(&other.logical_x_words);
            self.logical_z_words.copy_from_slice(&other.logical_z_words);
            self.weight = other.weight;
            self.sector = other.sector;
        }
    }

    fn copy_from_witness(&mut self, other: &LogicalWitness) {
        self.valid = true;
        self.x_words.copy_from_slice(&other.x_words);
        self.z_words.copy_from_slice(&other.z_words);
        self.logical_x_words.copy_from_slice(&other.logical_x_words);
        self.logical_z_words.copy_from_slice(&other.logical_z_words);
        self.weight = other.weight;
        self.sector = other.sector;
    }

    fn to_owned(&self) -> Option<LogicalWitness> {
        self.valid.then(|| LogicalWitness {
            x_words: self.x_words.clone(),
            z_words: self.z_words.clone(),
            logical_x_words: self.logical_x_words.clone(),
            logical_z_words: self.logical_z_words.clone(),
            weight: self.weight,
            sector: self.sector,
        })
    }
}

#[derive(Clone, Debug)]
struct TrialScratch {
    representation: SearchRepresentation,
    nqubits: usize,
    word_count: usize,
    physical_width: usize,
    tag_words: usize,
    rows: usize,
    physical_words: Vec<u64>,
    tags: Vec<u64>,
    current_words: Vec<u64>,
    current_tag: Vec<u64>,
    evaluation_words: Vec<u64>,
    evaluation_tag: Vec<u64>,
    refinement_words: Vec<u64>,
    refinement_tag: Vec<u64>,
    coordinates_x: Vec<u64>,
    coordinates_z: Vec<u64>,
    zero_words: Vec<u64>,
    maps: Vec<u8>,
    permutation: Vec<usize>,
    candidate_indices: [usize; 4],
    candidate_count: usize,
    trial_best: BestBuffer,
}

impl TrialScratch {
    fn new(
        problem: &PreparedSearch,
        requested_sketch_rows: usize,
        nlogical: usize,
    ) -> Result<Self, PauliError> {
        let rows = problem.resolved_rows(requested_sketch_rows);
        let word_count = problem.basis.word_count;
        let tag_words = problem.basis.tag_words;
        let physical_width = match problem.representation {
            SearchRepresentation::General => checked_product(
                word_count,
                2,
                "allocating general logical-search physical row width",
            )?,
            SearchRepresentation::CssX | SearchRepresentation::CssZ => word_count,
        };
        let physical_word_count = checked_product(
            rows,
            physical_width,
            "allocating logical-search trial physical words",
        )?;
        let tag_capacity =
            checked_product(rows, tag_words, "allocating logical-search trial tags")?;
        let physical_columns = match problem.representation {
            SearchRepresentation::General => {
                problem
                    .basis
                    .nqubits
                    .checked_mul(2)
                    .ok_or(PauliError::Overflow {
                        context: "allocating general logical-search column permutation",
                    })?
            }
            SearchRepresentation::CssX | SearchRepresentation::CssZ => problem.basis.nqubits,
        };
        let logical_word_count = packed_count(nlogical);
        let scratch_words = vec![0; physical_width];
        Ok(Self {
            representation: problem.representation,
            nqubits: problem.basis.nqubits,
            word_count,
            physical_width,
            tag_words,
            rows,
            physical_words: vec![0; physical_word_count],
            tags: vec![0; tag_capacity],
            current_words: scratch_words.clone(),
            current_tag: vec![0; tag_words],
            evaluation_words: scratch_words.clone(),
            evaluation_tag: vec![0; tag_words],
            refinement_words: scratch_words,
            refinement_tag: vec![0; tag_words],
            coordinates_x: vec![0; logical_word_count],
            coordinates_z: vec![0; logical_word_count],
            zero_words: match problem.representation {
                SearchRepresentation::General => Vec::new(),
                SearchRepresentation::CssX | SearchRepresentation::CssZ => vec![0; word_count],
            },
            maps: match problem.representation {
                SearchRepresentation::General => vec![0; problem.basis.nqubits],
                SearchRepresentation::CssX | SearchRepresentation::CssZ => Vec::new(),
            },
            permutation: vec![0; physical_columns],
            candidate_indices: [0; 4],
            candidate_count: 0,
            trial_best: BestBuffer::new(word_count, logical_word_count),
        })
    }

    fn reset_trial(&mut self) {
        self.trial_best.clear();
        self.candidate_count = 0;
    }

    fn run_trial(
        &mut self,
        problem: &PreparedSearch,
        code: &StabilizerCode,
        seed: u64,
        trial_index: usize,
        requested_sketch_rows: usize,
    ) -> Result<(), PauliError> {
        self.reset_trial();
        let resolved_rows = problem.resolved_rows(requested_sketch_rows);
        debug_assert_eq!(resolved_rows, self.rows);
        let mut random = RandomStream::new(
            seed,
            LOGICAL_SEARCH_DOMAIN.wrapping_add(problem.direction),
            trial_index,
            0,
        );
        if resolved_rows == problem.basis.dimension {
            for row in 0..resolved_rows {
                self.copy_basis_row(problem, row);
            }
        } else {
            self.fill_random_sketch(problem, &mut random)?;
        }
        self.sample_scramble_and_permutation(&mut random)?;
        self.reduce_tagged_rows();
        self.select_candidate_rows();
        for candidate in 0..self.candidate_count {
            self.refine_candidate(self.candidate_indices[candidate]);
            match self.representation {
                SearchRepresentation::General => {
                    let (x_words, z_words) = self.current_words.split_at_mut(self.word_count);
                    apply_maps_to_row(x_words, z_words, &self.maps, &LOCAL_MAP_INVERSE);
                    self.trial_best.consider_tagged(
                        code,
                        problem.sector,
                        x_words,
                        z_words,
                        &self.current_tag,
                        &mut self.coordinates_x,
                        &mut self.coordinates_z,
                    );
                }
                SearchRepresentation::CssX => self.trial_best.consider_tagged(
                    code,
                    problem.sector,
                    &self.current_words,
                    &self.zero_words,
                    &self.current_tag,
                    &mut self.coordinates_x,
                    &mut self.coordinates_z,
                ),
                SearchRepresentation::CssZ => self.trial_best.consider_tagged(
                    code,
                    problem.sector,
                    &self.zero_words,
                    &self.current_words,
                    &self.current_tag,
                    &mut self.coordinates_x,
                    &mut self.coordinates_z,
                ),
            }
        }
        Ok(())
    }

    fn copy_basis_row(&mut self, problem: &PreparedSearch, row: usize) {
        let row_words =
            &mut self.physical_words[row * self.physical_width..(row + 1) * self.physical_width];
        match self.representation {
            SearchRepresentation::General => {
                row_words[..self.word_count].copy_from_slice(problem.basis.row_x(row));
                row_words[self.word_count..].copy_from_slice(problem.basis.row_z(row));
            }
            SearchRepresentation::CssX => row_words.copy_from_slice(problem.basis.row_x(row)),
            SearchRepresentation::CssZ => row_words.copy_from_slice(problem.basis.row_z(row)),
        }
        self.tags[row * self.tag_words..(row + 1) * self.tag_words]
            .copy_from_slice(problem.basis.row_tag(row));
    }

    fn fill_random_sketch(
        &mut self,
        problem: &PreparedSearch,
        random: &mut RandomStream,
    ) -> Result<(), PauliError> {
        let stabilizer_dimension = problem
            .basis
            .dimension
            .checked_sub(problem.logical_rows.len())
            .ok_or(PauliError::InvalidStabilizer {
                context: "logical-search basis dimension is inconsistent",
            })?;
        for row in 0..self.rows {
            let row_words = &mut self.physical_words
                [row * self.physical_width..(row + 1) * self.physical_width];
            let row_tag = &mut self.tags[row * self.tag_words..(row + 1) * self.tag_words];
            row_words.fill(0);
            row_tag.fill(0);
            if row == 0 {
                for basis_row in 0..stabilizer_dimension {
                    if random.next_u64() & 1 != 0 {
                        xor_basis_physical(row_words, problem, basis_row, self.representation);
                        xor_assign(row_tag, problem.basis.row_tag(basis_row));
                    }
                }
                // Rejection sampling makes the logical coefficient vector
                // uniform over all nonzero vectors, including dimensions
                // wider than one machine word.
                loop {
                    let mut nonzero = false;
                    for logical in 0..problem.logical_rows.len() {
                        if random.next_u64() & 1 != 0 {
                            let basis_row = stabilizer_dimension + logical;
                            xor_basis_physical(row_words, problem, basis_row, self.representation);
                            xor_assign(row_tag, problem.basis.row_tag(basis_row));
                            nonzero = true;
                        }
                    }
                    if nonzero {
                        break;
                    }
                }
            } else {
                for basis_row in 0..problem.basis.dimension {
                    if random.next_u64() & 1 != 0 {
                        xor_basis_physical(row_words, problem, basis_row, self.representation);
                        xor_assign(row_tag, problem.basis.row_tag(basis_row));
                    }
                }
            }
        }
        Ok(())
    }

    fn sample_scramble_and_permutation(
        &mut self,
        random: &mut RandomStream,
    ) -> Result<(), PauliError> {
        if self.representation == SearchRepresentation::General {
            for map in &mut self.maps {
                *map = u8::try_from(random.bounded(LOCAL_MAP_CODES.len())?).map_err(|_| {
                    PauliError::Overflow {
                        context: "converting logical-search local map",
                    }
                })?;
            }
        }
        for (column, value) in self.permutation.iter_mut().enumerate() {
            *value = column;
        }
        for index in (1..self.permutation.len()).rev() {
            let swap = random.bounded(index + 1)?;
            self.permutation.swap(index, swap);
        }
        if self.representation == SearchRepresentation::General {
            for row in 0..self.rows {
                let words = &mut self.physical_words
                    [row * self.physical_width..(row + 1) * self.physical_width];
                let (x_words, z_words) = words.split_at_mut(self.word_count);
                apply_maps_to_row(x_words, z_words, &self.maps, &IDENTITY_MAPS);
            }
        }
        Ok(())
    }

    fn reduce_tagged_rows(&mut self) {
        let mut rank = 0;
        for permutation_index in 0..self.permutation.len() {
            let column = self.permutation[permutation_index];
            let Some(pivot) = (rank..self.rows).find(|&row| self.matrix_has_column(row, column))
            else {
                continue;
            };
            self.swap_rows(rank, pivot);
            for row in 0..self.rows {
                if row != rank && self.matrix_has_column(row, column) {
                    self.xor_rows(row, rank);
                }
            }
            rank += 1;
            if rank == self.rows {
                break;
            }
        }
    }

    fn matrix_has_column(&self, row: usize, column: usize) -> bool {
        match self.representation {
            SearchRepresentation::General => {
                let (segment, local_column) = if column < self.nqubits {
                    (0, column)
                } else {
                    (self.word_count, column - self.nqubits)
                };
                self.physical_words[row * self.physical_width + segment + local_column / 64]
                    & (1_u64 << (local_column % 64))
                    != 0
            }
            SearchRepresentation::CssX | SearchRepresentation::CssZ => {
                matrix_bit(&self.physical_words, row, self.physical_width, column)
            }
        }
    }

    fn swap_rows(&mut self, left: usize, right: usize) {
        if left == right {
            return;
        }
        for word in 0..self.physical_width {
            self.physical_words.swap(
                left * self.physical_width + word,
                right * self.physical_width + word,
            );
        }
        for word in 0..self.tag_words {
            self.tags
                .swap(left * self.tag_words + word, right * self.tag_words + word);
        }
    }

    fn xor_rows(&mut self, target: usize, pivot: usize) {
        xor_disjoint_rows(&mut self.physical_words, target, pivot, self.physical_width);
        xor_disjoint_rows(&mut self.tags, target, pivot, self.tag_words);
    }

    fn select_candidate_rows(&mut self) {
        self.candidate_count = 0;
        for row in 0..self.rows {
            let tag = &self.tags[row * self.tag_words..(row + 1) * self.tag_words];
            if tag.iter().all(|&word| word == 0) {
                continue;
            }
            let words =
                &self.physical_words[row * self.physical_width..(row + 1) * self.physical_width];
            if represented_weight(words, self.representation, self.word_count) == 0 {
                continue;
            }
            let insertion = (0..self.candidate_count)
                .find(|&index| self.row_precedes(row, self.candidate_indices[index]))
                .unwrap_or(self.candidate_count);
            if self.candidate_count < self.candidate_indices.len() {
                for index in (insertion..self.candidate_count).rev() {
                    self.candidate_indices[index + 1] = self.candidate_indices[index];
                }
                self.candidate_indices[insertion] = row;
                self.candidate_count += 1;
            } else if insertion < self.candidate_indices.len() {
                for index in (insertion..self.candidate_indices.len() - 1).rev() {
                    self.candidate_indices[index + 1] = self.candidate_indices[index];
                }
                self.candidate_indices[insertion] = row;
            }
        }
    }

    fn row_precedes(&self, left: usize, right: usize) -> bool {
        let left_words =
            &self.physical_words[left * self.physical_width..(left + 1) * self.physical_width];
        let right_words =
            &self.physical_words[right * self.physical_width..(right + 1) * self.physical_width];
        candidate_is_better_representation(
            represented_weight(left_words, self.representation, self.word_count),
            left_words,
            self.representation,
            represented_weight(right_words, self.representation, self.word_count),
            right_words,
            self.representation,
            self.nqubits,
        )
    }

    fn refine_candidate(&mut self, row: usize) {
        self.current_words.copy_from_slice(
            &self.physical_words[row * self.physical_width..(row + 1) * self.physical_width],
        );
        self.current_tag
            .copy_from_slice(&self.tags[row * self.tag_words..(row + 1) * self.tag_words]);
        for _ in 0..2 {
            let current_weight =
                represented_weight(&self.current_words, self.representation, self.word_count);
            self.refinement_tag.fill(0);
            let mut found = false;
            let mut best_weight = current_weight;
            for other in 0..self.rows {
                xor_into(
                    &mut self.evaluation_words,
                    &self.current_words,
                    &self.physical_words
                        [other * self.physical_width..(other + 1) * self.physical_width],
                );
                xor_into(
                    &mut self.evaluation_tag,
                    &self.current_tag,
                    &self.tags[other * self.tag_words..(other + 1) * self.tag_words],
                );
                if self.evaluation_tag.iter().all(|&word| word == 0) {
                    continue;
                }
                let weight = represented_weight(
                    &self.evaluation_words,
                    self.representation,
                    self.word_count,
                );
                if weight == 0 || weight >= best_weight {
                    continue;
                }
                found = true;
                best_weight = weight;
                self.refinement_words
                    .copy_from_slice(&self.evaluation_words);
                self.refinement_tag.copy_from_slice(&self.evaluation_tag);
            }
            if !found {
                break;
            }
            self.current_words.copy_from_slice(&self.refinement_words);
            self.current_tag.copy_from_slice(&self.refinement_tag);
        }
    }
}

#[derive(Clone, Debug)]
struct WorkerState {
    scratch: TrialScratch,
    local_best: BestBuffer,
}

impl WorkerState {
    fn new(
        problem: &PreparedSearch,
        requested_sketch_rows: usize,
        nlogical: usize,
    ) -> Result<Self, PauliError> {
        let scratch = TrialScratch::new(problem, requested_sketch_rows, nlogical)?;
        let local_best = BestBuffer::new(problem.basis.word_count, packed_count(nlogical));
        Ok(Self {
            scratch,
            local_best,
        })
    }

    fn run_indices(
        &mut self,
        problem: &PreparedSearch,
        code: &StabilizerCode,
        indices: &[usize],
        requested_sketch_rows: usize,
        seed: u64,
    ) -> Result<Option<LogicalWitness>, PauliError> {
        self.local_best.clear();
        for &trial_index in indices {
            self.scratch
                .run_trial(problem, code, seed, trial_index, requested_sketch_rows)?;
            if self.scratch.trial_best.valid
                && (!self.local_best.valid
                    || candidate_is_better(
                        self.scratch.trial_best.weight,
                        &self.scratch.trial_best.x_words,
                        &self.scratch.trial_best.z_words,
                        self.local_best.weight,
                        &self.local_best.x_words,
                        &self.local_best.z_words,
                        problem.basis.nqubits,
                    ))
            {
                self.local_best.copy_from(&self.scratch.trial_best);
            }
        }
        Ok(self.local_best.to_owned())
    }
}

/// Search for a concrete non-trivial logical witness.
pub fn search_logical_witness(
    code: &StabilizerCode,
    sector: LogicalSearchSector,
    max_trials: usize,
    sketch_rows: usize,
    stop_below_weight: Option<usize>,
    seed: u64,
    max_bytes: u128,
) -> Result<LogicalWitnessSearchResult, PauliError> {
    if code.nlogical() == 0 {
        return Err(PauliError::InvalidStabilizer {
            context: "logical-witness search requires at least one logical qubit",
        });
    }
    if max_trials == 0 {
        return Err(PauliError::InvalidIndex {
            context: "logical-witness max_trials must be positive",
        });
    }
    if sketch_rows == 0 {
        return Err(PauliError::InvalidIndex {
            context: "logical-witness sketch_rows must be positive",
        });
    }
    if stop_below_weight == Some(0) {
        return Err(PauliError::InvalidIndex {
            context: "logical-witness stop_below_weight must be positive",
        });
    }

    let setup_start = Instant::now();
    let (prepared, css_analysis) = match sector {
        LogicalSearchSector::General => (SearchProblems::One(cached_general(code)?), None),
        LogicalSearchSector::X => {
            let analysis = cached_css_analysis(code)?;
            (
                SearchProblems::One(cached_css_direction(code, true)?),
                Some(analysis),
            )
        }
        LogicalSearchSector::Z => {
            let analysis = cached_css_analysis(code)?;
            (
                SearchProblems::One(cached_css_direction(code, false)?),
                Some(analysis),
            )
        }
        LogicalSearchSector::Auto => match cached_css_analysis(code) {
            Ok(analysis) => (
                SearchProblems::Two(
                    cached_css_direction(code, true)?,
                    cached_css_direction(code, false)?,
                ),
                Some(analysis),
            ),
            Err(PauliError::InvalidSector { .. }) => {
                (SearchProblems::One(cached_general(code)?), None)
            }
            Err(error) => return Err(error),
        },
    };
    let setup_nanos = elapsed_nanos(setup_start);

    let worker_count = enforce_search_budget(
        &prepared,
        code.nlogical(),
        sketch_rows,
        max_trials,
        max_bytes,
        css_analysis,
    )?;
    let mut initial_best = BestBuffer::new(code.word_count(), packed_count(code.nlogical()));
    let initialization_start = Instant::now();
    let mut initialization_candidates = 0;
    let mut early_stopped = false;
    match &prepared {
        SearchProblems::One(problem) => {
            let initialization = deterministic_initialization(problem, code, stop_below_weight)?;
            initialization_candidates = initialization.candidates;
            if let Some(candidate) = initialization.best {
                initial_best.copy_from_witness(&candidate);
            }
            early_stopped = initialization.early_stopped;
        }
        SearchProblems::Two(x, z) => {
            for problem in [x, z] {
                let initialization =
                    deterministic_initialization(problem, code, stop_below_weight)?;
                initialization_candidates = initialization_candidates
                    .checked_add(initialization.candidates)
                    .ok_or(PauliError::Overflow {
                        context: "counting logical-witness initialization candidates",
                    })?;
                if let Some(candidate) = initialization.best {
                    update_best(&mut initial_best, &candidate, code.nqubits());
                }
                if initialization.early_stopped {
                    early_stopped = true;
                    break;
                }
            }
        }
    }
    let initialization_nanos = elapsed_nanos(initialization_start);
    if !initial_best.valid {
        return Err(PauliError::InvalidStabilizer {
            context: "logical-witness initialization found no valid logical",
        });
    }

    let initial_incumbent_weight = initial_best.weight;
    let mut trials_run = 0;
    if threshold_reached(&initial_best, stop_below_weight) {
        early_stopped = true;
    }
    while !early_stopped && trials_run < max_trials {
        let chunk_end = trials_run
            .checked_add(TRIAL_CHUNK_SIZE.min(max_trials - trials_run))
            .ok_or(PauliError::Overflow {
                context: "advancing logical-witness trial budget",
            })?;
        let candidate = match &prepared {
            SearchProblems::One(problem) => run_trial_batch(
                problem,
                code,
                &(trials_run..chunk_end).collect::<Vec<_>>(),
                sketch_rows,
                seed,
                worker_count,
            )?,
            SearchProblems::Two(x, z) => {
                let mut x_indices = Vec::new();
                let mut z_indices = Vec::new();
                for index in trials_run..chunk_end {
                    if index % 2 == 0 {
                        x_indices.push(index);
                    } else {
                        z_indices.push(index);
                    }
                }
                let x_candidate =
                    run_trial_batch(x, code, &x_indices, sketch_rows, seed, worker_count)?;
                let z_candidate =
                    run_trial_batch(z, code, &z_indices, sketch_rows, seed, worker_count)?;
                merge_candidates(x_candidate, z_candidate, code.nqubits())
            }
        };
        if let Some(candidate) = candidate {
            update_best(&mut initial_best, &candidate, code.nqubits());
        }
        trials_run = chunk_end;
        if threshold_reached(&initial_best, stop_below_weight) {
            early_stopped = true;
        }
    }

    let best = initial_best
        .to_owned()
        .ok_or(PauliError::InvalidStabilizer {
            context: "logical-witness search lost its incumbent",
        })?;
    validate_witness(code, &best)?;
    Ok(LogicalWitnessSearchResult {
        best_weight: best.weight,
        best,
        sector,
        algorithm: LOGICAL_SEARCH_ALGORITHM,
        sketch_rows,
        max_trials,
        trials_run,
        stop_below_weight,
        seed,
        early_stopped,
        initial_incumbent_weight,
        initialization_candidates,
        resolved_sketch_rows: resolved_rows(&prepared, sketch_rows),
        setup_nanos,
        initialization_nanos,
    })
}

enum SearchProblems<'a> {
    One(&'a PreparedSearch),
    Two(&'a PreparedSearch, &'a PreparedSearch),
}

fn cached_general(code: &StabilizerCode) -> Result<&PreparedSearch, PauliError> {
    code.general_search_cache()
        .get_or_init(|| prepare_general(code))
        .as_ref()
        .map_err(Clone::clone)
}

fn cached_css_analysis(code: &StabilizerCode) -> Result<&CssAnalysis, PauliError> {
    code.css_analysis_cache()
        .get_or_init(|| prepare_css_analysis(code))
        .as_ref()
        .map_err(Clone::clone)
}

fn cached_css_direction(
    code: &StabilizerCode,
    x_sector: bool,
) -> Result<&PreparedSearch, PauliError> {
    let analysis = cached_css_analysis(code)?;
    let result = if x_sector {
        code.css_x_search_cache()
            .get_or_init(|| prepare_css_direction(code, analysis, true))
    } else {
        code.css_z_search_cache()
            .get_or_init(|| prepare_css_direction(code, analysis, false))
    };
    result.as_ref().map_err(Clone::clone)
}

fn prepare_general(code: &StabilizerCode) -> Result<PreparedSearch, PauliError> {
    let stabilizers = build_canonical_stabilizer_rows(code)?;
    let mut logical_rows =
        Vec::with_capacity(code.nlogical().checked_mul(2).ok_or(PauliError::Overflow {
            context: "building general logical-search quotient",
        })?);
    for row in code.logical_x_word_rows() {
        logical_rows.push(PhysicalRow {
            x_words: row[..code.word_count()].to_vec(),
            z_words: row[code.word_count()..].to_vec(),
        });
    }
    for row in code.logical_z_word_rows() {
        logical_rows.push(PhysicalRow {
            x_words: row[..code.word_count()].to_vec(),
            z_words: row[code.word_count()..].to_vec(),
        });
    }
    let basis = TaggedBasis::from_rows(code.nqubits(), &stabilizers, &logical_rows)?;
    drop(stabilizers);
    Ok(PreparedSearch {
        sector: WitnessSector::Mixed,
        direction: GENERAL_DIRECTION,
        basis,
        logical_rows,
        representation: SearchRepresentation::General,
    })
}

#[cfg(test)]
fn prepare_css_pair(code: &StabilizerCode) -> Result<(PreparedSearch, PreparedSearch), PauliError> {
    let analysis = prepare_css_analysis(code)?;
    Ok((
        prepare_css_direction(code, &analysis, true)?,
        prepare_css_direction(code, &analysis, false)?,
    ))
}

fn prepare_css_analysis(code: &StabilizerCode) -> Result<CssAnalysis, PauliError> {
    let canonical = build_canonical_stabilizer_rows(code)?;
    let pure_x_rank = pure_stabilizer_rank(&canonical, code.nqubits(), true)?;
    let pure_z_rank = pure_stabilizer_rank(&canonical, code.nqubits(), false)?;
    if pure_x_rank.checked_add(pure_z_rank) != Some(canonical.len()) {
        return Err(PauliError::InvalidSector {
            context: "requested CSS logical search for a non-CSS stabilizer row space",
        });
    }
    Ok(CssAnalysis {
        canonical,
        pure_x_rank,
        pure_z_rank,
    })
}

fn prepare_css_direction(
    code: &StabilizerCode,
    analysis: &CssAnalysis,
    x_sector: bool,
) -> Result<PreparedSearch, PauliError> {
    let expected_rank = if x_sector {
        analysis.pure_x_rank
    } else {
        analysis.pure_z_rank
    };
    let pure_stabilizers = pure_stabilizers(&analysis.canonical, code.nqubits(), x_sector);
    if pure_stabilizers.len() != expected_rank {
        return Err(PauliError::InvalidStabilizer {
            context: "CSS pure stabilizer rank changed during directional setup",
        });
    }
    let logical_rows = quotient_logicals(
        &analysis.canonical,
        &pure_stabilizers,
        x_sector,
        code.nqubits(),
        code.nlogical(),
    )?;
    let basis = TaggedBasis::from_rows(code.nqubits(), &pure_stabilizers, &logical_rows)?;
    Ok(PreparedSearch {
        sector: if x_sector {
            WitnessSector::X
        } else {
            WitnessSector::Z
        },
        direction: if x_sector { X_DIRECTION } else { Z_DIRECTION },
        basis,
        logical_rows,
        representation: if x_sector {
            SearchRepresentation::CssX
        } else {
            SearchRepresentation::CssZ
        },
    })
}

fn build_canonical_stabilizer_rows(code: &StabilizerCode) -> Result<Vec<PhysicalRow>, PauliError> {
    let canonical_bits = code.canonical_stabilizer_bit_rows();
    if canonical_bits.len() != code.rank() {
        return Err(PauliError::InvalidStabilizer {
            context: "logical-search stabilizer rank disagrees with the code handle",
        });
    }
    let expected_bit_words =
        packed_count(code.nqubits().checked_mul(2).ok_or(PauliError::Overflow {
            context: "checking retained logical-search stabilizer bit width",
        })?);
    canonical_bits
        .iter()
        .map(|bits| {
            if bits.len() != expected_bit_words {
                return Err(PauliError::InvalidStabilizer {
                    context: "retained logical-search stabilizer bit width is inconsistent",
                });
            }
            let (x_words, z_words) = words_from_bits(bits, code.nqubits());
            Ok(PhysicalRow { x_words, z_words })
        })
        .collect()
}

fn pure_stabilizers(rows: &[PhysicalRow], nqubits: usize, x_sector: bool) -> Vec<PhysicalRow> {
    if rows.is_empty() {
        return Vec::new();
    }
    let constraints = coefficient_constraints(rows, nqubits, x_sector);
    let coefficients = null_space(&constraints, rows.len()).0;
    coefficients
        .iter()
        .map(|coefficient| combine_rows(rows, coefficient))
        .filter(|row| {
            if x_sector {
                row.z_words.iter().all(|&word| word == 0)
            } else {
                row.x_words.iter().all(|&word| word == 0)
            }
        })
        .collect()
}

fn pure_stabilizer_rank(
    rows: &[PhysicalRow],
    nqubits: usize,
    x_sector: bool,
) -> Result<usize, PauliError> {
    let constraints = coefficient_constraints(rows, nqubits, x_sector);
    rows.len()
        .checked_sub(rank(&constraints, rows.len()))
        .ok_or(PauliError::InvalidStabilizer {
            context: "CSS pure stabilizer rank exceeds stabilizer rank",
        })
}

fn quotient_logicals(
    stabilizers: &[PhysicalRow],
    pure_stabilizers: &[PhysicalRow],
    x_sector: bool,
    nqubits: usize,
    expected: usize,
) -> Result<Vec<PhysicalRow>, PauliError> {
    let constraints = stabilizers
        .iter()
        .map(|row| {
            if x_sector {
                row.z_words.clone()
            } else {
                row.x_words.clone()
            }
        })
        .collect::<Vec<_>>();
    let normalizer = null_space(&constraints, nqubits);
    let mut span = RrefBasis::from_reduced_rows(Vec::new(), Vec::new());
    for row in pure_stabilizers {
        let component = if x_sector { &row.x_words } else { &row.z_words };
        span.insert(component);
    }
    let mut logicals = Vec::with_capacity(expected);
    for vector in normalizer.0 {
        if span.insert(&vector) {
            let mut x_words = vec![0_u64; packed_count(nqubits)];
            let mut z_words = vec![0_u64; x_words.len()];
            if x_sector {
                x_words.copy_from_slice(&vector);
            } else {
                z_words.copy_from_slice(&vector);
            }
            logicals.push(PhysicalRow { x_words, z_words });
            if logicals.len() == expected {
                break;
            }
        }
    }
    if logicals.len() != expected {
        return Err(PauliError::InvalidStabilizer {
            context: "CSS logical quotient has an inconsistent dimension",
        });
    }
    Ok(logicals)
}

fn coefficient_constraints(rows: &[PhysicalRow], nqubits: usize, use_z: bool) -> Vec<Vec<u64>> {
    let coefficient_words = packed_count(rows.len());
    let mut constraints = vec![vec![0_u64; coefficient_words]; nqubits];
    for (row_index, row) in rows.iter().enumerate() {
        let component = if use_z { &row.z_words } else { &row.x_words };
        for (qubit, constraint) in constraints.iter_mut().enumerate().take(nqubits) {
            if get_bit(component, qubit) {
                set_bit(constraint, row_index, true);
            }
        }
    }
    constraints
}

fn combine_rows(rows: &[PhysicalRow], coefficients: &[u64]) -> PhysicalRow {
    let word_count = rows.first().map_or(0, |row| row.x_words.len());
    let mut x_words = vec![0_u64; word_count];
    let mut z_words = vec![0_u64; word_count];
    for (index, row) in rows.iter().enumerate() {
        if get_bit(coefficients, index) {
            xor_assign(&mut x_words, &row.x_words);
            xor_assign(&mut z_words, &row.z_words);
        }
    }
    PhysicalRow { x_words, z_words }
}

struct InitializationResult {
    best: Option<LogicalWitness>,
    candidates: usize,
    early_stopped: bool,
}

fn deterministic_initialization(
    problem: &PreparedSearch,
    code: &StabilizerCode,
    stop_below_weight: Option<usize>,
) -> Result<InitializationResult, PauliError> {
    let word_count = problem.basis.word_count;
    let logical_word_count = packed_count(code.nlogical());
    let mut best = BestBuffer::new(word_count, logical_word_count);
    let mut coordinates_x = vec![0_u64; packed_count(code.nlogical())];
    let mut coordinates_z = vec![0_u64; packed_count(code.nlogical())];
    let mut candidates = 0_usize;
    for row in &problem.logical_rows {
        candidates = candidates.checked_add(1).ok_or(PauliError::Overflow {
            context: "counting logical-witness initialization candidates",
        })?;
        best.consider_known_logical(
            code,
            problem.sector,
            &row.x_words,
            &row.z_words,
            &mut coordinates_x,
            &mut coordinates_z,
        );
        if threshold_reached(&best, stop_below_weight) {
            return Ok(InitializationResult {
                best: best.to_owned(),
                candidates,
                early_stopped: true,
            });
        }
    }
    if problem.logical_rows.len() <= 20 {
        let combination_count = 1_usize << problem.logical_rows.len();
        let mut x_words = vec![0_u64; word_count];
        let mut z_words = vec![0_u64; word_count];
        let mut previous_gray = 0_usize;
        for index in 1..combination_count {
            let gray = index ^ (index >> 1);
            let changed = gray ^ previous_gray;
            let logical = changed.trailing_zeros() as usize;
            xor_assign(&mut x_words, &problem.logical_rows[logical].x_words);
            xor_assign(&mut z_words, &problem.logical_rows[logical].z_words);
            candidates = candidates.checked_add(1).ok_or(PauliError::Overflow {
                context: "counting logical-witness initialization candidates",
            })?;
            best.consider_known_logical(
                code,
                problem.sector,
                &x_words,
                &z_words,
                &mut coordinates_x,
                &mut coordinates_z,
            );
            previous_gray = gray;
            if threshold_reached(&best, stop_below_weight) {
                return Ok(InitializationResult {
                    best: best.to_owned(),
                    candidates,
                    early_stopped: true,
                });
            }
        }
    }
    Ok(InitializationResult {
        best: best.to_owned(),
        candidates,
        early_stopped: false,
    })
}

fn update_best(target: &mut BestBuffer, candidate: &LogicalWitness, nqubits: usize) {
    if !target.valid
        || candidate_is_better(
            candidate.weight,
            &candidate.x_words,
            &candidate.z_words,
            target.weight,
            &target.x_words,
            &target.z_words,
            nqubits,
        )
    {
        target.copy_from_witness(candidate);
    }
}

fn resolved_rows(prepared: &SearchProblems<'_>, requested: usize) -> usize {
    match prepared {
        SearchProblems::One(problem) => problem.resolved_rows(requested),
        SearchProblems::Two(x, z) => x.resolved_rows(requested).max(z.resolved_rows(requested)),
    }
}

fn elapsed_nanos(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn enforce_search_budget(
    prepared: &SearchProblems<'_>,
    nlogical: usize,
    sketch_rows: usize,
    max_trials: usize,
    max_bytes: u128,
    css_analysis: Option<&CssAnalysis>,
) -> Result<usize, PauliError> {
    let (prepared_bytes, worker_bytes, word_count) = match prepared {
        SearchProblems::One(problem) => (
            problem.estimated_bytes()?,
            worker_bytes(problem, sketch_rows, nlogical)?,
            problem.basis.word_count,
        ),
        SearchProblems::Two(x, z) => (
            x.estimated_bytes()?
                .checked_add(z.estimated_bytes()?)
                .ok_or(PauliError::Overflow {
                    context: "estimating CSS logical-search prepared bases",
                })?,
            worker_bytes(x, sketch_rows, nlogical)?.max(worker_bytes(z, sketch_rows, nlogical)?),
            x.basis.word_count,
        ),
    };
    let retained = prepared_bytes
        .checked_add(
            css_analysis
                .map(|analysis| analysis.estimated_bytes(word_count))
                .transpose()?
                .unwrap_or(0),
        )
        .ok_or(PauliError::Overflow {
            context: "estimating cached CSS logical-search analysis",
        })?;
    let logical_words = packed_count(nlogical);
    let output_bytes = checked_bytes(
        word_count
            .checked_mul(2)
            .and_then(|value| value.checked_add(logical_words.checked_mul(2)?))
            .ok_or(PauliError::Overflow {
                context: "estimating logical-search result",
            })?,
        size_of::<u64>(),
        "estimating logical-search result",
    )?;
    let base = checked_sum(&[retained, output_bytes])?;
    let thread_limit = rayon::current_num_threads().max(1).min(max_trials);
    if base.checked_add(worker_bytes).ok_or(PauliError::Overflow {
        context: "estimating one logical-search worker",
    })? > max_bytes
    {
        return Err(PauliError::MemoryLimit {
            requested: base.checked_add(worker_bytes).ok_or(PauliError::Overflow {
                context: "estimating one logical-search worker",
            })?,
            limit: max_bytes,
        });
    }
    let available = max_bytes.saturating_sub(base);
    let budget_workers = if worker_bytes == 0 {
        thread_limit
    } else {
        available
            .checked_div(worker_bytes)
            .unwrap_or(0)
            .min(usize::MAX as u128) as usize
    };
    Ok(thread_limit.min(budget_workers.max(1)))
}

fn worker_bytes(
    problem: &PreparedSearch,
    sketch_rows: usize,
    nlogical: usize,
) -> Result<u128, PauliError> {
    let word_count = problem.basis.word_count;
    let tag_words = problem.basis.tag_words;
    let rows = problem.resolved_rows(sketch_rows);
    let physical_width = match problem.representation {
        SearchRepresentation::General => checked_product(
            word_count,
            2,
            "estimating general logical-search physical row width",
        )?,
        SearchRepresentation::CssX | SearchRepresentation::CssZ => word_count,
    };
    let physical_columns = match problem.representation {
        SearchRepresentation::General => {
            problem
                .basis
                .nqubits
                .checked_mul(2)
                .ok_or(PauliError::Overflow {
                    context: "estimating general logical-search column permutation",
                })?
        }
        SearchRepresentation::CssX | SearchRepresentation::CssZ => problem.basis.nqubits,
    };
    let logical_words = packed_count(nlogical);
    let matrix_words = checked_product(
        rows,
        physical_width,
        "estimating logical-search trial matrix",
    )?;
    let zero_words = match problem.representation {
        SearchRepresentation::General => 0,
        SearchRepresentation::CssX | SearchRepresentation::CssZ => word_count,
    };
    let maps = match problem.representation {
        SearchRepresentation::General => problem.basis.nqubits,
        SearchRepresentation::CssX | SearchRepresentation::CssZ => 0,
    };
    checked_sum(&[
        checked_bytes(
            matrix_words,
            size_of::<u64>(),
            "estimating logical-search trial matrix",
        )?,
        checked_bytes(
            checked_product(rows, tag_words, "estimating logical-search trial tags")?,
            size_of::<u64>(),
            "estimating logical-search trial tags",
        )?,
        checked_bytes(
            checked_product(
                physical_width,
                3,
                "estimating logical-search physical scratch",
            )?
            .checked_add(zero_words)
            .ok_or(PauliError::Overflow {
                context: "estimating logical-search physical scratch",
            })?,
            size_of::<u64>(),
            "estimating logical-search physical scratch",
        )?,
        checked_bytes(
            checked_product(tag_words, 3, "estimating logical-search tag scratch")?,
            size_of::<u64>(),
            "estimating logical-search tag scratch",
        )?,
        checked_bytes(
            checked_product(
                logical_words,
                2,
                "estimating logical-search quotient scratch",
            )?,
            size_of::<u64>(),
            "estimating logical-search quotient scratch",
        )?,
        checked_bytes(
            maps,
            size_of::<u8>(),
            "estimating logical-search local maps",
        )?,
        checked_bytes(
            physical_columns,
            size_of::<usize>(),
            "estimating logical-search column permutation",
        )?,
        checked_bytes(
            word_count
                .checked_mul(2)
                .and_then(|value| value.checked_add(logical_words.checked_mul(2)?))
                .ok_or(PauliError::Overflow {
                    context: "estimating logical-search local incumbent",
                })?,
            size_of::<u64>(),
            "estimating logical-search local incumbent",
        )?,
    ])
}

fn run_trial_batch(
    problem: &PreparedSearch,
    code: &StabilizerCode,
    indices: &[usize],
    sketch_rows: usize,
    seed: u64,
    worker_count: usize,
) -> Result<Option<LogicalWitness>, PauliError> {
    if indices.is_empty() {
        return Ok(None);
    }
    let subchunk = indices
        .len()
        .div_ceil(worker_count.saturating_mul(4).max(1))
        .max(1);
    if worker_count <= 1 {
        let mut state = WorkerState::new(problem, sketch_rows, code.nlogical())?;
        return state.run_indices(problem, code, indices, sketch_rows, seed);
    }
    let run_parallel = || {
        let ranges = indices.chunks(subchunk).collect::<Vec<_>>();
        let results = ranges
            .par_iter()
            .map_init(
                || WorkerState::new(problem, sketch_rows, code.nlogical()),
                |state, range| match state {
                    Ok(worker) => worker.run_indices(problem, code, range, sketch_rows, seed),
                    Err(error) => Err(error.clone()),
                },
            )
            .collect::<Result<Vec<_>, PauliError>>()?;
        let mut best = None;
        for candidate in results.into_iter().flatten() {
            best = merge_candidates(best, Some(candidate), code.nqubits());
        }
        Ok::<Option<LogicalWitness>, PauliError>(best)
    };
    if worker_count < rayon::current_num_threads() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .map_err(|_| PauliError::InvalidStabilizer {
                context: "failed to create the bounded logical-search worker pool",
            })?;
        pool.install(run_parallel)
    } else {
        run_parallel()
    }
}

fn merge_candidates(
    left: Option<LogicalWitness>,
    right: Option<LogicalWitness>,
    nqubits: usize,
) -> Option<LogicalWitness> {
    match (left, right) {
        (None, candidate) | (candidate, None) => candidate,
        (Some(left), Some(right)) => {
            if candidate_is_better(
                right.weight,
                &right.x_words,
                &right.z_words,
                left.weight,
                &left.x_words,
                &left.z_words,
                nqubits,
            ) {
                Some(right)
            } else {
                Some(left)
            }
        }
    }
}

fn threshold_reached(best: &BestBuffer, threshold: Option<usize>) -> bool {
    threshold.is_some_and(|value| best.valid && best.weight < value)
}

fn validate_witness(code: &StabilizerCode, witness: &LogicalWitness) -> Result<(), PauliError> {
    if witness.x_words.len() != code.word_count() || witness.z_words.len() != code.word_count() {
        return Err(PauliError::InvalidStabilizer {
            context: "logical-witness physical width is invalid",
        });
    }
    if witness.logical_x_words.len() != packed_count(code.nlogical())
        || witness.logical_z_words.len() != packed_count(code.nlogical())
    {
        return Err(PauliError::InvalidStabilizer {
            context: "logical-witness quotient width is invalid",
        });
    }
    if let Some(mask) = tail_mask(code.nqubits()) {
        if code.word_count() != 0
            && (witness.x_words[code.word_count() - 1] & !mask != 0
                || witness.z_words[code.word_count() - 1] & !mask != 0)
        {
            return Err(PauliError::InvalidStabilizer {
                context: "logical-witness has nonzero unused physical bits",
            });
        }
    }
    if !code.is_in_normalizer_words(&witness.x_words, &witness.z_words) {
        return Err(PauliError::InvalidStabilizer {
            context: "logical-witness has a nonzero syndrome",
        });
    }
    let mut coordinates_x = vec![0_u64; packed_count(code.nlogical())];
    let mut coordinates_z = vec![0_u64; packed_count(code.nlogical())];
    code.logical_coordinates_packed_words(
        &witness.x_words,
        &witness.z_words,
        &mut coordinates_x,
        &mut coordinates_z,
    );
    if coordinates_x.iter().all(|&word| word == 0) && coordinates_z.iter().all(|&word| word == 0) {
        return Err(PauliError::InvalidStabilizer {
            context: "logical-witness is a stabilizer",
        });
    }
    if coordinates_x != witness.logical_x_words || coordinates_z != witness.logical_z_words {
        return Err(PauliError::InvalidStabilizer {
            context: "logical-witness quotient coordinates are inconsistent",
        });
    }
    if physical_weight(&witness.x_words, &witness.z_words) != witness.weight {
        return Err(PauliError::InvalidStabilizer {
            context: "logical-witness weight is inconsistent",
        });
    }
    Ok(())
}

fn candidate_is_better(
    left_weight: usize,
    left_x: &[u64],
    left_z: &[u64],
    right_weight: usize,
    right_x: &[u64],
    right_z: &[u64],
    nqubits: usize,
) -> bool {
    if left_weight != right_weight {
        return left_weight < right_weight;
    }
    compare_pauli_words(left_x, left_z, right_x, right_z, nqubits) == Ordering::Less
}

fn compare_pauli_words(
    left_x: &[u64],
    left_z: &[u64],
    right_x: &[u64],
    right_z: &[u64],
    nqubits: usize,
) -> Ordering {
    for qubit in 0..nqubits {
        let left = pauli_code_at(left_x, left_z, qubit);
        let right = pauli_code_at(right_x, right_z, qubit);
        if left != right {
            return left.cmp(&right);
        }
    }
    Ordering::Equal
}

fn pauli_code_at(x_words: &[u64], z_words: &[u64], qubit: usize) -> u8 {
    let pair = u8::from(get_bit(x_words, qubit)) | (u8::from(get_bit(z_words, qubit)) << 1);
    PAULI_CODE_FROM_PAIR[pair as usize]
}

fn physical_weight(x_words: &[u64], z_words: &[u64]) -> usize {
    x_words
        .iter()
        .zip(z_words)
        .map(|(&x, &z)| (x | z).count_ones() as usize)
        .sum()
}

fn xor_basis_physical(
    target: &mut [u64],
    problem: &PreparedSearch,
    row: usize,
    representation: SearchRepresentation,
) {
    match representation {
        SearchRepresentation::General => {
            let (target_x, target_z) = target.split_at_mut(problem.basis.word_count);
            xor_assign(target_x, problem.basis.row_x(row));
            xor_assign(target_z, problem.basis.row_z(row));
        }
        SearchRepresentation::CssX => xor_assign(target, problem.basis.row_x(row)),
        SearchRepresentation::CssZ => xor_assign(target, problem.basis.row_z(row)),
    }
}

fn represented_weight(
    words: &[u64],
    representation: SearchRepresentation,
    word_count: usize,
) -> usize {
    match representation {
        SearchRepresentation::General => {
            let (x_words, z_words) = words.split_at(word_count);
            physical_weight(x_words, z_words)
        }
        SearchRepresentation::CssX | SearchRepresentation::CssZ => {
            words.iter().map(|word| word.count_ones() as usize).sum()
        }
    }
}

fn represented_pauli_code(
    words: &[u64],
    representation: SearchRepresentation,
    word_count: usize,
    qubit: usize,
) -> u8 {
    match representation {
        SearchRepresentation::General => {
            pauli_code_at(&words[..word_count], &words[word_count..], qubit)
        }
        SearchRepresentation::CssX => u8::from(get_bit(words, qubit)),
        SearchRepresentation::CssZ => {
            if get_bit(words, qubit) {
                3
            } else {
                0
            }
        }
    }
}

fn candidate_is_better_representation(
    left_weight: usize,
    left_words: &[u64],
    left_representation: SearchRepresentation,
    right_weight: usize,
    right_words: &[u64],
    right_representation: SearchRepresentation,
    nqubits: usize,
) -> bool {
    if left_weight != right_weight {
        return left_weight < right_weight;
    }
    for qubit in 0..nqubits {
        let left_word_count = match left_representation {
            SearchRepresentation::General => left_words.len() / 2,
            SearchRepresentation::CssX | SearchRepresentation::CssZ => left_words.len(),
        };
        let right_word_count = match right_representation {
            SearchRepresentation::General => right_words.len() / 2,
            SearchRepresentation::CssX | SearchRepresentation::CssZ => right_words.len(),
        };
        let left = represented_pauli_code(left_words, left_representation, left_word_count, qubit);
        let right =
            represented_pauli_code(right_words, right_representation, right_word_count, qubit);
        if left != right {
            return left < right;
        }
    }
    false
}

fn resolved_witness_sector(
    requested: WitnessSector,
    x_words: &[u64],
    z_words: &[u64],
) -> WitnessSector {
    if requested != WitnessSector::Mixed {
        return requested;
    }
    if z_words.iter().all(|&word| word == 0) {
        WitnessSector::X
    } else if x_words.iter().all(|&word| word == 0) {
        WitnessSector::Z
    } else {
        WitnessSector::Mixed
    }
}

fn apply_maps_to_row(
    x_words: &mut [u64],
    z_words: &mut [u64],
    maps: &[u8],
    inverse_maps: &[usize; 6],
) {
    for (qubit, &map) in maps.iter().enumerate() {
        let word = qubit / 64;
        let bit = 1_u64 << (qubit % 64);
        let input_pair =
            u8::from(x_words[word] & bit != 0) | (u8::from(z_words[word] & bit != 0) << 1);
        let input_code = PAULI_CODE_FROM_PAIR[input_pair as usize];
        let output_code = LOCAL_MAP_CODES[inverse_maps[map as usize]][input_code as usize];
        let output_pair = match output_code {
            0 => 0,
            1 => 1,
            2 => 3,
            3 => 2,
            _ => unreachable!(),
        };
        if output_pair & 1 != 0 {
            x_words[word] |= bit;
        } else {
            x_words[word] &= !bit;
        }
        if output_pair & 2 != 0 {
            z_words[word] |= bit;
        } else {
            z_words[word] &= !bit;
        }
    }
}

const IDENTITY_MAPS: [usize; 6] = [0, 1, 2, 3, 4, 5];

fn xor_into(target: &mut [u64], left: &[u64], right: &[u64]) {
    for ((target, &left), &right) in target.iter_mut().zip(left).zip(right) {
        *target = left ^ right;
    }
}

fn xor_disjoint_rows(words: &mut [u64], target: usize, pivot: usize, row_width: usize) {
    debug_assert_ne!(target, pivot);
    if target < pivot {
        let (before, after) = words.split_at_mut(pivot * row_width);
        xor_assign(
            &mut before[target * row_width..(target + 1) * row_width],
            &after[..row_width],
        );
    } else {
        let (before, after) = words.split_at_mut(target * row_width);
        xor_assign(
            &mut after[..row_width],
            &before[pivot * row_width..(pivot + 1) * row_width],
        );
    }
}

fn matrix_bit(words: &[u64], row: usize, row_width: usize, column: usize) -> bool {
    words[row * row_width + column / 64] & (1_u64 << (column % 64)) != 0
}

fn tail_mask(nqubits: usize) -> Option<u64> {
    let remainder = nqubits % 64;
    (remainder != 0).then(|| (1_u64 << remainder) - 1)
}

fn checked_product(left: usize, right: usize, context: &'static str) -> Result<usize, PauliError> {
    left.checked_mul(right)
        .ok_or(PauliError::Overflow { context })
}

fn checked_bytes(
    count: usize,
    element_size: usize,
    context: &'static str,
) -> Result<u128, PauliError> {
    (count as u128)
        .checked_mul(element_size as u128)
        .ok_or(PauliError::Overflow { context })
}

fn checked_sum(values: &[u128]) -> Result<u128, PauliError> {
    values.iter().try_fold(0_u128, |sum, &value| {
        sum.checked_add(value).ok_or(PauliError::Overflow {
            context: "estimating logical-search memory",
        })
    })
}

#[cfg(test)]
mod tests {
    use super::{
        deterministic_initialization, prepare_css_pair, prepare_general, run_trial_batch,
        search_logical_witness, TrialScratch,
    };
    use crate::{LogicalSearchSector, StabilizerCode};
    use rayon::ThreadPoolBuilder;

    #[test]
    fn empty_stabilizer_search_returns_weight_one_logicals() {
        for nqubits in [1, 2, 65] {
            let code = StabilizerCode::from_css_binary(nqubits, &[], &[], &[], &[], u128::MAX)
                .expect("unconstrained code");
            for sector in [
                LogicalSearchSector::Auto,
                LogicalSearchSector::X,
                LogicalSearchSector::Z,
                LogicalSearchSector::General,
            ] {
                let result = search_logical_witness(&code, sector, 1, 2, None, 0, u128::MAX)
                    .expect("logical search for an unconstrained code");
                assert_eq!(result.best_weight, 1);
                assert_eq!(result.best.weight, 1);
                assert_eq!(result.best.x_words.len(), code.word_count());
                assert_eq!(result.best.z_words.len(), code.word_count());
                assert!(code.is_in_normalizer_words(&result.best.x_words, &result.best.z_words));
                if sector == LogicalSearchSector::X {
                    assert!(result.best.z_words.iter().all(|&word| word == 0));
                } else if sector == LogicalSearchSector::Z {
                    assert!(result.best.x_words.iter().all(|&word| word == 0));
                }
            }
        }
    }

    #[test]
    fn repetition_css_setup_has_x_and_z_incumbents() {
        let code =
            StabilizerCode::from_css_binary(3, &[], &[1, 1, 0, 0, 1, 1], &[], &[1, 1], u128::MAX)
                .expect("repetition code");
        let (x, z) = prepare_css_pair(&code).expect("CSS setup");
        assert_eq!(x.logical_rows.len(), 1);
        assert_eq!(z.logical_rows.len(), 1);
        assert_eq!(x.logical_rows[0].x_words, vec![7]);
        assert_eq!(z.logical_rows[0].z_words, vec![1]);
        assert!(deterministic_initialization(&x, &code, None)
            .unwrap()
            .best
            .is_some());
        assert!(deterministic_initialization(&z, &code, None)
            .unwrap()
            .best
            .is_some());
    }

    #[test]
    fn repetition_search_returns_a_valid_incumbent() {
        let code =
            StabilizerCode::from_css_binary(3, &[], &[1, 1, 0, 0, 1, 1], &[], &[1, 1], u128::MAX)
                .expect("repetition code");
        let result =
            search_logical_witness(&code, LogicalSearchSector::Auto, 2, 4, None, 7, u128::MAX)
                .expect("search");
        assert!(result.best_weight >= 1);
        assert!(code.is_in_normalizer_words(&result.best.x_words, &result.best.z_words));
    }

    #[test]
    fn search_caches_are_lazy_until_search() {
        let code =
            StabilizerCode::from_css_binary(3, &[], &[1, 1, 0, 0, 1, 1], &[], &[1, 1], u128::MAX)
                .expect("repetition code");
        assert!(code.general_search_cache().get().is_none());
        assert!(code.css_analysis_cache().get().is_none());

        search_logical_witness(
            &code,
            LogicalSearchSector::General,
            1,
            4,
            None,
            7,
            u128::MAX,
        )
        .expect("general search");
        assert!(code.general_search_cache().get().is_some());
        assert!(code.css_analysis_cache().get().is_none());
    }

    #[test]
    fn search_caches_are_shared_across_clones() {
        let code =
            StabilizerCode::from_css_binary(3, &[], &[1, 1, 0, 0, 1, 1], &[], &[1, 1], u128::MAX)
                .expect("repetition code");
        let clone = code.clone();
        search_logical_witness(
            &clone,
            LogicalSearchSector::General,
            1,
            4,
            None,
            7,
            u128::MAX,
        )
        .expect("general search");
        assert!(code.general_search_cache().get().is_some());
    }

    #[test]
    fn seeded_trial_batch_is_stable_across_worker_counts() {
        let code =
            StabilizerCode::from_css_binary(3, &[], &[1, 1, 0, 0, 1, 1], &[], &[1, 1], u128::MAX)
                .expect("repetition code");
        let problem = prepare_general(&code).expect("general setup");
        let indices = (0..64).collect::<Vec<_>>();
        let serial =
            run_trial_batch(&problem, &code, &indices, 4, 23, 1).expect("serial trial batch");
        let parallel = ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("test worker pool")
            .install(|| {
                run_trial_batch(&problem, &code, &indices, 4, 23, 2).expect("parallel trial batch")
            });
        assert_eq!(serial, parallel);
    }

    #[test]
    fn css_trial_scratch_uses_one_sector_and_n_columns() {
        let code =
            StabilizerCode::from_css_binary(3, &[], &[1, 1, 0, 0, 1, 1], &[], &[1, 1], u128::MAX)
                .expect("repetition code");
        let (x, _) = prepare_css_pair(&code).expect("CSS setup");
        let css = TrialScratch::new(&x, 4, code.nlogical()).expect("CSS scratch");
        assert_eq!(css.representation, super::SearchRepresentation::CssX);
        assert_eq!(css.physical_width, code.word_count());
        assert_eq!(css.physical_words.len(), css.rows * code.word_count());
        assert_eq!(css.permutation.len(), code.nqubits());
        assert!(css.maps.is_empty());

        let general = prepare_general(&code).expect("general setup");
        let general_scratch =
            TrialScratch::new(&general, 4, code.nlogical()).expect("general scratch");
        assert_eq!(general_scratch.physical_width, 2 * code.word_count());
        assert_eq!(general_scratch.permutation.len(), 2 * code.nqubits());
        assert_eq!(general_scratch.maps.len(), code.nqubits());
    }

    #[test]
    fn directional_css_search_lazily_prepares_only_requested_direction() {
        let code =
            StabilizerCode::from_css_binary(3, &[], &[1, 1, 0, 0, 1, 1], &[], &[1, 1], u128::MAX)
                .expect("repetition code");
        search_logical_witness(&code, LogicalSearchSector::X, 1, 4, None, 7, u128::MAX)
            .expect("X search");
        assert!(code.css_analysis_cache().get().is_some());
        assert!(code.css_x_search_cache().get().is_some());
        assert!(code.css_z_search_cache().get().is_none());
        search_logical_witness(&code, LogicalSearchSector::Z, 1, 4, None, 7, u128::MAX)
            .expect("Z search");
        assert!(code.css_z_search_cache().get().is_some());
    }
}
