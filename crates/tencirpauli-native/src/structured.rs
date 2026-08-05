use std::cmp::Ordering;
use std::collections::BTreeMap;

use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{
    binary_boson_terms, binary_fermion_terms, binary_hybrid_terms, canonicalize_boson_terms,
    canonicalize_fermion_terms, canonicalize_hybrid_terms, fermion_to_majorana_terms,
    jordan_wigner_hybrid_terms, jordan_wigner_terms, multiply_boson_terms, multiply_fermion_terms,
    multiply_hybrid_terms, structured_dense_matrix, structured_mvp_plan, structured_sparse_matrix,
    BosonCanonicalResult, Complex64, FermionBatch, FermionCanonicalResult, HybridBatch,
    HybridLayout, HybridRawBatch, StructuredMvpPlan as CoreStructuredMvpPlan, StructuredOperation,
};

use crate::convert::{complex_coefficients, map_error, split_complex};
use crate::majorana::NativeMajoranaOperatorHandle;
use crate::operator::NativePauliOperatorHandle;

type NumpyFermionOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
type NumpyBosonOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
type JordanWignerOutput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>);
type FermionInput = (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<f64>, Vec<f64>);
type BosonInput = (Vec<Vec<(u32, u32, u32)>>, Vec<f64>, Vec<f64>);
type HybridInput = (
    Vec<bool>,
    Vec<Vec<u32>>,
    Vec<Vec<u32>>,
    Vec<bool>,
    Vec<Vec<(u32, u32, u32)>>,
    Vec<Vec<u8>>,
    Vec<bool>,
    Vec<Vec<u8>>,
    Vec<bool>,
    Vec<Vec<(u32, u32, u32)>>,
    Vec<f64>,
    Vec<f64>,
);
type HybridRawInput = (
    Vec<Vec<(usize, u8)>>,
    Vec<Vec<(usize, u8)>>,
    Vec<Vec<u8>>,
    Vec<bool>,
    Vec<Vec<(u32, u32, u32)>>,
    Vec<f64>,
    Vec<f64>,
);
type StructuredSparseOutput = (usize, Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>);
type HybridFlatParts = (
    Vec<u8>,
    Vec<u32>,
    Vec<usize>,
    Vec<u32>,
    Vec<usize>,
    Vec<u32>,
    Vec<usize>,
    Vec<u8>,
    Vec<u8>,
    Vec<u32>,
    Vec<usize>,
    Vec<Complex64>,
);
type NumpyHybridOutput<'py> = (
    usize,
    usize,
    (
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<usize>>,
    ),
    (
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<NumpyComplex128>>,
    ),
);

fn flatten_words(words: &[Vec<u32>]) -> (Vec<u32>, Vec<usize>) {
    let mut payload = Vec::new();
    let mut offsets = Vec::with_capacity(words.len() + 1);
    offsets.push(0);
    for word in words {
        payload.extend_from_slice(word);
        offsets.push(payload.len());
    }
    (payload, offsets)
}

fn flatten_blocks(words: &[Vec<(u32, u32, u32)>]) -> (Vec<u32>, Vec<usize>) {
    let mut payload = Vec::new();
    let mut offsets = Vec::with_capacity(words.len() + 1);
    offsets.push(0);
    for word in words {
        for &(mode, create, annihilate) in word {
            payload.extend([mode, create, annihilate]);
        }
        offsets.push(payload.len() / 3);
    }
    (payload, offsets)
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeFermionOperatorHandle {
    n_modes: usize,
    creation: Vec<Vec<u32>>,
    annihilation: Vec<Vec<u32>>,
    coefficients: Vec<Complex64>,
}

impl NativeFermionOperatorHandle {
    pub(crate) fn from_result(n_modes: usize, result: FermionCanonicalResult) -> Self {
        Self {
            n_modes,
            creation: result.0,
            annihilation: result.1,
            coefficients: result.2,
        }
    }

    fn to_hybrid_result(&self) -> NativeHybridOperatorHandle {
        let mut result = empty_hybrid_result(self.coefficients.len());
        for index in 0..self.coefficients.len() {
            result.fermion_present.push(true);
            result.fermion_creation.push(self.creation[index].clone());
            result
                .fermion_annihilation
                .push(self.annihilation[index].clone());
            result.boson_present.push(false);
            result.boson_blocks.push(Vec::new());
            result.qubit_codes.push(Vec::new());
            result.mapped_present.push(false);
            result.mapped_codes.push(Vec::new());
            result.qudit_present.push(false);
            result.qudit_triples.push(Vec::new());
            result.coefficients.push(self.coefficients[index]);
        }
        NativeHybridOperatorHandle::from_result(
            HybridLayout {
                n_modes: self.n_modes,
                n_bosons: 0,
                nqubits: 0,
                n_qudit_sites: 0,
                qudit_dimension: 0,
            },
            result,
        )
    }
}

#[pymethods]
impl NativeFermionOperatorHandle {
    #[getter]
    fn n_modes(&self) -> usize {
        self.n_modes
    }

    #[getter]
    fn term_count(&self) -> usize {
        self.coefficients.len()
    }

    fn add(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.n_modes != other.n_modes {
            return Err(PyValueError::new_err("fermion handle mode counts differ"));
        }
        let result = py
            .allow_threads(|| merge_fermion_handles(self, other, max_bytes as u128))
            .map_err(map_error)?;
        Ok(result)
    }

    fn scale(&self, py: Python<'_>, scalar_re: f64, scalar_im: f64) -> PyResult<Self> {
        py.allow_threads(|| scale_fermion_handle(self, Complex64::new(scalar_re, scalar_im)))
            .map_err(map_error)
    }

    fn multiply(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.n_modes != other.n_modes {
            return Err(PyValueError::new_err("fermion handle mode counts differ"));
        }
        let result = py
            .allow_threads(|| {
                multiply_fermion_terms(
                    self.n_modes,
                    FermionBatch {
                        creation: &self.creation,
                        annihilation: &self.annihilation,
                        coefficients: &self.coefficients,
                    },
                    FermionBatch {
                        creation: &other.creation,
                        annihilation: &other.annihilation,
                        coefficients: &other.coefficients,
                    },
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.n_modes, result))
    }

    fn commutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.n_modes != other.n_modes {
            return Err(PyValueError::new_err("fermion handle mode counts differ"));
        }
        let result = py
            .allow_threads(|| {
                binary_fermion_terms(
                    self.n_modes,
                    FermionBatch {
                        creation: &self.creation,
                        annihilation: &self.annihilation,
                        coefficients: &self.coefficients,
                    },
                    FermionBatch {
                        creation: &other.creation,
                        annihilation: &other.annihilation,
                        coefficients: &other.coefficients,
                    },
                    max_bytes as u128,
                    -1,
                )
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.n_modes, result))
    }

    fn anticommutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.n_modes != other.n_modes {
            return Err(PyValueError::new_err("fermion handle mode counts differ"));
        }
        let result = py
            .allow_threads(|| {
                binary_fermion_terms(
                    self.n_modes,
                    FermionBatch {
                        creation: &self.creation,
                        annihilation: &self.annihilation,
                        coefficients: &self.coefficients,
                    },
                    FermionBatch {
                        creation: &other.creation,
                        annihilation: &other.annihilation,
                        coefficients: &other.coefficients,
                    },
                    max_bytes as u128,
                    1,
                )
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.n_modes, result))
    }

    fn adjoint(&self, py: Python<'_>, max_bytes: usize) -> PyResult<Self> {
        let factors = self
            .annihilation
            .iter()
            .zip(&self.creation)
            .map(|(annihilation, creation)| {
                annihilation
                    .iter()
                    .rev()
                    .map(|&mode| (mode as usize, 0))
                    .chain(creation.iter().rev().map(|&mode| (mode as usize, 1)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let coefficients = self
            .coefficients
            .iter()
            .map(|value| value.conj())
            .collect::<Vec<_>>();
        let result = py
            .allow_threads(|| {
                canonicalize_fermion_terms(self.n_modes, &factors, &coefficients, max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.n_modes, result))
    }

    fn materialize<'py>(&self, py: Python<'py>) -> NumpyFermionOutput<'py> {
        let (creation, annihilation, coefficients) = py.allow_threads(|| {
            (
                flatten_words(&self.creation),
                flatten_words(&self.annihilation),
                self.coefficients.clone(),
            )
        });
        (
            self.term_count(),
            PyArray1::from_vec(py, creation.0),
            PyArray1::from_vec(py, creation.1),
            PyArray1::from_vec(py, annihilation.0),
            PyArray1::from_vec(py, annihilation.1),
            PyArray1::from_vec(py, coefficients),
        )
    }

    fn jordan_wigner(
        &self,
        py: Python<'_>,
        max_bytes: usize,
    ) -> PyResult<NativePauliOperatorHandle> {
        let (structures, coefficients) = py
            .allow_threads(|| {
                jordan_wigner_terms(
                    self.n_modes,
                    &self.creation,
                    &self.annihilation,
                    &self.coefficients,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        let operator =
            tencir_pauli_core::PauliOperator::from_terms(self.n_modes, &structures, &coefficients)
                .map_err(map_error)?;
        Ok(NativePauliOperatorHandle::from_operator(operator))
    }

    fn to_majorana(
        &self,
        py: Python<'_>,
        max_bytes: usize,
    ) -> PyResult<NativeMajoranaOperatorHandle> {
        let result = py
            .allow_threads(|| {
                fermion_to_majorana_terms(
                    self.n_modes,
                    &self.creation,
                    &self.annihilation,
                    &self.coefficients,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        Ok(NativeMajoranaOperatorHandle::from_result(
            self.n_modes,
            result,
        ))
    }

    fn to_hybrid(&self, py: Python<'_>) -> NativeHybridOperatorHandle {
        py.allow_threads(|| self.to_hybrid_result())
    }

    fn is_hermitian(&self, py: Python<'_>, tolerance: f64) -> bool {
        py.allow_threads(|| {
            let factors = self
                .annihilation
                .iter()
                .zip(&self.creation)
                .map(|(annihilation, creation)| {
                    annihilation
                        .iter()
                        .rev()
                        .map(|&mode| (mode as usize, 0))
                        .chain(creation.iter().rev().map(|&mode| (mode as usize, 1)))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let coefficients = self
                .coefficients
                .iter()
                .map(|value| value.conj())
                .collect::<Vec<_>>();
            let Ok((creation, annihilation, adjoint_coefficients)) = canonicalize_fermion_terms(
                self.n_modes,
                &factors,
                &coefficients,
                usize::MAX as u128,
            ) else {
                return false;
            };
            same_structured_coefficients(
                &self.creation,
                &self.annihilation,
                &self.coefficients,
                &creation,
                &annihilation,
                &adjoint_coefficients,
                tolerance,
            )
        })
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeBosonOperatorHandle {
    n_modes: usize,
    blocks: Vec<Vec<(u32, u32, u32)>>,
    coefficients: Vec<Complex64>,
}

impl NativeBosonOperatorHandle {
    fn from_result(n_modes: usize, result: BosonCanonicalResult) -> Self {
        Self {
            n_modes,
            blocks: result.0,
            coefficients: result.1,
        }
    }

    fn to_hybrid_result(&self) -> NativeHybridOperatorHandle {
        let mut result = empty_hybrid_result(self.coefficients.len());
        for index in 0..self.coefficients.len() {
            result.fermion_present.push(false);
            result.fermion_creation.push(Vec::new());
            result.fermion_annihilation.push(Vec::new());
            result.boson_present.push(true);
            result.boson_blocks.push(self.blocks[index].clone());
            result.qubit_codes.push(Vec::new());
            result.mapped_present.push(false);
            result.mapped_codes.push(Vec::new());
            result.qudit_present.push(false);
            result.qudit_triples.push(Vec::new());
            result.coefficients.push(self.coefficients[index]);
        }
        NativeHybridOperatorHandle::from_result(
            HybridLayout {
                n_modes: 0,
                n_bosons: self.n_modes,
                nqubits: 0,
                n_qudit_sites: 0,
                qudit_dimension: 0,
            },
            result,
        )
    }
}

fn empty_hybrid_result(capacity: usize) -> tencir_pauli_core::HybridCanonicalResult {
    tencir_pauli_core::HybridCanonicalResult {
        fermion_present: Vec::with_capacity(capacity),
        fermion_creation: Vec::with_capacity(capacity),
        fermion_annihilation: Vec::with_capacity(capacity),
        boson_present: Vec::with_capacity(capacity),
        boson_blocks: Vec::with_capacity(capacity),
        qubit_codes: Vec::with_capacity(capacity),
        mapped_present: Vec::with_capacity(capacity),
        mapped_codes: Vec::with_capacity(capacity),
        qudit_present: Vec::with_capacity(capacity),
        qudit_triples: Vec::with_capacity(capacity),
        coefficients: Vec::with_capacity(capacity),
    }
}

fn check_native_output(
    term_count: usize,
    bytes_per_term: usize,
    max_bytes: u128,
    context: &'static str,
) -> Result<(), tencir_pauli_core::PauliError> {
    let requested = (term_count.max(1) as u128)
        .checked_mul(bytes_per_term as u128)
        .ok_or(tencir_pauli_core::PauliError::Overflow { context })?;
    if requested > max_bytes {
        return Err(tencir_pauli_core::PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    Ok(())
}

fn coefficient_close(left: Complex64, right: Complex64, tolerance: f64) -> bool {
    (left.re - right.re).abs() <= tolerance && (left.im - right.im).abs() <= tolerance
}

fn same_coefficient_maps<K: Ord>(
    left: &BTreeMap<K, Complex64>,
    right: &BTreeMap<K, Complex64>,
    tolerance: f64,
) -> bool {
    left.len() == right.len()
        && left.iter().all(|(key, value)| {
            right
                .get(key)
                .is_some_and(|other| coefficient_close(*value, *other, tolerance))
        })
}

fn same_structured_coefficients(
    left_creation: &[Vec<u32>],
    left_annihilation: &[Vec<u32>],
    left_coefficients: &[Complex64],
    right_creation: &[Vec<u32>],
    right_annihilation: &[Vec<u32>],
    right_coefficients: &[Complex64],
    tolerance: f64,
) -> bool {
    if left_creation.len() != right_creation.len()
        || left_annihilation.len() != right_annihilation.len()
        || left_coefficients.len() != right_coefficients.len()
    {
        return false;
    }
    left_creation
        .iter()
        .zip(left_annihilation)
        .zip(left_coefficients)
        .zip(
            right_creation
                .iter()
                .zip(right_annihilation)
                .zip(right_coefficients),
        )
        .all(
            |(((left_c, left_a), left_value), ((right_c, right_a), right_value))| {
                left_c == right_c
                    && left_a == right_a
                    && coefficient_close(*left_value, *right_value, tolerance)
            },
        )
}

fn boson_adjoint(input: &NativeBosonOperatorHandle) -> NativeBosonOperatorHandle {
    NativeBosonOperatorHandle {
        n_modes: input.n_modes,
        blocks: input
            .blocks
            .iter()
            .map(|word| {
                word.iter()
                    .map(|&(mode, creation, annihilation)| (mode, annihilation, creation))
                    .collect()
            })
            .collect(),
        coefficients: input
            .coefficients
            .iter()
            .map(|value| value.conj())
            .collect(),
    }
}

fn same_hybrid_coefficients(
    left: &NativeHybridOperatorHandle,
    right: &NativeHybridOperatorHandle,
    tolerance: f64,
) -> bool {
    let left_map = (0..left.term_count())
        .map(|index| (left.key_at(index), left.result.coefficients[index]))
        .collect::<BTreeMap<_, _>>();
    let right_map = (0..right.term_count())
        .map(|index| (right.key_at(index), right.result.coefficients[index]))
        .collect::<BTreeMap<_, _>>();
    same_coefficient_maps(&left_map, &right_map, tolerance)
}

fn merge_fermion_handles(
    left: &NativeFermionOperatorHandle,
    right: &NativeFermionOperatorHandle,
    max_bytes: u128,
) -> Result<NativeFermionOperatorHandle, tencir_pauli_core::PauliError> {
    check_native_output(
        left.term_count() + right.term_count(),
        left.n_modes.div_ceil(64) * 16 + 32,
        max_bytes,
        "estimating fermion operator addition",
    )?;
    let capacity = left.term_count() + right.term_count();
    let mut creation = Vec::with_capacity(capacity);
    let mut annihilation = Vec::with_capacity(capacity);
    let mut coefficients = Vec::with_capacity(capacity);
    let mut index = 0;
    let mut other_index = 0;
    while index < left.term_count() && other_index < right.term_count() {
        let left_key = (&left.creation[index], &left.annihilation[index]);
        let right_key = (
            &right.creation[other_index],
            &right.annihilation[other_index],
        );
        match left_key.cmp(&right_key) {
            Ordering::Less => {
                creation.push(left.creation[index].clone());
                annihilation.push(left.annihilation[index].clone());
                coefficients.push(left.coefficients[index]);
                index += 1;
            }
            Ordering::Greater => {
                creation.push(right.creation[other_index].clone());
                annihilation.push(right.annihilation[other_index].clone());
                coefficients.push(right.coefficients[other_index]);
                other_index += 1;
            }
            Ordering::Equal => {
                let value = left.coefficients[index] + right.coefficients[other_index];
                if value.re != 0.0 || value.im != 0.0 {
                    creation.push(left.creation[index].clone());
                    annihilation.push(left.annihilation[index].clone());
                    coefficients.push(value);
                }
                index += 1;
                other_index += 1;
            }
        }
    }
    while index < left.term_count() {
        creation.push(left.creation[index].clone());
        annihilation.push(left.annihilation[index].clone());
        coefficients.push(left.coefficients[index]);
        index += 1;
    }
    while other_index < right.term_count() {
        creation.push(right.creation[other_index].clone());
        annihilation.push(right.annihilation[other_index].clone());
        coefficients.push(right.coefficients[other_index]);
        other_index += 1;
    }
    Ok(NativeFermionOperatorHandle::from_result(
        left.n_modes,
        (creation, annihilation, coefficients),
    ))
}

fn scale_fermion_handle(
    input: &NativeFermionOperatorHandle,
    scalar: Complex64,
) -> Result<NativeFermionOperatorHandle, tencir_pauli_core::PauliError> {
    let mut creation = Vec::with_capacity(input.term_count());
    let mut annihilation = Vec::with_capacity(input.term_count());
    let mut coefficients = Vec::with_capacity(input.term_count());
    if scalar.re == 0.0 && scalar.im == 0.0 {
        return Ok(NativeFermionOperatorHandle::from_result(
            input.n_modes,
            (creation, annihilation, coefficients),
        ));
    }
    for index in 0..input.term_count() {
        creation.push(input.creation[index].clone());
        annihilation.push(input.annihilation[index].clone());
        coefficients.push(input.coefficients[index] * scalar);
    }
    Ok(NativeFermionOperatorHandle::from_result(
        input.n_modes,
        (creation, annihilation, coefficients),
    ))
}

fn merge_boson_handles(
    left: &NativeBosonOperatorHandle,
    right: &NativeBosonOperatorHandle,
    max_bytes: u128,
) -> Result<NativeBosonOperatorHandle, tencir_pauli_core::PauliError> {
    check_native_output(
        left.term_count() + right.term_count(),
        left.n_modes.div_ceil(64) * 16 + 32,
        max_bytes,
        "estimating boson operator addition",
    )?;
    let mut blocks = Vec::with_capacity(left.term_count() + right.term_count());
    let mut coefficients = Vec::with_capacity(left.term_count() + right.term_count());
    let mut index = 0;
    let mut other_index = 0;
    while index < left.term_count() && other_index < right.term_count() {
        match left.blocks[index].cmp(&right.blocks[other_index]) {
            Ordering::Less => {
                blocks.push(left.blocks[index].clone());
                coefficients.push(left.coefficients[index]);
                index += 1;
            }
            Ordering::Greater => {
                blocks.push(right.blocks[other_index].clone());
                coefficients.push(right.coefficients[other_index]);
                other_index += 1;
            }
            Ordering::Equal => {
                let value = left.coefficients[index] + right.coefficients[other_index];
                if value.re != 0.0 || value.im != 0.0 {
                    blocks.push(left.blocks[index].clone());
                    coefficients.push(value);
                }
                index += 1;
                other_index += 1;
            }
        }
    }
    while index < left.term_count() {
        blocks.push(left.blocks[index].clone());
        coefficients.push(left.coefficients[index]);
        index += 1;
    }
    while other_index < right.term_count() {
        blocks.push(right.blocks[other_index].clone());
        coefficients.push(right.coefficients[other_index]);
        other_index += 1;
    }
    Ok(NativeBosonOperatorHandle {
        n_modes: left.n_modes,
        blocks,
        coefficients,
    })
}

fn scale_boson_handle(
    input: &NativeBosonOperatorHandle,
    scalar: Complex64,
) -> Result<NativeBosonOperatorHandle, tencir_pauli_core::PauliError> {
    let mut blocks = Vec::with_capacity(input.term_count());
    let mut coefficients = Vec::with_capacity(input.term_count());
    if scalar.re == 0.0 && scalar.im == 0.0 {
        return Ok(NativeBosonOperatorHandle {
            n_modes: input.n_modes,
            blocks,
            coefficients,
        });
    }
    for index in 0..input.term_count() {
        blocks.push(input.blocks[index].clone());
        coefficients.push(input.coefficients[index] * scalar);
    }
    Ok(NativeBosonOperatorHandle {
        n_modes: input.n_modes,
        blocks,
        coefficients,
    })
}

fn append_hybrid_term(
    result: &mut tencir_pauli_core::HybridCanonicalResult,
    source: &NativeHybridOperatorHandle,
    index: usize,
    coefficient: Complex64,
) {
    result
        .fermion_present
        .push(source.result.fermion_present[index]);
    result
        .fermion_creation
        .push(source.result.fermion_creation[index].clone());
    result
        .fermion_annihilation
        .push(source.result.fermion_annihilation[index].clone());
    result
        .boson_present
        .push(source.result.boson_present[index]);
    result
        .boson_blocks
        .push(source.result.boson_blocks[index].clone());
    result
        .qubit_codes
        .push(source.result.qubit_codes[index].clone());
    result
        .mapped_present
        .push(source.result.mapped_present[index]);
    result
        .mapped_codes
        .push(source.result.mapped_codes[index].clone());
    result
        .qudit_present
        .push(source.result.qudit_present[index]);
    result
        .qudit_triples
        .push(source.result.qudit_triples[index].clone());
    result.coefficients.push(coefficient);
}

fn merge_hybrid_handles(
    left: &NativeHybridOperatorHandle,
    right: &NativeHybridOperatorHandle,
    max_bytes: u128,
) -> Result<NativeHybridOperatorHandle, tencir_pauli_core::PauliError> {
    check_native_output(
        left.term_count() + right.term_count(),
        128,
        max_bytes,
        "estimating hybrid operator addition",
    )?;
    let mut result = empty_hybrid_result(left.term_count() + right.term_count());
    let mut index = 0;
    let mut other_index = 0;
    while index < left.term_count() && other_index < right.term_count() {
        match left.key_at(index).cmp(&right.key_at(other_index)) {
            Ordering::Less => {
                append_hybrid_term(&mut result, left, index, left.result.coefficients[index]);
                index += 1;
            }
            Ordering::Greater => {
                append_hybrid_term(
                    &mut result,
                    right,
                    other_index,
                    right.result.coefficients[other_index],
                );
                other_index += 1;
            }
            Ordering::Equal => {
                let value =
                    left.result.coefficients[index] + right.result.coefficients[other_index];
                if value.re != 0.0 || value.im != 0.0 {
                    append_hybrid_term(&mut result, left, index, value);
                }
                index += 1;
                other_index += 1;
            }
        }
    }
    while index < left.term_count() {
        append_hybrid_term(&mut result, left, index, left.result.coefficients[index]);
        index += 1;
    }
    while other_index < right.term_count() {
        append_hybrid_term(
            &mut result,
            right,
            other_index,
            right.result.coefficients[other_index],
        );
        other_index += 1;
    }
    Ok(NativeHybridOperatorHandle::from_result(left.layout, result))
}

fn scale_hybrid_handle(
    input: &NativeHybridOperatorHandle,
    scalar: Complex64,
) -> Result<NativeHybridOperatorHandle, tencir_pauli_core::PauliError> {
    let mut result = empty_hybrid_result(input.term_count());
    if scalar.re == 0.0 && scalar.im == 0.0 {
        return Ok(NativeHybridOperatorHandle::from_result(
            input.layout,
            result,
        ));
    }
    for index in 0..input.term_count() {
        append_hybrid_term(
            &mut result,
            input,
            index,
            input.result.coefficients[index] * scalar,
        );
    }
    Ok(NativeHybridOperatorHandle::from_result(
        input.layout,
        result,
    ))
}

fn hybrid_to_pauli(
    layout: &HybridLayout,
    result: &tencir_pauli_core::HybridCanonicalResult,
    axes: &[(u8, usize)],
    max_bytes: u128,
) -> Result<tencir_pauli_core::PauliOperator, tencir_pauli_core::PauliError> {
    if layout.n_bosons != 0 || layout.n_qudit_sites != 0 {
        return Err(tencir_pauli_core::PauliError::InvalidSector {
            context: "hybrid-to-Pauli projection requires a pure fermion/qubit space",
        });
    }
    check_native_output(
        result.coefficients.len(),
        axes.len().div_ceil(64) * 16 + 16,
        max_bytes,
        "estimating hybrid-to-Pauli projection",
    )?;
    let mut structures = Vec::with_capacity(result.coefficients.len());
    for term_index in 0..result.coefficients.len() {
        if result.fermion_present[term_index] {
            return Err(tencir_pauli_core::PauliError::InvalidSector {
                context: "hybrid-to-Pauli projection received raw fermion terms",
            });
        }
        let mut codes = Vec::with_capacity(axes.len());
        for &(domain, index) in axes {
            let code = match domain {
                0 => {
                    if index >= layout.n_modes {
                        return Err(tencir_pauli_core::PauliError::InvalidIndex {
                            context: "hybrid fermion axis",
                        });
                    }
                    if result.mapped_present[term_index] {
                        result.mapped_codes[term_index][index]
                    } else {
                        0
                    }
                }
                1 => {
                    if index >= layout.nqubits {
                        return Err(tencir_pauli_core::PauliError::InvalidIndex {
                            context: "hybrid qubit axis",
                        });
                    }
                    result.qubit_codes[term_index][index]
                }
                _ => {
                    return Err(tencir_pauli_core::PauliError::InvalidSector {
                        context: "hybrid-to-Pauli projection received a non-qubit axis",
                    })
                }
            };
            codes.push(code);
        }
        structures.push(codes);
    }
    tencir_pauli_core::PauliOperator::from_terms(axes.len(), &structures, &result.coefficients)
}

type HybridNativeKey = (
    Option<(Vec<u32>, Vec<u32>)>,
    Option<Vec<(u32, u32, u32)>>,
    Vec<u8>,
    Option<Vec<u8>>,
    Option<Vec<(u32, u32, u32)>>,
);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeHybridOperatorHandle {
    layout: HybridLayout,
    result: tencir_pauli_core::HybridCanonicalResult,
}

impl NativeHybridOperatorHandle {
    pub(crate) fn from_result(
        layout: HybridLayout,
        mut result: tencir_pauli_core::HybridCanonicalResult,
    ) -> Self {
        for (present, codes) in result
            .mapped_present
            .iter()
            .copied()
            .zip(&mut result.mapped_codes)
        {
            if !present {
                *codes = vec![0; layout.n_modes];
            }
        }
        Self { layout, result }
    }

    pub(crate) fn batch(&self) -> HybridBatch<'_> {
        HybridBatch {
            fermion_present: &self.result.fermion_present,
            fermion_creation: &self.result.fermion_creation,
            fermion_annihilation: &self.result.fermion_annihilation,
            boson_present: &self.result.boson_present,
            boson_blocks: &self.result.boson_blocks,
            qubit_codes: &self.result.qubit_codes,
            mapped_present: &self.result.mapped_present,
            mapped_codes: &self.result.mapped_codes,
            qudit_present: &self.result.qudit_present,
            qudit_triples: &self.result.qudit_triples,
            coefficients: &self.result.coefficients,
        }
    }

    pub(crate) fn layout(&self) -> HybridLayout {
        self.layout
    }

    fn key_at(&self, index: usize) -> HybridNativeKey {
        (
            self.result.fermion_present[index].then(|| {
                (
                    self.result.fermion_creation[index].clone(),
                    self.result.fermion_annihilation[index].clone(),
                )
            }),
            self.result.boson_present[index].then(|| self.result.boson_blocks[index].clone()),
            self.result.qubit_codes[index].clone(),
            self.result.mapped_present[index].then(|| self.result.mapped_codes[index].clone()),
            self.result.qudit_present[index].then(|| self.result.qudit_triples[index].clone()),
        )
    }

    fn flat_parts(&self) -> HybridFlatParts {
        let (fermion_creation, fermion_creation_offsets) =
            flatten_words(&self.result.fermion_creation);
        let (fermion_annihilation, fermion_annihilation_offsets) =
            flatten_words(&self.result.fermion_annihilation);
        let (boson_blocks, boson_offsets) = flatten_blocks(&self.result.boson_blocks);
        let (qudit_triples, qudit_offsets) = flatten_blocks(&self.result.qudit_triples);
        let mut flags = Vec::with_capacity(self.term_count() * 4);
        for index in 0..self.term_count() {
            flags.extend([
                u8::from(self.result.fermion_present[index]),
                u8::from(self.result.boson_present[index]),
                u8::from(self.result.mapped_present[index]),
                u8::from(self.result.qudit_present[index]),
            ]);
        }
        let mut qubit_codes = Vec::new();
        let mut mapped_codes = Vec::new();
        for index in 0..self.term_count() {
            qubit_codes.extend_from_slice(&self.result.qubit_codes[index]);
            mapped_codes.extend_from_slice(&self.result.mapped_codes[index]);
        }
        (
            flags,
            fermion_creation,
            fermion_creation_offsets,
            fermion_annihilation,
            fermion_annihilation_offsets,
            boson_blocks,
            boson_offsets,
            qubit_codes,
            mapped_codes,
            qudit_triples,
            qudit_offsets,
            self.result.coefficients.clone(),
        )
    }

    fn has_raw_fermions_inner(&self) -> bool {
        self.result.fermion_present.iter().any(|&present| present)
    }

    fn has_mapped_fermions_inner(&self) -> bool {
        self.result.mapped_present.iter().any(|&present| present)
    }

    fn has_mixed_fermion_roles_inner(&self) -> bool {
        self.result
            .fermion_present
            .iter()
            .zip(&self.result.mapped_present)
            .any(|(&raw, &mapped)| raw && mapped)
    }

    fn adjoint_result(&self) -> Result<Self, String> {
        if self.layout.n_qudit_sites != 0 && self.layout.qudit_dimension < 3 {
            return Err("invalid qudit dimension".to_owned());
        }
        let dimension = self.layout.qudit_dimension;
        let mut result = self.result.clone();
        let mut coefficients = Vec::with_capacity(self.term_count());
        for index in 0..self.term_count() {
            if self.result.fermion_present[index] && self.result.mapped_present[index] {
                return Err(
                    "cannot take the adjoint of a term containing both raw and mapped fermion factors"
                        .to_owned(),
                );
            }
            if self.result.fermion_present[index] {
                result.fermion_creation[index] = self.result.fermion_annihilation[index]
                    .iter()
                    .rev()
                    .copied()
                    .collect();
                result.fermion_annihilation[index] = self.result.fermion_creation[index]
                    .iter()
                    .rev()
                    .copied()
                    .collect();
            }
            if self.result.boson_present[index] {
                for block in &mut result.boson_blocks[index] {
                    std::mem::swap(&mut block.1, &mut block.2);
                }
            }
            let mut phase = Complex64::new(1.0, 0.0);
            if self.result.qudit_present[index] {
                let exponent: u128 = self.result.qudit_triples[index]
                    .iter()
                    .map(|&(_, a, b)| u128::from(a) * u128::from(b))
                    .sum::<u128>()
                    % dimension as u128;
                for triple in &mut result.qudit_triples[index] {
                    triple.1 = (dimension as u32 - triple.1) % dimension as u32;
                    triple.2 = (dimension as u32 - triple.2) % dimension as u32;
                }
                let angle = 2.0 * std::f64::consts::PI * exponent as f64 / dimension as f64;
                phase = Complex64::from_polar(1.0, angle);
            }
            coefficients.push(self.result.coefficients[index].conj() * phase);
        }
        result.coefficients = coefficients;
        Ok(Self::from_result(self.layout, result))
    }
}

#[pymethods]
impl NativeHybridOperatorHandle {
    #[getter]
    fn term_count(&self) -> usize {
        self.result.coefficients.len()
    }

    fn has_raw_fermions(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.has_raw_fermions_inner())
    }

    fn has_mapped_fermions(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.has_mapped_fermions_inner())
    }

    fn has_mixed_fermion_roles(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.has_mixed_fermion_roles_inner())
    }

    fn add(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.layout != other.layout {
            return Err(PyValueError::new_err("hybrid handle layouts differ"));
        }
        py.allow_threads(|| merge_hybrid_handles(self, other, max_bytes as u128))
            .map_err(map_error)
    }

    fn scale(&self, py: Python<'_>, scalar_re: f64, scalar_im: f64) -> PyResult<Self> {
        py.allow_threads(|| scale_hybrid_handle(self, Complex64::new(scalar_re, scalar_im)))
            .map_err(map_error)
    }

    fn multiply(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.layout != other.layout {
            return Err(PyValueError::new_err("hybrid handle layouts differ"));
        }
        let result = py
            .allow_threads(|| {
                multiply_hybrid_terms(self.layout, self.batch(), other.batch(), max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.layout, result))
    }

    fn commutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.layout != other.layout {
            return Err(PyValueError::new_err("hybrid handle layouts differ"));
        }
        let result = py
            .allow_threads(|| {
                binary_hybrid_terms(
                    self.layout,
                    self.batch(),
                    other.batch(),
                    max_bytes as u128,
                    1,
                    -1,
                )
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.layout, result))
    }

    fn anticommutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.layout != other.layout {
            return Err(PyValueError::new_err("hybrid handle layouts differ"));
        }
        let result = py
            .allow_threads(|| {
                binary_hybrid_terms(
                    self.layout,
                    self.batch(),
                    other.batch(),
                    max_bytes as u128,
                    1,
                    1,
                )
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.layout, result))
    }

    fn jordan_wigner(&self, py: Python<'_>, max_bytes: usize) -> PyResult<Self> {
        if self.has_mixed_fermion_roles_inner() {
            return Err(PyValueError::new_err(
                "cannot map a term containing both raw and mapped fermion factors",
            ));
        }
        let result = py
            .allow_threads(|| {
                jordan_wigner_hybrid_terms(self.layout, self.batch(), max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.layout, result))
    }

    fn adjoint(&self, py: Python<'_>) -> PyResult<Self> {
        py.allow_threads(|| self.adjoint_result())
            .map_err(PyValueError::new_err)
    }

    fn is_hermitian(&self, py: Python<'_>, tolerance: f64) -> PyResult<bool> {
        py.allow_threads(|| {
            let adjoint = self.adjoint_result()?;
            Ok::<bool, String>(same_hybrid_coefficients(self, &adjoint, tolerance))
        })
        .map_err(PyValueError::new_err)
    }

    fn materialize<'py>(&self, py: Python<'py>) -> NumpyHybridOutput<'py> {
        let parts = py.allow_threads(|| self.flat_parts());
        (
            self.term_count(),
            self.layout.nqubits,
            (
                PyArray1::from_vec(py, parts.0),
                PyArray1::from_vec(py, parts.1),
                PyArray1::from_vec(py, parts.2),
                PyArray1::from_vec(py, parts.3),
                PyArray1::from_vec(py, parts.4),
                PyArray1::from_vec(py, parts.5),
                PyArray1::from_vec(py, parts.6),
            ),
            (
                PyArray1::from_vec(py, parts.7),
                PyArray1::from_vec(py, parts.8),
                PyArray1::from_vec(py, parts.9),
                PyArray1::from_vec(py, parts.10),
                PyArray1::from_vec(py, parts.11),
            ),
        )
    }

    fn to_pauli(
        &self,
        py: Python<'_>,
        axes: Vec<(u8, usize)>,
        max_bytes: usize,
    ) -> PyResult<NativePauliOperatorHandle> {
        let operator = py
            .allow_threads(|| hybrid_to_pauli(&self.layout, &self.result, &axes, max_bytes as u128))
            .map_err(map_error)?;
        Ok(NativePauliOperatorHandle::from_operator(operator))
    }
}

#[pymethods]
impl NativeBosonOperatorHandle {
    #[getter]
    fn n_modes(&self) -> usize {
        self.n_modes
    }

    #[getter]
    fn term_count(&self) -> usize {
        self.coefficients.len()
    }

    fn add(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.n_modes != other.n_modes {
            return Err(PyValueError::new_err("boson handle mode counts differ"));
        }
        py.allow_threads(|| merge_boson_handles(self, other, max_bytes as u128))
            .map_err(map_error)
    }

    fn scale(&self, py: Python<'_>, scalar_re: f64, scalar_im: f64) -> PyResult<Self> {
        py.allow_threads(|| scale_boson_handle(self, Complex64::new(scalar_re, scalar_im)))
            .map_err(map_error)
    }

    fn multiply(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.n_modes != other.n_modes {
            return Err(PyValueError::new_err("boson handle mode counts differ"));
        }
        let result = py
            .allow_threads(|| {
                multiply_boson_terms(
                    self.n_modes,
                    &self.blocks,
                    &self.coefficients,
                    &other.blocks,
                    &other.coefficients,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.n_modes, result))
    }

    fn commutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.n_modes != other.n_modes {
            return Err(PyValueError::new_err("boson handle mode counts differ"));
        }
        let result = py
            .allow_threads(|| {
                binary_boson_terms(
                    self.n_modes,
                    &self.blocks,
                    &self.coefficients,
                    &other.blocks,
                    &other.coefficients,
                    max_bytes as u128,
                    -1,
                )
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.n_modes, result))
    }

    fn anticommutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.n_modes != other.n_modes {
            return Err(PyValueError::new_err("boson handle mode counts differ"));
        }
        let result = py
            .allow_threads(|| {
                binary_boson_terms(
                    self.n_modes,
                    &self.blocks,
                    &self.coefficients,
                    &other.blocks,
                    &other.coefficients,
                    max_bytes as u128,
                    1,
                )
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.n_modes, result))
    }

    fn adjoint(&self, py: Python<'_>) -> Self {
        py.allow_threads(|| boson_adjoint(self))
    }

    fn is_hermitian(&self, py: Python<'_>, tolerance: f64) -> bool {
        py.allow_threads(|| {
            let adjoint = boson_adjoint(self);
            let left = self
                .blocks
                .iter()
                .zip(&self.coefficients)
                .map(|(key, value)| (key.clone(), *value))
                .collect::<BTreeMap<_, _>>();
            let right = adjoint
                .blocks
                .iter()
                .zip(&adjoint.coefficients)
                .map(|(key, value)| (key.clone(), *value))
                .collect::<BTreeMap<_, _>>();
            same_coefficient_maps(&left, &right, tolerance)
        })
    }

    fn materialize<'py>(&self, py: Python<'py>) -> NumpyBosonOutput<'py> {
        let (blocks, coefficients) =
            py.allow_threads(|| (flatten_blocks(&self.blocks), self.coefficients.clone()));
        (
            self.term_count(),
            PyArray1::from_vec(py, blocks.0),
            PyArray1::from_vec(py, blocks.1),
            PyArray1::from_vec(py, coefficients),
        )
    }

    fn to_hybrid(&self, py: Python<'_>) -> NativeHybridOperatorHandle {
        py.allow_threads(|| self.to_hybrid_result())
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct StructuredMvpPlan {
    plan: CoreStructuredMvpPlan,
}

#[pymethods]
impl StructuredMvpPlan {
    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan.estimated_bytes()
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        if state_slice.len() != self.dimension() {
            return Err(PyValueError::new_err(format!(
                "state must have shape ({},), got ({},)",
                self.dimension(),
                state_slice.len()
            )));
        }
        let output = py
            .allow_threads(|| self.plan.apply(state_slice, max_bytes))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, output))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        mut output: PyReadwriteArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<()> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output_slice = output
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output must be C-contiguous"))?;
        py.allow_threads(|| self.plan.apply_into(state_slice, output_slice, max_bytes))
            .map_err(map_error)
    }
}

#[pyfunction]
pub(crate) fn structured_fermion_canonicalize(
    py: Python<'_>,
    n_modes: usize,
    factors: Vec<Vec<(usize, u8)>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<NativeFermionOperatorHandle> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let result = py
        .allow_threads(|| canonicalize_fermion_terms(n_modes, &factors, &coefficients, max_bytes))
        .map_err(map_error)?;
    Ok(NativeFermionOperatorHandle::from_result(n_modes, result))
}

#[pyfunction]
pub(crate) fn structured_fermion_multiply(
    py: Python<'_>,
    n_modes: usize,
    left: FermionInput,
    right: FermionInput,
    max_bytes: u128,
) -> PyResult<NativeFermionOperatorHandle> {
    let (left_creation, left_annihilation, left_re, left_im) = left;
    let (right_creation, right_annihilation, right_re, right_im) = right;
    let left_coefficients = complex_coefficients(left_re, left_im)?;
    let right_coefficients = complex_coefficients(right_re, right_im)?;
    let result = py
        .allow_threads(|| {
            multiply_fermion_terms(
                n_modes,
                FermionBatch {
                    creation: &left_creation,
                    annihilation: &left_annihilation,
                    coefficients: &left_coefficients,
                },
                FermionBatch {
                    creation: &right_creation,
                    annihilation: &right_annihilation,
                    coefficients: &right_coefficients,
                },
                max_bytes,
            )
        })
        .map_err(map_error)?;
    Ok(NativeFermionOperatorHandle::from_result(n_modes, result))
}

#[pyfunction]
pub(crate) fn structured_fermion_jordan_wigner(
    py: Python<'_>,
    n_modes: usize,
    creation: Vec<Vec<u32>>,
    annihilation: Vec<Vec<u32>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<JordanWignerOutput> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let result = py
        .allow_threads(|| {
            jordan_wigner_terms(n_modes, &creation, &annihilation, &coefficients, max_bytes)
        })
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&result.1);
    Ok((result.0, real, imaginary))
}

#[pyfunction]
pub(crate) fn structured_boson_canonicalize(
    py: Python<'_>,
    n_modes: usize,
    factors: Vec<Vec<(usize, u8)>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<NativeBosonOperatorHandle> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let result = py
        .allow_threads(|| canonicalize_boson_terms(n_modes, &factors, &coefficients, max_bytes))
        .map_err(map_error)?;
    Ok(NativeBosonOperatorHandle::from_result(n_modes, result))
}

#[pyfunction]
pub(crate) fn structured_boson_multiply(
    py: Python<'_>,
    n_modes: usize,
    left: BosonInput,
    right: BosonInput,
    max_bytes: u128,
) -> PyResult<NativeBosonOperatorHandle> {
    let (left_blocks, left_re, left_im) = left;
    let (right_blocks, right_re, right_im) = right;
    let left_coefficients = complex_coefficients(left_re, left_im)?;
    let right_coefficients = complex_coefficients(right_re, right_im)?;
    let result = py
        .allow_threads(|| {
            multiply_boson_terms(
                n_modes,
                &left_blocks,
                &left_coefficients,
                &right_blocks,
                &right_coefficients,
                max_bytes,
            )
        })
        .map_err(map_error)?;
    Ok(NativeBosonOperatorHandle::from_result(n_modes, result))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_hybrid_multiply(
    py: Python<'_>,
    n_modes: usize,
    n_bosons: usize,
    nqubits: usize,
    n_qudit_sites: usize,
    qudit_dimension: usize,
    left: HybridInput,
    right: HybridInput,
    max_bytes: u128,
) -> PyResult<NativeHybridOperatorHandle> {
    let (
        left_fermion_present,
        left_fermion_creation,
        left_fermion_annihilation,
        left_boson_present,
        left_boson_blocks,
        left_qubit_codes,
        left_mapped_present,
        left_mapped_codes,
        left_qudit_present,
        left_qudit_triples,
        left_re,
        left_im,
    ) = left;
    let (
        right_fermion_present,
        right_fermion_creation,
        right_fermion_annihilation,
        right_boson_present,
        right_boson_blocks,
        right_qubit_codes,
        right_mapped_present,
        right_mapped_codes,
        right_qudit_present,
        right_qudit_triples,
        right_re,
        right_im,
    ) = right;
    let left_coefficients = complex_coefficients(left_re, left_im)?;
    let right_coefficients = complex_coefficients(right_re, right_im)?;
    let layout = HybridLayout {
        n_modes,
        n_bosons,
        nqubits,
        n_qudit_sites,
        qudit_dimension,
    };
    let result = py
        .allow_threads(|| {
            multiply_hybrid_terms(
                layout,
                HybridBatch {
                    fermion_present: &left_fermion_present,
                    fermion_creation: &left_fermion_creation,
                    fermion_annihilation: &left_fermion_annihilation,
                    boson_present: &left_boson_present,
                    boson_blocks: &left_boson_blocks,
                    qubit_codes: &left_qubit_codes,
                    mapped_present: &left_mapped_present,
                    mapped_codes: &left_mapped_codes,
                    qudit_present: &left_qudit_present,
                    qudit_triples: &left_qudit_triples,
                    coefficients: &left_coefficients,
                },
                HybridBatch {
                    fermion_present: &right_fermion_present,
                    fermion_creation: &right_fermion_creation,
                    fermion_annihilation: &right_fermion_annihilation,
                    boson_present: &right_boson_present,
                    boson_blocks: &right_boson_blocks,
                    qubit_codes: &right_qubit_codes,
                    mapped_present: &right_mapped_present,
                    mapped_codes: &right_mapped_codes,
                    qudit_present: &right_qudit_present,
                    qudit_triples: &right_qudit_triples,
                    coefficients: &right_coefficients,
                },
                max_bytes,
            )
        })
        .map_err(map_error)?;
    Ok(NativeHybridOperatorHandle::from_result(layout, result))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_hybrid_canonicalize(
    py: Python<'_>,
    n_modes: usize,
    n_bosons: usize,
    nqubits: usize,
    n_qudit_sites: usize,
    qudit_dimension: usize,
    input: HybridRawInput,
    max_bytes: u128,
) -> PyResult<NativeHybridOperatorHandle> {
    let (
        fermion_factors,
        boson_factors,
        qubit_codes,
        qudit_present,
        qudit_triples,
        coefficients_re,
        coefficients_im,
    ) = input;
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let layout = HybridLayout {
        n_modes,
        n_bosons,
        nqubits,
        n_qudit_sites,
        qudit_dimension,
    };
    let result = py
        .allow_threads(|| {
            canonicalize_hybrid_terms(
                layout,
                HybridRawBatch {
                    fermion_factors: &fermion_factors,
                    boson_factors: &boson_factors,
                    qubit_codes: &qubit_codes,
                    qudit_present: &qudit_present,
                    qudit_triples: &qudit_triples,
                    coefficients: &coefficients,
                },
                max_bytes,
            )
        })
        .map_err(map_error)?;
    Ok(NativeHybridOperatorHandle::from_result(layout, result))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_hybrid_jordan_wigner(
    py: Python<'_>,
    n_modes: usize,
    n_bosons: usize,
    nqubits: usize,
    n_qudit_sites: usize,
    qudit_dimension: usize,
    input: HybridInput,
    max_bytes: u128,
) -> PyResult<NativeHybridOperatorHandle> {
    let (
        fermion_present,
        fermion_creation,
        fermion_annihilation,
        boson_present,
        boson_blocks,
        qubit_codes,
        mapped_present,
        mapped_codes,
        qudit_present,
        qudit_triples,
        coefficients_re,
        coefficients_im,
    ) = input;
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let layout = HybridLayout {
        n_modes,
        n_bosons,
        nqubits,
        n_qudit_sites,
        qudit_dimension,
    };
    let result = py
        .allow_threads(|| {
            jordan_wigner_hybrid_terms(
                layout,
                HybridBatch {
                    fermion_present: &fermion_present,
                    fermion_creation: &fermion_creation,
                    fermion_annihilation: &fermion_annihilation,
                    boson_present: &boson_present,
                    boson_blocks: &boson_blocks,
                    qubit_codes: &qubit_codes,
                    mapped_present: &mapped_present,
                    mapped_codes: &mapped_codes,
                    qudit_present: &qudit_present,
                    qudit_triples: &qudit_triples,
                    coefficients: &coefficients,
                },
                max_bytes,
            )
        })
        .map_err(map_error)?;
    Ok(NativeHybridOperatorHandle::from_result(layout, result))
}

#[pyfunction]
pub(crate) fn structured_dense(
    py: Python<'_>,
    local_dimensions: Vec<usize>,
    operations: Vec<Vec<(usize, u8, u32, u32)>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<(usize, Bound<'_, PyArray1<NumpyComplex128>>)> {
    if operations.len() != coefficients_re.len() {
        return Err(PyValueError::new_err(
            "operation and coefficient lengths differ",
        ));
    }
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let operations = operations
        .into_iter()
        .map(|term| {
            term.into_iter()
                .map(|(axis, kind, p, q)| StructuredOperation { axis, kind, p, q })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (dimension, values) = py
        .allow_threads(|| {
            structured_dense_matrix(&local_dimensions, &operations, &coefficients, max_bytes)
        })
        .map_err(map_error)?;
    Ok((dimension, PyArray1::from_vec(py, values)))
}

#[pyfunction]
pub(crate) fn structured_sparse(
    py: Python<'_>,
    local_dimensions: Vec<usize>,
    operations: Vec<Vec<(usize, u8, u32, u32)>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<StructuredSparseOutput> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let operations = operations
        .into_iter()
        .map(|term| {
            term.into_iter()
                .map(|(axis, kind, p, q)| StructuredOperation { axis, kind, p, q })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let result = py
        .allow_threads(|| {
            structured_sparse_matrix(&local_dimensions, &operations, &coefficients, max_bytes)
        })
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&result.values);
    Ok((
        result.dimension,
        result.rows,
        result.columns,
        real,
        imaginary,
    ))
}

#[pyfunction]
pub(crate) fn structured_sparse_plan(
    py: Python<'_>,
    local_dimensions: Vec<usize>,
    operations: Vec<Vec<(usize, u8, u32, u32)>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<StructuredMvpPlan> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let operations = operations
        .into_iter()
        .map(|term| {
            term.into_iter()
                .map(|(axis, kind, p, q)| StructuredOperation { axis, kind, p, q })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let plan = py
        .allow_threads(|| {
            structured_mvp_plan(local_dimensions, operations, coefficients, max_bytes)
        })
        .map_err(map_error)?;
    Ok(StructuredMvpPlan { plan })
}

fn lower_hybrid_operations(
    result: &tencir_pauli_core::HybridCanonicalResult,
    axes: &[(u8, usize)],
) -> Result<Vec<Vec<StructuredOperation>>, tencir_pauli_core::PauliError> {
    let count = result.coefficients.len();
    let mut operations = Vec::with_capacity(count);
    for term_index in 0..count {
        if result.fermion_present[term_index] {
            return Err(tencir_pauli_core::PauliError::InvalidSector {
                context: "finite compilation requires mapped fermion factors",
            });
        }
        let boson_blocks = &result.boson_blocks[term_index];
        let qudit_triples = &result.qudit_triples[term_index];
        let mut term_operations = Vec::new();
        for (position, &(domain, index)) in axes.iter().enumerate() {
            match domain {
                0 => {
                    if result.mapped_present[term_index] {
                        let code = *result.mapped_codes[term_index].get(index).ok_or(
                            tencir_pauli_core::PauliError::InvalidIndex {
                                context: "mapped fermion compilation axis",
                            },
                        )?;
                        if code != 0 {
                            term_operations.push(StructuredOperation {
                                axis: position,
                                kind: 0,
                                p: u32::from(code),
                                q: 0,
                            });
                        }
                    }
                }
                1 => {
                    if let Some((creation, annihilation)) = boson_blocks
                        .iter()
                        .find(|&&(mode, _, _)| mode as usize == index)
                        .map(|&(_, creation, annihilation)| (creation, annihilation))
                    {
                        term_operations.push(StructuredOperation {
                            axis: position,
                            kind: 1,
                            p: creation,
                            q: annihilation,
                        });
                    }
                }
                2 => {
                    let code = *result.qubit_codes[term_index].get(index).ok_or(
                        tencir_pauli_core::PauliError::InvalidIndex {
                            context: "qubit compilation axis",
                        },
                    )?;
                    if code != 0 {
                        term_operations.push(StructuredOperation {
                            axis: position,
                            kind: 0,
                            p: u32::from(code),
                            q: 0,
                        });
                    }
                }
                3 => {
                    if let Some((a, b)) = qudit_triples
                        .iter()
                        .find(|&&(site, _, _)| site as usize == index)
                        .map(|&(_, a, b)| (a, b))
                    {
                        term_operations.push(StructuredOperation {
                            axis: position,
                            kind: 2,
                            p: a,
                            q: b,
                        });
                    }
                }
                _ => {
                    return Err(tencir_pauli_core::PauliError::InvalidCode {
                        code: domain,
                        index: position,
                    })
                }
            }
        }
        operations.push(term_operations);
    }
    Ok(operations)
}

#[pyfunction]
pub(crate) fn structured_dense_handle<'py>(
    py: Python<'py>,
    handle: &NativeHybridOperatorHandle,
    local_dimensions: Vec<usize>,
    axes: Vec<(u8, usize)>,
    max_bytes: u128,
) -> PyResult<(usize, Bound<'py, PyArray1<NumpyComplex128>>)> {
    let (dimension, values) = py
        .allow_threads(|| {
            let operations = lower_hybrid_operations(&handle.result, &axes)?;
            let coefficients = handle.result.coefficients.clone();
            structured_dense_matrix(&local_dimensions, &operations, &coefficients, max_bytes)
        })
        .map_err(map_error)?;
    Ok((dimension, PyArray1::from_vec(py, values)))
}

#[pyfunction]
pub(crate) fn structured_sparse_handle(
    py: Python<'_>,
    handle: &NativeHybridOperatorHandle,
    local_dimensions: Vec<usize>,
    axes: Vec<(u8, usize)>,
    max_bytes: u128,
) -> PyResult<StructuredSparseOutput> {
    let result = py
        .allow_threads(|| {
            let operations = lower_hybrid_operations(&handle.result, &axes)?;
            let coefficients = handle.result.coefficients.clone();
            structured_sparse_matrix(&local_dimensions, &operations, &coefficients, max_bytes)
        })
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&result.values);
    Ok((
        result.dimension,
        result.rows,
        result.columns,
        real,
        imaginary,
    ))
}

#[pyfunction]
pub(crate) fn structured_sparse_plan_handle(
    py: Python<'_>,
    handle: &NativeHybridOperatorHandle,
    local_dimensions: Vec<usize>,
    axes: Vec<(u8, usize)>,
    max_bytes: u128,
) -> PyResult<StructuredMvpPlan> {
    let plan = py
        .allow_threads(|| {
            let operations = lower_hybrid_operations(&handle.result, &axes)?;
            let coefficients = handle.result.coefficients.clone();
            structured_mvp_plan(local_dimensions, operations, coefficients, max_bytes)
        })
        .map_err(map_error)?;
    Ok(StructuredMvpPlan { plan })
}
