use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{
    canonicalize_boson_terms, canonicalize_fermion_terms, canonicalize_hybrid_terms,
    jordan_wigner_hybrid_terms, jordan_wigner_terms, multiply_boson_terms, multiply_fermion_terms,
    multiply_hybrid_terms, structured_dense_matrix, structured_mvp_plan, structured_sparse_matrix,
    FermionBatch, HybridBatch, HybridLayout, HybridRawBatch,
    StructuredMvpPlan as CoreStructuredMvpPlan, StructuredOperation,
};

use crate::convert::{complex_coefficients, map_error, split_complex};

type FermionOutput = (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<f64>, Vec<f64>);
type BosonOutput = (Vec<Vec<(u32, u32, u32)>>, Vec<f64>, Vec<f64>);
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
type HybridOutput = (
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
) -> PyResult<FermionOutput> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let result = py
        .allow_threads(|| canonicalize_fermion_terms(n_modes, &factors, &coefficients, max_bytes))
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&result.2);
    Ok((result.0, result.1, real, imaginary))
}

#[pyfunction]
pub(crate) fn structured_fermion_multiply(
    py: Python<'_>,
    n_modes: usize,
    left: FermionInput,
    right: FermionInput,
    max_bytes: u128,
) -> PyResult<FermionOutput> {
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
    let (real, imaginary) = split_complex(&result.2);
    Ok((result.0, result.1, real, imaginary))
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
) -> PyResult<BosonOutput> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let result = py
        .allow_threads(|| canonicalize_boson_terms(n_modes, &factors, &coefficients, max_bytes))
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&result.1);
    Ok((result.0, real, imaginary))
}

#[pyfunction]
pub(crate) fn structured_boson_multiply(
    py: Python<'_>,
    n_modes: usize,
    left: BosonInput,
    right: BosonInput,
    max_bytes: u128,
) -> PyResult<BosonOutput> {
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
    let (real, imaginary) = split_complex(&result.1);
    Ok((result.0, real, imaginary))
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
) -> PyResult<HybridOutput> {
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
    let result = py
        .allow_threads(|| {
            multiply_hybrid_terms(
                HybridLayout {
                    n_modes,
                    n_bosons,
                    nqubits,
                    n_qudit_sites,
                    qudit_dimension,
                },
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
    let (real, imaginary) = split_complex(&result.coefficients);
    Ok((
        result.fermion_present,
        result.fermion_creation,
        result.fermion_annihilation,
        result.boson_present,
        result.boson_blocks,
        result.qubit_codes,
        result.mapped_present,
        result.mapped_codes,
        result.qudit_present,
        result.qudit_triples,
        real,
        imaginary,
    ))
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
) -> PyResult<HybridOutput> {
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
    let result = py
        .allow_threads(|| {
            canonicalize_hybrid_terms(
                HybridLayout {
                    n_modes,
                    n_bosons,
                    nqubits,
                    n_qudit_sites,
                    qudit_dimension,
                },
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
    let (real, imaginary) = split_complex(&result.coefficients);
    Ok((
        result.fermion_present,
        result.fermion_creation,
        result.fermion_annihilation,
        result.boson_present,
        result.boson_blocks,
        result.qubit_codes,
        result.mapped_present,
        result.mapped_codes,
        result.qudit_present,
        result.qudit_triples,
        real,
        imaginary,
    ))
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
) -> PyResult<HybridOutput> {
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
    let result = py
        .allow_threads(|| {
            jordan_wigner_hybrid_terms(
                HybridLayout {
                    n_modes,
                    n_bosons,
                    nqubits,
                    n_qudit_sites,
                    qudit_dimension,
                },
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
    let (real, imaginary) = split_complex(&result.coefficients);
    Ok((
        result.fermion_present,
        result.fermion_creation,
        result.fermion_annihilation,
        result.boson_present,
        result.boson_blocks,
        result.qubit_codes,
        result.mapped_present,
        result.mapped_codes,
        result.qudit_present,
        result.qudit_triples,
        real,
        imaginary,
    ))
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
