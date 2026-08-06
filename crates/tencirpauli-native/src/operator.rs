use numpy::{
    Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use std::collections::BTreeSet;
use tencir_pauli_core::{Complex64, PauliOperator};

use crate::convert::{
    build_operator, code_rows, complex_coefficients, map_error, operator_codes_flat_output,
    operator_packed_flat_output, operator_strings_flat_output, phase_code, CanonicalizeBatchOutput,
    NumpyCanonicalizeBatchOutput, NumpyPauliCodesOutput, NumpyPauliPackedOutput,
    NumpyPauliStringsOutput,
};

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativePauliOperatorHandle {
    operator: PauliOperator,
}

impl NativePauliOperatorHandle {
    pub(crate) fn from_operator(operator: PauliOperator) -> Self {
        Self { operator }
    }

    pub(crate) fn core(&self) -> &PauliOperator {
        &self.operator
    }
}

#[pymethods]
impl NativePauliOperatorHandle {
    #[getter]
    fn nqubits(&self) -> usize {
        self.operator.nqubits()
    }

    #[getter]
    fn term_count(&self) -> usize {
        self.operator.terms().len()
    }

    fn distinct_x_mask_count(&self, py: Python<'_>) -> usize {
        py.allow_threads(|| {
            self.operator
                .terms()
                .iter()
                .map(|term| term.word.x_words().to_vec())
                .collect::<BTreeSet<_>>()
                .len()
        })
    }

    fn add(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        let operator = py
            .allow_threads(|| {
                self.operator
                    .add_with_limit(&other.operator, max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(Self { operator })
    }

    fn scale(&self, py: Python<'_>, scalar_re: f64, scalar_im: f64) -> PyResult<Self> {
        let operator = py
            .allow_threads(|| self.operator.scale(Complex64::new(scalar_re, scalar_im)))
            .map_err(map_error)?;
        Ok(Self { operator })
    }

    fn multiply(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        let operator = py
            .allow_threads(|| {
                self.operator
                    .multiply_with_limit(&other.operator, max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(Self { operator })
    }

    fn commutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        let operator = py
            .allow_threads(|| {
                self.operator
                    .commutator_with_limit(&other.operator, max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(Self { operator })
    }

    fn anticommutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        let operator = py
            .allow_threads(|| {
                self.operator
                    .anticommutator_with_limit(&other.operator, max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(Self { operator })
    }

    fn materialize<'py>(&self, py: Python<'py>) -> NumpyPauliPackedOutput<'py> {
        let (term_count, word_count, x_words, z_words, coefficients) =
            py.allow_threads(|| operator_packed_flat_output(&self.operator));
        (
            term_count,
            word_count,
            PyArray1::from_vec(py, x_words),
            PyArray1::from_vec(py, z_words),
            PyArray1::from_vec(py, coefficients),
        )
    }

    fn materialize_arrays<'py>(&self, py: Python<'py>) -> NumpyPauliCodesOutput<'py> {
        let (term_count, nqubits, codes, coefficients) =
            py.allow_threads(|| operator_codes_flat_output(&self.operator));
        (
            term_count,
            nqubits,
            PyArray1::from_vec(py, codes),
            PyArray1::from_vec(py, coefficients),
        )
    }

    fn materialize_strings<'py>(&self, py: Python<'py>) -> NumpyPauliStringsOutput<'py> {
        let (strings, coefficients) =
            py.allow_threads(|| operator_strings_flat_output(&self.operator));
        (strings, PyArray1::from_vec(py, coefficients))
    }

    fn adjoint(&self, py: Python<'_>) -> Self {
        let operator = py.allow_threads(|| self.operator.adjoint());
        Self { operator }
    }

    fn is_hermitian(&self, py: Python<'_>, tolerance: f64) -> PyResult<bool> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(PyValueError::new_err(
                "Hermiticity tolerance must be finite and non-negative",
            ));
        }
        Ok(py.allow_threads(|| self.operator.is_hermitian(tolerance)))
    }

    fn termwise_conserves_charge(&self, py: Python<'_>, qubit_levels: Vec<(f64, f64)>) -> bool {
        py.allow_threads(|| self.operator.termwise_conserves_charge(&qubit_levels))
    }

    fn content_eq(&self, py: Python<'_>, other: &Self) -> bool {
        py.allow_threads(|| self.operator == other.operator)
    }

    fn content_hash(&self, py: Python<'_>) -> u64 {
        py.allow_threads(|| self.operator.content_hash())
    }
}

#[pyfunction]
#[pyo3(signature = (nqubits, structures, coefficients_re, coefficients_im, max_bytes))]
pub(crate) fn pauli_operator_native(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<NativePauliOperatorHandle> {
    let operator = py
        .allow_threads(|| {
            let operator = build_operator(
                nqubits,
                &structures,
                &coefficients_re,
                &coefficients_im,
            )?;
            let word_count = nqubits
                .checked_add(63)
                .ok_or_else(|| PyOverflowError::new_err("native operator size overflow"))?
                / 64;
            let bytes_per_term = (word_count as u128)
                .checked_mul(16)
                .and_then(|value| value.checked_add(16))
                .ok_or_else(|| PyOverflowError::new_err("native operator size overflow"))?;
            let requested = (operator.terms().len() as u128)
                .checked_mul(bytes_per_term)
                .ok_or_else(|| PyOverflowError::new_err("native operator size overflow"))?;
            if requested > max_bytes as u128 {
                return Err(PyMemoryError::new_err(format!(
                    "native Pauli operator requires approximately {requested} bytes, exceeding max_bytes={max_bytes}"
                )));
            }
            Ok::<PauliOperator, PyErr>(operator)
        })?;
    Ok(NativePauliOperatorHandle { operator })
}

#[pyfunction]
pub(crate) fn pauli_operator_native_array(
    py: Python<'_>,
    nqubits: usize,
    structures: PyReadonlyArray2<'_, u8>,
    coefficients: PyReadonlyArray1<'_, NumpyComplex128>,
    max_bytes: usize,
) -> PyResult<NativePauliOperatorHandle> {
    let shape = structures.shape();
    if shape[1] != nqubits {
        return Err(PyValueError::new_err(format!(
            "expected structure width {nqubits}, got {}",
            shape[1]
        )));
    }
    if shape[0] != coefficients.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} coefficients, got {}",
            shape[0],
            coefficients.len()
        )));
    }
    let code_slice = structures
        .as_slice()
        .map_err(|_| PyValueError::new_err("structures must be C-contiguous"))?;
    let coefficient_slice = coefficients
        .as_slice()
        .map_err(|_| PyValueError::new_err("coefficients must be C-contiguous"))?;
    let operator = py.allow_threads(|| {
        let rows = code_rows(code_slice, shape[0], nqubits);
        let operator =
            PauliOperator::from_terms(nqubits, &rows, coefficient_slice).map_err(map_error)?;
        check_native_operator_limit(&operator, nqubits, max_bytes)?;
        Ok::<PauliOperator, PyErr>(operator)
    })?;
    Ok(NativePauliOperatorHandle { operator })
}

fn check_native_operator_limit(
    operator: &PauliOperator,
    nqubits: usize,
    max_bytes: usize,
) -> PyResult<()> {
    let word_count = nqubits
        .checked_add(63)
        .ok_or_else(|| PyOverflowError::new_err("native operator size overflow"))?
        / 64;
    let bytes_per_term = (word_count as u128)
        .checked_mul(16)
        .and_then(|value| value.checked_add(16))
        .ok_or_else(|| PyOverflowError::new_err("native operator size overflow"))?;
    let requested = (operator.terms().len() as u128)
        .checked_mul(bytes_per_term)
        .ok_or_else(|| PyOverflowError::new_err("native operator size overflow"))?;
    if requested > max_bytes as u128 {
        return Err(PyMemoryError::new_err(format!(
            "native Pauli operator requires approximately {requested} bytes, exceeding max_bytes={max_bytes}"
        )));
    }
    Ok(())
}

#[pyfunction]
pub(crate) fn pauli_canonicalize_batch(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
) -> PyResult<CanonicalizeBatchOutput> {
    let result = py.allow_threads(|| {
        let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
        PauliOperator::canonicalize(nqubits, &structures, &coefficients).map_err(map_error)
    })?;
    let mut result_structures = Vec::with_capacity(result.terms.len());
    let mut result_re = Vec::with_capacity(result.terms.len());
    let mut result_im = Vec::with_capacity(result.terms.len());
    for term in result.terms {
        result_structures.push(term.word.codes());
        result_re.push(term.coefficient.re);
        result_im.push(term.coefficient.im);
    }
    Ok((
        result_structures,
        result_re,
        result_im,
        result.input_to_canonical,
        result
            .phase_multipliers
            .into_iter()
            .map(phase_code)
            .collect(),
    ))
}

#[pyfunction]
pub(crate) fn pauli_canonicalize_batch_array(
    py: Python<'_>,
    nqubits: usize,
    structures: PyReadonlyArray2<'_, u8>,
    coefficients: PyReadonlyArray1<'_, NumpyComplex128>,
) -> PyResult<CanonicalizeBatchOutput> {
    let shape = structures.shape();
    if shape[1] != nqubits {
        return Err(PyValueError::new_err(format!(
            "expected structure width {nqubits}, got {}",
            shape[1]
        )));
    }
    if shape[0] != coefficients.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} coefficients, got {}",
            shape[0],
            coefficients.len()
        )));
    }
    let code_slice = structures
        .as_slice()
        .map_err(|_| PyValueError::new_err("structures must be C-contiguous"))?;
    let coefficient_slice = coefficients
        .as_slice()
        .map_err(|_| PyValueError::new_err("coefficients must be C-contiguous"))?;
    let result = py
        .allow_threads(|| {
            let rows = code_rows(code_slice, shape[0], nqubits);
            PauliOperator::canonicalize(nqubits, &rows, coefficient_slice)
        })
        .map_err(map_error)?;
    let mut result_structures = Vec::with_capacity(result.terms.len());
    let mut result_re = Vec::with_capacity(result.terms.len());
    let mut result_im = Vec::with_capacity(result.terms.len());
    for term in result.terms {
        result_structures.push(term.word.codes());
        result_re.push(term.coefficient.re);
        result_im.push(term.coefficient.im);
    }
    Ok((
        result_structures,
        result_re,
        result_im,
        result.input_to_canonical,
        result
            .phase_multipliers
            .into_iter()
            .map(phase_code)
            .collect(),
    ))
}

#[pyfunction]
pub(crate) fn pauli_canonicalize_batch_numpy<'py>(
    py: Python<'py>,
    nqubits: usize,
    structures: PyReadonlyArray2<'py, u8>,
    coefficients: PyReadonlyArray1<'py, NumpyComplex128>,
) -> PyResult<NumpyCanonicalizeBatchOutput<'py>> {
    let shape = structures.shape();
    if shape[1] != nqubits {
        return Err(PyValueError::new_err(format!(
            "expected structure width {nqubits}, got {}",
            shape[1]
        )));
    }
    if shape[0] != coefficients.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} coefficients, got {}",
            shape[0],
            coefficients.len()
        )));
    }
    let code_slice = structures
        .as_slice()
        .map_err(|_| PyValueError::new_err("structures must be C-contiguous"))?;
    let coefficient_slice = coefficients
        .as_slice()
        .map_err(|_| PyValueError::new_err("coefficients must be C-contiguous"))?;
    let result = py
        .allow_threads(|| {
            let rows = code_rows(code_slice, shape[0], nqubits);
            PauliOperator::canonicalize(nqubits, &rows, coefficient_slice)
        })
        .map_err(map_error)?;
    let canonical_count = result.terms.len();
    let mut canonical_codes = Vec::with_capacity(canonical_count.saturating_mul(nqubits));
    let mut canonical_coefficients = Vec::with_capacity(canonical_count);
    for term in result.terms {
        canonical_codes.extend(term.word.codes());
        canonical_coefficients.push(term.coefficient);
    }
    Ok((
        canonical_count,
        PyArray1::from_vec(py, canonical_codes),
        PyArray1::from_vec(py, canonical_coefficients),
        PyArray1::from_vec(py, result.input_to_canonical),
        PyArray1::from_vec(
            py,
            result
                .phase_multipliers
                .into_iter()
                .map(phase_code)
                .collect(),
        ),
    ))
}
