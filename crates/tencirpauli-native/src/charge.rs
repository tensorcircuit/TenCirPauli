use numpy::{
    Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use tencir_pauli_core::{
    compile_charge_transitions, ChargeTransitionLayout, ChargeTransitionTerm, Complex64,
};

use crate::convert::{map_error, split_complex};

type ChargeTransitionOutput = (Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>);

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub(crate) fn charge_compile_transitions<'py>(
    py: Python<'py>,
    dimension: usize,
    basis: PyReadonlyArray2<'py, u64>,
    local_dimensions: Vec<u64>,
    fermion_positions: Vec<u64>,
    boson_positions: Vec<u64>,
    qubit_positions: Vec<u64>,
    qudit_positions: Vec<u64>,
    fermion_creation: Vec<Vec<u32>>,
    fermion_annihilation: Vec<Vec<u32>>,
    boson_blocks: Vec<Vec<(u32, u32, u32)>>,
    qubit_codes: Vec<Vec<u8>>,
    mapped_present: Vec<bool>,
    mapped_codes: Vec<Vec<u8>>,
    qudit_present: Vec<bool>,
    qudit_triples: Vec<Vec<(u32, u32, u32)>>,
    coefficients: PyReadonlyArray1<'py, NumpyComplex128>,
    qudit_dimension: u64,
    max_bytes: usize,
) -> PyResult<ChargeTransitionOutput> {
    let basis_shape = basis.shape();
    if basis_shape.len() != 2 || basis_shape[0] != dimension {
        return Err(PyValueError::new_err(
            "charge basis must have shape (dimension, axis_count)",
        ));
    }
    let basis_values = basis
        .as_slice()
        .map_err(|_| PyValueError::new_err("charge basis must be C-contiguous"))?;
    let coefficient_values = coefficients
        .as_slice()
        .map_err(|_| PyValueError::new_err("charge coefficients must be C-contiguous"))?;
    let term_count = coefficient_values.len();
    if fermion_creation.len() != term_count
        || fermion_annihilation.len() != term_count
        || boson_blocks.len() != term_count
        || qubit_codes.len() != term_count
        || mapped_present.len() != term_count
        || mapped_codes.len() != term_count
        || qudit_present.len() != term_count
        || qudit_triples.len() != term_count
    {
        return Err(PyValueError::new_err(
            "charge transition term arrays have inconsistent lengths",
        ));
    }
    let terms = coefficient_values
        .iter()
        .enumerate()
        .map(|(index, coefficient)| ChargeTransitionTerm {
            fermion_creation: fermion_creation[index].clone(),
            fermion_annihilation: fermion_annihilation[index].clone(),
            boson_blocks: boson_blocks[index].clone(),
            qubit_codes: qubit_codes[index].clone(),
            mapped_present: mapped_present[index],
            mapped_codes: mapped_codes[index].clone(),
            qudit_present: qudit_present[index],
            qudit_triples: qudit_triples[index].clone(),
            coefficient: Complex64::new(coefficient.re, coefficient.im),
        })
        .collect::<Vec<_>>();
    let result = py
        .allow_threads(|| {
            compile_charge_transitions(
                ChargeTransitionLayout {
                    dimension,
                    basis: basis_values,
                    local_dimensions: &local_dimensions,
                    fermion_positions: &fermion_positions,
                    boson_positions: &boson_positions,
                    qubit_positions: &qubit_positions,
                    qudit_positions: &qudit_positions,
                    qudit_dimension,
                    max_bytes: max_bytes as u128,
                },
                &terms,
            )
        })
        .map_err(map_error)?;
    let (rows, columns, coefficients) = result;
    let (real, imaginary) = split_complex(&coefficients);
    Ok((rows, columns, real, imaginary))
}

/// Apply a deterministic restricted transition list in one coarse native call.
///
/// The transition schema is constructed by the pure Rust core. This binding
/// only validates NumPy views, releases the GIL, and materializes the output.
#[pyfunction]
pub(crate) fn charge_mvp_apply<'py>(
    py: Python<'py>,
    dimension: usize,
    rows: PyReadonlyArray1<'py, u64>,
    columns: PyReadonlyArray1<'py, u64>,
    coefficients: PyReadonlyArray1<'py, NumpyComplex128>,
    state: PyReadonlyArray1<'py, NumpyComplex128>,
    max_bytes: usize,
) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
    let row_values = rows
        .as_slice()
        .map_err(|_| PyValueError::new_err("charge rows must be C-contiguous"))?;
    let column_values = columns
        .as_slice()
        .map_err(|_| PyValueError::new_err("charge columns must be C-contiguous"))?;
    let coefficient_values = coefficients
        .as_slice()
        .map_err(|_| PyValueError::new_err("charge coefficients must be C-contiguous"))?;
    let state_values = state
        .as_slice()
        .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
    if state_values.len() != dimension {
        return Err(PyValueError::new_err(format!(
            "state must have shape ({dimension},), got ({},)",
            state_values.len()
        )));
    }
    if row_values.len() != column_values.len() || row_values.len() != coefficient_values.len() {
        return Err(PyValueError::new_err(
            "restricted transition arrays must have equal lengths",
        ));
    }
    let output_bytes = dimension
        .checked_mul(std::mem::size_of::<NumpyComplex128>())
        .ok_or_else(|| PyMemoryError::new_err("charge MVP output size overflow"))?;
    if output_bytes > max_bytes {
        return Err(PyMemoryError::new_err(format!(
            "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
        )));
    }
    let values = py.allow_threads(|| {
        let mut output = vec![Complex64::new(0.0, 0.0); dimension];
        for ((&row, &column), &coefficient) in
            row_values.iter().zip(column_values).zip(coefficient_values)
        {
            let row = usize::try_from(row)
                .map_err(|_| PyValueError::new_err("restricted row index overflow"))?;
            let column = usize::try_from(column)
                .map_err(|_| PyValueError::new_err("restricted column index overflow"))?;
            if row >= dimension || column >= dimension {
                return Err(PyValueError::new_err(
                    "restricted transition index is outside the sector dimension",
                ));
            }
            output[row] += coefficient * state_values[column];
        }
        Ok::<Vec<Complex64>, PyErr>(output)
    })?;
    Ok(PyArray1::from_vec(py, values))
}
