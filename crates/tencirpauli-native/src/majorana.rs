use numpy::{Complex64 as NumpyComplex128, PyArray1};
use pyo3::prelude::*;
use std::hash::{Hash, Hasher};
use tencir_pauli_core::{
    binary_majorana_terms, canonicalize_majorana_terms, hash_complex, majorana_to_fermion_terms,
    multiply_majorana_terms, Complex64, MajoranaBatch,
};

use crate::convert::{complex_coefficients, map_error};
use crate::structured::NativeFermionOperatorHandle;

type NumpyMajoranaOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeMajoranaOperatorHandle {
    n_modes: usize,
    indices: Vec<Vec<u64>>,
    coefficients: Vec<Complex64>,
}

impl NativeMajoranaOperatorHandle {
    pub(crate) fn from_result(n_modes: usize, result: (Vec<Vec<u64>>, Vec<Complex64>)) -> Self {
        Self {
            n_modes,
            indices: result.0,
            coefficients: result.1,
        }
    }

    pub(crate) fn native_parts(&self) -> (&[Vec<u64>], &[Complex64]) {
        (&self.indices, &self.coefficients)
    }

    fn content_hash_inner(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.n_modes.hash(&mut hasher);
        self.indices.hash(&mut hasher);
        for coefficient in &self.coefficients {
            hash_complex(*coefficient, &mut hasher);
        }
        hasher.finish()
    }
}

#[pymethods]
impl NativeMajoranaOperatorHandle {
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
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Majorana handle mode counts differ",
            ));
        }
        py.allow_threads(|| merge_majorana_handles(self, other, max_bytes as u128))
            .map_err(map_error)
    }

    fn scale(&self, py: Python<'_>, scalar_re: f64, scalar_im: f64) -> PyResult<Self> {
        py.allow_threads(|| scale_majorana_handle(self, Complex64::new(scalar_re, scalar_im)))
            .map_err(map_error)
    }

    fn multiply(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        if self.n_modes != other.n_modes {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Majorana handle mode counts differ",
            ));
        }
        let result = py
            .allow_threads(|| {
                multiply_majorana_terms(
                    self.n_modes,
                    MajoranaBatch {
                        indices: &self.indices,
                        coefficients: &self.coefficients,
                    },
                    MajoranaBatch {
                        indices: &other.indices,
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
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Majorana handle mode counts differ",
            ));
        }
        let result = py
            .allow_threads(|| {
                binary_majorana_terms(
                    self.n_modes,
                    MajoranaBatch {
                        indices: &self.indices,
                        coefficients: &self.coefficients,
                    },
                    MajoranaBatch {
                        indices: &other.indices,
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
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Majorana handle mode counts differ",
            ));
        }
        let result = py
            .allow_threads(|| {
                binary_majorana_terms(
                    self.n_modes,
                    MajoranaBatch {
                        indices: &self.indices,
                        coefficients: &self.coefficients,
                    },
                    MajoranaBatch {
                        indices: &other.indices,
                        coefficients: &other.coefficients,
                    },
                    max_bytes as u128,
                    1,
                )
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.n_modes, result))
    }

    fn adjoint(&self, py: Python<'_>) -> Self {
        py.allow_threads(|| Self {
            n_modes: self.n_modes,
            indices: self.indices.clone(),
            coefficients: self
                .indices
                .iter()
                .zip(&self.coefficients)
                .map(|(word, value)| {
                    let sign = if (word.len() * (word.len() - 1) / 2) & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    value.conj() * sign
                })
                .collect(),
        })
    }

    fn is_hermitian(&self, py: Python<'_>, tolerance: f64) -> bool {
        py.allow_threads(|| {
            self.indices.len() == self.coefficients.len()
                && self
                    .indices
                    .iter()
                    .zip(&self.coefficients)
                    .all(|(word, value)| {
                        let sign = if (word.len() * (word.len() - 1) / 2) & 1 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        let target = value.conj() * sign;
                        (value.re - target.re).abs() <= tolerance
                            && (value.im - target.im).abs() <= tolerance
                    })
        })
    }

    fn content_eq(&self, py: Python<'_>, other: &Self) -> bool {
        py.allow_threads(|| {
            self.n_modes == other.n_modes
                && self.indices == other.indices
                && self.coefficients == other.coefficients
        })
    }

    fn content_hash(&self, py: Python<'_>) -> u64 {
        py.allow_threads(|| self.content_hash_inner())
    }

    fn materialize<'py>(&self, py: Python<'py>) -> NumpyMajoranaOutput<'py> {
        let (indices, offsets, coefficients) = py.allow_threads(|| {
            let mut payload = Vec::new();
            let mut offsets = Vec::with_capacity(self.indices.len() + 1);
            offsets.push(0);
            for word in &self.indices {
                payload.extend_from_slice(word);
                offsets.push(payload.len());
            }
            (payload, offsets, self.coefficients.clone())
        });
        (
            self.term_count(),
            PyArray1::from_vec(py, indices),
            PyArray1::from_vec(py, offsets),
            PyArray1::from_vec(py, coefficients),
        )
    }

    fn to_fermion(
        &self,
        py: Python<'_>,
        max_bytes: usize,
    ) -> PyResult<NativeFermionOperatorHandle> {
        let result = py
            .allow_threads(|| {
                majorana_to_fermion_terms(
                    self.n_modes,
                    &self.indices,
                    &self.coefficients,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        Ok(NativeFermionOperatorHandle::from_result(
            self.n_modes,
            result,
        ))
    }
}

fn merge_majorana_handles(
    left: &NativeMajoranaOperatorHandle,
    right: &NativeMajoranaOperatorHandle,
    max_bytes: u128,
) -> Result<NativeMajoranaOperatorHandle, tencir_pauli_core::PauliError> {
    let requested = ((left.term_count() + right.term_count()).max(1) as u128)
        .checked_mul(192)
        .ok_or(tencir_pauli_core::PauliError::Overflow {
            context: "estimating Majorana operator addition",
        })?;
    if requested > max_bytes {
        return Err(tencir_pauli_core::PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    let mut indices = Vec::with_capacity(left.term_count() + right.term_count());
    let mut coefficients = Vec::with_capacity(left.term_count() + right.term_count());
    let mut index = 0;
    let mut other_index = 0;
    while index < left.term_count() && other_index < right.term_count() {
        match left.indices[index].cmp(&right.indices[other_index]) {
            std::cmp::Ordering::Less => {
                indices.push(left.indices[index].clone());
                coefficients.push(left.coefficients[index]);
                index += 1;
            }
            std::cmp::Ordering::Greater => {
                indices.push(right.indices[other_index].clone());
                coefficients.push(right.coefficients[other_index]);
                other_index += 1;
            }
            std::cmp::Ordering::Equal => {
                let value = left.coefficients[index] + right.coefficients[other_index];
                if value.re != 0.0 || value.im != 0.0 {
                    indices.push(left.indices[index].clone());
                    coefficients.push(value);
                }
                index += 1;
                other_index += 1;
            }
        }
    }
    while index < left.term_count() {
        indices.push(left.indices[index].clone());
        coefficients.push(left.coefficients[index]);
        index += 1;
    }
    while other_index < right.term_count() {
        indices.push(right.indices[other_index].clone());
        coefficients.push(right.coefficients[other_index]);
        other_index += 1;
    }
    Ok(NativeMajoranaOperatorHandle::from_result(
        left.n_modes,
        (indices, coefficients),
    ))
}

fn scale_majorana_handle(
    input: &NativeMajoranaOperatorHandle,
    scalar: Complex64,
) -> Result<NativeMajoranaOperatorHandle, tencir_pauli_core::PauliError> {
    let mut indices = Vec::with_capacity(input.term_count());
    let mut coefficients = Vec::with_capacity(input.term_count());
    if scalar.re == 0.0 && scalar.im == 0.0 {
        return Ok(NativeMajoranaOperatorHandle::from_result(
            input.n_modes,
            (indices, coefficients),
        ));
    }
    for index in 0..input.term_count() {
        indices.push(input.indices[index].clone());
        coefficients.push(input.coefficients[index] * scalar);
    }
    Ok(NativeMajoranaOperatorHandle::from_result(
        input.n_modes,
        (indices, coefficients),
    ))
}

#[pyfunction]
pub(crate) fn majorana_canonicalize(
    py: Python<'_>,
    n_modes: usize,
    indices: Vec<Vec<u64>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<NativeMajoranaOperatorHandle> {
    let result = py.allow_threads(|| {
        let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
        canonicalize_majorana_terms(n_modes, &indices, &coefficients, max_bytes).map_err(map_error)
    })?;
    Ok(NativeMajoranaOperatorHandle::from_result(n_modes, result))
}
