use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{SPPSEngine, SPPSEstimate, SPPSValueEstimate};

use crate::convert::{build_canonical_operator, map_error};
use crate::propagation::{compile_operation, compile_state};

type SppsOutput<'py> = (
    f64,
    Bound<'py, PyArray1<f64>>,
    f64,
    usize,
    Vec<usize>,
    usize,
    u64,
    Option<f64>,
    Option<Vec<f64>>,
    Option<bool>,
);
type SppsValueOutput = (f64, f64, usize, Vec<usize>, usize, u64);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeSPPSEngine {
    engine: SPPSEngine,
}

#[pymethods]
impl NativeSPPSEngine {
    #[getter]
    fn nqubits(&self) -> usize {
        self.engine.nqubits()
    }

    #[getter]
    fn nparameters(&self) -> usize {
        self.engine.nparameters()
    }

    #[getter]
    fn gate_count(&self) -> usize {
        self.engine.gate_count()
    }

    #[getter]
    fn observable_terms(&self) -> usize {
        self.engine.observable_terms()
    }

    #[getter]
    fn smoothing(&self) -> f64 {
        self.engine.smoothing()
    }

    fn expectation(
        &self,
        py: Python<'_>,
        parameters: PyReadonlyArray1<'_, f64>,
        samples_per_term: usize,
        seed: u64,
    ) -> PyResult<SppsValueOutput> {
        let values = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let result = py
            .allow_threads(|| self.engine.expectation(values, samples_per_term, seed))
            .map_err(map_error)?;
        Ok(materialize_value(result))
    }

    fn value_and_grad<'py>(
        &self,
        py: Python<'py>,
        parameters: PyReadonlyArray1<'py, f64>,
        samples_per_term: usize,
        seed: u64,
    ) -> PyResult<SppsOutput<'py>> {
        let values = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let result = py
            .allow_threads(|| self.engine.value_and_grad(values, samples_per_term, seed))
            .map_err(map_error)?;
        Ok(materialize(py, result))
    }

    fn value_and_grad_adaptive<'py>(
        &self,
        py: Python<'py>,
        parameters: PyReadonlyArray1<'py, f64>,
        initial_samples_per_term: usize,
        max_samples_per_term: usize,
        gradient_tolerance: f64,
        seed: u64,
    ) -> PyResult<SppsOutput<'py>> {
        let values = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let result = py
            .allow_threads(|| {
                self.engine.value_and_grad_adaptive(
                    values,
                    initial_samples_per_term,
                    max_samples_per_term,
                    gradient_tolerance,
                    seed,
                )
            })
            .map_err(map_error)?;
        Ok(materialize(py, result))
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (nqubits, operations, structures, coefficients_re, coefficients_im, state_kind, state_bits, state_values, smoothing=0.01, max_bytes=None))]
pub(crate) fn pauli_spps_engine(
    py: Python<'_>,
    nqubits: usize,
    operations: Vec<(u8, usize, usize, i64, f64, Vec<f64>)>,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    state_kind: u8,
    state_bits: Vec<u8>,
    state_values: Vec<f64>,
    smoothing: f64,
    max_bytes: Option<usize>,
) -> PyResult<NativeSPPSEngine> {
    let engine = py.allow_threads(|| {
        let compiled = operations
            .into_iter()
            .map(|(kind, wire0, wire1, parameter, angle, matrix)| {
                compile_operation(nqubits, kind, wire0, wire1, parameter, angle, &matrix)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        let state = compile_state(nqubits, state_kind, state_bits, state_values)?;
        SPPSEngine::new(
            nqubits,
            compiled,
            operator,
            state,
            smoothing,
            max_bytes.map(|value| value as u128),
        )
        .map_err(map_error)
    })?;
    Ok(NativeSPPSEngine { engine })
}

fn materialize<'py>(py: Python<'py>, result: SPPSEstimate) -> SppsOutput<'py> {
    (
        result.value,
        PyArray1::from_vec(py, result.gradient),
        result.value_standard_error,
        result.replicates,
        result.samples_per_replicate,
        result.total_paths,
        result.seed,
        result.gradient_error_proxy,
        result.term_gradient_error_proxies,
        result.converged,
    )
}

fn materialize_value(result: SPPSValueEstimate) -> SppsValueOutput {
    (
        result.value,
        result.value_standard_error,
        result.replicates,
        result.samples_per_replicate,
        result.total_paths,
        result.seed,
    )
}
