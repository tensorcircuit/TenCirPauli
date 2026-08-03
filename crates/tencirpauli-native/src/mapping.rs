use pyo3::prelude::*;
use tencir_pauli_core::{build_mapping_plan, MappingPlan};

use crate::convert::{complex_coefficients, map_error, operator_output};

type MappingOutput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeMappingPlan {
    plan: MappingPlan,
}

#[pymethods]
impl NativeMappingPlan {
    #[getter]
    fn n_modes(&self) -> usize {
        self.plan.n_modes()
    }

    #[getter]
    fn encoding(&self) -> Vec<Vec<u8>> {
        self.plan.encoding().to_vec()
    }

    #[getter]
    fn inverse_encoding(&self) -> Vec<Vec<u8>> {
        self.plan.inverse_encoding().to_vec()
    }

    #[getter]
    fn cnot_operations(&self) -> Vec<(usize, usize)> {
        self.plan.cnot_operations().to_vec()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan.estimated_bytes()
    }

    fn transform(
        &self,
        py: Python<'_>,
        structures: Vec<Vec<u8>>,
        coefficients_re: Vec<f64>,
        coefficients_im: Vec<f64>,
        max_bytes: u128,
    ) -> PyResult<MappingOutput> {
        let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
        let operator = py
            .allow_threads(|| {
                self.plan
                    .map_pauli_terms(&structures, &coefficients, max_bytes)
            })
            .map_err(map_error)?;
        Ok(operator_output(&operator))
    }
}

#[pyfunction]
pub(crate) fn mapping_plan(
    py: Python<'_>,
    mapping: String,
    n_modes: usize,
    max_bytes: u128,
) -> PyResult<NativeMappingPlan> {
    let plan = py
        .allow_threads(|| build_mapping_plan(&mapping, n_modes, max_bytes))
        .map_err(map_error)?;
    Ok(NativeMappingPlan { plan })
}
