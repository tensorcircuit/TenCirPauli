use pyo3::prelude::*;
use tencir_pauli_core::{build_mapping_plan, HybridBatch, HybridLayout, MappingPlan};

use crate::convert::{complex_coefficients, map_error, operator_output};

type MappingOutput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>);
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
type HybridOutput = HybridInput;

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

    fn transform_majorana(
        &self,
        py: Python<'_>,
        indices: Vec<Vec<u64>>,
        coefficients_re: Vec<f64>,
        coefficients_im: Vec<f64>,
        max_bytes: u128,
    ) -> PyResult<MappingOutput> {
        let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
        let operator = py
            .allow_threads(|| {
                self.plan
                    .map_majorana_terms(&indices, &coefficients, max_bytes)
            })
            .map_err(map_error)?;
        Ok(operator_output(&operator))
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_hybrid(
        &self,
        py: Python<'_>,
        n_bosons: usize,
        n_qubits: usize,
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
                self.plan.map_hybrid_terms(
                    HybridLayout {
                        n_modes: self.plan.n_modes(),
                        n_bosons,
                        n_qubits,
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
        let (real, imaginary) = crate::convert::split_complex(&result.coefficients);
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
