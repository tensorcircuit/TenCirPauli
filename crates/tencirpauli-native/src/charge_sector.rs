use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use tencir_pauli_core::{
    apply_charge_mvp_from_plan, build_charge_sector_plan, build_compact_charge_sector_plan,
    compile_charge_transitions_from_plan, ChargeSectorPlan, ChargeTransitionPlanLayout,
    ChargeTransitionTerm, Complex64,
};

use crate::convert::{map_error, split_complex};

type ChargeTransitionOutput = (Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeSectorPlan {
    plan: ChargeSectorPlan,
}

#[pymethods]
impl NativeChargeSectorPlan {
    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan.estimated_bytes()
    }

    fn rank(&self, occupations: Vec<u64>) -> PyResult<u64> {
        self.plan
            .rank(&occupations)
            .map_err(crate::convert::map_error)
    }

    fn unrank(&self, index: u64) -> PyResult<Vec<u64>> {
        self.plan.unrank(index).map_err(crate::convert::map_error)
    }

    fn basis_states<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let values = py
            .allow_threads(|| self.plan.basis_states(max_bytes))
            .map_err(crate::convert::map_error)?;
        Ok(PyArray1::from_vec(py, values))
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_transitions<'py>(
        &self,
        py: Python<'py>,
        dimension: usize,
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
        max_bytes: u128,
    ) -> PyResult<ChargeTransitionOutput> {
        let coefficient_values = coefficients.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge coefficients must be C-contiguous")
        })?;
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
            return Err(pyo3::exceptions::PyValueError::new_err(
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
                compile_charge_transitions_from_plan(
                    &self.plan,
                    ChargeTransitionPlanLayout {
                        dimension,
                        local_dimensions: &local_dimensions,
                        fermion_positions: &fermion_positions,
                        boson_positions: &boson_positions,
                        qubit_positions: &qubit_positions,
                        qudit_positions: &qudit_positions,
                        qudit_dimension,
                        max_bytes,
                    },
                    &terms,
                )
            })
            .map_err(map_error)?;
        let (rows, columns, values) = result;
        let (real, imaginary) = split_complex(&values);
        Ok((rows, columns, real, imaginary))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        dimension,
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        qudit_positions,
        fermion_creation,
        fermion_annihilation,
        boson_blocks,
        qubit_codes,
        mapped_present,
        mapped_codes,
        qudit_present,
        qudit_triples,
        coefficients,
        state,
        qudit_dimension,
        termwise_conserved,
        max_bytes,
        fast_fermion_particles=None
    ))]
    fn apply_lazy<'py>(
        &self,
        py: Python<'py>,
        dimension: usize,
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
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        qudit_dimension: u64,
        termwise_conserved: bool,
        max_bytes: u128,
        fast_fermion_particles: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let coefficient_values = coefficients.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge coefficients must be C-contiguous")
        })?;
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        if state_values.len() != dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "charge state must have shape ({dimension},), got ({},)",
                state_values.len()
            )));
        }
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
            return Err(pyo3::exceptions::PyValueError::new_err(
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
                apply_charge_mvp_from_plan(
                    &self.plan,
                    ChargeTransitionPlanLayout {
                        dimension,
                        local_dimensions: &local_dimensions,
                        fermion_positions: &fermion_positions,
                        boson_positions: &boson_positions,
                        qubit_positions: &qubit_positions,
                        qudit_positions: &qudit_positions,
                        qudit_dimension,
                        max_bytes,
                    },
                    &terms,
                    state_values,
                    termwise_conserved,
                    fast_fermion_particles,
                )
            })
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, result))
    }
}

#[pyfunction]
pub(crate) fn charge_sector_plan(
    py: Python<'_>,
    local_dimensions: Vec<usize>,
    contributions: Vec<Vec<Vec<i128>>>,
    target: Vec<i128>,
    max_bytes: u128,
) -> PyResult<NativeChargeSectorPlan> {
    let plan = py
        .allow_threads(|| {
            build_charge_sector_plan(local_dimensions, contributions, target, max_bytes)
        })
        .map_err(crate::convert::map_error)?;
    Ok(NativeChargeSectorPlan { plan })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn charge_sector_plan_compact(
    py: Python<'_>,
    local_dimensions: Vec<usize>,
    axis_kinds: Vec<u8>,
    axis_indices: Vec<usize>,
    fermion_weights: Vec<Vec<i128>>,
    boson_weights: Vec<Vec<i128>>,
    qubit_levels: Vec<Vec<(i128, i128)>>,
    target: Vec<i128>,
    max_bytes: u128,
) -> PyResult<NativeChargeSectorPlan> {
    let plan = py
        .allow_threads(|| {
            build_compact_charge_sector_plan(
                local_dimensions,
                axis_kinds,
                axis_indices,
                fermion_weights,
                boson_weights,
                qubit_levels,
                target,
                max_bytes,
            )
        })
        .map_err(crate::convert::map_error)?;
    Ok(NativeChargeSectorPlan { plan })
}
