use std::sync::Arc;

use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use tencir_pauli_core::{
    build_compact_charge_sector_plan, charge_execution_variant,
    compile_charge_csr_from_prepared_plan, estimate_charge_transition_terms_bytes,
    group_charge_transition_terms, prepare_charge_mvp_plan, prepare_charge_transition_plan_layout,
    ChargeCsrResult, ChargeSectorPlan, ChargeTransitionTerm, Complex64, CsrApplyPolicy, CsrGraph,
    PauliError, PreparedChargeMvpPlan,
};

use crate::convert::map_error;
use crate::operator::NativePauliOperatorHandle;
use crate::structured::NativeHybridOperatorHandle;

type ChargeCsrArrays<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
type ChargeCooArrays<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);

/// Native-only physical-axis layout retained with the exact sector plan.
/// Operator compilation receives the handle directly, so Python never
/// rebuilds and resends this O(axis-count) metadata for every restriction.
#[derive(Clone)]
struct NativeChargeLayout {
    local_dimensions: Vec<u64>,
    fermion_positions: Vec<u64>,
    boson_positions: Vec<u64>,
    qubit_positions: Vec<u64>,
    qudit_positions: Vec<u64>,
    qudit_dimension: u64,
}

impl NativeChargeLayout {
    fn from_compact(
        local_dimensions: &[usize],
        axis_kinds: &[u8],
        axis_indices: &[usize],
    ) -> Result<Self, PauliError> {
        if local_dimensions.len() != axis_kinds.len()
            || local_dimensions.len() != axis_indices.len()
        {
            return Err(PauliError::InvalidStructureLength {
                expected: local_dimensions.len(),
                actual: axis_kinds.len(),
            });
        }
        fn positions_for_kind(
            kind: u8,
            axis_kinds: &[u8],
            axis_indices: &[usize],
        ) -> Result<Vec<u64>, PauliError> {
            let count = axis_kinds
                .iter()
                .zip(axis_indices)
                .filter_map(|(&candidate, &index)| (candidate == kind).then_some(index))
                .max()
                .map_or(Ok(0usize), |index| {
                    index.checked_add(1).ok_or(PauliError::Overflow {
                        context: "sizing charge-sector axis layout",
                    })
                })?;
            let mut positions = vec![u64::MAX; count];
            for (physical, (&candidate, &index)) in axis_kinds.iter().zip(axis_indices).enumerate()
            {
                if candidate != kind {
                    continue;
                }
                let position = positions.get_mut(index).ok_or(PauliError::InvalidSector {
                    context: "invalid charge-sector axis index",
                })?;
                if *position != u64::MAX {
                    return Err(PauliError::InvalidSector {
                        context: "duplicate charge-sector axis index",
                    });
                }
                *position = u64::try_from(physical).map_err(|_| PauliError::Overflow {
                    context: "converting charge-sector axis position",
                })?;
            }
            if positions.contains(&u64::MAX) {
                return Err(PauliError::InvalidSector {
                    context: "missing charge-sector axis index",
                });
            }
            Ok(positions)
        }

        let qudit_positions = positions_for_kind(3, axis_kinds, axis_indices)?;
        let qudit_dimension = qudit_positions.first().map_or(Ok(0_u64), |&position| {
            let dimension = *local_dimensions
                .get(usize::try_from(position).map_err(|_| PauliError::Overflow {
                    context: "converting charge-sector qudit position",
                })?)
                .ok_or(PauliError::InvalidSector {
                    context: "invalid charge-sector qudit position",
                })?;
            u64::try_from(dimension).map_err(|_| PauliError::Overflow {
                context: "converting charge-sector qudit dimension",
            })
        })?;
        for &position in &qudit_positions {
            let dimension = *local_dimensions
                .get(usize::try_from(position).map_err(|_| PauliError::Overflow {
                    context: "converting charge-sector qudit position",
                })?)
                .ok_or(PauliError::InvalidSector {
                    context: "invalid charge-sector qudit position",
                })?;
            if u64::try_from(dimension).ok() != Some(qudit_dimension) {
                return Err(PauliError::InvalidSector {
                    context: "charge-sector qudit dimensions must be uniform",
                });
            }
        }
        Ok(Self {
            local_dimensions: local_dimensions
                .iter()
                .copied()
                .map(|dimension| {
                    u64::try_from(dimension).map_err(|_| PauliError::Overflow {
                        context: "converting charge-sector local dimension",
                    })
                })
                .collect::<Result<_, _>>()?,
            fermion_positions: positions_for_kind(0, axis_kinds, axis_indices)?,
            boson_positions: positions_for_kind(1, axis_kinds, axis_indices)?,
            qubit_positions: positions_for_kind(2, axis_kinds, axis_indices)?,
            qudit_positions,
            qudit_dimension,
        })
    }

    fn estimated_bytes(&self) -> Result<u128, PauliError> {
        let entry_count = self
            .local_dimensions
            .len()
            .checked_add(self.fermion_positions.len())
            .and_then(|count| count.checked_add(self.boson_positions.len()))
            .and_then(|count| count.checked_add(self.qubit_positions.len()))
            .and_then(|count| count.checked_add(self.qudit_positions.len()))
            .ok_or(PauliError::Overflow {
                context: "sizing retained charge-sector layout",
            })?;
        let entries_bytes = (entry_count as u128)
            .checked_mul(std::mem::size_of::<u64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating retained charge-sector layout",
            })?;
        (std::mem::size_of::<NativeChargeLayout>() as u128)
            .checked_add(entries_bytes)
            .ok_or(PauliError::Overflow {
                context: "estimating retained charge-sector layout",
            })
    }
}

fn charge_plan_base_bytes(
    plan: &ChargeSectorPlan,
    layout: &NativeChargeLayout,
    terms: &[ChargeTransitionTerm],
) -> Result<u128, PauliError> {
    plan.estimated_bytes()
        .checked_add(layout.estimated_bytes()?)
        .and_then(|bytes| bytes.checked_add(estimate_charge_transition_terms_bytes(terms).ok()?))
        .ok_or(PauliError::Overflow {
            context: "estimating native charge MVP plan",
        })
}

fn charge_remaining_budget(max_bytes: u128, base_bytes: u128) -> Result<u128, PauliError> {
    max_bytes
        .checked_sub(base_bytes)
        .ok_or(PauliError::MemoryLimit {
            requested: base_bytes,
            limit: max_bytes,
        })
}

fn charge_terms_from_pauli_handle(handle: &NativePauliOperatorHandle) -> Vec<ChargeTransitionTerm> {
    handle
        .core()
        .terms()
        .iter()
        .map(|term| ChargeTransitionTerm {
            fermion_creation: Vec::new(),
            fermion_annihilation: Vec::new(),
            boson_blocks: Vec::new(),
            qubit_codes: term.word.codes(),
            mapped_present: false,
            mapped_codes: vec![0; 0],
            qudit_present: false,
            qudit_triples: Vec::new(),
            coefficient: term.coefficient,
        })
        .collect()
}

fn charge_terms_from_hybrid_handle(
    handle: &NativeHybridOperatorHandle,
) -> Vec<ChargeTransitionTerm> {
    let batch = handle.batch();
    (0..batch.coefficients.len())
        .map(|index| ChargeTransitionTerm {
            fermion_creation: batch.fermion_creation[index].clone(),
            fermion_annihilation: batch.fermion_annihilation[index].clone(),
            boson_blocks: batch.boson_blocks[index].clone(),
            qubit_codes: batch.qubit_codes[index].clone(),
            mapped_present: batch.mapped_present[index],
            mapped_codes: batch.mapped_codes[index].clone(),
            qudit_present: batch.qudit_present[index],
            qudit_triples: batch.qudit_triples[index].clone(),
            coefficient: batch.coefficients[index],
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_native_charge_schedule_layout(
    plan: &Arc<ChargeSectorPlan>,
    layout_descriptor: NativeChargeLayout,
    terms: Vec<ChargeTransitionTerm>,
    max_bytes: u128,
) -> PyResult<(
    tencir_pauli_core::PreparedChargeTransitionPlanLayout,
    Box<[ChargeTransitionTerm]>,
    u128,
)> {
    let dimension = plan.dimension();
    let terms = group_charge_transition_terms(&terms).map_err(map_error)?;
    let base_bytes = charge_plan_base_bytes(plan, &layout_descriptor, &terms).map_err(map_error)?;
    let layout_budget = charge_remaining_budget(max_bytes, base_bytes).map_err(map_error)?;
    let layout = prepare_charge_transition_plan_layout(
        plan,
        dimension,
        layout_descriptor.local_dimensions,
        &layout_descriptor.fermion_positions,
        &layout_descriptor.boson_positions,
        &layout_descriptor.qubit_positions,
        &layout_descriptor.qudit_positions,
        layout_descriptor.qudit_dimension,
        &terms,
        layout_budget,
    )
    .map_err(map_error)?;
    let retained_bytes = base_bytes
        .checked_add(layout.estimated_bytes())
        .ok_or(PauliError::Overflow {
            context: "estimating native charge schedule layout",
        })
        .map_err(map_error)?;
    Ok((layout, terms.into_boxed_slice(), retained_bytes))
}

fn build_native_charge_prepared_plan(
    plan: Arc<ChargeSectorPlan>,
    layout_descriptor: NativeChargeLayout,
    terms: Vec<ChargeTransitionTerm>,
    termwise_conserved: bool,
    max_bytes: u128,
) -> PyResult<PreparedChargeMvpPlan> {
    let (layout, terms, retained_bytes) =
        build_native_charge_schedule_layout(&plan, layout_descriptor, terms, max_bytes)?;
    let kernel_budget = charge_remaining_budget(max_bytes, retained_bytes).map_err(map_error)?;
    prepare_charge_mvp_plan(plan, layout, terms, termwise_conserved, kernel_budget)
        .map_err(map_error)
}

fn build_native_charge_eager_plan(
    plan: Arc<ChargeSectorPlan>,
    layout_descriptor: NativeChargeLayout,
    terms: Vec<ChargeTransitionTerm>,
    _termwise_conserved: bool,
    max_bytes: u128,
) -> PyResult<NativeChargeEagerMvpPlan> {
    let sector_layout_bytes = layout_descriptor.estimated_bytes().map_err(map_error)?;
    let (layout, terms, retained_bytes) =
        build_native_charge_schedule_layout(&plan, layout_descriptor, terms, max_bytes)?;
    let eager_budget = max_bytes
        .checked_sub(retained_bytes)
        .ok_or(PauliError::MemoryLimit {
            requested: retained_bytes,
            limit: max_bytes,
        })
        .map_err(map_error)?;
    let result = compile_charge_csr_from_prepared_plan(&plan, &layout, &terms, eager_budget)
        .map_err(map_error)?;
    NativeChargeEagerMvpPlan::from_csr(result, charge_execution_variant(&plan), sector_layout_bytes)
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeSectorPlan {
    plan: Arc<ChargeSectorPlan>,
    layout: NativeChargeLayout,
    layout_estimated_bytes: u128,
}

#[pymethods]
impl NativeChargeSectorPlan {
    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan
            .estimated_bytes()
            .saturating_add(self.layout_estimated_bytes)
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

    #[pyo3(signature = (handle, termwise_conserved, max_bytes))]
    fn compile_mvp_pauli_handle(
        &self,
        py: Python<'_>,
        handle: &NativePauliOperatorHandle,
        termwise_conserved: bool,
        max_bytes: u128,
    ) -> PyResult<NativeChargeMvpPlan> {
        let plan = Arc::clone(&self.plan);
        let layout = self.layout.clone();
        let sector_layout_bytes = self.layout_estimated_bytes;
        py.allow_threads(move || {
            build_native_charge_prepared_plan(
                plan,
                layout,
                charge_terms_from_pauli_handle(handle),
                termwise_conserved,
                max_bytes,
            )
            .map(|plan| NativeChargeMvpPlan {
                plan,
                sector_layout_bytes,
            })
        })
    }

    #[pyo3(signature = (handle, termwise_conserved, max_bytes))]
    fn compile_mvp_hybrid_handle(
        &self,
        py: Python<'_>,
        handle: &NativeHybridOperatorHandle,
        termwise_conserved: bool,
        max_bytes: u128,
    ) -> PyResult<NativeChargeMvpPlan> {
        let plan = Arc::clone(&self.plan);
        let layout = self.layout.clone();
        let sector_layout_bytes = self.layout_estimated_bytes;
        py.allow_threads(move || {
            build_native_charge_prepared_plan(
                plan,
                layout,
                charge_terms_from_hybrid_handle(handle),
                termwise_conserved,
                max_bytes,
            )
            .map(|plan| NativeChargeMvpPlan {
                plan,
                sector_layout_bytes,
            })
        })
    }

    #[pyo3(signature = (handle, termwise_conserved, max_bytes))]
    fn compile_eager_pauli_handle(
        &self,
        py: Python<'_>,
        handle: &NativePauliOperatorHandle,
        termwise_conserved: bool,
        max_bytes: u128,
    ) -> PyResult<NativeChargeEagerMvpPlan> {
        let plan = Arc::clone(&self.plan);
        let layout = self.layout.clone();
        py.allow_threads(move || {
            build_native_charge_eager_plan(
                plan,
                layout,
                charge_terms_from_pauli_handle(handle),
                termwise_conserved,
                max_bytes,
            )
        })
    }

    #[pyo3(signature = (handle, termwise_conserved, max_bytes))]
    fn compile_eager_hybrid_handle(
        &self,
        py: Python<'_>,
        handle: &NativeHybridOperatorHandle,
        termwise_conserved: bool,
        max_bytes: u128,
    ) -> PyResult<NativeChargeEagerMvpPlan> {
        let plan = Arc::clone(&self.plan);
        let layout = self.layout.clone();
        py.allow_threads(move || {
            build_native_charge_eager_plan(
                plan,
                layout,
                charge_terms_from_hybrid_handle(handle),
                termwise_conserved,
                max_bytes,
            )
        })
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeMvpPlan {
    plan: PreparedChargeMvpPlan,
    sector_layout_bytes: u128,
}

#[pymethods]
impl NativeChargeMvpPlan {
    fn apply_adjoint<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output = py
            .allow_threads(|| self.plan.apply_adjoint(state_slice, max_bytes))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, output))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan
            .estimated_bytes()
            .ok()
            .and_then(|bytes| bytes.checked_add(self.sector_layout_bytes))
            .unwrap_or(u128::MAX)
    }

    /// Return the executor selected by native plan construction.
    ///
    /// Python must not infer `product` from constraint shapes when the native
    /// selector actually fell through to the generic executor.
    #[getter]
    fn execution_family(&self) -> &'static str {
        self.plan.execution_family()
    }

    #[getter]
    fn native_variant(&self) -> &'static str {
        self.plan.native_variant()
    }

    fn debug_counters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self
            .plan
            .debug_counters()
            .unwrap_or_else(|| self.plan.generic_debug_counters().to_vec());
        PyTuple::new(py, values)
    }

    #[getter]
    fn schedule_count(&self) -> usize {
        self.plan.schedule_count()
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        if state_values.len() != self.dimension() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "charge state must have shape ({},), got ({},)",
                self.dimension(),
                state_values.len()
            )));
        }
        let result = py
            .allow_threads(|| self.plan.apply(state_values, max_bytes))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, result))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        mut output: PyReadwriteArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<()> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let output_values = output.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge output must be C-contiguous")
        })?;
        let output_bytes = (self.dimension() as u128)
            .checked_mul(std::mem::size_of::<NumpyComplex128>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge MVP output size overflow")
            })?;
        let effective_max = max_bytes.saturating_add(output_bytes);
        py.allow_threads(|| {
            self.plan
                .apply_into(state_values, output_values, effective_max)
        })
        .map_err(map_error)
    }

    fn compile_eager<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<NativeChargeEagerMvpPlan> {
        py.allow_threads(|| {
            let result = self.plan.compile_eager(max_bytes).map_err(map_error)?;
            NativeChargeEagerMvpPlan::from_csr(
                result,
                self.native_variant(),
                self.sector_layout_bytes,
            )
        })
    }

    #[pyo3(signature = (max_bytes, chunk_capacity))]
    fn compile_eager_with_chunk_capacity(
        &self,
        py: Python<'_>,
        max_bytes: u128,
        chunk_capacity: usize,
    ) -> PyResult<NativeChargeEagerMvpPlan> {
        py.allow_threads(|| {
            let result = self
                .plan
                .compile_eager_with_chunk_capacity(max_bytes, chunk_capacity)
                .map_err(map_error)?;
            NativeChargeEagerMvpPlan::from_csr(
                result,
                self.native_variant(),
                self.sector_layout_bytes,
            )
        })
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeEagerMvpPlan {
    graph: CsrGraph<usize>,
    estimated_bytes: u128,
    estimated_construction_peak_bytes: u128,
    schedule_count: usize,
    native_variant: &'static str,
    generation_nanos: u128,
    chunk_count: usize,
    sorted_entries: usize,
    merged_entries: usize,
    heap_operations: usize,
    final_array_allocation_entries: usize,
    chunk_metadata_bytes: u128,
    sort_nanos: u128,
    merge_nanos: u128,
    final_array_allocation_nanos: u128,
}

impl NativeChargeEagerMvpPlan {
    fn from_csr(
        result: ChargeCsrResult,
        native_variant: &'static str,
        sector_layout_bytes: u128,
    ) -> PyResult<Self> {
        let graph = CsrGraph::try_from_u64(
            result.dimension,
            result.indptr,
            result.columns,
            result.values,
        )
        .map_err(map_error)?;
        Ok(Self {
            graph,
            estimated_bytes: result
                .estimated_bytes
                .checked_add(sector_layout_bytes)
                .ok_or(PauliError::Overflow {
                    context: "estimating retained eager charge plan",
                })
                .map_err(map_error)?,
            estimated_construction_peak_bytes: result
                .estimated_construction_peak_bytes
                .checked_add(sector_layout_bytes)
                .ok_or(PauliError::Overflow {
                    context: "estimating eager charge construction peak",
                })
                .map_err(map_error)?,
            schedule_count: result.schedule_count,
            native_variant,
            generation_nanos: result.generation_nanos,
            chunk_count: result.chunk_count,
            sorted_entries: result.sorted_entries,
            merged_entries: result.merged_entries,
            heap_operations: result.heap_operations,
            final_array_allocation_entries: result.final_array_allocation_entries,
            chunk_metadata_bytes: result.chunk_metadata_bytes,
            sort_nanos: result.sort_nanos,
            merge_nanos: result.merge_nanos,
            final_array_allocation_nanos: result.final_array_allocation_nanos,
        })
    }

    fn apply_values(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        parallel: bool,
    ) -> PyResult<()> {
        let policy = if parallel {
            CsrApplyPolicy::parallel(1 << 14)
        } else {
            CsrApplyPolicy::serial()
        };
        self.graph
            .apply_into(state, output, policy)
            .map_err(map_error)
    }
}

#[pymethods]
impl NativeChargeEagerMvpPlan {
    fn apply_adjoint<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output = py
            .allow_threads(|| self.graph.apply_adjoint(state_slice, max_bytes))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, output))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.graph.dimension()
    }

    #[getter]
    fn transition_count(&self) -> usize {
        self.graph.transition_count()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    #[getter]
    fn estimated_construction_peak_bytes(&self) -> u128 {
        self.estimated_construction_peak_bytes
    }

    #[getter]
    fn schedule_count(&self) -> usize {
        self.schedule_count
    }

    #[getter]
    fn native_variant(&self) -> &'static str {
        self.native_variant
    }

    fn construction_profile(
        &self,
    ) -> (
        u128,
        usize,
        usize,
        usize,
        usize,
        usize,
        u128,
        u128,
        u128,
        u128,
    ) {
        (
            self.generation_nanos,
            self.chunk_count,
            self.sorted_entries,
            self.merged_entries,
            self.heap_operations,
            self.final_array_allocation_entries,
            self.chunk_metadata_bytes,
            self.sort_nanos,
            self.merge_nanos,
            self.final_array_allocation_nanos,
        )
    }

    fn csr<'py>(&self, py: Python<'py>, max_bytes: u128) -> PyResult<ChargeCsrArrays<'py>> {
        let (indptr, columns, values) = py
            .allow_threads(|| self.graph.to_u64_arrays(max_bytes))
            .map_err(map_error)?;
        Ok((
            PyArray1::from_vec(py, indptr),
            PyArray1::from_vec(py, columns),
            PyArray1::from_vec(py, values),
        ))
    }

    fn coo<'py>(&self, py: Python<'py>, max_bytes: u128) -> PyResult<ChargeCooArrays<'py>> {
        let (rows, columns, values) = self.graph.to_coo_u64(max_bytes).map_err(map_error)?;
        Ok((
            PyArray1::from_vec(py, rows),
            PyArray1::from_vec(py, columns),
            PyArray1::from_vec(py, values),
        ))
    }

    fn dense<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let dimension = self.graph.dimension();
        let entries = (dimension as u128)
            .checked_mul(dimension as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge dense size overflow")
            })?;
        let output_bytes = entries
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge dense size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge dense matrix requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let values = vec![
            Complex64::default();
            usize::try_from(entries).map_err(|_| {
                pyo3::exceptions::PyMemoryError::new_err("charge dense dimension overflow")
            })?
        ];
        let values = py.allow_threads(|| {
            let mut values = values;
            for row in 0..dimension {
                for index in self.graph.indptr()[row]..self.graph.indptr()[row + 1] {
                    values[row * dimension + self.graph.columns()[index]] =
                        self.graph.values()[index];
                }
            }
            values
        });
        Ok(PyArray1::from_vec(py, values))
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let dimension = self.graph.dimension();
        let output_bytes = (dimension as u128)
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge output size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let mut output = vec![Complex64::default(); dimension];
        py.allow_threads(|| self.apply_values(state_values, &mut output, true))?;
        Ok(PyArray1::from_vec(py, output))
    }

    fn apply_with_parallelism<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let dimension = self.graph.dimension();
        let output_bytes = (dimension as u128)
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge output size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let mut output = vec![Complex64::default(); dimension];
        py.allow_threads(|| self.apply_values(state_values, &mut output, parallel))?;
        Ok(PyArray1::from_vec(py, output))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        mut output: PyReadwriteArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<()> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let output_values = output.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge output must be C-contiguous")
        })?;
        let output_bytes = (self.graph.dimension() as u128)
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge output size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        py.allow_threads(|| self.apply_values(state_values, output_values, true))
    }
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
    qudit_weights: Vec<Vec<i128>>,
    target: Vec<i128>,
    max_bytes: u128,
) -> PyResult<NativeChargeSectorPlan> {
    let (plan, layout) = py
        .allow_threads(|| {
            let layout =
                NativeChargeLayout::from_compact(&local_dimensions, &axis_kinds, &axis_indices)?;
            let layout_bytes = layout.estimated_bytes()?;
            let plan_budget =
                max_bytes
                    .checked_sub(layout_bytes)
                    .ok_or(PauliError::MemoryLimit {
                        requested: layout_bytes,
                        limit: max_bytes,
                    })?;
            let plan = build_compact_charge_sector_plan(
                local_dimensions,
                axis_kinds,
                axis_indices,
                fermion_weights,
                boson_weights,
                qubit_levels,
                qudit_weights,
                target,
                plan_budget,
            )?;
            Ok((plan, layout))
        })
        .map_err(crate::convert::map_error)?;
    Ok(NativeChargeSectorPlan {
        plan: Arc::new(plan),
        layout_estimated_bytes: layout
            .estimated_bytes()
            .map_err(crate::convert::map_error)?,
        layout,
    })
}
