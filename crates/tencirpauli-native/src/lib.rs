//! Private PyO3 extension for the public `tencirpauli` Python package.

mod charge_analysis;
mod charge_sector;
mod convert;
mod grouping;
mod hamiltonian;
mod majorana;
mod mapping;
mod operator;
mod propagation;
mod spps;
mod structured;
mod symmetry;
mod u1_circuit;
mod word;

use pyo3::prelude::*;

use charge_analysis::pauli_analyze_charge_handle;
use charge_sector::{
    charge_sector_plan_compact, NativeChargeEagerMvpPlan, NativeChargeMvpPlan,
    NativeChargeSectorPlan,
};
use grouping::{
    pauli_compatibility_matrix_handle, pauli_group_handle, pauli_incompatibility_edges_handle,
    pauli_qwc_group_handle, NativeQwcGroupingHandle,
};
use hamiltonian::{
    pauli_backend_plan_handle, pauli_coo_handle, pauli_csr_handle, pauli_dense_handle,
    pauli_mvp_handle, pauli_mvp_plan_handle, NativeMvpPlan,
};
use majorana::{majorana_canonicalize, NativeMajoranaOperatorHandle};
use mapping::{mapping_plan, NativeMappingPlan};
use operator::{
    pauli_canonicalize_batch, pauli_canonicalize_batch_array, pauli_canonicalize_batch_numpy,
    pauli_operator_native, pauli_operator_native_array, NativePauliOperatorHandle,
};
use propagation::{
    pauli_gate_tape, pauli_propagation_batch_handles_tape, pauli_propagation_engine_tape,
    NativeGateTape, NativePropagationBatch, NativePropagationEngine,
};
use spps::{pauli_spps_engine_tape, NativeSPPSEngine};
use structured::{
    structured_boson_canonicalize, structured_dense_handle, structured_fermion_canonicalize,
    structured_hybrid_canonicalize, structured_hybrid_jordan_wigner, structured_sparse_handle,
    structured_sparse_plan_handle, NativeBosonOperatorHandle, NativeFermionOperatorHandle,
    NativeHybridOperatorHandle, StructuredMvpPlan,
};
use symmetry::{
    pauli_find_z2_symmetries_handle, pauli_restrict_u1_handle, pauli_restrict_u1_lazy_handle,
    pauli_z2_tapering_plan, u1_basis_words, NativeU1LazyMvpPlan, NativeU1MvpPlan,
    NativeU1RestrictedOperator, NativeZ2TaperingPlan,
};
use u1_circuit::{u1_circuit_plan, NativeU1CircuitPlan, NativeU1FinalState};
use word::{
    pauli_batch_from_codes, pauli_codes, pauli_commutes, pauli_from_codes, pauli_multiply,
    pauli_support, pauli_symplectic_inner_product, pauli_weight,
};

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<StructuredMvpPlan>()?;
    module.add_class::<NativeFermionOperatorHandle>()?;
    module.add_class::<NativeBosonOperatorHandle>()?;
    module.add_class::<NativeHybridOperatorHandle>()?;
    module.add_class::<NativeMajoranaOperatorHandle>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<NativeMvpPlan>()?;
    module.add_class::<NativePauliOperatorHandle>()?;
    module.add_class::<NativeChargeSectorPlan>()?;
    module.add_class::<NativeChargeMvpPlan>()?;
    module.add_class::<NativeChargeEagerMvpPlan>()?;
    module.add_class::<NativeZ2TaperingPlan>()?;
    module.add_class::<NativeU1RestrictedOperator>()?;
    module.add_class::<NativeU1MvpPlan>()?;
    module.add_class::<NativeU1LazyMvpPlan>()?;
    module.add_class::<NativeU1CircuitPlan>()?;
    module.add_class::<NativeU1FinalState>()?;
    module.add_class::<NativePropagationEngine>()?;
    module.add_class::<NativeGateTape>()?;
    module.add_class::<NativePropagationBatch>()?;
    module.add_class::<NativeSPPSEngine>()?;
    module.add_class::<NativeMappingPlan>()?;
    module.add_class::<NativeQwcGroupingHandle>()?;
    module.add_function(wrap_pyfunction!(pauli_weight, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_support, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_from_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_batch_from_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_multiply, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_symplectic_inner_product, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_commutes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize_batch, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize_batch_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize_batch_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_native, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_native_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_dense_handle, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_coo_handle, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_csr_handle, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_mvp_handle, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_mvp_plan_handle, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_backend_plan_handle, module)?)?;
    module.add_function(wrap_pyfunction!(charge_sector_plan_compact, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_analyze_charge_handle, module)?)?;
    module.add_function(wrap_pyfunction!(majorana_canonicalize, module)?)?;
    module.add_function(wrap_pyfunction!(mapping_plan, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_group_handle, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_qwc_group_handle, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_compatibility_matrix_handle, module)?)?;
    module.add_function(wrap_pyfunction!(
        pauli_incompatibility_edges_handle,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(pauli_find_z2_symmetries_handle, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_z2_tapering_plan, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_restrict_u1_handle, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_restrict_u1_lazy_handle, module)?)?;
    module.add_function(wrap_pyfunction!(u1_basis_words, module)?)?;
    module.add_function(wrap_pyfunction!(u1_circuit_plan, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_gate_tape, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_propagation_engine_tape, module)?)?;
    module.add_function(wrap_pyfunction!(
        pauli_propagation_batch_handles_tape,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(pauli_spps_engine_tape, module)?)?;
    module.add_function(wrap_pyfunction!(structured_dense_handle, module)?)?;
    module.add_function(wrap_pyfunction!(structured_sparse_handle, module)?)?;
    module.add_function(wrap_pyfunction!(structured_sparse_plan_handle, module)?)?;
    module.add_function(wrap_pyfunction!(structured_fermion_canonicalize, module)?)?;
    module.add_function(wrap_pyfunction!(structured_boson_canonicalize, module)?)?;
    module.add_function(wrap_pyfunction!(structured_hybrid_canonicalize, module)?)?;
    module.add_function(wrap_pyfunction!(structured_hybrid_jordan_wigner, module)?)?;
    Ok(())
}
