//! Private PyO3 extension for the public `tencirpauli` Python package.

mod charge;
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

use charge::{charge_compile_transitions, charge_mvp_apply};
use charge_sector::{charge_sector_plan, charge_sector_plan_compact, NativeChargeSectorPlan};
use grouping::{pauli_compatibility_matrix, pauli_group, pauli_incompatibility_edges};
use hamiltonian::{
    pauli_backend_plan, pauli_coo, pauli_coo_array, pauli_csr, pauli_csr_array, pauli_dense,
    pauli_dense_array, pauli_mvp_array, pauli_mvp_plan, NativeMvpPlan,
};
use majorana::{
    fermion_to_majorana, majorana_canonicalize, majorana_multiply, majorana_to_fermion,
};
use mapping::{mapping_plan, NativeMappingPlan};
use operator::{
    pauli_canonicalize, pauli_canonicalize_array, pauli_canonicalize_batch,
    pauli_canonicalize_batch_array, pauli_canonicalize_batch_numpy, pauli_operator_adjoint,
    pauli_operator_binary, pauli_operator_is_hermitian, pauli_operator_scale,
};
use propagation::{
    pauli_propagation_batch, pauli_propagation_engine, NativePropagationBatch,
    NativePropagationEngine,
};
use spps::{pauli_spps_engine, NativeSPPSEngine};
use structured::{
    structured_boson_canonicalize, structured_boson_multiply, structured_dense,
    structured_fermion_canonicalize, structured_fermion_jordan_wigner, structured_fermion_multiply,
    structured_hybrid_canonicalize, structured_hybrid_jordan_wigner, structured_hybrid_multiply,
    structured_sparse, structured_sparse_plan, StructuredMvpPlan,
};
use symmetry::{
    pauli_find_z2_symmetries, pauli_restrict_u1, pauli_z2_tapering_plan, u1_basis_words,
    NativeU1MvpPlan, NativeU1RestrictedOperator, NativeZ2TaperingPlan,
};
use u1_circuit::{u1_circuit_plan, NativeU1CircuitPlan, NativeU1FinalState};
use word::{
    pauli_batch_from_codes, pauli_codes, pauli_commutes, pauli_from_codes, pauli_multiply,
    pauli_support, pauli_symplectic_inner_product, pauli_weight,
};

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<StructuredMvpPlan>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<NativeMvpPlan>()?;
    module.add_class::<NativeChargeSectorPlan>()?;
    module.add_class::<NativeZ2TaperingPlan>()?;
    module.add_class::<NativeU1RestrictedOperator>()?;
    module.add_class::<NativeU1MvpPlan>()?;
    module.add_class::<NativeU1CircuitPlan>()?;
    module.add_class::<NativeU1FinalState>()?;
    module.add_class::<NativePropagationEngine>()?;
    module.add_class::<NativePropagationBatch>()?;
    module.add_class::<NativeSPPSEngine>()?;
    module.add_class::<NativeMappingPlan>()?;
    module.add_function(wrap_pyfunction!(pauli_weight, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_support, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_from_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_batch_from_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_multiply, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_symplectic_inner_product, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_commutes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize_batch, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize_batch_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize_batch_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_binary, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_scale, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_adjoint, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_is_hermitian, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_dense, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_dense_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_coo, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_coo_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_csr, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_csr_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_mvp_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_mvp_plan, module)?)?;
    module.add_function(wrap_pyfunction!(charge_mvp_apply, module)?)?;
    module.add_function(wrap_pyfunction!(charge_compile_transitions, module)?)?;
    module.add_function(wrap_pyfunction!(charge_sector_plan, module)?)?;
    module.add_function(wrap_pyfunction!(charge_sector_plan_compact, module)?)?;
    module.add_function(wrap_pyfunction!(majorana_canonicalize, module)?)?;
    module.add_function(wrap_pyfunction!(majorana_multiply, module)?)?;
    module.add_function(wrap_pyfunction!(majorana_to_fermion, module)?)?;
    module.add_function(wrap_pyfunction!(fermion_to_majorana, module)?)?;
    module.add_function(wrap_pyfunction!(mapping_plan, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_backend_plan, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_group, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_compatibility_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_incompatibility_edges, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_find_z2_symmetries, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_z2_tapering_plan, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_restrict_u1, module)?)?;
    module.add_function(wrap_pyfunction!(u1_basis_words, module)?)?;
    module.add_function(wrap_pyfunction!(u1_circuit_plan, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_propagation_engine, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_propagation_batch, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_spps_engine, module)?)?;
    module.add_function(wrap_pyfunction!(structured_dense, module)?)?;
    module.add_function(wrap_pyfunction!(structured_sparse, module)?)?;
    module.add_function(wrap_pyfunction!(structured_sparse_plan, module)?)?;
    module.add_function(wrap_pyfunction!(structured_fermion_canonicalize, module)?)?;
    module.add_function(wrap_pyfunction!(structured_fermion_multiply, module)?)?;
    module.add_function(wrap_pyfunction!(structured_fermion_jordan_wigner, module)?)?;
    module.add_function(wrap_pyfunction!(structured_boson_canonicalize, module)?)?;
    module.add_function(wrap_pyfunction!(structured_boson_multiply, module)?)?;
    module.add_function(wrap_pyfunction!(structured_hybrid_multiply, module)?)?;
    module.add_function(wrap_pyfunction!(structured_hybrid_canonicalize, module)?)?;
    module.add_function(wrap_pyfunction!(structured_hybrid_jordan_wigner, module)?)?;
    Ok(())
}
