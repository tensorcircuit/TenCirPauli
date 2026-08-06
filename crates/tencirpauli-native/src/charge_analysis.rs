use crate::operator::NativePauliOperatorHandle;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Analyze a pure-Pauli additive charge without exporting operator terms.
#[pyfunction]
pub(crate) fn pauli_analyze_charge_handle(
    py: Python<'_>,
    operator: &NativePauliOperatorHandle,
    qubit_levels: Vec<(f64, f64)>,
    max_bytes: usize,
) -> PyResult<(bool, usize)> {
    if qubit_levels.len() != operator.core().nqubits() {
        return Err(PyValueError::new_err("charge and operator layouts differ"));
    }
    py.allow_threads(|| {
        operator
            .core()
            .analyze_charge(&qubit_levels, max_bytes as u128)
            .map_err(crate::convert::map_error)
    })
}
