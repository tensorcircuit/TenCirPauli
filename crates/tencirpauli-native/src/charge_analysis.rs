use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use tencir_pauli_core::{Complex64, PauliWord};

use crate::operator::NativePauliOperatorHandle;

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
    let term_count = operator.core().terms().len();
    let estimated = (term_count as u128)
        .checked_mul(operator.core().nqubits().saturating_add(1) as u128)
        .and_then(|value| value.checked_mul(128))
        .ok_or_else(|| PyValueError::new_err("charge analysis size overflow"))?;
    if estimated > max_bytes as u128 {
        return Err(PyMemoryError::new_err(format!(
            "exact additive-charge analysis requires approximately {estimated} bytes, exceeding max_bytes={max_bytes}"
        )));
    }
    py.allow_threads(|| {
        let mut aggregate = HashMap::<PauliWord, Complex64>::new();
        for term in operator.core().terms() {
            for (index, code) in term.word.codes().into_iter().enumerate() {
                if code != 1 && code != 2 {
                    continue;
                }
                let difference = qubit_levels[index].0 - qubit_levels[index].1;
                if difference == 0.0 {
                    continue;
                }
                let mut changed = term.word.codes();
                changed[index] = if code == 1 { 2 } else { 1 };
                let word = PauliWord::from_codes(operator.core().nqubits(), &changed)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                let scale = if code == 1 { -difference } else { difference };
                let contribution = term.coefficient * Complex64::new(0.0, scale);
                let entry = aggregate.entry(word).or_default();
                *entry += contribution;
            }
        }
        let nonzero = aggregate
            .values()
            .filter(|value| value.re != 0.0 || value.im != 0.0)
            .count();
        Ok::<(bool, usize), PyErr>((nonzero == 0, nonzero))
    })
}
