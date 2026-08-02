"""TensorCircuit NumPy/JAX adapter smoke tests."""

from __future__ import annotations

import numpy as np
import pytest
import tensorcircuit as tc

from tencirpauli import PauliOperator
from tencirpauli.integrations.tensorcircuit import (
    backend_mvp,
    gate_tape_from_circuit,
    require_tensorcircuit,
    u1_circuit_from_tensorcircuit,
)


def test_tensorcircuit_runtime_dependency_is_available() -> None:
    assert require_tensorcircuit() is tc


def test_numpy_backend_plan_smoke() -> None:
    tc.set_backend("numpy")
    operator = PauliOperator.from_terms(2, (("XY", 0.5), ("ZI", -1.25j)))
    plan = operator.backend_mvp_plan()
    state = np.arange(4, dtype=np.float64) + 1j * np.arange(4, dtype=np.float64)
    result = backend_mvp(plan)(state)
    np.testing.assert_allclose(result, operator.dense() @ state, rtol=1e-12, atol=1e-12)
    with pytest.raises(MemoryError, match="TensorCircuit MVP adapter"):
        backend_mvp(plan, max_bytes=1)


def test_jax_backend_plan_smoke() -> None:
    pytest.importorskip("jax")
    tc = require_tensorcircuit()
    tc.set_backend("jax")
    operator = PauliOperator.from_terms(2, (("XX", 0.5), ("YZ", 1.25)))
    plan = operator.backend_mvp_plan()
    state = np.arange(4, dtype=np.float64) + 1j * np.arange(4, dtype=np.float64)
    result = tc.backend.numpy(backend_mvp(plan)(tc.backend.convert_to_tensor(state)))
    np.testing.assert_allclose(result, operator.dense() @ state, rtol=1e-12, atol=1e-12)


def test_numeric_qir_tape_conversion() -> None:
    circuit = tc.Circuit(2)
    circuit.h(0)
    circuit.rx(1, theta=0.23)
    circuit.cnot(0, 1)
    converted = gate_tape_from_circuit(circuit)
    assert converted.tape.nqubits == 2
    assert len(converted.tape) == 3
    assert converted.parameters == ()


def test_symbol_qir_tape_conversion_and_order() -> None:
    sympy = pytest.importorskip("sympy")
    theta, phi = sympy.symbols("theta phi")
    circuit = tc.SymbolCircuit(2)
    circuit.rx(0, theta=theta)
    circuit.ry(1, theta=phi)
    converted = gate_tape_from_circuit(circuit, parameter_order=(phi, theta))
    assert converted.parameters == (phi, theta)
    assert converted.tape.nparameters == 2


def test_u1_qir_conversion_matches_native_state() -> None:
    tc.set_dtype("complex128")
    tc.set_backend("numpy")
    circuit = tc.U1Circuit(3, k=1, filled=[0])
    circuit.rz(0, theta=0.31)
    circuit.iswap(0, 1, theta=0.63)
    converted = u1_circuit_from_tensorcircuit(circuit)
    np.testing.assert_allclose(
        converted.circuit.state(),
        np.asarray(circuit.state()),
        atol=1e-11,
        rtol=1e-10,
    )
