"""TensorCircuit gate factories determine QIR semantics, not display labels."""

from __future__ import annotations

import numpy as np
import pytest
import tensorcircuit as tc

import tencirpauli as tcp


@pytest.mark.parametrize("gate, expected", [("s", -1.0), ("sd", 1.0)])
@pytest.mark.parametrize("circuit_type", [tcp.PropagationCircuit, tcp.SPPSCircuit])
def test_inverse_phase_gate_uses_factory(
    circuit_type: type[tcp.PropagationCircuit] | type[tcp.SPPSCircuit],
    gate: str,
    expected: float,
) -> None:
    with tc.runtime_backend("numpy"), tc.runtime_dtype("complex128"):
        reference = tc.Circuit(1)
        getattr(reference, gate)(0)
        reference.h(0)
        reference = reference.inverse()
        converted = circuit_type.from_circuit(reference)
        observable = tcp.PauliOperator.from_terms(1, [("Y", 1.0)])
        kwargs = (
            {"samples_per_term": 2, "seed": 0}
            if circuit_type is tcp.SPPSCircuit
            else {}
        )
        result = converted.expectation(observable, **kwargs)
        value = result if isinstance(result, float) else result.value
        np.testing.assert_allclose(reference.expectation_ps(y=[0]), expected)
        np.testing.assert_allclose(value, expected)


def test_custom_display_names_preserve_gate_semantics() -> None:
    with tc.runtime_backend("numpy"), tc.runtime_dtype("complex128"):
        reference = tc.Circuit(1)
        reference.h(0, name="prepare")
        reference.sd(0, name="s")
        reference.rx(0, theta=0.37, name="rz")
        converted = tcp.PropagationCircuit.from_circuit(reference)
        for code in [1, 2, 3]:
            observable = tcp.PauliOperator.from_terms(1, [([code], 1.0)])
            np.testing.assert_allclose(
                converted.expectation(observable),
                reference.expectation_ps(ps=[code]),
                atol=1e-12,
            )


def test_unsupported_gate_cannot_use_a_supported_display_name() -> None:
    with tc.runtime_backend("numpy"):
        reference = tc.Circuit(1)
        reference.t(0, name="s")
        with pytest.raises(ValueError, match="unsupported TensorCircuit gate 't'"):
            tcp.PropagationCircuit.from_circuit(reference)
