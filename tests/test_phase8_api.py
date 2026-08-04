"""Phase 8 public API contract tests."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import tencirpauli as tcp
from tencirpauli import advanced


def test_top_level_and_advanced_manifests_are_separated() -> None:
    advanced_names = {
        "NativeMVPPlan",
        "BackendMVPPlan",
        "U1MvpPlan",
        "ChargeMvpPlan",
        "PropagationEngine",
        "SPPSEngine",
        "GateTape",
        "OperatorBuilder",
        "U1CircuitPlan",
    }
    assert advanced_names.isdisjoint(set(tcp.__all__))
    assert advanced_names <= set(advanced.__all__)
    assert not hasattr(tcp, "NativeMVPPlan")
    assert not hasattr(tcp, "GateTape")


def test_operator_counts_and_unified_code_validation() -> None:
    zero = tcp.PauliOperator.empty(2)
    assert zero.term_count == 0
    assert len(zero) == 0
    assert not zero
    assert tcp.PauliWord.from_codes([np.int64(1)]).to_string() == "X"
    with pytest.raises(TypeError):
        tcp.PauliWord.from_codes([True])
    with pytest.raises(TypeError):
        tcp.PauliWord.from_codes([1.0])
    with pytest.raises(ValueError, match=r"0\.\.4"):
        tcp.PauliWord.from_codes([4])
    with pytest.raises(TypeError):
        tcp.PauliOperator.from_code_arrays(np.asarray([[True]], dtype=bool), [1.0])
    with pytest.raises(TypeError):
        tcp.PauliOperator.from_code_arrays(np.asarray([[1.0]]), [1.0])


def test_grouping_metadata_is_canonical_and_mode_defaults_are_safe() -> None:
    operator = tcp.PauliOperator.from_terms(2, [("XX", 1.0), ("ZZ", 1.0)])
    qwc = operator.group_commuting()
    general = operator.group_commuting(mode="general")
    assert qwc.group_count == len(qwc.groups)
    assert qwc.term_count == len(operator)
    assert len(qwc.term_to_group) == qwc.term_count
    assert qwc.mode == "qubit_wise" and qwc.measurement_ready
    assert general.mode == "general" and not general.measurement_ready
    assert operator.compatibility_matrix().shape == (2, 2)


def test_mvp_plan_contract_and_factory_only_construction() -> None:
    operator = tcp.PauliOperator.from_terms(1, [("X", 1.0)])
    plan = operator.compile(target="native_mvp")
    assert plan.target == "native_mvp"
    assert plan.dimension == 2 and plan.term_count == 1
    assert plan.apply(np.asarray([1.0, 0.0]), max_bytes=None).shape == (2,)
    with pytest.raises(TypeError, match="factory"):
        advanced.NativeMVPPlan(1, 1, "term_direct", object())
    assert "max_bytes" in inspect.signature(plan.apply).parameters
    assert (
        inspect.signature(plan.apply).parameters["max_bytes"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_circuit_capabilities_hermiticity_and_u1_state_contract() -> None:
    spps = tcp.SPPSCircuit(1)
    assert not isinstance(spps, tcp.PropagationCircuit)
    assert not hasattr(spps, "ptm")
    assert not hasattr(spps, "propagate_operator")
    assert not hasattr(spps, "profile")
    assert type(tcp.SPPSCircuit.from_qir([], {"nqubits": 1})) is tcp.SPPSCircuit

    nonhermitian = tcp.PauliOperator.from_terms(1, [("X", 1.0j)])
    propagation = tcp.PropagationCircuit(1)
    with pytest.raises(ValueError, match="Hermitian"):
        propagation.expectation(nonhermitian)
    with pytest.raises(ValueError, match="Hermitian"):
        spps.compile(nonhermitian)

    circuit = tcp.U1Circuit(3, particle_number=1, occupied=[2])
    assert not hasattr(circuit, "k")
    assert not hasattr(circuit, "to_dense")
    assert circuit.state().shape == (3,)
    assert circuit.state_full().shape == (8,)
    assert circuit.sector.unrank(0) == (0, 0, 1)
    basis = circuit.sector.basis_states()
    packed = circuit.sector.basis_words_packed()
    assert basis.shape == (3, 3) and basis.dtype == np.uint8
    assert packed.shape == (3, 1) and packed.dtype == np.uint64
    assert not basis.flags.writeable and not packed.flags.writeable


def test_mapping_name_and_keyword_contract() -> None:
    mapping = tcp.FermionQubitMapping.from_name("jordan_wigner", 1)
    assert mapping.name == "jordan_wigner"
    assert not hasattr(mapping, "mapping_name")
    assert not hasattr(mapping, "mapping")
    operator = tcp.FermionOperator.from_terms(
        1, [(((0, "create"), (0, "annihilate")), 1.0)]
    )
    assert operator.compile(target="dense", mapping=mapping).shape == (2, 2)
