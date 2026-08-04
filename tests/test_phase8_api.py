"""Phase 8 public API contract tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import tencirpauli as tcp
from tencirpauli import advanced


def test_top_level_and_advanced_manifests_are_separated() -> None:
    assert tuple(tcp.__all__) == (
        "DEFAULT_MAX_BYTES",
        "AdditiveCharge",
        "AdditiveSymmetryAnalysis",
        "BosonOperator",
        "BosonTerm",
        "BosonWord",
        "COOMatrix",
        "CSRMatrix",
        "CanonicalizationResult",
        "ChargeSector",
        "ComputationalBasisState",
        "FermionOperator",
        "FermionQubitMapping",
        "FermionTerm",
        "FermionWord",
        "GeneralCommutingGroupingResult",
        "HybridOperator",
        "HybridTerm",
        "MVPPlan",
        "MajoranaOperator",
        "MajoranaProduct",
        "MajoranaTerm",
        "MajoranaWord",
        "OperatorSpace",
        "Parameter",
        "ParameterExpr",
        "PauliOperator",
        "PauliPhase",
        "PauliProduct",
        "PauliTerm",
        "PauliWord",
        "ProductBlochState",
        "ProfiledExpectation",
        "PropagationBatch",
        "PropagationBatchValueAndGradient",
        "PropagationCircuit",
        "PropagationProfile",
        "PropagationValueAndGradient",
        "QWCGroupingResult",
        "QuditProduct",
        "QuditWeylOperator",
        "QuditWeylTerm",
        "QuditWeylWord",
        "SPPSCircuit",
        "SPPSEstimate",
        "SPPSValueEstimate",
        "U1Circuit",
        "U1CircuitValueAndGradient",
        "U1Sector",
        "Z2SymmetryAnalysis",
        "ZeroState",
        "__version__",
        "backend_mvp",
    )
    assert tuple(advanced.__all__) == (
        "BackendMVPPlan",
        "CanonicalizationArrayResult",
        "ChargeLazyMvpPlan",
        "ChargeMvpPlan",
        "ChargeRestrictedOperator",
        "GateTape",
        "NativeMVPPlan",
        "OperatorBuilder",
        "PropagationCircuitPlan",
        "PropagationEngine",
        "SPPSCircuitPlan",
        "SPPSEngine",
        "U1CircuitPlan",
        "U1MvpPlan",
        "U1RestrictedOperator",
        "Z2TaperingPlan",
    )
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

    for dtype in (np.bool_, np.float64, object):
        with pytest.raises(TypeError):
            tcp.PauliOperator.from_code_arrays(
                np.empty((0, 2), dtype=dtype), np.empty(0)
            )
        with pytest.raises(TypeError):
            tcp.PauliOperator.from_code_arrays(
                np.empty((1, 0), dtype=dtype), np.zeros(1)
            )
    for dtype in (np.int8, np.uint8, np.int64, np.uint64):
        assert (
            tcp.PauliOperator.from_code_arrays(
                np.empty((0, 2), dtype=dtype), np.empty(0)
            ).term_count
            == 0
        )
        assert (
            tcp.PauliOperator.from_code_arrays(
                np.empty((1, 0), dtype=dtype), np.ones(1)
            ).term_count
            == 1
        )


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


def test_all_four_mvp_plans_share_flat_apply_contract() -> None:
    operator = tcp.PauliOperator.from_terms(1, [("I", 1.0)])
    native = operator.compile(target="native_mvp")
    backend = operator.compile(target="backend_mvp")
    u1 = operator.restrict_u1(tcp.U1Sector(1, 0)).mvp_plan()
    space = tcp.OperatorSpace(qubits=1)
    charge = tcp.AdditiveCharge(space, qubits={0: (0, 1)})
    charged = operator.restrict_charge(charge.sector(0)).mvp_plan()
    plans = (native, backend, u1, charged)
    for candidate in plans:
        assert candidate.dimension == 1 or candidate.dimension == 2
        assert candidate.term_count == 1
        assert candidate.estimated_bytes >= 0
        assert candidate.target == "native_mvp" or candidate.target == "backend_mvp"
        result = candidate.apply(np.ones(candidate.dimension, dtype=np.complex128))
        assert result.shape == (candidate.dimension,)
        assert result.dtype == np.complex128
        assert result.flags.c_contiguous and result.flags.owndata
        assert result.flags.writeable
        np.testing.assert_allclose(candidate(np.ones(candidate.dimension)), result)


def test_circuit_capabilities_hermiticity_and_u1_state_contract() -> None:
    spps = tcp.SPPSCircuit(1)
    assert not isinstance(spps, tcp.PropagationCircuit)
    assert not hasattr(spps, "ptm")
    assert not hasattr(spps, "propagate_operator")
    assert not hasattr(spps, "profile")
    assert not hasattr(tcp.SPPSCircuit, "ptm")
    assert not hasattr(tcp.SPPSCircuit, "propagate_operator")
    assert not hasattr(tcp.SPPSCircuit, "profile")
    assert type(tcp.SPPSCircuit.from_qir([], {"nqubits": 1})) is tcp.SPPSCircuit
    with pytest.raises(ValueError, match="PTM"):
        tcp.SPPSCircuit.from_qir(
            [
                {
                    "name": "ptm",
                    "index": [0],
                    "matrix": np.eye(4, dtype=np.float64),
                }
            ],
            {"nqubits": 1},
        )
    with pytest.raises(AttributeError):
        tcp.SPPSCircuit.ptm(spps, [0], np.eye(4, dtype=np.float64))

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
    with pytest.raises(TypeError, match="named factory"):
        tcp.FermionQubitMapping("jordan_wigner", 1, ((1,),))
    assert (
        inspect.signature(tcp.FermionQubitMapping).parameters["max_bytes"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_propagation_engine_term_count_is_consumed_canonical_count() -> None:
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0), ("X", -1.0), ("Z", 0.25)])
    engine = advanced.PropagationEngine(advanced.GateTape(1), observable)
    assert observable.term_count == 1
    assert engine.term_count == 1


def test_u1_exact_hermiticity_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    import tencirpauli.pauli as pauli_module

    calls = 0
    original = pauli_module._native.pauli_operator_is_hermitian

    def counted(*args: object, **kwargs: object) -> bool:
        nonlocal calls
        calls += 1
        return bool(original(*args, **kwargs))

    monkeypatch.setattr(pauli_module._native, "pauli_operator_is_hermitian", counted)
    observable = tcp.PauliOperator.from_terms(1, [("Z", 1.0)])
    circuit = tcp.U1Circuit(1, particle_number=0)
    circuit.value_and_grad(observable)
    circuit.value_and_grad(observable)
    plan = circuit.compile()
    plan.value_and_grad(None, observable)
    assert calls == 1


def test_advanced_reference_is_published() -> None:
    assert Path("docs/advanced-api.md").is_file()
