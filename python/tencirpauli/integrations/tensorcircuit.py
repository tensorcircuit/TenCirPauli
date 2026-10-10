"""TensorCircuit backend-plan and circuit conversion boundary.

TensorCircuit is a required Python runtime dependency. It remains behind this
Python boundary so that the Rust core never depends on the framework.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, cast

import numpy as np

from ..hamiltonian import (
    DEFAULT_MAX_BYTES,
    BackendMVPPlan,
    _check_allocation,
    _dimension,
)
from ..propagation import GateTape
from ..u1_circuit import U1Circuit


@dataclass(frozen=True)
class TensorCircuitTapeConversion:
    """A supported TensorCircuit QIR converted to a native ``GateTape``."""

    tape: GateTape


@dataclass(frozen=True)
class TensorCircuitU1Conversion:
    """A supported TensorCircuit U(1) QIR converted to ``U1Circuit``."""

    circuit: U1Circuit


def require_tensorcircuit() -> Any:
    """Import TensorCircuit or report an incomplete runtime installation."""
    try:
        return importlib.import_module("tensorcircuit")
    except ImportError as error:
        raise ImportError(
            "TenCirPauli requires the 'tensorcircuit-ng' runtime dependency; "
            "reinstall the package with its dependencies"
        ) from error


def _qir_gate_name(instruction: dict[str, Any]) -> str:
    """Read gate semantics from the factory rather than its display label."""
    if "gatef" not in instruction or instruction.get("is_channel"):
        name = str(instruction.get("name", "")).lower()
    else:
        factory = instruction["gatef"]
        name = str(getattr(factory, "n", getattr(factory, "__name__", ""))).lower()
        if name.endswith("_gate"):
            name = name[:-5]
    return {"sd": "sdg"}.get(name, name)


def gate_tape_from_circuit(
    circuit: Any,
) -> TensorCircuitTapeConversion:
    """Convert concrete TensorCircuit QIR to a native gate tape."""
    require_tensorcircuit()
    if not hasattr(circuit, "to_qir"):
        raise TypeError("circuit must provide TensorCircuit to_qir()")
    qir = circuit.to_qir()
    nqubits = getattr(circuit, "_nqubits", getattr(circuit, "nqubits", None))
    if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
        raise ValueError("could not determine circuit qubit count")

    operations: list[tuple[str, tuple[int, ...], object]] = []
    fixed = {"x", "y", "z", "h", "s", "sdg", "cnot", "cz", "swap"}
    rotations = {"rx", "ry", "rz", "rxx", "ryy", "rzz"}
    channels = {
        "depolarizing": ("px", "py", "pz"),
        "phase_damping": ("gamma",),
        "amplitude_damping": ("gamma", "p"),
    }
    for instruction in qir:
        if not isinstance(instruction, dict):
            raise ValueError("TensorCircuit QIR entries must be dictionaries")
        name = _qir_gate_name(instruction)
        wires = tuple(instruction.get("index", ()))
        if name in fixed:
            if instruction.get("parameters"):
                raise ValueError(f"unsupported parameters on TensorCircuit gate {name}")
            operations.append((name, wires, None))
            continue
        if name in channels:
            if instruction.get("is_channel") is not True:
                raise ValueError(
                    f"TensorCircuit channel {name} requires is_channel=True"
                )
            if not isinstance(instruction.get("channel_unitary"), bool):
                raise ValueError("TensorCircuit channel_unitary must be a boolean")
            parameters = instruction.get("channel_parameters", {})
            expected = channels[name]
            if not isinstance(parameters, dict) or set(parameters) != set(expected):
                raise ValueError(f"{name} requires channel parameters {expected}")
            values: list[float] = []
            for parameter_name in expected:
                value = parameters[parameter_name]
                if isinstance(value, (bool, np.bool_)):
                    raise TypeError("TensorCircuit channel parameters must be real")
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "TensorCircuit channel parameters must be concrete"
                    ) from error
                if not math.isfinite(numeric):
                    raise ValueError("TensorCircuit channel parameters must be finite")
                values.append(numeric)
            operations.append((name, wires, tuple(values)))
            continue
        if name not in rotations:
            raise ValueError(f"unsupported TensorCircuit gate {name!r}")
        parameters = instruction.get("parameters", {})
        if not isinstance(parameters, dict) or set(parameters) != {"theta"}:
            raise ValueError(f"{name} requires one direct theta parameter")
        angle = parameters["theta"]
        if isinstance(angle, (bool, np.bool_)):
            raise TypeError("TensorCircuit angles must be finite real values")
        try:
            numeric_angle = float(angle)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "TensorCircuit angles must be concrete numeric values"
            ) from error
        if not math.isfinite(numeric_angle):
            raise ValueError("TensorCircuit angles must be finite")
        operations.append((name, wires, numeric_angle))

    tape = GateTape(nqubits)
    for name, wires, payload in operations:
        if name in fixed:
            if name in {"x", "y", "z", "h", "s", "sdg"}:
                getattr(tape, name)(wires[0])
            else:
                getattr(tape, name)(wires[0], wires[1])
            continue
        if name in channels:
            assert isinstance(payload, tuple)
            method = {
                "depolarizing": "depolarizing",
                "phase_damping": "phasedamping",
                "amplitude_damping": "amplitudedamping",
            }[name]
            parameter_names = channels[name]
            getattr(tape, method)(
                wires[0], **dict(zip(parameter_names, payload, strict=True))
            )
            continue
        if name in {"rx", "ry", "rz"}:
            append = getattr(tape, name)
            wire_args: tuple[int, ...] = (wires[0],)
        else:
            append = getattr(tape, name)
            wire_args = (wires[0], wires[1])
        assert payload is not None
        append(*wire_args, angle=cast(float, payload))
    return TensorCircuitTapeConversion(tape=tape)


def u1_circuit_from_tensorcircuit(
    circuit: Any,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> TensorCircuitU1Conversion:
    """Convert TensorCircuit U1Circuit QIR without importing it at module load."""
    require_tensorcircuit()
    if not hasattr(circuit, "to_qir"):
        raise TypeError("circuit must provide TensorCircuit to_qir()")
    nqubits = getattr(circuit, "_nqubits", getattr(circuit, "nqubits", None))
    if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
        raise ValueError("could not determine circuit qubit count")
    circuit_params = getattr(circuit, "circuit_param", {})
    if not isinstance(circuit_params, dict):
        raise ValueError("TensorCircuit U1 circuit parameters are unavailable")
    if "k" not in circuit_params and "filled" not in circuit_params:
        raise ValueError("TensorCircuit U1 circuit must expose k or filled")

    qir: list[dict[str, object]] = []
    supported = {"rz", "rzz", "cz", "cphase", "swap", "iswap", "diagonal"}
    for instruction in circuit.to_qir():
        if not isinstance(instruction, dict):
            raise ValueError("TensorCircuit QIR entries must be dictionaries")
        name = str(instruction.get("name", "")).lower()
        if name not in supported:
            raise ValueError(f"unsupported U1Circuit gate {name!r}")
        wires = tuple(instruction.get("index", ()))
        parameters = instruction.get("parameters", {})
        item: dict[str, object] = {"name": name, "index": wires}
        if name in {"rz", "rzz", "cphase", "iswap"}:
            if not isinstance(parameters, dict) or set(parameters) != {"theta"}:
                raise ValueError(f"{name} requires one theta parameter")
            theta = parameters["theta"]
            if isinstance(theta, (bool, np.bool_)):
                raise TypeError("U1Circuit angles must be finite real values")
            try:
                numeric = float(theta)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "U1Circuit angles must be concrete numeric values"
                ) from error
            if not math.isfinite(numeric):
                raise ValueError("U1Circuit angles must be finite")
            theta = numeric
            item["parameters"] = {"theta": theta}
        if name == "diagonal":
            gate = instruction.get("gate")
            payload = getattr(gate, "tensor", gate)
            values = np.asarray(payload, dtype=np.complex128).reshape(-1)
            item["diagonal"] = values
        qir.append(item)

    params = dict(circuit_params)
    params["nqubits"] = nqubits
    params["particle_number"] = params.pop("k", None)
    params["occupied"] = params.pop("filled", None)
    params["initial_state"] = params.pop("inputs", None)
    converted = U1Circuit.from_qir(qir, params, max_bytes=max_bytes)
    return TensorCircuitU1Conversion(converted)


def backend_mvp(
    plan: BackendMVPPlan,
    coefficients: Optional[Sequence[complex]] = None,
    backend: Any = None,
    *,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> Any:
    """Return a TensorCircuit-backend MVP callable for a pure-array plan.

    The returned callable accepts a flat or local-rank state. Binary Pauli plans
    use packed masks; uniform Weyl plans use factorized phase-and-cyclic-shift
    operations directly on the backend tensor.
    Plan structure is fixed before tracing; coefficients may be replaced by a
    backend tensor for a differentiable parameter buffer.
    """
    tensorcircuit = require_tensorcircuit()
    runtime_backend = backend if backend is not None else tensorcircuit.backend
    if plan.plan_kind == "direct_weyl":
        return _direct_weyl_backend_mvp(plan, coefficients, runtime_backend, max_bytes)
    dimension = _dimension(plan.nqubits)
    estimated_bytes = dimension * ((len(plan.coefficients) + 1) * 8 + 48)
    _check_allocation(estimated_bytes, max_bytes, "TensorCircuit MVP adapter")
    coefficient_values: Any
    if coefficients is None:
        coefficient_values = plan.coefficients
    elif hasattr(coefficients, "shape"):
        coefficient_values = coefficients
    else:
        coefficient_values = np.asarray(coefficients, dtype=np.complex128)
    if len(plan.coefficients) != int(np.shape(coefficient_values)[0]):
        raise ValueError("coefficient buffer length does not match backend MVP plan")
    coefficient_cache: dict[str, Any] = {}
    phase_cache: dict[str, tuple[Any, ...]] = {}

    term_masks: list[Any] = []
    flip_axes: list[tuple[int, ...]] = []
    y_counts: list[int] = []
    for term_index in range(len(plan.coefficients)):
        z_mask: np.ndarray[Any, Any] = np.ones((1,) * plan.nqubits, dtype=np.float64)
        flips = []
        y_count = 0
        for qubit in range(plan.nqubits):
            code = _plan_code(plan, term_index, qubit)
            if code in (2, 3):
                local = np.array([1.0, -1.0]).reshape(
                    (1,) * qubit + (2,) + (1,) * (plan.nqubits - qubit - 1)
                )
                z_mask = z_mask * local
            if code in (1, 2):
                flips.append(qubit)
            if code == 2:
                y_count += 1
        term_masks.append(runtime_backend.convert_to_tensor(z_mask))
        flip_axes.append(tuple(flips))
        y_counts.append(y_count)

    def mvp(state: Any) -> Any:
        state_shape = runtime_backend.shape_tuple(state)
        if len(state_shape) == 1:
            if state_shape[0] != 2**plan.nqubits:
                raise ValueError("flat state has incompatible length")
            tensor_state = runtime_backend.reshape(state, (2,) * plan.nqubits)
            flat = True
        elif tuple(state_shape) == (2,) * plan.nqubits:
            tensor_state = state
            flat = False
        else:
            raise ValueError("state must be flat or rank-nqubits with binary axes")
        dtype = runtime_backend.dtype(tensor_state)
        coefficient_tensor = _cached_backend_coefficients(
            coefficient_values, runtime_backend, dtype, coefficient_cache
        )
        phases = phase_cache.get(dtype)
        if phases is None:
            phases = tuple(
                runtime_backend.cast(1j**y_count, dtype) for y_count in y_counts
            )
            phase_cache[dtype] = phases
        total = runtime_backend.zeros_like(tensor_state)
        for term_index, mask in enumerate(term_masks):
            term_state = tensor_state * runtime_backend.cast(mask, dtype)
            if flip_axes[term_index]:
                slices = tuple(
                    (
                        slice(None, None, -1)
                        if axis in flip_axes[term_index]
                        else slice(None)
                    )
                    for axis in range(plan.nqubits)
                )
                term_state = term_state[slices]
            weight = coefficient_tensor[term_index] * phases[term_index]
            total = total + term_state * weight
        return runtime_backend.reshape(total, (-1,)) if flat else total

    return mvp


def _direct_weyl_backend_mvp(
    plan: BackendMVPPlan,
    coefficients: Optional[Sequence[complex]],
    runtime_backend: Any,
    max_bytes: Optional[int],
) -> Any:
    """Create a pure-array executor for a uniform Weyl plan.

    For the direct convention ``W(a,b)=X**a Z**b``, a source basis value is
    multiplied by ``omega**(b*k)`` and then moved to ``k+a mod d``. All slices
    and phase masks are static, which keeps the executor compatible with JAX
    tracing without building a circuit or a sparse matrix.
    """
    dimensions = tuple(plan.local_dimensions)
    dimension = plan.dimension
    n_sites = len(dimensions)
    d = plan.qudit_dimension
    estimated_bytes = dimension * 32 + int(plan.estimated_bytes)
    _check_allocation(estimated_bytes, max_bytes, "TensorCircuit Weyl MVP adapter")
    coefficient_values: Any
    if coefficients is None:
        coefficient_values = plan.coefficients
    elif hasattr(coefficients, "shape"):
        coefficient_values = coefficients
    else:
        coefficient_values = np.asarray(coefficients, dtype=np.complex128)
    if np.shape(coefficient_values) != (len(plan.coefficients),):
        raise ValueError("coefficient buffer length does not match backend MVP plan")
    coefficient_cache: dict[str, Any] = {}
    phase_masks: list[list[Optional[Any]]] = []
    omega = np.exp(2j * np.pi / d)
    for term_index in range(len(plan.coefficients)):
        masks: list[Optional[Any]] = []
        for axis in range(n_sites):
            b = int(plan.b_exponents[term_index, axis])
            if b == 0:
                masks.append(None)
                continue
            local = omega ** (b * np.arange(d))
            shape = [1] * n_sites
            shape[axis] = d
            masks.append(runtime_backend.convert_to_tensor(local.reshape(shape)))
        phase_masks.append(masks)

    def mvp(state: Any) -> Any:
        state_shape = runtime_backend.shape_tuple(state)
        if len(state_shape) == 1 and state_shape[0] == dimension:
            tensor_state = runtime_backend.reshape(state, dimensions)
            flat = True
        elif tuple(state_shape) == dimensions:
            tensor_state = state
            flat = False
        else:
            raise ValueError(
                f"state must be flat with length {dimension} or have shape {dimensions}"
            )
        dtype = runtime_backend.dtype(tensor_state)
        coefficient_tensor = _cached_backend_coefficients(
            coefficient_values, runtime_backend, dtype, coefficient_cache
        )
        total = runtime_backend.zeros_like(tensor_state)
        for term_index in range(len(plan.coefficients)):
            term_state = tensor_state
            for axis, mask in enumerate(phase_masks[term_index]):
                if mask is not None:
                    term_state = term_state * runtime_backend.cast(mask, dtype)
                shift = int(plan.a_exponents[term_index, axis]) % d
                if shift:
                    tail = [slice(None)] * n_sites
                    head = [slice(None)] * n_sites
                    tail[axis] = slice(-shift, None)
                    head[axis] = slice(None, -shift)
                    term_state = runtime_backend.concat(
                        (term_state[tuple(tail)], term_state[tuple(head)]), axis=axis
                    )
            total = total + term_state * coefficient_tensor[term_index]
        return runtime_backend.reshape(total, (-1,)) if flat else total

    return mvp


def _cast_backend_coefficients(
    coefficients: Any, runtime_backend: Any, dtype: str
) -> Any:
    """Cast complex coefficients to the TensorCircuit state dtype."""
    if dtype not in {"complex64", "complex128"}:
        raise TypeError(
            "TensorCircuit backend MVP requires a complex64 or complex128 state"
        )
    if isinstance(coefficients, np.ndarray):
        host_dtype = np.complex64 if dtype == "complex64" else np.complex128
        coefficients = np.asarray(coefficients, dtype=host_dtype)
        return runtime_backend.convert_to_tensor(coefficients, dtype=dtype)
    return runtime_backend.cast(coefficients, dtype)


def _cached_backend_coefficients(
    coefficients: Any,
    runtime_backend: Any,
    dtype: str,
    cache: dict[str, Any],
) -> Any:
    """Reuse casts for static host coefficients without freezing backend tensors."""
    if not isinstance(coefficients, np.ndarray):
        return _cast_backend_coefficients(coefficients, runtime_backend, dtype)
    cached = cache.get(dtype)
    if cached is None:
        cached = _cast_backend_coefficients(coefficients, runtime_backend, dtype)
        cache[dtype] = cached
    return cached


def _plan_code(plan: BackendMVPPlan, term_index: int, qubit: int) -> int:
    x = (int(plan.x_words[term_index, qubit // 64]) >> (qubit % 64)) & 1
    z = (int(plan.z_words[term_index, qubit // 64]) >> (qubit % 64)) & 1
    return {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}[(x, z)]
