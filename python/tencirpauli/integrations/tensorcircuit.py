"""Optional TensorCircuit backend-plan adapter.

TensorCircuit is imported only when an adapter is explicitly requested. The
Rust core and public top-level package remain independent of that dependency.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..hamiltonian import (
    DEFAULT_MAX_BYTES,
    BackendMVPPlan,
    _check_allocation,
    _dimension,
)
from ..propagation import GateTape


@dataclass(frozen=True)
class TensorCircuitTapeConversion:
    """A supported TensorCircuit QIR converted to a native ``GateTape``."""

    tape: GateTape
    parameters: tuple[Any, ...]


def require_tensorcircuit() -> Any:
    """Import TensorCircuit or fail with an actionable optional-dependency error."""
    try:
        return importlib.import_module("tensorcircuit")
    except ImportError as error:
        raise ImportError(
            "TensorCircuit integration requires the optional 'tensorcircuit-ng' "
            "dependency; install tencirpauli[tensorcircuit]"
        ) from error


def gate_tape_from_circuit(
    circuit: Any,
    *,
    parameter_order: Optional[Sequence[Any]] = None,
) -> TensorCircuitTapeConversion:
    """Convert supported numeric/direct-symbol TensorCircuit QIR to a tape.

    Only fixed Clifford gates and the six Pauli rotations with a direct
    numeric angle or one direct SymPy symbol are accepted.  The adapter does
    not connect the resulting tape to TensorCircuit autodiff or tracing.
    """
    require_tensorcircuit()
    if not hasattr(circuit, "to_qir"):
        raise TypeError("circuit must provide TensorCircuit to_qir()")
    qir = circuit.to_qir()
    nqubits = getattr(circuit, "_nqubits", getattr(circuit, "nqubits", None))
    if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
        raise ValueError("could not determine circuit qubit count")

    try:
        sympy_module: Any = importlib.import_module("sympy")
    except ImportError:
        sympy_module = None

    def is_symbol(value: Any) -> bool:
        return sympy_module is not None and isinstance(value, sympy_module.Symbol)

    operations: list[tuple[str, tuple[int, ...], Any]] = []
    symbols: list[Any] = []
    fixed = {"x", "y", "z", "h", "s", "sdg", "cnot", "cz", "swap"}
    rotations = {"rx", "ry", "rz", "rxx", "ryy", "rzz"}
    for instruction in qir:
        if not isinstance(instruction, dict):
            raise ValueError("TensorCircuit QIR entries must be dictionaries")
        name = str(instruction.get("name", "")).lower()
        wires = tuple(instruction.get("index", ()))
        if name in fixed:
            if instruction.get("parameters"):
                raise ValueError(f"unsupported parameters on TensorCircuit gate {name}")
            operations.append((name, wires, None))
            continue
        if name not in rotations:
            raise ValueError(f"unsupported TensorCircuit gate {name!r}")
        parameters = instruction.get("parameters", {})
        if not isinstance(parameters, dict) or set(parameters) != {"theta"}:
            raise ValueError(f"{name} requires one direct theta parameter")
        angle = parameters["theta"]
        if is_symbol(angle):
            if angle not in symbols:
                symbols.append(angle)
            operations.append((name, wires, angle))
        else:
            if isinstance(angle, (bool, np.bool_)):
                raise TypeError("TensorCircuit angles must be finite real values")
            try:
                numeric_angle = float(angle)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "TensorCircuit angles must be numeric or direct SymPy symbols"
                ) from error
            if not math.isfinite(numeric_angle):
                raise ValueError("TensorCircuit angles must be finite")
            operations.append((name, wires, numeric_angle))

    if parameter_order is None:
        ordered_symbols = tuple(symbols)
    else:
        ordered_symbols = tuple(parameter_order)
        if len(set(ordered_symbols)) != len(ordered_symbols):
            raise ValueError("parameter_order must not contain duplicates")
        if set(ordered_symbols) != set(symbols) or any(
            not is_symbol(symbol) for symbol in ordered_symbols
        ):
            raise ValueError("parameter_order must exactly cover direct QIR symbols")
    symbol_slots = {symbol: index for index, symbol in enumerate(ordered_symbols)}

    tape = GateTape(nqubits)
    for name, wires, angle in operations:
        if name in fixed:
            if name in {"x", "y", "z", "h", "s", "sdg"}:
                getattr(tape, name)(wires[0])
            else:
                getattr(tape, name)(wires[0], wires[1])
            continue
        if name in {"rx", "ry", "rz"}:
            append = getattr(tape, name)
            wire_args: tuple[int, ...] = (wires[0],)
        else:
            append = getattr(tape, name)
            wire_args = (wires[0], wires[1])
        if is_symbol(angle):
            append(*wire_args, parameter=symbol_slots[angle])
        else:
            append(*wire_args, angle=angle)
    return TensorCircuitTapeConversion(tape=tape, parameters=ordered_symbols)


def backend_mvp(
    plan: BackendMVPPlan,
    coefficients: Optional[Sequence[complex]] = None,
    backend: Any = None,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> Any:
    """Return a TensorCircuit-backend MVP callable for a pure-array plan.

    The returned callable accepts a flat ``2**n`` state or a rank-``n`` tensor.
    Plan structure is fixed before tracing; coefficients may be replaced by a
    backend tensor for a differentiable parameter buffer.
    """
    tensorcircuit = require_tensorcircuit()
    runtime_backend = backend if backend is not None else tensorcircuit.backend
    dimension = _dimension(plan.nqubits)
    estimated_bytes = dimension * ((len(plan.coefficients) + 1) * 8 + 48)
    _check_allocation(estimated_bytes, max_bytes, "TensorCircuit MVP adapter")
    if coefficients is None:
        coefficient_values = runtime_backend.convert_to_tensor(plan.coefficients)
    elif hasattr(coefficients, "shape"):
        coefficient_values = coefficients
    else:
        coefficient_values = runtime_backend.convert_to_tensor(coefficients)
    if len(plan.coefficients) != int(
        runtime_backend.shape_tuple(coefficient_values)[0]
    ):
        raise ValueError("coefficient buffer length does not match backend MVP plan")

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
            weight = coefficient_values[term_index] * (1j ** y_counts[term_index])
            total = total + term_state * weight
        return runtime_backend.reshape(total, (-1,)) if flat else total

    return mvp


def _plan_code(plan: BackendMVPPlan, term_index: int, qubit: int) -> int:
    x = (int(plan.x_words[term_index, qubit // 64]) >> (qubit % 64)) & 1
    z = (int(plan.z_words[term_index, qubit // 64]) >> (qubit % 64)) & 1
    return {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}[(x, z)]
