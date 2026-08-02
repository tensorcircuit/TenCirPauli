"""Matched complex128 JAX reference for finite Pauli-weight propagation.

This module is benchmark-only. It uses an independent local matrix table and a
global k-local Pauli basis, then lets JAX compile only the repeated coefficient
recurrence. It is intentionally not imported by the public package.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np


PAULIS = (
    np.array([[1, 0], [0, 1]], dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
)


def _local_decompose(matrix: np.ndarray, width: int) -> dict[tuple[int, ...], complex]:
    result = {}
    for codes in product(range(4), repeat=width):
        basis = np.array([[1.0 + 0.0j]])
        for code in codes:
            basis = np.kron(basis, PAULIS[code])
        coefficient = np.trace(basis @ matrix) / (2**width)
        if abs(coefficient) > 1e-12:
            result[codes] = complex(coefficient)
    return result


def _local_gate(operation: tuple[object, ...]) -> tuple[np.ndarray, int]:
    kind = str(operation[0])
    if kind in ("x", "y", "z", "h", "s", "sdg"):
        if kind == "h":
            matrix = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        elif kind in ("s", "sdg"):
            matrix = np.diag([1.0, 1j if kind == "s" else -1j])
        else:
            matrix = PAULIS[{"x": 1, "y": 2, "z": 3}[kind]]
        return matrix, 1
    if kind in ("cnot", "cz", "swap"):
        if kind == "cnot":
            matrix = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=np.complex128,
            )
        elif kind == "cz":
            matrix = np.diag([1, 1, 1, -1]).astype(np.complex128)
        else:
            matrix = np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=np.complex128,
            )
        return matrix, 2
    if kind in ("rx", "ry", "rz", "rxx", "ryy", "rzz"):
        axis = {"rx": 1, "ry": 2, "rz": 3, "rxx": 1, "ryy": 2, "rzz": 3}[kind]
        width = 1 if kind in ("rx", "ry", "rz") else 2
        generator = PAULIS[axis]
        if width == 2:
            generator = np.kron(generator, generator)
        theta = float(operation[-1])
        return (
            np.cos(theta / 2) * np.eye(2**width) - 1j * np.sin(theta / 2) * generator,
            width,
        )
    raise ValueError(f"unknown reference gate {kind}")


def _rotation_transitions(kind: str, width: int) -> tuple[
    dict[int, dict[int, complex]],
    dict[int, dict[int, complex]],
    dict[int, dict[int, complex]],
]:
    axis = {"rx": 1, "ry": 2, "rz": 3, "rxx": 1, "ryy": 2, "rzz": 3}[kind]
    generator = PAULIS[axis]
    if width == 2:
        generator = np.kron(generator, generator)
    commute = {}
    cosine = {}
    sine = {}
    for input_code, input_codes in enumerate(product(range(4), repeat=width)):
        input_matrix = np.array([[1.0 + 0.0j]])
        for code in input_codes:
            input_matrix = np.kron(input_matrix, PAULIS[code])
        commutes = np.allclose(generator @ input_matrix, input_matrix @ generator)
        if commutes:
            commute[input_code] = {input_code: 1.0}
            cosine[input_code] = {}
            sine[input_code] = {}
        else:
            commute[input_code] = {}
            cosine[input_code] = {input_code: 1.0}
            branch = _local_decompose(1j * generator @ input_matrix, width)
            sine[input_code] = {
                sum(
                    code * (4 ** (width - index - 1))
                    for index, code in enumerate(codes)
                ): value
                for codes, value in branch.items()
            }
    return commute, cosine, sine


@dataclass
class MatchedJaxPropagation:
    basis: tuple[tuple[int, ...], ...]
    transitions: tuple[tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...], ...]
    initial: np.ndarray
    expectation_mask: np.ndarray
    function: object
    gradient_function: object

    @classmethod
    def build(
        cls,
        nqubits: int,
        operations: Sequence[tuple[object, ...]],
        structures: Iterable[Sequence[int]],
        coefficients: Sequence[complex],
        max_weight: int,
    ) -> "MatchedJaxPropagation":
        basis = tuple(
            codes
            for codes in product(range(4), repeat=nqubits)
            if sum(code != 0 for code in codes) <= max_weight
        )
        positions = {codes: index for index, codes in enumerate(basis)}
        transitions = []
        for operation in operations:
            kind = str(operation[0])
            width = (
                1
                if kind in ("rx", "ry", "rz")
                else 2 if kind in ("rxx", "ryy", "rzz") else _local_gate(operation)[1]
            )
            wire0 = int(operation[1])
            wire1 = int(operation[2]) if width == 2 else None
            gate_transitions = []
            if kind in ("rx", "ry", "rz", "rxx", "ryy", "rzz"):
                commute, cosine, sine = _rotation_transitions(kind, width)
                local_transitions = (commute, cosine, sine)
            else:
                matrix, width = _local_gate(operation)
                local_transitions = ({}, {})
                for input_codes in product(range(4), repeat=width):
                    input_matrix = np.array([[1.0 + 0.0j]])
                    for code in input_codes:
                        input_matrix = np.kron(input_matrix, PAULIS[code])
                    output = _local_decompose(
                        matrix.conj().T @ input_matrix @ matrix, width
                    )
                    local_index = sum(
                        code * (4 ** (width - index - 1))
                        for index, code in enumerate(input_codes)
                    )
                    local_transitions[0][local_index] = {
                        sum(
                            code * (4 ** (width - index - 1))
                            for index, code in enumerate(codes)
                        ): value
                        for codes, value in output.items()
                    }
            for local_transitions_for_kind in local_transitions:
                sources = []
                destinations = []
                values = []
                for source, codes in enumerate(basis):
                    local_codes = (
                        (codes[wire0],) if width == 1 else (codes[wire0], codes[wire1])
                    )
                    local_source = sum(
                        code * (4 ** (width - index - 1))
                        for index, code in enumerate(local_codes)
                    )
                    for local_destination, value in local_transitions_for_kind.get(
                        local_source, {}
                    ).items():
                        output_codes = list(codes)
                        if width == 1:
                            output_codes[wire0] = local_destination
                        else:
                            output_codes[wire0], output_codes[wire1] = divmod(
                                local_destination, 4
                            )
                        output = tuple(output_codes)
                        destination = positions.get(output)
                        if destination is not None:
                            sources.append(source)
                            destinations.append(destination)
                            values.append(value)
                gate_transitions.append(
                    (
                        np.asarray(sources, dtype=np.int32),
                        np.asarray(destinations, dtype=np.int32),
                        np.asarray(values, dtype=np.complex128),
                    )
                )
            transitions.append(tuple(gate_transitions))
        initial = np.zeros(len(basis), dtype=np.complex128)
        for structure, coefficient in zip(structures, coefficients):
            initial[positions[tuple(structure)]] += coefficient
        expectation_mask = np.asarray(
            [all(code in (0, 3) for code in codes) for codes in basis], dtype=np.float64
        )
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        static_transitions = tuple(
            tuple((jnp.asarray(s), jnp.asarray(d), jnp.asarray(v)) for s, d, v in gate)
            for gate in transitions
        )

        def run(parameters: object) -> object:
            values = jnp.asarray(initial)
            for operation, gate in zip(
                reversed(operations), reversed(static_transitions)
            ):
                next_values = jnp.zeros_like(values)
                kind = str(operation[0])
                if kind in ("rx", "ry", "rz", "rxx", "ryy", "rzz") and len(gate) == 3:
                    theta = (
                        parameters[int(operation[3])]
                        if int(operation[3]) >= 0
                        else float(operation[-1])
                    )
                    cosine, sine = jnp.cos(theta), jnp.sin(theta)
                    for transition, scale in zip(gate, (1.0, cosine, sine)):
                        source, destination, coefficient = transition
                        next_values = next_values.at[destination].add(
                            scale * coefficient * values[source]
                        )
                else:
                    source, destination, coefficient = gate[0]
                    next_values = next_values.at[destination].add(
                        coefficient * values[source]
                    )
                values = next_values
            return jnp.real(jnp.sum(values * jnp.asarray(expectation_mask)))

        compiled = jax.jit(run)
        compiled_gradient = jax.jit(jax.value_and_grad(run))
        return cls(
            basis,
            tuple(transitions),
            initial,
            expectation_mask,
            compiled,
            compiled_gradient,
        )

    def expectation(self, parameters: Sequence[float]) -> float:
        value = self.function(np.asarray(parameters, dtype=np.float64))
        return float(value.block_until_ready())

    def value_and_gradient(
        self, parameters: Sequence[float]
    ) -> tuple[float, np.ndarray]:
        value, gradient = self.gradient_function(
            np.asarray(parameters, dtype=np.float64)
        )
        value = value.block_until_ready()
        gradient = gradient.block_until_ready()
        return float(value), np.asarray(gradient, dtype=np.float64)
