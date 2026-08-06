"""Optional end-to-end JAX VQE example smoke test."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_pyscf_example_runs_jax_adam() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    pytest.importorskip("pyscf")

    environment = os.environ.copy()
    python_path = [str(ROOT / "python")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quantum_chemistry_pyscf.py")],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    energy_match = re.search(
        r"H2 JAX VQE: ([+-]?[0-9.]+) -> ([+-]?[0-9.]+)", result.stdout
    )
    assert energy_match is not None
    initial_energy, final_energy = (float(value) for value in energy_match.groups())
    assert final_energy < initial_energy - 1.0e-3

    gradient_match = re.search(
        r"final gradient norm: ([0-9.]+e[+-][0-9]+)", result.stdout
    )
    assert gradient_match is not None
    assert float(gradient_match.group(1)) > 0.0
