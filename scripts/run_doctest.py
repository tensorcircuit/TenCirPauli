"""Run doctests against the package selected by the active Python environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run pytest doctests from the imported package directory."""
    import tencirpauli

    package_file = tencirpauli.__file__
    if package_file is None:
        raise RuntimeError("cannot locate the imported tencirpauli package")
    package_dir = str(Path(package_file).resolve().parent)
    return subprocess.call(
        [sys.executable, "-m", "pytest", "--doctest-modules", package_dir]
    )


if __name__ == "__main__":
    raise SystemExit(main())
