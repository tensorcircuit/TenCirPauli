"""Reject numbered development labels outside the vibe documentation tree."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".git",
    ".benchmarks",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "site",
    "target",
    ".conda",
}
LABEL_PATTERN = re.compile(
    r"(?:(?i:phase|stage|milestone)[ _-]?\d+(?:\.\d+)?|\bP\d+\b)"
)


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative_parts = path.relative_to(ROOT).parts
        if not path.is_file() or relative_parts[:2] == ("docs", "vibe"):
            continue
        if SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return files


def main() -> int:
    violations: list[str] = []
    for path in iter_files():
        relative = path.relative_to(ROOT)
        if path.name == "AGENTS.local.md":
            continue
        if LABEL_PATTERN.search(str(relative)):
            violations.append(f"{relative}: numbered label in path")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if LABEL_PATTERN.search(line):
                violations.append(f"{relative}:{line_number}: {line.strip()}")
    if violations:
        print("numbered development labels found outside docs/vibe:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
