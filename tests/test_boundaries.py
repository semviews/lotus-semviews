# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Repository rules enforced as tests: import boundary, tape access, headers, no secrets."""

from __future__ import annotations

import ast
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ["oracle", "catalog", "relate", "certify", "operators", "advisor"]


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def test_core_does_not_import_lotus():
    for pkg in CORE:
        for f in (ROOT / "src" / "semviews" / pkg).rglob("*.py"):
            assert "lotus" not in _imports(f), f"{f} imports lotus"


def test_only_oracle_reads_tapes():
    """No module outside semviews.oracle and the tape builder opens tape files."""
    allowed = {"oracle", "adapters"}
    for f in (ROOT / "src" / "semviews").rglob("*.py"):
        rel = f.relative_to(ROOT / "src" / "semviews").parts[0]
        if rel in allowed:
            continue
        src = f.read_text()
        assert "data/tapes" not in src and "__oracle-" not in src, f"{f} references tape files"
    for f in (ROOT / "src" / "semviews" / "baselines").rglob("*.py"):
        assert "ground_truth" not in f.read_text(), f"{f} reads ground truth"


def test_apache_headers():
    for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        if f.stat().st_size == 0:
            continue
        assert "Apache License, Version 2.0" in f.read_text()[:400], f"missing header: {f}"


def test_no_identifying_strings():
    """No author names, employers, or internal system names in tracked text files."""
    banned = [s for s in os.environ.get("SEMVIEWS_BANNED", "").split(",") if s]
    if not banned:
        return
    pat = re.compile("|".join(re.escape(b) for b in banned), re.I)
    for f in ROOT.rglob("*"):
        if any(p in f.parts for p in (".git", ".venv", "data", "models", "docs")) or not f.is_file():
            continue
        if f.suffix in {".py", ".md", ".tex", ".yaml", ".yml", ".toml", ".bib"}:
            assert not pat.search(f.read_text(errors="ignore")), f"identifying string in {f}"
