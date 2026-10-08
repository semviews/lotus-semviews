# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Stable identities for rows, predicates, and views (see docs/ARCHITECTURE.md)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

PROMPT_VERSION_TEMPLATE = "lotus-{version}-sem_filter-default"


def row_id(values: Sequence[object]) -> str:
    """Content hash of the langex columns' values, in langex order."""
    payload = json.dumps([str(v) for v in values], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def view_key(langex: str, prompt_version: str, model: str) -> str:
    payload = json.dumps([langex, prompt_version, model], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
