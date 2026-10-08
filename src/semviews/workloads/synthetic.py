# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Synthetic tapes with controlled predicate relations, for validity and leakage tests.

Rows carry latent binary attributes. Predicates are attributes, unions, or intersections
of attributes, corrupted by a small flip rate, so containment holds only approximately
(epsilon-containment), as with real LLM labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n_rows: int = 4000, n_attr: int = 6, flip: float = 0.02, seed: int = 0):
    """Return (tape_df, helper_df, texts) for a small family of related predicates."""
    rng = np.random.default_rng(seed)
    base_rate = rng.uniform(0.08, 0.35, size=n_attr)
    attrs = rng.random((n_rows, n_attr)) < base_rate
    preds: dict[str, np.ndarray] = {}
    for j in range(n_attr):
        preds[f"a{j}"] = attrs[:, j]
    for j in range(0, n_attr - 1, 2):
        preds[f"a{j}_or_a{j+1}"] = attrs[:, j] | attrs[:, j + 1]
        preds[f"a{j}_and_a{j+1}"] = attrs[:, j] & attrs[:, j + 1]
    row_ids = np.array([f"r{i:06d}" for i in range(n_rows)])
    tape, helper = [], []
    for pid, y in preds.items():
        y = y ^ (rng.random(n_rows) < flip)
        tape.append(pd.DataFrame({"row_id": row_ids, "predicate_id": f"syn/{pid}", "model": "syn-oracle",
                                  "label": y.astype(np.int8), "tokens_in": 100, "tokens_out": 4}))
        logit = 2.2 * (y.astype(float) - 0.5) + rng.normal(0, 1.2, size=n_rows)
        helper.append(pd.DataFrame({"row_id": row_ids, "predicate_id": f"syn/{pid}", "score": 1 / (1 + np.exp(-logit))}))
    texts = {f"syn/{p}": f"{{text}} has property {p.replace('_', ' ')}" for p in preds}
    return pd.concat(tape, ignore_index=True), pd.concat(helper, ignore_index=True), texts, row_ids
