# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""CertifiedFilter: validity over seeds, savings from views, and leakage on synthetic tapes."""

from __future__ import annotations

import numpy as np
import pytest

from semviews.catalog import Catalog
from semviews.operators.region_filter import CertConfig, CertifiedFilter
from semviews.operators.reuse_filter import Targets
from semviews.oracle import HelperScores, TapeOracle
from semviews.workloads.synthetic import make_synthetic

T = Targets(0.9, 0.9, 0.1)


@pytest.fixture(scope="module")
def synth():
    return make_synthetic(n_rows=4000, flip=0.01, seed=4)


def run(synth, seed, cfg=None, oracle_cls=TapeOracle, tape=None):
    tape_df, helper_df, texts, row_ids = synth
    oracle = oracle_cls(tape if tape is not None else tape_df, "syn-oracle", row_ids=row_ids)
    cf = CertifiedFilter(oracle, Catalog("syn", len(row_ids)), texts, helper=HelperScores(helper_df, row_ids), cfg=cfg or CertConfig())
    rng = np.random.default_rng(seed)
    res = []
    for pid in sorted(texts):
        with oracle.meter.scope(pid):
            c0 = oracle.meter.calls
            r = cf.run(pid, T, rng)
            calls = oracle.meter.calls - c0
        y = oracle.ground_truth(pid).astype(bool)
        tp = (r.mask & y).sum()
        res.append(dict(calls=calls, recall=tp / max(y.sum(), 1), precision=tp / max(r.mask.sum(), 1) if r.mask.sum() else 1.0,
                        mask=r.mask))
    return res, oracle


def test_validity_over_seeds(synth):
    misses = total = 0
    for seed in range(80):
        res, _ = run(synth, seed)
        total += len(res)
        misses += sum((r["recall"] < 0.9) or (r["precision"] < 0.9) for r in res)
    assert misses / total <= T.delta


def test_views_save_calls(synth):
    a, _ = run(synth, 1)
    b, _ = run(synth, 1, cfg=CertConfig(use_views=False))
    assert sum(r["calls"] for r in a) < sum(r["calls"] for r in b)


def test_leakage(synth):
    tape_df, helper_df, texts, row_ids = synth
    seen: dict = {}

    class Recording(TapeOracle):
        def label(self, predicate_id, rows, *, purpose):
            seen.setdefault(predicate_id, set()).update(np.asarray(rows).tolist())
            return super().label(predicate_id, rows, purpose=purpose)

    res1, _ = run(synth, 9, oracle_cls=Recording)
    pos = {r: i for i, r in enumerate(row_ids)}
    rp = tape_df.row_id.map(pos).to_numpy()
    unpaid = np.array([x not in seen.get(p, ()) for p, x in zip(tape_df.predicate_id, rp)])
    flipped = tape_df.copy()
    flipped.loc[unpaid, "label"] = 1 - flipped.loc[unpaid, "label"]
    res2, _ = run(synth, 9, tape=flipped)
    assert unpaid.sum() > 0
    for a, b in zip(res1, res2):
        assert np.array_equal(a["mask"], b["mask"])
