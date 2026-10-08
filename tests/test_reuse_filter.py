# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""ReuseFilter: validity on synthetic tapes, savings from views, and leakage."""

from __future__ import annotations

import numpy as np
import pytest

from semviews.catalog import Catalog
from semviews.operators.reuse_filter import ReuseConfig, ReuseFilter, Targets
from semviews.oracle import HelperScores, TapeOracle
from semviews.workloads.synthetic import make_synthetic


@pytest.fixture(scope="module")
def synth():
    return make_synthetic(n_rows=4000, seed=3)


def _run_workload(synth, seed, cfg=None, targets=Targets(0.9, 0.9, 0.1), order=None, tape=None):
    tape_df, helper_df, texts, row_ids = synth
    tape_df = tape if tape is not None else tape_df
    oracle = TapeOracle(tape_df, "syn-oracle", row_ids=row_ids)
    helper = HelperScores(helper_df, row_ids)
    cat = Catalog("syn", len(row_ids))
    rf = ReuseFilter(oracle, cat, texts, helper=helper, cfg=cfg or ReuseConfig())
    rng = np.random.default_rng(seed)
    pids = order or sorted(texts)
    out = []
    for pid in pids:
        with oracle.meter.scope(pid):
            before = oracle.meter.calls
            res = rf.run(pid, targets, rng)
            calls = oracle.meter.calls - before
        y = oracle.ground_truth(pid).astype(bool)
        tp = (res.mask & y).sum()
        rec = tp / max(y.sum(), 1)
        prec = tp / max(res.mask.sum(), 1) if res.mask.sum() else 1.0
        out.append(dict(pid=pid, calls=calls, recall=rec, precision=prec, mask=res.mask.copy(), stats=res.stats))
    return out, oracle


def test_validity_over_seeds(synth):
    """Missed-target rate over 60 seeds x 12 queries stays at or below delta = 0.1."""
    misses = total = 0
    for seed in range(60):
        res, _ = _run_workload(synth, seed)
        for r in res:
            total += 1
            misses += (r["recall"] < 0.9) or (r["precision"] < 0.9)
    assert misses / total <= 0.1


def test_views_save_calls(synth):
    with_views, _ = _run_workload(synth, 1)
    no_views, _ = _run_workload(synth, 1, cfg=ReuseConfig(use_views=False))
    assert sum(r["calls"] for r in with_views) < sum(r["calls"] for r in no_views)


def test_leakage_unpaid_labels_do_not_matter(synth):
    """Flipping every label the method never paid for leaves its output unchanged."""
    tape_df, helper_df, texts, row_ids = synth
    order = sorted(texts)[:6]
    res1, oracle1 = _run_workload(synth, 5, order=order)
    paid = {pid: arr.copy() for pid, arr in oracle1._charged.items()}  # charged in the last scope only
    # recompute the full paid set by replaying with a recording wrapper
    seen: dict[str, set] = {}

    class Recording(TapeOracle):
        def label(self, predicate_id, rows, *, purpose):
            seen.setdefault(predicate_id, set()).update(np.asarray(rows).tolist())
            return super().label(predicate_id, rows, purpose=purpose)

    oracle = Recording(tape_df, "syn-oracle", row_ids=row_ids)
    cat = Catalog("syn", len(row_ids))
    rf = ReuseFilter(oracle, cat, texts, helper=HelperScores(helper_df, row_ids), cfg=ReuseConfig())
    rng = np.random.default_rng(5)
    masks1 = [rf.run(pid, Targets(0.9, 0.9, 0.1), rng).mask for pid in order]
    flipped = tape_df.copy()
    pos = {r: i for i, r in enumerate(row_ids)}
    rowpos = flipped.row_id.map(pos).to_numpy()
    unpaid = np.array([rp not in seen.get(p, ()) for p, rp in zip(flipped.predicate_id, rowpos)])
    flipped.loc[unpaid, "label"] = 1 - flipped.loc[unpaid, "label"]
    oracle2 = TapeOracle(flipped, "syn-oracle", row_ids=row_ids)
    rf2 = ReuseFilter(oracle2, Catalog("syn", len(row_ids)), texts, helper=HelperScores(helper_df, row_ids), cfg=ReuseConfig())
    rng = np.random.default_rng(5)
    masks2 = [rf2.run(pid, Targets(0.9, 0.9, 0.1), rng).mask for pid in order]
    assert unpaid.sum() > 0
    for a, b in zip(masks1, masks2):
        assert np.array_equal(a, b)
    assert paid is not None
