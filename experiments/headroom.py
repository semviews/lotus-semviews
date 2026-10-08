# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Headroom and scale studies of the operator. Replays the tape; evaluation only.

    python experiments/headroom.py

* perfect: the score is the oracle label itself (a floor no real score can beat; the label never
  enters a bound, so the operator stays valid).
* warm: before every query, the catalog holds the complete oracle column of every *other*
  predicate (an upper bound on what views can offer this learner).
* scale: the 5,000 tuples are tiled k times (k = 4, 20). Duplicated tuples keep the label
  distribution and the scores; only the fixed per-query costs are amortized differently. This is
  a simulation, not a measurement on larger data.
Writes experiments/results/HEADROOM.parquet.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process, before numpy loads

import sys  # noqa: E402
from collections import defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_all  # noqa: E402
from improvements import FULL, TUNE3  # noqa: E402

from semviews.catalog import DERIVED, Catalog, View  # noqa: E402
from semviews.operators.region_filter import CertConfig, CertifiedFilter  # noqa: E402
from semviews.operators.reuse_filter import Targets  # noqa: E402
from semviews.workloads.runner import OPERATOR  # noqa: E402

GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)


class PerfectScore(CertifiedFilter):
    truth: dict = {}

    def _features(self, pid, views):
        y = self.truth[pid].astype(float)
        return np.stack([4 * y - 2 + 0.01 * np.random.default_rng(0).standard_normal(len(y))], 1)


class _Meter:
    def __init__(self):
        self.calls, self.helper_calls, self.calls_by_purpose = 0, 0, defaultdict(int)


class TiledOracle:
    model = "tiled"

    def __init__(self, truth: dict, k: int):
        self.truth, self.k, self.meter = truth, k, _Meter()

    def label(self, pid, rows, purpose):
        rows = np.asarray(rows)
        self.meter.calls += len(rows)
        self.meter.calls_by_purpose[purpose] += len(rows)
        return np.tile(self.truth[pid], self.k)[rows].astype(np.int8)


class TiledHelper:
    def __init__(self, h, k: int):
        self.h, self.k = h, k

    def has(self, pid):
        return self.h.has(pid)

    def score(self, pid, rows):
        return np.tile(self.h.score(pid), self.k)[rows]


VARIANTS = {
    "base": {}, "noviews": dict(use_views=False), "best": FULL["best"], "best_noviews": FULL["best_noviews"],
    **{k: v for k, v in TUNE3.items() if k != "base"},
    "paper": OPERATOR,  # the paper's operator (configs/experiments/sprint.yaml)
    "paper_nv": {**OPERATOR, "use_views": False},
}


def job(a: dict) -> list[dict]:
    ds, variant, seed, k, mode = a["dataset"], a["variant"], a["seed"], a["k"], a["mode"]
    b = run_all.bench(ds)
    cfg = CertConfig(**{**VARIANTS[variant], **({"n_grid": GRID} if k > 1 else {}),
                        **({"shape_n": 200 * k // 5} if mode == "scale_shape" else {})})
    order = run_all.order_for(b.preds, seed)
    N = k * len(b.row_ids)
    oracle, helper = TiledOracle(b.truth, k), TiledHelper(b.helper0, k)
    rowf = np.tile(b.row_pca, (k, 1))
    rng = np.random.default_rng(seed)
    t = Targets(0.9, 0.9, 0.05)
    cat = Catalog(ds, N)
    cls = CertifiedFilter
    if mode == "perfect":
        cls = type("P", (PerfectScore,), {"truth": {p: np.tile(b.truth[p], k) for p in b.preds}})
    out = []
    for qi, pid in enumerate(order):
        if mode == "warm":
            cat = Catalog(ds, N)
            for p in b.preds:
                if p != pid:
                    lab = np.tile(b.truth[p], k).astype(np.int8)
                    cat.register(View(p, b.texts[p], "init", lab, np.full(N, DERIVED, np.int8)))
        f = cls(oracle, cat, b.texts, helper=helper, embedder=b.embedder, cfg=cfg, row_features=rowf)
        c0 = oracle.meter.calls
        res = f.run(pid, t, rng)
        y = np.tile(b.truth[pid], k)
        tp = int((res.mask & y).sum())
        rec = tp / y.sum() if y.sum() else 1.0
        prec = tp / res.mask.sum() if res.mask.sum() else 1.0
        out.append(dict(dataset=ds, variant=variant, mode=mode, k=k, seed=seed, query_idx=qi, predicate_id=pid,
                        oracle_calls=oracle.meter.calls - c0, n_rows=N, selectivity=float(y.mean()),
                        met=bool(prec >= 0.9 and rec >= 0.9)))
    return out


def scale_fix() -> None:
    """Tiled scale check of the third tuning round (pilot datasets only)."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [dict(dataset=d, variant=v, seed=s, k=k, mode="scale") for d in ("goemotions", "pubmed")
            for v in TUNE3 for s in range(2) for k in (4, 20)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as ex:
        for r in ex.map(job, jobs):
            rows += r
    pd.DataFrame(rows).to_parquet(os.path.join(run_all.RESULTS, "HEADROOM_fix.parquet"), index=False)


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    ds = ["pubmed", "goemotions", "dbpedia", "reviews"]
    jobs = []
    for d in ds:
        for s in range(4):
            jobs += [dict(dataset=d, variant=v, seed=s, k=1, mode="perfect") for v in ("base", "best")]
            jobs += [dict(dataset=d, variant=v, seed=s, k=1, mode="warm") for v in ("base", "best")]
            for k in (1, 4, 20):
                jobs += [dict(dataset=d, variant=v, seed=s, k=k, mode="scale") for v in VARIANTS]
            for k in (4, 20):
                jobs += [dict(dataset=d, variant=v, seed=s, k=k, mode="scale_shape") for v in ("base", "best")]
            jobs += [dict(dataset=d, variant="base", seed=s, k=20, mode="perfect")]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as ex:
        for r in ex.map(job, jobs):
            rows += r
    pd.DataFrame(rows).to_parquet(os.path.join(run_all.RESULTS, "HEADROOM.parquet"), index=False)
    print(len(rows), "rows")


def paper() -> None:
    """Headroom and tiled scale for the paper's operator -> results/op2/HEADROOM.parquet."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for d in ["pubmed", "goemotions", "dbpedia", "reviews"]:
        for s in range(4):
            jobs += [dict(dataset=d, variant="paper", seed=s, k=1, mode=m) for m in ("perfect", "warm")]
            jobs += [dict(dataset=d, variant="paper", seed=s, k=k, mode="scale") for k in (1, 4, 20)]
            jobs += [dict(dataset=d, variant="paper", seed=s, k=k, mode="scale_shape") for k in (4, 20)]
            jobs += [dict(dataset=d, variant="paper", seed=s, k=20, mode="perfect")]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as ex:
        for r in ex.map(job, jobs):
            rows += r
    os.makedirs(os.path.join(run_all.RESULTS, "op2"), exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(run_all.RESULTS, "op2", "HEADROOM.parquet"), index=False)


if __name__ == "__main__":
    {"fix": scale_fix, "paper": paper}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
