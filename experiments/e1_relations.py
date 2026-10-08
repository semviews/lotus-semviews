# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""E1: prevalence of epsilon-containment, equivalence, and exclusion among real predicates.

For every ordered pair (p, q) on the same table, from the oracle tape:
  eps(p in q)   = P[q = 0 | p = 1]       (containment error)
  excl(p, q)    = P[q = 1 | p = 1]       (exclusion error)
Also compares with the LLM judge's logical relation and with human-label containment (W3).
Writes experiments/results/E1_pairs.parquet and E1_summary.parquet.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semviews.workloads.runner import ROOT, Bench  # noqa: E402

EPS = (0.01, 0.05, 0.1)


def human_truth(b: Bench, pid: str) -> np.ndarray | None:
    h = b.pool[pid].get("human")
    if not h:
        return None
    m = np.zeros(len(b.rows), dtype=bool)
    for col, vals in h.items():
        if col == "emotions":
            em = b.rows["emotions"].map(json.loads)
            m |= em.map(lambda xs, vals=vals: any(v in xs for v in vals)).to_numpy()
        else:
            m |= b.rows[col].isin(vals).to_numpy()
    return m


def pairs(dataset: str) -> pd.DataFrame:
    b = Bench(dataset)
    preds = [p for p in b.oracle0.predicate_ids if p in b.pool]
    y = {p: b.oracle0.ground_truth(p).astype(bool) for p in preds}
    rel_path = os.path.join(ROOT, "data", "relations", f"{dataset}.json")
    rel = json.load(open(rel_path)) if os.path.exists(rel_path) else {}
    hum = {p: human_truth(b, p) for p in preds}
    rows = []
    for p, q in itertools.permutations(preds, 2):
        yp, yq = y[p], y[q]
        npos = yp.sum()
        eps = float((~yq[yp]).mean()) if npos else np.nan
        excl = float(yq[yp].mean()) if npos else np.nan
        r = dict(dataset=dataset, p=p, q=q, sel_p=float(yp.mean()), sel_q=float(yq.mean()), eps=eps, excl=excl,
                 judge=rel.get(f"{p}||{q}"))
        if hum[p] is not None and hum[q] is not None:
            hp, hq = hum[p], hum[q]
            r["human_eps"] = float((~hq[hp]).mean()) if hp.sum() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main(datasets: list[str]) -> None:
    out_dir = os.path.join(ROOT, "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.concat([pairs(d) for d in datasets], ignore_index=True)
    df.to_parquet(os.path.join(out_dir, "E1_pairs.parquet"), index=False)
    summ = []
    for d, g in df.groupby("dataset"):
        n_pred = g.p.nunique()
        s = dict(dataset=d, n_pred=n_pred, n_ordered_pairs=len(g))
        for e in EPS:
            s[f"contain@{e}"] = int((g.eps <= e).sum())
            s[f"excl@{e}"] = int((g.excl <= e).sum())
            s[f"queries_with_container@{e}"] = int(g[g.eps <= e].q.nunique())  # some p contained in q
            s[f"queries_with_contained@{e}"] = int(g[g.eps <= e].p.nunique())  # q contained in some p
        # judged implies (p => q) versus measured containment error
        imp = g[g.judge == "implies"]
        s["judge_implies"] = len(imp)
        s["judge_implies_median_eps"] = float(imp.eps.median()) if len(imp) else np.nan
        s["judge_implies_share_eps_gt_0.1"] = float((imp.eps > 0.1).mean()) if len(imp) else np.nan
        summ.append(s)
    summ = pd.DataFrame(summ)
    summ.to_parquet(os.path.join(out_dir, "E1_summary.parquet"), index=False)
    pd.set_option("display.width", 250)
    print(summ.T)


if __name__ == "__main__":
    main(sys.argv[1:] or ["goemotions", "dbpedia", "reviews", "pubmed"])
