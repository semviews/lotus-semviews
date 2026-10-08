# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Exploration-session workloads (W_S): the same predicate pools, issued as refinement sessions.

A session starts at a random predicate; each next predicate is drawn uniformly from the three
unused predicates whose *text* is most similar to the current one (sentence embeddings of the
predicate text, no labels). After `length` queries a new session starts at a random unused
predicate; the catalog persists across sessions. Every predicate is issued exactly once, as in the
random-order workloads, so only the order differs.

    python experiments/sessions.py   -> experiments/results/op2/ES.parquet
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import run_all  # noqa: E402

METHODS = ["semviews", "semviews_noviews", "bargain_pr", "oracle_all"]


def session_order(b, seed: int, length: int = 8, k: int = 3) -> list[str]:
    rng = np.random.default_rng(20_000 + seed)
    preds = list(b.preds)
    E = np.stack([b.pred_emb[p] for p in preds])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    unused = set(range(len(preds)))
    order: list[int] = []
    while unused:
        cur = int(rng.choice(sorted(unused)))
        for _ in range(length):
            order.append(cur)
            unused.discard(cur)
            if not unused:
                break
            cand = sorted(unused, key=lambda j: -S[cur, j])[:k]
            cur = int(rng.choice(cand))
    return [preds[i] for i in order]


def main() -> None:
    cfg = run_all.yaml.safe_load(open(os.path.join(run_all.ROOT, "configs", "experiments", "sprint.yaml")))
    T = [cfg["targets"]["precision"], cfg["targets"]["recall"], cfg["targets"]["delta"]]
    jobs = []
    for d in cfg["datasets"]:
        b = run_all.bench(d)
        for s in range(cfg["orders"]):
            order = session_order(b, s)
            jobs += [dict(dataset=d, method=m, seed=s, targets=T, workload="WS", experiment="ES", order=order)
                     for m in METHODS]
    run_all.RESULTS = os.path.join(run_all.RESULTS, "op2")
    run_all.run_jobs(jobs, "ES", max(1, (os.cpu_count() or 4) - 2))


if __name__ == "__main__":
    main()
