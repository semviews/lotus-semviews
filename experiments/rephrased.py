# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Workload W_R: the same questions asked again in different words.

The pool of each dataset (Reviews, GoEmotions) plus 15 reworded predicates
(configs/predicates/<dataset>_rephrased.yaml, labeled by the same oracle into the `rw` tape),
issued in random order. A query is a *repeat* when its twin (original or rewording) was issued
earlier in the order. Exact caches never hit a rewording; similarity caches and SemWeave reuse
the twin's answer without certification; SemViews uses the twin's view only to plan.

    python experiments/rephrased.py cost      -> results/op2/ER.parquet   (20 orders)
    python experiments/rephrased.py valid     -> results/op2/ER_valid.parquet (200 orders)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import run_all  # noqa: E402

DATASETS = ["reviews", "goemotions"]
COST = ["oracle_all", "semviews", "semviews_noviews", "bargain_pr", "lotus_cascade", "exact_cache",
        "sim_cache@0.7", "sim_cache@0.8", "sim_cache@0.9", "semweave"]
VALID = ["semviews", "semviews_noviews", "bargain_pr", "lotus_cascade", "sim_cache@0.8", "sim_cache@0.9", "semweave"]


def main() -> None:
    stage = sys.argv[1]
    cfg = run_all.yaml.safe_load(open(os.path.join(run_all.ROOT, "configs", "experiments", "sprint.yaml")))
    T = [cfg["targets"]["precision"], cfg["targets"]["recall"], cfg["targets"]["delta"]]
    methods, seeds, name = (COST, range(cfg["orders"]), "ER") if stage == "cost" else (VALID, range(200), "ER_valid")
    jobs = []
    for d in DATASETS:
        b = run_all.bench(d, rephrased=True)
        for s in seeds:
            order = run_all.order_for(b.preds, s)
            jobs += [dict(dataset=d, method=m, seed=s, targets=T, workload="WR", experiment=name, order=order,
                          rephrased=True) for m in methods]
    run_all.RESULTS = os.path.join(run_all.RESULTS, "op2")
    run_all.run_jobs(jobs, name, max(1, (os.cpu_count() or 4) - 2))


if __name__ == "__main__":
    main()
