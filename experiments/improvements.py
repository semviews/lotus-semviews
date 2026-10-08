# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in improvements to CertifiedFilter, evaluated on the tape.

    python experiments/improvements.py tune      # pilot datasets only (GoEmotions, PubMed)
    python experiments/improvements.py full      # frozen candidates on all datasets, E2 + E3 protocol

Variants are CertConfig overrides on top of the CertConfig defaults. Results go to
experiments/results/IMP_<stage>.parquet; the paper's result files are not touched.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import run_all  # noqa: E402  (sets BLAS threads before numpy is imported)

ACT = lambda r, c: dict(active_rounds=r, active_chunk=c)  # noqa: E731
GRID = (100, 200, 400, 800, 1600, 3200)

TUNE: dict[str, dict] = {
    "base": {},
    "wor": dict(without_replacement=True),
    "margin0.5": dict(plan_margin=0.5),
    "margin1": dict(plan_margin=1.0),
    "split": dict(delta_splits=(0.5, 0.7, 0.9)),
    "grid3200": dict(n_grid=GRID),
    **{f"act{r}x{c}": ACT(r, c) for r in (2, 4, 8) for c in (100, 250, 500)},
    "shape400": dict(shape_n=400),
}

# Second tuning round: combinations around the best single change.
TUNE2: dict[str, dict] = {
    "base": {},
    **{f"act{r}x{c}": ACT(r, c) for r, c in ((8, 250), (12, 250), (16, 250), (16, 150), (24, 150))},
    "act8x250+wor": dict(**ACT(8, 250), without_replacement=True),
    "act8x250+margin0.5": dict(**ACT(8, 250), plan_margin=0.5),
    "act8x250+split": dict(**ACT(8, 250), delta_splits=(0.5, 0.7, 0.9)),
    "act8x250+shape400": dict(**ACT(8, 250), shape_n=400),
    "act8x250+wor+margin0.5+split": dict(**ACT(8, 250), without_replacement=True, plan_margin=0.5,
                                         delta_splits=(0.5, 0.7, 0.9)),
}

# Third round (after the second round under-performed at larger table sizes): calibration on the uniform
# shaping labels and a refit batch proportional to the table. Pilot datasets, fresh orders 12-17.
_PLUS = dict(**ACT(8, 250), without_replacement=True, plan_margin=0.5, delta_splits=(0.5, 0.7, 0.9))
TUNE3: dict[str, dict] = {
    "base": {},
    "plus": _PLUS,
    "plus+cal": dict(**_PLUS, calibrate_uniform=True),
    "plus+frac": dict(**_PLUS, active_chunk_frac=0.05),
    "plus+frac+cal": dict(**_PLUS, active_chunk_frac=0.05, calibrate_uniform=True),
}

# Frozen after the tuning stage (pilot datasets only).
FULL: dict[str, dict] = {
    "base": {},
    "noviews": dict(use_views=False),
    # frozen from IMP_tune2 (pilot datasets, orders 6-11): the cheapest combination
    "best": dict(**ACT(8, 250), without_replacement=True, plan_margin=0.5, delta_splits=(0.5, 0.7, 0.9)),
    "best_noviews": dict(**ACT(8, 250), without_replacement=True, plan_margin=0.5, delta_splits=(0.5, 0.7, 0.9),
                         use_views=False),
}


def jobs_for(variants: dict[str, dict], datasets: list[str], seeds: range, experiment: str) -> list[dict]:
    """Variants are relative to the first-draft operator (V1), whatever the paper's operator is now."""
    from semviews.workloads.runner import V1

    T = [0.9, 0.9, 0.05]
    return [dict(dataset=d, method="semviews", seed=s, targets=T, workload="W", experiment=experiment, variant=v,
                 cfg={**V1, **c})
            for d in datasets for v, c in variants.items() for s in seeds]


def main() -> None:
    stage = sys.argv[1]
    workers = max(1, (os.cpu_count() or 4) - 2)
    if stage == "tune":
        run_all.run_jobs(jobs_for(TUNE, ["goemotions", "pubmed"], range(6), "IMP_tune"), "IMP_tune", workers)
    elif stage == "tune2":
        run_all.run_jobs(jobs_for(TUNE2, ["goemotions", "pubmed"], range(6, 12), "IMP_tune2"), "IMP_tune2", workers)
    elif stage == "tune3":
        run_all.run_jobs(jobs_for(TUNE3, ["goemotions", "pubmed"], range(12, 18), "IMP_tune3"), "IMP_tune3", workers)
    elif stage == "full":
        run_all.run_jobs(jobs_for(FULL, run_all.yaml.safe_load(open(os.path.join(run_all.ROOT, "configs", "experiments",
                                                                                 "sprint.yaml")))["datasets"], range(20), "IMP_E2"),
                         "IMP_E2", workers)
    elif stage == "valid":
        for d in sys.argv[2:] or ["pubmed", "goemotions", "dbpedia", "reviews"]:
            jobs = jobs_for({"best": FULL["best"]}, [d], range(1000), "IMP_E3")
            jobs += jobs_for({"best_noviews": FULL["best_noviews"]}, [d], range(200), "IMP_E3")
            run_all.run_jobs(jobs, f"IMP_E3__{d}", workers)


if __name__ == "__main__":
    main()
