# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Run the paper's experiments from the frozen config. Everything replays the tape.

    python experiments/run_all.py --config configs/experiments/sprint.yaml [--only E2 E3 ...]

Outputs experiments/results/<E>.parquet in the result schema described in docs/DATA.md.
"""

from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")  # one BLAS thread per worker process

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semviews.operators.reuse_filter import Targets  # noqa: E402
from semviews.workloads.runner import ROOT, Bench  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results")
_BENCH: dict = {}


def bench(dataset: str, model: str = "oracle-llama", n_rows: int | None = None, rephrased: bool = False) -> Bench:
    key = (dataset, model, n_rows, rephrased)
    if key not in _BENCH:
        logging.getLogger("lotus").setLevel(logging.ERROR)
        _BENCH[key] = Bench(dataset, model, n_rows, rephrased=rephrased)
    return _BENCH[key]


def relations(dataset: str, rephrased: bool = False) -> dict:
    raw: dict = {}
    for name in [dataset] + ([f"{dataset}_rephrased"] if rephrased else []):
        p = os.path.join(ROOT, "data", "relations", f"{name}.json")
        if os.path.exists(p):
            raw.update(json.load(open(p)))
    return {tuple(k.split("||")): v for k, v in raw.items()}


def order_for(preds: list[str], seed: int) -> list[str]:
    o = list(preds)
    np.random.default_rng(10_000 + seed).shuffle(o)
    return o


def w0_subset(b: Bench, size: int) -> tuple[list[str], float]:
    """Unrelated control: greedily pick `size` predicates minimizing the largest pairwise |phi|
    on the tape. Returns the subset and the largest |phi| it contains."""
    preds = [p for p in b.preds if 0.02 <= b.truth[p].mean() <= 0.6]
    Y = np.stack([b.truth[p].astype(float) for p in preds], axis=1)
    C = np.abs(np.corrcoef(Y.T))
    np.fill_diagonal(C, 0)
    start = int(np.argmin(C.max(axis=1)))
    chosen = [start]
    while len(chosen) < min(size, len(preds)):
        rest = [i for i in range(len(preds)) if i not in chosen]
        nxt = min(rest, key=lambda i: C[i, chosen].max())
        chosen.append(nxt)
    worst = float(C[np.ix_(chosen, chosen)].max())
    return [preds[i] for i in chosen], worst


def _job(args: dict) -> list[dict]:
    b = bench(args["dataset"], args.get("model", "oracle-llama"), args.get("n_rows"), args.get("rephrased", False))
    t = Targets(*args["targets"])
    order = args.get("order") or order_for(args.get("preds") or b.preds, args["seed"])
    init = args.get("init_views")
    if init in ("adversarial", "inverted"):
        # inverted: each predicate's labels inverted (20% noise) under the same text;
        # adversarial: same text, labels random at the true selectivity (matched first, useless)
        rng = np.random.default_rng(99)
        init_kind, init = init, {}
        for p in b.preds:
            y = b.truth[p]
            if init_kind == "inverted":
                flip = rng.random(len(y)) < 0.2
                init[f"{p}@adv"] = np.where(flip, y, ~y).astype(np.int8)
            else:
                init[f"{p}@adv"] = (rng.random(len(y)) < y.mean()).astype(np.int8)
            b.texts[f"{p}@adv"] = b.texts[p]
            b.pred_emb[f"{p}@adv"] = b.pred_emb[p]
        for i in range(10):
            init[f"rand{i}"] = (rng.random(len(b.row_ids)) < 0.3).astype(np.int8)
            b.texts[f"rand{i}"] = f"{{text}} random property {i}"
            b.pred_emb[f"rand{i}"] = b.embedder.embed([f"random property {i}"])[0]
    elif init == "appended":  # views recorded before the last 20% of rows were appended
        init = {}
        cut = int(0.8 * len(b.row_ids))
        for p in b.preds:
            lab = b.truth[p].astype(np.int8).copy()
            lab[cut:] = -1
            init[f"{p}@old"] = lab
            b.texts[f"{p}@old"] = b.texts[p]
            b.pred_emb[f"{p}@old"] = b.pred_emb[p]
    elif init == "llama":  # E7: views recorded under the primary oracle, reused under another
        src = bench(args["dataset"], "oracle-llama", args.get("n_rows"))
        init = {f"{p}@llama": src.truth[p].astype(np.int8) for p in src.preds}
        for p in src.preds:
            b.texts[f"{p}@llama"] = b.texts[p]
            b.pred_emb[f"{p}@llama"] = b.pred_emb[p]
    rows = b.run(args["method"], order, args["seed"], t, args["workload"], args["experiment"],
                 relations=relations(args["dataset"], args.get("rephrased", False)), cfg_overrides=args.get("cfg"),
                 init_views=init, view_coverage=args.get("coverage", 1.0))
    for r in rows:
        r.update({k: args[k] for k in ("variant",) if k in args})
    return rows


SUFFIX = ""
SEMVIEWS_ONLY = False  # rerun only the operator's own methods (baselines are deterministic replays)


def run_jobs(jobs: list[dict], name: str, workers: int) -> pd.DataFrame:
    t0 = time.time()
    if SEMVIEWS_ONLY:
        jobs = [j for j in jobs if j["method"].startswith("semviews") and j["method"] != "semviews_strata"]
    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rows in ex.map(_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))):
            out.extend(rows)
    df = pd.DataFrame(out)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_parquet(os.path.join(RESULTS, f"{name}{SUFFIX}.parquet"), index=False)
    print(f"{name}: {len(jobs)} runs, {len(df)} rows, {time.time() - t0:.0f}s", flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "experiments", "sprint.yaml"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--methods", nargs="*", help="override the method list of E2/E3")
    ap.add_argument("--tag", default="", help="suffix for result files (e.g. b for E2b)")
    ap.add_argument("--outdir", default="", help="subdirectory of experiments/results for the output files")
    ap.add_argument("--semviews-only", action="store_true", help="run only semviews* methods (not the strata ablation)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    T = [cfg["targets"]["precision"], cfg["targets"]["recall"], cfg["targets"]["delta"]]
    datasets = a.datasets or list(cfg["datasets"])
    global SUFFIX, RESULTS, SEMVIEWS_ONLY
    RESULTS = os.path.join(RESULTS, a.outdir) if a.outdir else RESULTS
    SEMVIEWS_ONLY = a.semviews_only
    SUFFIX = ("__" + "_".join(datasets) if a.datasets else "") + (f"__{a.tag}" if a.tag else "")
    if a.methods:
        cfg["methods_main"], cfg["ablations"], cfg["methods_validity"] = a.methods, [], a.methods
    want = lambda e: not a.only or e in a.only  # noqa: E731

    if want("E2"):  # cumulative calls per method over workloads (also E9 on W0)
        jobs = []
        for d in datasets:
            w = cfg["datasets"][d]
            for m in cfg["methods_main"] + cfg["ablations"]:
                for s in range(cfg["orders"]):
                    jobs.append(dict(dataset=d, method=m, seed=s, targets=T, workload=w, experiment="E2"))
        run_jobs(jobs, "E2", a.workers)

    if want("E9"):  # unrelated control W0
        jobs = []
        for d in datasets:
            b = bench(d)
            sub, worst = w0_subset(b, cfg["w0"]["size"])
            print(d, "W0 size", len(sub), "max |phi|", round(worst, 3), flush=True)
            for m in ["lotus_cascade", "lotus_cascade_full", "semviews_noviews", "semviews", "oracle_all"]:
                for s in range(cfg["orders"]):
                    jobs.append(dict(dataset=d, method=m, seed=s, targets=T, workload="W0", experiment="E9", preds=sub))
        run_jobs(jobs, "E9", a.workers)

    if want("E3"):  # validity over many seeds
        jobs = []
        for d in datasets:
            for m in cfg["methods_validity"]:
                n_seeds = cfg["validity_seeds"] if m.startswith("semviews") else cfg["validity_seeds_baselines"]
                for s in range(n_seeds):
                    jobs.append(dict(dataset=d, method=m, seed=s, targets=T, workload=cfg["datasets"][d], experiment="E3"))
        run_jobs(jobs, "E3", a.workers)

    if want("E4"):  # sensitivity: targets, delta, k, table size
        jobs = []
        sens = cfg["sensitivity"]
        for d in datasets:
            w = cfg["datasets"][d]
            for m in ["lotus_cascade", "semviews_noviews", "semviews"]:
                for s in range(cfg["orders"]):
                    for gp, gr in sens["targets"]:
                        jobs.append(dict(dataset=d, method=m, seed=s, targets=[gp, gr, T[2]], workload=w, experiment="E4",
                                         variant=f"targets={gp},{gr}"))
                    for dl in sens["delta"]:
                        jobs.append(dict(dataset=d, method=m, seed=s, targets=[T[0], T[1], dl], workload=w, experiment="E4",
                                         variant=f"delta={dl}"))
                    for n in sens["table_rows"]:
                        jobs.append(dict(dataset=d, method=m, seed=s, targets=T, workload=w, experiment="E4",
                                         variant=f"rows={n}", n_rows=n))
        run_jobs(jobs, "E4", a.workers)

    if want("E5"):  # partial views
        jobs = []
        for d in datasets:
            for c in cfg["partial_coverage"]:
                for s in range(cfg["orders"]):
                    jobs.append(dict(dataset=d, method="semviews", seed=s, targets=T, workload=cfg["datasets"][d],
                                     experiment="E5", coverage=c, variant=f"coverage={c}"))
        run_jobs(jobs, "E5", a.workers)

    if want("E11"):  # appends: views cover only the first 80% of rows
        jobs = []
        for d in datasets:
            for m in ["semviews", "semviews_noviews", "sim_cache@0.9"]:
                for s in range(20):
                    jobs.append(dict(dataset=d, method=m, seed=s, targets=T, workload=cfg["datasets"][d], experiment="E11",
                                     init_views="appended", variant="appended rows"))
        run_jobs(jobs, "E11", a.workers)

    if want("E12"):  # can the LOTUS cascade be made valid by asking for stricter internal targets?
        jobs = []
        for d in datasets:
            for m in ["lotus_strict", "lotus_strict2"]:
                for s in range(100):
                    jobs.append(dict(dataset=d, method=m, seed=s, targets=T, workload=cfg["datasets"][d], experiment="E12"))
        run_jobs(jobs, "E12", a.workers)

    if want("E10"):  # adversarial catalog: validity must hold whatever the views say
        jobs = []
        for d in datasets:
            for m in ["semviews", "semviews_noviews", "sim_cache@0.9", "inferred_reuse"]:
                for kind in ("adversarial", "inverted"):
                    for s in range(200 if m.startswith("semviews") else 20):
                        jobs.append(dict(dataset=d, method=m, seed=s, targets=T, workload=cfg["datasets"][d], experiment="E10",
                                         init_views=kind, variant=kind))
        run_jobs(jobs, "E10", a.workers)

    if want("E7"):  # robustness: views from one oracle reused under another; appended rows
        jobs = []
        d = "goemotions"
        for m in ["semviews_noviews", "semviews", "lotus_cascade", "inferred_reuse", "sim_cache@0.9"]:
            for s in range(cfg["orders"]):
                jobs.append(dict(dataset=d, model="oracle-gptoss", n_rows=1000, method=m, seed=s, targets=T,
                                 workload="W3", experiment="E7", init_views="llama", variant="cross-model views"))
                jobs.append(dict(dataset=d, model="oracle-gptoss", n_rows=1000, method=m, seed=s, targets=T,
                                 workload="W3", experiment="E7", variant="same model"))
        run_jobs(jobs, "E7", min(a.workers, 8))


if __name__ == "__main__":
    main()
