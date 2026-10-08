# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""E13: the paper's operator on a real relation of 100,000 tuples (appendix).

    python experiments/scale_real.py data     # build data/datasets/dbpedia_100k.parquet
    python experiments/scale_real.py          # human-label oracle -> experiments/results/op2/E13_scale.parquet
    python experiments/scale_real.py reviews  # build data/datasets/reviews_100k.parquet
    python experiments/scale_real.py llm      # LLM oracle tapes -> experiments/results/op2/E13_llm__<dataset>.parquet

Relation: 100,000 tuples sampled (seed 20261001, same cleaning as the main tables) from the
*train* split of DBPedia_Classes, so it shares no source rows with the 5,000-tuple DBpedia table
(test split). Labelling 100,000 tuples per predicate with the LLM oracle is beyond the budget, so
the oracle here is the human DBpedia ontology label of each tuple, through each predicate's
`human` mapping in configs/predicates/dbpedia.yaml. That makes ground truth exact and free. The
operator sees labels only through `HumanOracle.label()`, which counts every call, exactly as with
the LLM oracle. The helper model is off (scoring 40 predicates x 100,000 tuples with it would take
about 18 hours locally), so the operator orders tuples by views and local row embeddings only.
The 5,000- and 20,000-tuple tables are prefixes of the same sample with the same oracle, so the
comparison isolates the table size. Evaluation only; nothing enters a frozen configuration.

LLM-oracle run (`llm`): after the budget cap was raised, 8 of the 40 predicates (seeded choice,
configs/predicates/dbpedia_100k.yaml) were labeled on all 100,000 tuples by the LLM oracle through the
usual tape builder; likewise 8 of the 40 Reviews predicates on a second relation of 100,000 Amazon
Electronics reviews (reviews_100k, disjoint from the 5,000-tuple Reviews table). This replays them with the standard harness (run_all._job) and the paper's
operator on the same three table sizes, again without the helper (its tape ran at 14 tuples/s, about
16 hours for 800,000 scores; on the 5,000-tuple DBpedia table, dropping it changes calls by 0.99x).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process, before numpy loads

import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_all  # noqa: E402
import yaml  # noqa: E402
from headroom import GRID  # noqa: E402

from semviews.catalog import Catalog  # noqa: E402
from semviews.operators.region_filter import CertConfig, CertifiedFilter  # noqa: E402
from semviews.operators.reuse_filter import Targets  # noqa: E402
from semviews.relate import Embedder, predicate_text  # noqa: E402
from semviews.workloads.runner import OPERATOR, ROOT, config_hash, git_commit  # noqa: E402

N_LARGE, SEED = 100_000, 20261001
DATA = os.path.join(ROOT, "data", "datasets", "dbpedia_100k.parquet")
ROWEMB = os.path.join(ROOT, "data", "cache", "rowemb_dbpedia_100k.npy")
SIZES = (5_000, 20_000, 100_000)
ORDERS = 4
ORACLE_ID = "human-dbpedia-ontology"
VARIANTS = {"semviews": {"use_helper": False}, "semviews_noviews": {"use_helper": False, "use_views": False}}


def build_data() -> None:
    from datasets import load_dataset

    from semviews.workloads.datasets import _finish

    df = load_dataset("DeveloperOats/DBPedia_Classes", split="train").to_pandas()
    out = _finish(df[["text", "l1", "l2", "l3"]], N_LARGE, SEED)
    out.to_parquet(DATA, index=False)
    print(len(df), "train rows ->", len(out), "sampled;", os.path.getsize(DATA) // 1_000_000, "MB")


def build_reviews() -> None:
    """reviews_100k: the first 250,000 lines of the upstream Electronics.jsonl (data/raw/, not
    shipped; its first 60,000 lines are the source of the 5,000-tuple table), with the same
    filtering as semviews.workloads.datasets.reviews, minus every text in the 5,000-tuple table."""
    import json

    from semviews.ids import row_id
    from semviews.workloads.datasets import MAX_CHARS

    rows = []
    with open(os.path.join(ROOT, "data", "raw", "electronics_head250k.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            body, title = (r.get("text") or "").strip(), (r.get("title") or "").strip()
            if len(body) >= 80:
                rows.append({"text": f"{title}. {body}" if title else body, "rating": r.get("rating")})
    df = pd.DataFrame(rows)
    small = set(pd.read_parquet(os.path.join(ROOT, "data", "datasets", "reviews.parquet")).row_id)
    # the normalization of datasets._finish, then exclude the 5,000-tuple table, then sample
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, MAX_CHARS)
    clean = df[df["text"].str.len() >= 20].drop_duplicates("text")
    clean = clean[~clean.text.map(lambda t: row_id([t])).isin(small)]
    out = clean.sample(n=N_LARGE, random_state=SEED).reset_index(drop=True)
    out.insert(0, "row_id", [row_id([t]) for t in out["text"]])
    path = os.path.join(ROOT, "data", "datasets", "reviews_100k.parquet")
    out.to_parquet(path, index=False)
    print(len(df), "reviews ->", len(clean), "after cleaning and excluding the 5,000-tuple table ->", len(out),
          "sampled;", os.path.getsize(path) // 1_000_000, "MB")


class _Meter:
    def __init__(self):
        self.calls, self.helper_calls, self.calls_by_purpose = 0, 0, defaultdict(int)


class HumanOracle:
    """Oracle whose label is the tuple's human ontology label under the predicate's mapping."""

    model = ORACLE_ID

    def __init__(self, truth: dict[str, np.ndarray]):
        self._truth, self.meter = truth, _Meter()

    def label(self, pid, rows, purpose):
        rows = np.asarray(rows)
        self.meter.calls += len(rows)
        self.meter.calls_by_purpose[purpose] += len(rows)
        return self._truth[pid][rows].astype(np.int8)


def human_truth(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    pool = yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", "dbpedia.yaml")))
    out = {}
    for p in pool["predicates"]:
        y = np.zeros(len(rows), dtype=bool)
        for col, vals in p["human"].items():
            y |= rows[col].isin(vals).to_numpy()
        out[f"dbpedia/{p['id']}"] = y
    return out


class Large:
    def __init__(self, n: int):
        rows = pd.read_parquet(DATA).iloc[:n]
        self.truth = human_truth(rows)
        pool = yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", "dbpedia.yaml")))
        self.texts = {f"dbpedia/{p['id']}": p["langex"] for p in pool["predicates"]}
        self.preds = list(self.texts)
        self.embedder = Embedder(cache_dir=os.path.join(ROOT, "data", "cache", "emb"))
        self.embedder.embed([predicate_text(t) for t in self.texts.values()])
        from sklearn.decomposition import PCA

        emb = np.load(ROWEMB)[:n]  # same PCA recipe as Bench.row_pca, fit on this table
        z = PCA(n_components=32, random_state=0).fit_transform(emb)
        self.row_pca = (z / z.std(axis=0, keepdims=True)).astype(np.float64) * 0.5


def row_embeddings(dataset: str = "dbpedia_100k") -> None:
    """Row embeddings at the path Bench.row_emb reads (data/cache/rowemb_<dataset>.npy)."""
    path = os.path.join(ROOT, "data", "cache", f"rowemb_{dataset}.npy")
    if os.path.exists(path):
        return
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))  # the worker processes use one thread each
    texts = pd.read_parquet(os.path.join(ROOT, "data", "datasets", f"{dataset}.parquet")).text.tolist()
    model = Embedder()._load()
    t0 = time.time()
    emb = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(emb, dtype=np.float32))
    print(f"embedded {len(texts)} rows in {time.time() - t0:.0f}s", flush=True)


def job(a: dict) -> list[dict]:
    n, variant, seed = a["n"], a["variant"], a["seed"]
    b = Large(n)
    over = {**VARIANTS[variant], **({"n_grid": GRID} if n > 5_000 else {})}  # as in the tiled simulation
    cfg = CertConfig(**{**OPERATOR, **over})
    oracle, cat, rng = HumanOracle(b.truth), Catalog("dbpedia_100k", n), np.random.default_rng(seed)
    t = Targets(0.9, 0.9, 0.05)
    meta = dict(git_commit=git_commit(), config_hash=config_hash([variant, over, OPERATOR]), model_ids=ORACLE_ID,
                lotus_version=__import__("importlib.metadata").metadata.version("lotus-ai"))
    out = []
    for qi, pid in enumerate(run_all.order_for(b.preds, seed)):
        f = CertifiedFilter(oracle, cat, b.texts, helper=None, embedder=b.embedder, cfg=cfg, row_features=b.row_pca)
        c0, t0 = oracle.meter.calls, time.time()
        res = f.run(pid, t, rng)
        y = b.truth[pid]
        tp = int((res.mask & y).sum())
        rec = tp / y.sum() if y.sum() else 1.0
        prec = tp / res.mask.sum() if res.mask.sum() else 1.0
        out.append(dict(dataset="dbpedia_100k", method=variant, n_rows=n, seed=seed, query_idx=qi, predicate_id=pid,
                        oracle_calls=oracle.meter.calls - c0, selectivity=float(y.mean()), positives=int(y.sum()),
                        precision=prec, recall=rec, met=bool(prec >= 0.9 and rec >= 0.9), wall_s=time.time() - t0, **meta))
    print(f"done n={n} {variant} seed={seed}: share {sum(r['oracle_calls'] for r in out) / (n * len(out)):.3f}, "
          f"met {np.mean([r['met'] for r in out]):.3f}", flush=True)
    return out


LLM_METHODS = ("semviews", "semviews_noviews")


LLM_DATASETS = {"dbpedia_100k": "W3", "reviews_100k": "W2a"}


def llm(datasets: tuple[str, ...] = tuple(LLM_DATASETS)) -> None:
    from concurrent.futures import ProcessPoolExecutor

    jobs = [dict(dataset=d, n_rows=None if n == N_LARGE else n, method=m, seed=s, targets=(0.9, 0.9, 0.05),
                 workload=LLM_DATASETS[d], experiment="E13_llm",
                 cfg={"use_helper": False, **({"n_grid": GRID} if n > 5_000 else {})})
            for d in datasets for n in reversed(SIZES) for m in LLM_METHODS for s in range(20)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as ex:
        for r in ex.map(run_all._job, jobs):
            rows += r
            print(f"done {r[0]['dataset']} n={r[0]['n_rows']} {r[0]['method']} seed={r[0]['seed']}: "
                  f"share {sum(x['oracle_calls'] for x in r) / (r[0]['n_rows'] * len(r)):.3f}, "
                  f"met {np.mean([x['met'] for x in r]):.3f}", flush=True)
    for d in datasets:
        part = pd.DataFrame([r for r in rows if r["dataset"] == d])
        part.to_parquet(os.path.join(run_all.RESULTS, "op2", f"E13_llm__{d}.parquet"), index=False)
    print(len(rows), "rows")


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    row_embeddings()
    jobs = [dict(n=n, variant=v, seed=s) for n in reversed(SIZES) for v in VARIANTS for s in range(ORDERS)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as ex:
        for r in ex.map(job, jobs):
            rows += r
    os.makedirs(os.path.join(run_all.RESULTS, "op2"), exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(run_all.RESULTS, "op2", "E13_scale.parquet"), index=False)
    print(len(rows), "rows")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "llm":
        llm(tuple(sys.argv[2:]) or tuple(LLM_DATASETS))
    else:
        {"data": build_data, "reviews": build_reviews}.get(arg, main)()
