# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Replay harness: run a method over a workload order on the tape and score every query.

Ground truth is read here, after a method returns, and never handed to a method.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import yaml

from semviews.catalog import Catalog
from semviews.operators.reuse_filter import ReuseConfig, ReuseFilter, Targets
from semviews.oracle import HelperScores, TapeOracle, load_tape
from semviews.relate import Embedder, mutual_information, predicate_text

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


# CertifiedFilter (regions) is the main operator; the per-signature strata operator is kept
# as the ablation "semviews_strata".
def _operator() -> dict:
    """The paper's operator configuration (configs/experiments/sprint.yaml: operator)."""
    path = os.path.join(ROOT, "configs", "experiments", "sprint.yaml")
    op = (yaml.safe_load(open(path)) or {}).get("operator") or {}
    return {k: tuple(v) if isinstance(v, list) else v for k, v in op.items()}


OPERATOR: dict = _operator()
# The operator before the refit rounds and planner refinements (first-draft operator, the `semviews_v1` ablation).
V1: dict = dict(active_rounds=0, active_chunk_frac=0.0, without_replacement=False, plan_margin=0.0, delta_splits=(0.5,))

CERT_VARIANTS: dict[str, dict] = {
    "semviews": {},
    "semviews_v1": V1,
    "semviews_noviews": {"use_views": False},
    "semviews_nohelper": {"use_helper": False},
    "semviews_allviews": {"view_selection": "all"},
    "semviews_shape100": {"shape_n": 100},
    "semviews_shape400": {"shape_n": 400},
    "semviews_shrink50": {"shrink": 50.0},
    "semviews_c10": {"l2": 10.0},
    "semviews_c01": {"l2": 0.1},
    "semviews_noemb": {"use_rowemb": False},
    "semviews_noviews_noemb": {"use_views": False, "use_rowemb": False},
}

SEMVIEWS_VARIANTS: dict[str, dict] = {
    "semviews_strata": {},
    "semviews_noviews": {"use_views": False},
    "semviews_nohelper": {"use_helper": False},
    "semviews_greedy": {"decision": "greedy"},
    "semviews_nonadaptive": {"adaptive": False},
    "semviews_k1": {"k_views": 1},
    "semviews_k2": {"k_views": 2},
    "semviews_k4": {"k_views": 4},
    "semviews_topembed": {"selection": "top_embed"},
}


@dataclass
class Bench:
    dataset: str
    oracle_model: str = "oracle-llama"
    n_rows: int | None = None
    use_helper_tape: bool = True
    rephrased: bool = False  # add the reworded predicates (configs/predicates/<dataset>_rephrased.yaml, tape tag rw)

    def __post_init__(self):
        rows = pd.read_parquet(os.path.join(ROOT, "data", "datasets", f"{self.dataset}.parquet"))
        if self.n_rows:
            rows = rows.iloc[: self.n_rows]
        self.rows = rows
        self.row_ids = rows.row_id.to_numpy()
        tape = load_tape(self.dataset, self.oracle_model)
        if self.rephrased:
            tape = pd.concat([tape, load_tape(self.dataset, f"{self.oracle_model}__rw")], ignore_index=True)
        tape = tape[tape.row_id.isin(set(rows.row_id))]
        self.oracle0 = TapeOracle(tape, self.oracle_model, row_ids=self.row_ids)
        try:
            if not self.use_helper_tape:
                raise FileNotFoundError("helper disabled")
            htape = load_tape(self.dataset, "helper")
            if self.rephrased:
                htape = pd.concat([htape, load_tape(self.dataset, "helper__rw")], ignore_index=True)
            htape = htape[htape.row_id.isin(set(rows.row_id))]
            self.helper0 = HelperScores(htape, self.row_ids)
            helper_preds = set(htape.predicate_id.unique())
        except (FileNotFoundError, ValueError):
            self.helper0, helper_preds = None, set()
        pool = yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", f"{self.dataset}.yaml")))
        self.pool = {f"{self.dataset}/{p['id']}": p for p in pool["predicates"]}
        if self.rephrased:
            rw = yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", f"{self.dataset}_rephrased.yaml")))
            self.pool.update({f"{self.dataset}/{p['id']}": p for p in rw["predicates"]})
        self.texts = {k: p["langex"] for k, p in self.pool.items()}
        have = set(self.oracle0.predicate_ids)
        self.preds = [k for k in self.pool if k in have and (k in helper_preds or self.helper0 is None)]
        self.embedder = Embedder(cache_dir=os.path.join(ROOT, "data", "cache", "emb"))
        self.pred_emb = dict(zip(self.texts, self.embedder.embed([predicate_text(t) for t in self.texts.values()])))
        self._row_emb = None
        self.truth = {p: self.oracle0.ground_truth(p).astype(bool) for p in self.preds}

    @property
    def row_emb(self) -> np.ndarray:
        if self._row_emb is None:
            path = os.path.join(ROOT, "data", "cache", f"rowemb_{self.dataset}.npy")
            if os.path.exists(path):
                full = np.load(path)
            else:
                allrows = pd.read_parquet(os.path.join(ROOT, "data", "datasets", f"{self.dataset}.parquet"))
                full = self.embedder.embed(allrows.text.tolist())
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, full)
            self._row_emb = full[: len(self.rows)]  # rows are a prefix of the dataset file
        return self._row_emb

    @property
    def row_pca(self) -> np.ndarray:
        if getattr(self, "_row_pca", None) is None:
            from sklearn.decomposition import PCA

            z = PCA(n_components=32, random_state=0).fit_transform(self.row_emb)
            self._row_pca = (z / z.std(axis=0, keepdims=True)).astype(np.float64) * 0.5
        return self._row_pca

    # ---------------------------------------------------------------------------------
    def _bargain_views(self, ctx, pid, targets, rng, helper):
        """BARGAIN_PR with the same proxy semviews uses: shaping sample (paid) + logistic score
        over view labels, helper score, and row embeddings. Isolates the certification step."""
        from semviews import baselines as B
        from semviews.catalog import UNKNOWN
        from semviews.operators.region_filter import CertConfig, CertifiedFilter

        cf = CertifiedFilter(ctx.oracle, ctx.catalog, self.texts, helper=helper, embedder=self.embedder,
                             cfg=CertConfig(), row_features=self.row_pca)
        n = ctx.catalog.n_rows
        known = np.full(n, UNKNOWN, dtype=np.int8)
        shape = rng.choice(n, size=min(cf.cfg.shape_n, int(cf.cfg.shape_frac_max * n)), replace=False)
        known[shape] = ctx.oracle.label(pid, shape, purpose="shape")
        F = cf._features(pid, cf._candidates(pid, None))
        score, _ = cf._fit_score(F, shape, known[shape], n)
        return B.bargain_with_scores(ctx, pid, targets, rng, score, {int(r): int(known[r]) for r in shape})

    def ideal_views(self, pid: str, catalog: Catalog, k: int = 3) -> list[str]:
        """Premise oracle (G1): pick views by mutual information on the *true* labels."""
        y = self.truth[pid].astype(np.int8)
        chosen: list[str] = []
        cur = 0.0
        for _ in range(k):
            best, gain = None, 0.01
            for v in catalog.others(pid):
                if v.predicate_id in chosen:
                    continue
                sig = np.stack([catalog.get(c).labels for c in chosen] + [v.labels], axis=1)
                g = mutual_information(y, sig) - cur
                if g > gain:
                    best, gain = v.predicate_id, g
            if best is None:
                break
            chosen.append(best)
            cur += gain
        return chosen

    def run(self, method: str, order: list[str], seed: int, targets: Targets, workload: str,
            experiment: str, relations: dict | None = None, cfg_overrides: dict | None = None,
            init_views: dict[str, np.ndarray] | None = None, view_coverage: float = 1.0) -> list[dict]:
        """Replay one workload order. `init_views` seeds the catalog with derived (untrusted)
        label columns, e.g. from another oracle model; `view_coverage` < 1 keeps only a random
        share of each registered view's labels (partial views, E5)."""
        from semviews import baselines as B
        from semviews.catalog import DERIVED, UNKNOWN, View

        oracle = self.oracle0.clone()
        helper = self.helper0.clone(oracle.meter) if self.helper0 is not None else None
        catalog = Catalog(self.dataset, len(self.row_ids))
        for vp, lab in (init_views or {}).items():
            lab = np.asarray(lab, dtype=np.int8)
            catalog.register(View(vp, self.texts.get(vp, vp), "init", lab,
                                  np.where(lab != UNKNOWN, DERIVED, 0).astype(np.int8)))
        rng = np.random.default_rng(seed)
        cov_rng = np.random.default_rng(seed + 7_919)
        is_sv = method.startswith("semviews")
        base = method.replace("_ideal", "")
        if is_sv and base in CERT_VARIANTS:
            from semviews.operators.region_filter import CertConfig, CertifiedFilter

            cfg = CertConfig(**{**OPERATOR, **CERT_VARIANTS[base], **(cfg_overrides or {})})
            rf = CertifiedFilter(oracle, catalog, self.texts, helper=helper, embedder=self.embedder, cfg=cfg,
                                 row_features=self.row_pca if cfg.use_rowemb else None)
        elif is_sv:
            cfg = ReuseConfig(**{**SEMVIEWS_VARIANTS[base], **(cfg_overrides or {})})
            rf = ReuseFilter(oracle, catalog, self.texts, helper=helper, embedder=self.embedder, cfg=cfg)
        ctx = B.Ctx(oracle, catalog, self.texts, helper, self.pred_emb, None, relations or {})
        if method == "proxy_lr":
            ctx.row_emb = self.row_emb
        out, cum = [], 0
        meta = dict(git_commit=git_commit(), config_hash=config_hash([method, asdict(targets), cfg_overrides]),
                    model_ids=self.oracle_model, lotus_version=__import__("importlib.metadata").metadata.version("lotus-ai"))
        for qi, pid in enumerate(order):
            with oracle.meter.scope(pid):
                c0, h0, s0 = oracle.meter.calls, oracle.meter.helper_calls, dict(oracle.meter.calls_by_purpose)
                u0 = oracle.meter.usd
                t0 = time.perf_counter()
                if method == "bargain_views":
                    res = self._bargain_views(ctx, pid, targets, rng, helper)
                elif is_sv:
                    gv = self.ideal_views(pid, catalog) if method.endswith("_ideal") else None
                    res = rf.run(pid, targets, rng, given_views=gv)
                else:
                    res = B.METHODS[method](ctx, pid, targets, rng)
                wall = 1000 * (time.perf_counter() - t0)
                calls = oracle.meter.calls - c0
                byp = {k: v - s0.get(k, 0) for k, v in oracle.meter.calls_by_purpose.items()}
            if view_coverage < 1.0 and pid in catalog:
                v = catalog.get(pid)
                drop = cov_rng.random(len(v.labels)) >= view_coverage
                v.labels = np.where(drop, UNKNOWN, v.labels).astype(np.int8)
                v.provenance = np.where(drop, 0, v.provenance).astype(np.int8)
            y = self.truth[pid]
            m = res.mask
            tp = int((m & y).sum())
            rec = tp / y.sum() if y.sum() else 1.0
            prec = tp / m.sum() if m.sum() else 1.0
            cum += calls
            out.append(dict(
                experiment=experiment, workload=workload, dataset=self.dataset, method=method, seed=seed, query_idx=qi,
                predicate_id=pid, oracle_calls=calls,
                oracle_calls_cert=int(byp.get("sample", 0) + byp.get("pilot", 0) + byp.get("train", 0)),
                cum_oracle_calls=cum, helper_calls=oracle.meter.helper_calls - h0, usd=oracle.meter.usd - u0,
                precision=prec, recall=rec, precision_target=targets.precision, recall_target=targets.recall,
                delta=targets.delta, met=bool(prec >= targets.precision and rec >= targets.recall),
                n_rows=len(y), n_returned=int(m.sum()), selectivity=float(y.mean()), wall_ms=wall,
                stats=json.dumps({k: v for k, v in res.stats.items() if not isinstance(v, np.ndarray)}, default=str), **meta,
            ))
        return out
