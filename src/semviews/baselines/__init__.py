# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Baselines B0-B8 over the Oracle protocol. Each sees labels only through Oracle.label().

- B0 oracle_all         every row to the oracle
- B1 lotus_cascade      LOTUS's own cascade code (importance sampling, threshold learning,
                        calibration) on taped helper scores; `full_range` variant samples all rows
- B3 exact_cache        oracle labels cached by (predicate text, row); B1 on misses
- B4 sim_cache@tau      reuse the labels of the most similar past predicate if cosine >= tau
- B5 sim_proxy          most similar past view's labels as the cascade proxy score (our construction)
- B6 proxy_lr           per-predicate logistic regression on row embeddings as the cascade proxy
- B8 inferred_reuse     reuse by LLM-judged containment without certification (SemWeave-style idea)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from lotus.sem_ops.cascade_utils import (
    calibrate_llm_logprobs,
    importance_sampling,
    learn_cascade_thresholds,
)
from lotus.types import CascadeArgs

from semviews.catalog import DERIVED, ORACLE, UNKNOWN, Catalog, View
from semviews.operators.reuse_filter import FilterResult, Targets


@dataclass
class Ctx:
    """Shared per-workload state handed to every baseline."""

    oracle: object
    catalog: Catalog
    texts: dict[str, str]
    helper: object | None = None
    pred_emb: dict[str, np.ndarray] = field(default_factory=dict)
    row_emb: np.ndarray | None = None
    relations: dict[tuple[str, str], str] = field(default_factory=dict)  # (q, p) -> relation of q to p


def _label_all(ctx: Ctx, pid: str, rows: np.ndarray, purpose: str) -> np.ndarray:
    return ctx.oracle.label(pid, rows, purpose=purpose) if len(rows) else np.zeros(0, dtype=np.int8)


def _register(ctx: Ctx, pid: str, mask: np.ndarray, oracle_rows: np.ndarray, oracle_labels: np.ndarray) -> None:
    n = ctx.catalog.n_rows
    labels = mask.astype(np.int8)
    prov = np.full(n, DERIVED, dtype=np.int8)
    labels[oracle_rows] = oracle_labels
    prov[oracle_rows] = ORACLE
    ctx.catalog.register(View(pid, ctx.texts[pid], getattr(ctx.oracle, "model", ""), labels, prov))


def oracle_all(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator) -> FilterResult:
    rows = np.arange(ctx.catalog.n_rows)
    y = _label_all(ctx, pid, rows, "baseline")
    _register(ctx, pid, y.astype(bool), rows, y)
    return FilterResult(y.astype(bool), {})


def cascade_with_scores(
    ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator, scores: np.ndarray, *, calibrate: bool = True,
    full_range: bool = False, sampling_percentage: float = 0.1, extra_known: tuple[np.ndarray, np.ndarray] | None = None,
) -> FilterResult:
    """LOTUS's filter cascade, with LOTUS's own sampling and threshold code, on given proxy scores."""
    n = len(scores)
    args = CascadeArgs(recall_target=t.recall, precision_target=t.precision, failure_probability=t.delta,
                       sampling_percentage=sampling_percentage, cascade_IS_random_seed=int(rng.integers(0, 2**31 - 1)))
    if full_range:
        args.cascade_IS_max_sample_range = n
    proxy = list(calibrate_llm_logprobs(list(scores), args)) if calibrate else list(scores)
    idx, corr = importance_sampling(proxy, args)
    y_s = _label_all(ctx, pid, idx, "sample")
    (tau_pos, tau_neg), _ = learn_cascade_thresholds([proxy[i] for i in idx], [bool(v) for v in y_s], corr[idx], args)
    p = np.asarray(proxy)
    mid = np.flatnonzero((p < tau_pos) & (p > tau_neg))
    mask = p >= tau_pos
    y_mid = _label_all(ctx, pid, mid, "residual")
    mask[mid] = y_mid.astype(bool)
    rows = np.concatenate([idx, mid])
    labs = np.concatenate([y_s, y_mid])
    _register(ctx, pid, mask, rows, labs)
    return FilterResult(mask, {"tau_pos": float(tau_pos), "tau_neg": float(tau_neg), "n_mid": len(mid), "n_sample": len(np.unique(idx))})


def lotus_cascade(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator, full_range: bool = False) -> FilterResult:
    return cascade_with_scores(ctx, pid, t, rng, ctx.helper.score(pid), full_range=full_range)


def exact_cache(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator) -> FilterResult:
    """Identical predicate text seen before: return its stored result. Otherwise LOTUS cascade."""
    for v in ctx.catalog.views.values():
        if v.text == ctx.texts[pid] and v.predicate_id != pid:
            return FilterResult(v.labels == 1, {"hit": v.predicate_id})
    return lotus_cascade(ctx, pid, t, rng)


def _nearest(ctx: Ctx, pid: str) -> tuple[str | None, float]:
    q = ctx.pred_emb[pid]
    best, sim = None, -1.0
    for k in ctx.catalog.views:
        if k == pid:
            continue
        s = float(ctx.pred_emb[k] @ q)
        if s > sim:
            best, sim = k, s
    return best, sim


def sim_cache(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator, tau: float = 0.8) -> FilterResult:
    """GPTCache-style: if the nearest past predicate is within tau, reuse its labels unchecked."""
    best, sim = _nearest(ctx, pid)
    if best is not None and sim >= tau:
        v = ctx.catalog.get(best)
        mask = v.labels == 1
        ctx.catalog.register(View(pid, ctx.texts[pid], "", mask.astype(np.int8), np.full(len(mask), DERIVED, dtype=np.int8)))
        return FilterResult(mask, {"hit": best, "sim": sim})
    return lotus_cascade(ctx, pid, t, rng)


def sim_proxy(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator) -> FilterResult:
    """Proxy score = nearest past view's label (ties broken by the helper score), then LOTUS cascade."""
    best, _ = _nearest(ctx, pid)
    h = ctx.helper.score(pid)
    if best is None:
        return cascade_with_scores(ctx, pid, t, rng, h)
    v = ctx.catalog.get(best).labels
    score = np.where(v == 1, 0.5, 0.0) + 0.5 * h + np.where(v == UNKNOWN, 0.25, 0.0)
    return cascade_with_scores(ctx, pid, t, rng, score, calibrate=True)


def proxy_lr(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator, train_frac: float = 0.05) -> FilterResult:
    """Per-predicate logistic regression on row embeddings, trained on a fresh labeled sample."""
    from sklearn.linear_model import LogisticRegression

    n = ctx.catalog.n_rows
    tr = rng.choice(n, size=max(50, int(train_frac * n)), replace=False)
    y_tr = _label_all(ctx, pid, tr, "train")
    if y_tr.min() == y_tr.max():
        score = np.full(n, float(y_tr[0]))
    else:
        clf = LogisticRegression(max_iter=500, C=1.0).fit(ctx.row_emb[tr], y_tr)
        score = clf.predict_proba(ctx.row_emb)[:, 1]
    return cascade_with_scores(ctx, pid, t, rng, score, calibrate=True)


def inferred_reuse(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator) -> FilterResult:
    """Reuse by judged containment, without certification.

    equivalent p: return p's labels.  q implies p: label only p's positives (p's negatives
    rejected).  p implies q: accept p's positives, label the rest.  Otherwise: LOTUS cascade.
    """
    n = ctx.catalog.n_rows
    best = None
    for k in ctx.catalog.views:
        if k == pid:
            continue
        rel = ctx.relations.get((pid, k))
        if rel in ("equivalent", "implies", "implied_by"):
            score = {"equivalent": 3, "implies": 2, "implied_by": 1}[rel]
            if best is None or score > best[0]:
                best = (score, k, rel)
    if best is None:
        return lotus_cascade(ctx, pid, t, rng)
    _, k, rel = best
    p_lab = ctx.catalog.get(k).labels
    if rel == "equivalent":
        mask = p_lab == 1
        ctx.catalog.register(View(pid, ctx.texts[pid], "", mask.astype(np.int8), np.full(n, DERIVED, dtype=np.int8)))
        return FilterResult(mask, {"reused": k, "rel": rel})
    if rel == "implies":  # q => p: q's positives lie inside p's positives
        rows = np.flatnonzero(p_lab == 1)
    else:  # p => q: p's positives are q positives; label the rest
        rows = np.flatnonzero(p_lab != 1)
    y = _label_all(ctx, pid, rows, "residual")
    mask = np.zeros(n, dtype=bool) if rel == "implies" else (p_lab == 1)
    mask[rows] = y.astype(bool)
    _register(ctx, pid, mask, rows, y)
    return FilterResult(mask, {"reused": k, "rel": rel})


METHODS = {
    "oracle_all": oracle_all,
    "lotus_cascade": lotus_cascade,
    "lotus_cascade_full": lambda c, p, t, r: lotus_cascade(c, p, t, r, full_range=True),
    "exact_cache": exact_cache,
    "sim_cache@0.7": lambda c, p, t, r: sim_cache(c, p, t, r, 0.7),
    "sim_cache@0.8": lambda c, p, t, r: sim_cache(c, p, t, r, 0.8),
    "sim_cache@0.9": lambda c, p, t, r: sim_cache(c, p, t, r, 0.9),
    "sim_proxy": sim_proxy,
    "proxy_lr": proxy_lr,
    "inferred_reuse": inferred_reuse,
}


def bargain_pr(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator) -> FilterResult:
    """B2: BARGAIN_PR (released code, MIT) with taped helper scores as the proxy.

    Joint precision and recall target (BARGAIN_PR takes one target; we use max of the two),
    finite-sample betting bounds. Every oracle label goes through Oracle.label().
    """
    from BARGAIN import BARGAIN_PR
    from BARGAIN.models.AbstractModels import Oracle as BOracle
    from BARGAIN.models.AbstractModels import Proxy as BProxy

    h = ctx.helper.score(pid)
    paid_rows: list[int] = []

    class P(BProxy):
        def proxy_func(self, x):
            s = float(h[int(x)])
            return (s >= 0.5), max(s, 1 - s)

    class O(BOracle):
        def oracle_func(self, x, proxy_output):
            y = bool(ctx.oracle.label(pid, np.array([int(x)]), purpose="bargain")[0])
            paid_rows.append(int(x))
            return (proxy_output == y), y

    n = ctx.catalog.n_rows
    target = max(t.precision, t.recall)
    b = BARGAIN_PR(P(verbose=False, max_workers=1), O(verbose=False, max_workers=1), delta=t.delta, target=target,
                   verbose=False, seed=int(rng.integers(0, 2**31 - 1)))
    idx = b.process([str(i) for i in range(n)])
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(list(idx), dtype=int)] = True
    rows = np.unique(np.asarray(paid_rows, dtype=int))
    labs = ctx.oracle.label(pid, rows, purpose="bargain") if len(rows) else np.zeros(0, np.int8)
    _register(ctx, pid, mask, rows, labs)
    return FilterResult(mask, {"n_labeled": len(rows)})


METHODS["bargain_pr"] = bargain_pr


def lotus_strict(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator, bump: float = 0.05, delta: float = 0.01,
                 full_range: bool = True) -> FilterResult:
    """LOTUS cascade asked for stricter internal targets, to see whether it can be made valid."""
    tt = Targets(min(0.999, t.precision + bump), min(0.999, t.recall + bump), delta)
    return cascade_with_scores(ctx, pid, tt, rng, ctx.helper.score(pid), full_range=full_range)


METHODS["lotus_strict"] = lotus_strict
METHODS["lotus_strict2"] = lambda c, p, t, r: lotus_strict(c, p, t, r, bump=0.08, delta=0.001)


def bargain_with_scores(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator, scores: np.ndarray,
                        known: dict[int, int] | None = None) -> FilterResult:
    """BARGAIN_PR on arbitrary proxy scores (probability of positive). Labels already paid in this
    query scope (e.g. a shaping sample) are re-requested through Oracle.label() at no extra charge."""
    from BARGAIN import BARGAIN_PR
    from BARGAIN.models.AbstractModels import Oracle as BOracle
    from BARGAIN.models.AbstractModels import Proxy as BProxy

    paid_rows: list[int] = []

    class P(BProxy):
        def proxy_func(self, x):
            s = float(scores[int(x)])
            return (s >= 0.5), max(s, 1 - s)

    class O(BOracle):
        def oracle_func(self, x, proxy_output):
            y = bool(ctx.oracle.label(pid, np.array([int(x)]), purpose="bargain")[0])
            paid_rows.append(int(x))
            return (proxy_output == y), y

    n = ctx.catalog.n_rows
    b = BARGAIN_PR(P(verbose=False, max_workers=1), O(verbose=False, max_workers=1), delta=t.delta,
                   target=max(t.precision, t.recall), verbose=False, seed=int(rng.integers(0, 2**31 - 1)))
    idx = b.process([str(i) for i in range(n)])
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(list(idx), dtype=int)] = True
    rows = np.unique(np.asarray(paid_rows + list((known or {}).keys()), dtype=int))
    labs = ctx.oracle.label(pid, rows, purpose="bargain") if len(rows) else np.zeros(0, np.int8)
    _register(ctx, pid, mask, rows, labs)
    return FilterResult(mask, {"n_labeled": len(rows)})


def semweave(ctx: Ctx, pid: str, t: Targets, rng: np.random.Generator, k: int = 5) -> FilterResult:
    """Reimplementation of SemWeave's Algorithm 1 (Mahmood et al., SIGMOD Companion 2026).

    For the new filter F_i, retrieve the top-k most similar past filters F_s (predicate embedding
    cosine), take their cached bidirectional relation from the LLM judge (one prompt per pair, as in
    SemWeave), and decide each row from any stored result:
      F_s <=> F_i           : F_i(x) = F_s(x)
      F_s  => F_i, F_s(x)    : pass
      F_i  => F_s, not F_s(x): reject
      F_s  => not F_i, F_s(x): reject (exclusive)
    Rows no relation decides go to the oracle. No targets or certification are used. SemWeave's
    logical consistency check is applied: a pair's relation is dropped if it contradicts an
    implication chain through another candidate (F_i => F_a => F_s but judged exclusive).
    """
    n = ctx.catalog.n_rows
    q = ctx.pred_emb[pid]
    cands = sorted((k2 for k2 in ctx.catalog.views if k2 != pid and k2 in ctx.pred_emb),
                   key=lambda k2: -float(ctx.pred_emb[k2] @ q))[:k]
    rel = {c: ctx.relations.get((pid, c)) for c in cands}  # relation of F_i (q) with respect to F_s (c)
    # consistency check: q => a and a => c imply q => c, which contradicts "q exclusive c"
    for c in cands:
        if rel[c] == "exclusive":
            for a in cands:
                if rel.get(a) in ("implies", "equivalent") and ctx.relations.get((a, c)) in ("implies", "equivalent"):
                    rel[c] = None
                    break
    decided = np.full(n, -1, dtype=np.int8)
    for c in cands:
        r = rel[c]
        if r is None or r in ("overlapping", "unrelated"):
            continue
        fs = ctx.catalog.get(c).labels
        und = decided < 0
        if r == "equivalent":
            m = und & (fs >= 0)
            decided[m] = fs[m]
        elif r == "implied_by":  # F_s => F_i: rows passing F_s pass F_i
            decided[und & (fs == 1)] = 1
        elif r == "implies":  # F_i => F_s: rows failing F_s fail F_i
            decided[und & (fs == 0)] = 0
        elif r == "exclusive":  # F_s => not F_i: rows passing F_s fail F_i
            decided[und & (fs == 1)] = 0
    rows = np.flatnonzero(decided < 0)
    y = _label_all(ctx, pid, rows, "residual")
    out = decided.copy()
    out[rows] = y
    mask = out == 1
    _register(ctx, pid, mask, rows, y)
    return FilterResult(mask, {"reused_rows": int(n - len(rows)), "relations": {c: rel[c] for c in cands}})


METHODS["semweave"] = semweave
