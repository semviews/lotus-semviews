# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""ReuseFilter: certified semantic filtering that reuses semantic views.

Soundness rests on one rule: only fresh oracle draws, made uniformly with replacement inside
strata fixed before the draws, enter a confidence bound. Views, helper scores, and pilot
labels only decide *where* to sample and how to stratify. A wrong or stale view therefore
costs oracle calls, never correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semviews.catalog import DERIVED, ORACLE, UNKNOWN, Catalog, View
from semviews.certify import (
    ACCEPT,
    REJECT,
    bonferroni_alpha,
    cp_lower,
    cp_upper,
    decide_exact,
    decide_greedy,
)
from semviews.certify import (
    ORACLE as O_ASSIGN,
)
from semviews.relate import Embedder, predicate_text, select_views


@dataclass
class Targets:
    precision: float = 0.9
    recall: float = 0.9
    delta: float = 0.05


@dataclass
class ReuseConfig:
    use_views: bool = True
    use_helper: bool = True
    k_views: int = 3
    n_candidates: int = 10
    pilot_n: int = 50
    looks: tuple[int, ...] = (40, 120, 360, 1080)
    max_frac: float = 0.6  # never draw more than this multiple of a stratum's size
    max_strata: int = 32
    min_stratum: int = 30  # smaller strata are labeled outright
    score_bins: int = 4
    decision: str = "exact"  # exact | greedy
    adaptive: bool = True
    selection: str = "mi"  # mi | top_embed
    min_gain: float = 0.01
    alpha_weighting: str = "sqrt_size"  # uniform | sqrt_size (Bonferroni weights fixed before draws)
    settle: bool = True  # label clearly-oracle strata first, then re-plan with exact counts


@dataclass
class FilterResult:
    mask: np.ndarray  # rows returned
    stats: dict = field(default_factory=dict)


class ReuseFilter:
    """Engine-neutral operator. `oracle` follows the Oracle protocol in semviews.oracle; `helper` is optional."""

    def __init__(
        self,
        oracle,
        catalog: Catalog,
        texts: dict[str, str],
        helper=None,
        embedder: Embedder | None = None,
        cfg: ReuseConfig | None = None,
    ):
        self.oracle = oracle
        self.catalog = catalog
        self.texts = texts  # predicate_id -> langex
        self.helper = helper
        self.embedder = embedder
        self.cfg = cfg or ReuseConfig()
        self.n = catalog.n_rows

    # -------------------------------------------------------------------------------------
    def run(
        self,
        pid: str,
        targets: Targets,
        rng: np.random.Generator,
        given_views: list[str] | None = None,
        universe: np.ndarray | None = None,
    ) -> FilterResult:
        """Filter the rows in `universe` (catalog positions; default all rows) by predicate `pid`."""
        cfg = self.cfg
        n = self.n = self.catalog.n_rows
        known = np.full(n, UNKNOWN, dtype=np.int8)  # oracle labels for q known so far
        stats: dict = {"views": [], "n_strata": 0}
        in_u = np.ones(n, dtype=bool) if universe is None else np.isin(np.arange(n), universe)

        # Step 0: oracle labels for q already in the catalog are exact and free.
        pos, lab = self.catalog.known_oracle(pid)
        known[pos] = lab
        # rows outside the universe are excluded by marking them as "known negative" locally
        outside = ~in_u
        saved = known[outside].copy()
        known[outside] = 0

        # Step 1: match candidate views.
        views: list[View] = []
        if cfg.use_views:
            if given_views is not None:
                views = [self.catalog.get(v) for v in given_views if v in self.catalog]
            else:
                views = self._match(pid, known, rng, stats)
        stats["views"] = [v.predicate_id for v in views]

        # Step 2: stratify the rows whose q label is unknown.
        strata = self._stratify(pid, views, known)
        # tiny strata are labeled outright: they join the exact block
        sampled_strata = []
        for st in strata:
            if len(st) < cfg.min_stratum:
                self._label(pid, st, known, "tiny")
            else:
                sampled_strata.append(st)
        strata = sampled_strata
        m, L = len(strata), len(cfg.looks)
        stats["n_strata"] = m
        sizes = np.array([len(st) for st in strata], dtype=float)
        k_pos = float((known == 1).sum())
        # Bonferroni over 2 sides x m strata x L looks, weighted by stratum size; weights
        # depend only on sizes, which are fixed before any fresh draw.
        if cfg.alpha_weighting == "sqrt_size" and m:
            w = np.sqrt(sizes) / np.sqrt(sizes).sum()
            alpha = targets.delta * w / (2 * L)
        else:
            alpha = np.full(m, bonferroni_alpha(targets.delta, m, L))

        # Steps 3-6: fresh draws with replacement on a fixed look schedule; decide; adapt.
        draws: list[list[int]] = [[] for _ in strata]
        draw_lab: list[list[int]] = [[] for _ in strata]
        look_n = np.array([[min(c, int(cfg.max_frac * s)) for c in cfg.looks] for s in sizes], dtype=int) if m else np.zeros((0, L), int)
        level = np.zeros(m, dtype=int)
        active = np.ones(m, dtype=bool)
        plan = None
        for look in range(L):
            for i in np.flatnonzero(active):
                level[i] = look
                need = look_n[i, look] - len(draws[i])
                if need > 0:
                    new = strata[i][rng.integers(0, len(strata[i]), size=need)]
                    labs = self._label(pid, new, known, "sample")
                    draws[i].extend(new.tolist())
                    draw_lab[i].extend(labs.tolist())
            x = np.array([sum(d) for d in draw_lab], dtype=float)
            nn = np.array([len(d) for d in draw_lab], dtype=float)
            lo, hi = cp_lower(x, nn, alpha) * sizes, cp_upper(x, nn, alpha) * sizes
            cost_o, dk, ek = self._known_counts(strata, known)
            plan = self._decide(sizes, lo, hi, cost_o, targets, k_pos, dk, ek)
            if not cfg.adaptive or look == L - 1 or m == 0:
                break
            # project the next look at the current point estimates
            nxt = look_n[:, look + 1]
            p_hat = np.where(nn > 0, x / np.maximum(nn, 1), 0.5)
            lo_p = cp_lower(np.round(p_hat * nxt), nxt, alpha) * sizes
            hi_p = cp_upper(np.round(p_hat * nxt), nxt, alpha) * sizes
            proj = self._decide(sizes, lo_p, hi_p, cost_o, targets, k_pos, dk, ek)
            grow = (proj.assign != O_ASSIGN) & (nxt > nn) & (plan.assign == O_ASSIGN)
            extra = float(np.sum((nxt - nn)[grow]))
            if not grow.any() or plan.cost - proj.cost - extra <= 0:
                break
            active = grow
        stats["looks_used"] = int(level.max() + 1) if m else 0

        # Settle: label oracle strata one at a time, most uncertain first, replacing their
        # bounds by exact counts and re-planning. Every plan is feasible on the same coverage
        # event, so the choice of order cannot break the guarantee.
        if m and cfg.settle:
            exact = np.zeros(m, dtype=bool)
            while True:
                todo = [i for i in range(m) if plan.assign[i] == O_ASSIGN and not exact[i]]
                if not todo:
                    break
                p_hat = np.array([np.mean(draw_lab[i]) if draw_lab[i] else 0.5 for i in todo])
                i = todo[int(np.argmin(np.abs(p_hat - 0.5)))]
                self._label(pid, strata[i], known, "residual")
                exact[i] = True
                c = float((known[strata[i]] == 1).sum())
                lo[i] = hi[i] = c
                cost_o, dk, ek = self._known_counts(strata, known)
                plan = self._decide(sizes, lo, hi, cost_o, targets, k_pos, dk, ek)

        # Step 7: residual — oracle strata are labeled in full.
        out = known == 1  # exact block and every drawn row with a positive label
        derived = np.full(n, UNKNOWN, dtype=np.int8)
        if m:
            for i, st in enumerate(strata):
                a = plan.assign[i]
                if a == O_ASSIGN:
                    self._label(pid, st, known, "residual")
                    out[st] = known[st] == 1
                elif a == ACCEPT:
                    keep = st[known[st] != 0]  # drop rows known to be negative
                    out[keep] = True
                    derived[st[known[st] == UNKNOWN]] = 1
                elif a == REJECT:
                    derived[st[known[st] == UNKNOWN]] = 0
            stats.update(
                recall_lb=plan.recall_lb,
                precision_lb=plan.precision_lb,
                n_accept=int((plan.assign == ACCEPT).sum()),
                n_reject=int((plan.assign == REJECT).sum()),
                n_oracle=int((plan.assign == O_ASSIGN).sum()),
                rows_accept=int(sizes[plan.assign == ACCEPT].sum()),
                rows_reject=int(sizes[plan.assign == REJECT].sum()),
            )

        # Step 8: register q's view. Oracle labels are trusted; decisions are derived.
        known[outside] = saved
        out &= in_u
        labels = np.where(known != UNKNOWN, known, derived).astype(np.int8)
        prov = np.where(known != UNKNOWN, ORACLE, np.where(derived != UNKNOWN, DERIVED, 0)).astype(np.int8)
        self.catalog.register(View(pid, self.texts.get(pid, pid), getattr(self.oracle, "model", ""), labels, prov,
                                   {"targets": [targets.precision, targets.recall, targets.delta]}))
        return FilterResult(out, stats)

    # -------------------------------------------------------------------------------------
    def _label(self, pid: str, rows: np.ndarray, known: np.ndarray, purpose: str) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        need = np.unique(rows[known[rows] == UNKNOWN])
        if len(need):
            known[need] = self.oracle.label(pid, need, purpose=purpose)
        return known[rows].astype(np.int8)

    def _decide(self, s, lo, hi, cost_o, t: Targets, k_pos: float, d=None, e=None):
        f = decide_exact if self.cfg.decision == "exact" else decide_greedy
        return f(s, lo, hi, cost_o, t.precision, t.recall, k_pos, d, e)

    @staticmethod
    def _known_counts(strata, known):
        """Per stratum: unlabeled rows (O cost), known positives d, known negatives e."""
        cost = np.array([np.sum(known[st] == UNKNOWN) for st in strata], dtype=float)
        d = np.array([np.sum(known[st] == 1) for st in strata], dtype=float)
        e = np.array([np.sum(known[st] == 0) for st in strata], dtype=float)
        return cost, d, e

    def _match(self, pid: str, known: np.ndarray, rng: np.random.Generator, stats: dict) -> list[View]:
        cfg = self.cfg
        cands = self.catalog.others(pid)
        if not cands:
            return []
        if self.embedder is not None and len(cands) > cfg.n_candidates:
            q = self.embedder.embed([predicate_text(self.texts[pid])])[0]
            e = self.embedder.embed([predicate_text(v.text) for v in cands])
            order = np.argsort(-(e @ q), kind="stable")[: cfg.n_candidates]
            cands = [cands[i] for i in order]
        if cfg.selection == "top_embed":
            return cands[: cfg.k_views]
        # pilot: a small uniform sample of unknown rows, used for selection only
        unk = np.flatnonzero(known == UNKNOWN)
        pilot = rng.choice(unk, size=min(cfg.pilot_n, len(unk)), replace=False)
        q_pilot = self._label(pid, pilot, known, "pilot")
        rows = np.flatnonzero(known != UNKNOWN)
        q_rows = known[rows]
        chosen = select_views(q_rows, [v.labels[rows] for v in cands], cfg.k_views, cfg.min_gain)
        stats["pilot_pos"] = int(q_pilot.sum())
        return [cands[j] for j in chosen]

    def _stratify(self, pid: str, views: list[View], known: np.ndarray) -> list[np.ndarray]:
        cfg = self.cfg
        unk = np.flatnonzero(known == UNKNOWN)
        if len(unk) == 0:
            return []
        if views:
            sig = np.stack([v.labels[unk] for v in views], axis=1)
            _, inv = np.unique(sig, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
        else:
            inv = np.zeros(len(unk), dtype=int)
        score = None
        if cfg.use_helper and self.helper is not None and self.helper.has(pid):
            score = self.helper.score(pid, unk)
        groups = [unk[inv == g] for g in range(inv.max() + 1)]
        strata: list[np.ndarray] = []
        # split large signature groups by helper-score quantiles
        n_groups = len(groups)
        bins_budget = max(1, cfg.max_strata // max(n_groups, 1))
        for g, rows in zip(range(n_groups), groups):
            b = min(cfg.score_bins, bins_budget, max(1, len(rows) // (2 * cfg.min_stratum)))
            if score is None or b <= 1:
                strata.append(rows)
                continue
            sc = score[inv == g]
            cuts = np.unique(np.quantile(sc, np.linspace(0, 1, b + 1)[1:-1]))
            which = np.searchsorted(cuts, sc, side="right")
            for k in range(len(cuts) + 1):
                part = rows[which == k]
                if len(part):
                    strata.append(part)
        # cap the number of strata by merging the smallest ones
        strata.sort(key=len)
        while len(strata) > cfg.max_strata:
            a, b2 = strata.pop(0), strata.pop(0)
            strata.append(np.concatenate([a, b2]))
            strata.sort(key=len)
        return strata
