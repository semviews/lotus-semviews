# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""CertifiedFilter: shape with past views, certify with fresh samples.

1. Match: candidate views for q from the catalog (embedding similarity).
2. Shape: label a small uniform shaping sample of q; fit a cheap score on the shaping labels
   using view labels and helper scores as features. Shaping labels are paid and returned
   exactly, but never enter a bound.
3. Regions: from the score, fix a reject region X (lowest scores), an accept region A
   (highest), and an oracle region O, by minimizing predicted oracle calls.
4. Certify: label O in full; draw fresh uniform samples from X and from A; one-sided bounds
   (Clopper-Pearson with replacement, or exact hypergeometric without replacement) at a split of
   delta give an upper bound on X's positives and a lower bound on A's positives.
5. Close the gap: while a target is not certified, label X's highest-scored rows (each found
   positive is returned and tightens the missed-positive bound) or A's lowest-scored rows (each
   found negative is dropped). Labeling everything is always feasible.
6. Register q's view: oracle labels as `oracle`, region decisions as `derived`.

Validity (Theorem 1 with three strata): X and A are fixed before the fresh draws, so on the
event that both bounds cover (probability >= 1 - delta), every configuration the algorithm
can stop in meets both targets. Views influence only the score, hence only cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semviews.catalog import DERIVED, ORACLE, UNKNOWN, Catalog, View
from semviews.certify import cp_lower, cp_upper, hg_lower_count, hg_upper_count
from semviews.operators.reuse_filter import FilterResult, Targets
from semviews.relate import Embedder, predicate_text


@dataclass
class CertConfig:
    use_views: bool = True
    use_helper: bool = True
    n_candidates: int = 10
    shape_n: int = 200
    shape_frac_max: float = 0.08
    n_grid: tuple[int, ...] = (100, 200, 400, 800, 1600)
    q_grid: int = 40  # score quantiles considered as region boundaries
    close_chunk: int = 50  # rows labeled per gap-closing step
    l2: float = 1.0
    view_selection: str = "embed"  # embed | all
    shrink: float = 0.0  # pseudo-count shrinking predicted probabilities toward the base rate
    use_rowemb: bool = True  # free local row-embedding features (PCA-reduced)
    # Refinements, off by default; the paper's configuration (configs/experiments/sprint.yaml) sets them.
    active_rounds: int = 0  # refit rounds before the regions are fixed
    active_chunk: int = 250  # oracle-region rows labeled per refit round (most uncertain first)
    without_replacement: bool = False  # certification draws without replacement, hypergeometric bounds
    plan_margin: float = 0.0  # planner assumes sample counts this many std. devs. worse than expected
    delta_splits: tuple[float, ...] = (0.5,)  # candidate shares of delta given to X when both regions exist
    active_chunk_frac: float = 0.0  # if > 0, refit batch = max(active_chunk, frac * table rows)
    calibrate_uniform: bool = False  # recalibrate the planner's probabilities on the uniform shaping labels


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class CertifiedFilter:
    def __init__(self, oracle, catalog: Catalog, texts: dict[str, str], helper=None,
                 embedder: Embedder | None = None, cfg: CertConfig | None = None, row_features: np.ndarray | None = None):
        self.row_features = row_features
        self.oracle = oracle
        self.catalog = catalog
        self.texts = texts
        self.helper = helper
        self.embedder = embedder
        self.cfg = cfg or CertConfig()

    # ------------------------------------------------------------------------------------
    def run(self, pid: str, t: Targets, rng: np.random.Generator, given_views: list[str] | None = None,
            universe: np.ndarray | None = None) -> FilterResult:
        cfg = self.cfg
        n = self.catalog.n_rows
        in_u = np.ones(n, dtype=bool) if universe is None else np.isin(np.arange(n), universe)
        known = np.full(n, UNKNOWN, dtype=np.int8)
        pos, lab = self.catalog.known_oracle(pid)
        known[pos] = lab
        outside = ~in_u
        saved = known[outside].copy()
        known[outside] = 0
        stats: dict = {}

        # 1. candidate views
        views = self._candidates(pid, given_views) if cfg.use_views else []
        stats["views"] = [v.predicate_id for v in views]

        # 2. shaping sample (uniform, without replacement, from unknown rows)
        unk = np.flatnonzero(known == UNKNOWN)
        n_shape = int(min(cfg.shape_n, cfg.shape_frac_max * len(unk), len(unk)))
        if n_shape > 0:
            shape = rng.choice(unk, size=n_shape, replace=False)
            self._label(pid, shape, known, "shape")
        self._shape_rows = shape if n_shape > 0 else np.zeros(0, dtype=np.int64)
        U = np.flatnonzero(known == UNKNOWN)
        stats["n_shape"] = n_shape
        if len(U) == 0:
            return self._finish(pid, t, known, outside, saved, np.zeros(0, int), np.zeros(0, int), stats)

        # score every row from views and helper, fitted on the rows with known labels
        X_feat = self._features(pid, views)
        order, cum_p, k_pos, best = self._fit_and_plan(X_feat, known, in_u, t)
        # optional refit rounds: label the most uncertain oracle-region rows, refit, replan. Every
        # label here is exact and drawn before the regions are fixed, so validity is unaffected.
        chunk = max(cfg.active_chunk, int(cfg.active_chunk_frac * int(in_u.sum())))  # share of the rows filtered
        n_known0 = int((known != UNKNOWN).sum())
        for _ in range(cfg.active_rounds):
            i_x, j_a = best[0], best[1]
            O = order[i_x:j_a]
            if len(O) <= chunk:
                break
            p_o = self._p_hat[O]
            pick = O[np.argsort(np.abs(p_o - 0.5), kind="stable")[:chunk]]
            self._label(pid, pick, known, "residual")
            if not (known == UNKNOWN).any():
                break
            order, cum_p, k_pos, best = self._fit_and_plan(X_feat, known, in_u, t)
        stats["n_refit"] = int((known != UNKNOWN).sum()) - n_known0
        U = np.flatnonzero(known == UNKNOWN)
        if len(U) == 0:
            return self._finish(pid, t, known, outside, saved, np.zeros(0, int), np.zeros(0, int), stats)
        i_x, j_a, n_x, n_a, f_x = best
        X = order[:i_x]
        A = order[j_a:]
        O = order[i_x:j_a]
        stats.update(plan_x=len(X), plan_a=len(A), plan_o=len(O), n_x=n_x, n_a=n_a)

        # 4. certify: exact O, fresh draws from X and A. delta is split over the regions that
        # exist; the split depends only on the plan, which is fixed before the fresh draws.
        self._label(pid, O, known, "residual")
        a_x, a_a = self._alphas(t.delta, len(X), len(A), f_x)
        ux, la = 0.0, 0.0
        if len(X):
            if cfg.without_replacement:
                m = min(n_x, len(X))
                yx = self._label(pid, rng.choice(X, size=m, replace=False), known, "sample")
                ux = hg_upper_count(int(yx.sum()), len(X), m, a_x)
            else:
                dx = X[rng.integers(0, len(X), size=n_x)]
                yx = self._label(pid, dx, known, "sample")
                ux = float(cp_upper(yx.sum(), len(yx), a_x)) * len(X)
        if len(A):
            if cfg.without_replacement:
                m = min(n_a, len(A))
                ya = self._label(pid, rng.choice(A, size=m, replace=False), known, "sample")
                la = hg_lower_count(int(ya.sum()), len(A), m, a_a)
            else:
                da = A[rng.integers(0, len(A), size=n_a)]
                ya = self._label(pid, da, known, "sample")
                la = float(cp_lower(ya.sum(), len(ya), a_a)) * len(A)

        # 5. close the gap with exact labels where needed
        X_desc = X[::-1]  # highest-scored rejects first
        A_asc = A  # lowest-scored accepts first
        steps = 0
        while True:
            ok_r, ok_p = self._check(known, in_u, O, X, A, ux, la, t)
            if ok_r and ok_p:
                break
            progressed = False
            if not ok_r:
                nxt = X_desc[known[X_desc] == UNKNOWN][: cfg.close_chunk]
                if len(nxt):
                    self._label(pid, nxt, known, "close")
                    progressed = True
            if not ok_p:
                nxt = A_asc[known[A_asc] == UNKNOWN][: cfg.close_chunk]
                if len(nxt):
                    self._label(pid, nxt, known, "close")
                    progressed = True
            steps += 1
            if not progressed:
                # both regions fully labeled: the answer is exact
                break
        stats["close_steps"] = steps
        return self._finish(pid, t, known, outside, saved, X, A, stats, ux=ux, la=la)

    # ------------------------------------------------------------------------------------
    def _check(self, known, in_u, O, X, A, ux, la, t: Targets) -> tuple[bool, bool]:
        """Certified lower bounds on recall and precision for the current state."""
        kx = known[X]
        ka = known[A]
        # positives known exactly anywhere outside A (exact block, O, and found in X)
        exact_pos = float(((known == 1) & in_u).sum() - (ka == 1).sum())
        d_x = float((kx == 1).sum())
        e_a = float((ka == 0).sum())
        # A: lower bound on positives, at least the known positives in A
        la_eff = max(la, float((ka == 1).sum()))
        # X: upper bound on positives, at least the known positives; missed = positives not found
        unk_x = float((kx == UNKNOWN).sum())
        missed = min(max(ux - d_x, 0.0), unk_x) if len(X) else 0.0
        T = exact_pos + la_eff
        rec = 1.0 if T + missed == 0 else T / (T + missed)
        returned = exact_pos + (len(A) - e_a)
        prec = 1.0 if returned == 0 else T / returned
        return rec >= t.recall - 1e-12, prec >= t.precision - 1e-12

    _cp_cache: dict = {}

    def _tables(self, alpha: float):
        key = (alpha, self.cfg.n_grid)
        if key not in self._cp_cache:
            self._cp_cache[key] = {nn: (cp_upper(np.arange(nn + 1), nn, alpha), cp_lower(np.arange(nn + 1), nn, alpha))
                                   for nn in self.cfg.n_grid}
        return self._cp_cache[key]

    @staticmethod
    def _alphas(delta: float, sx: int, sa: int, f_x: float) -> tuple[float, float]:
        if sx and sa:
            return delta * f_x, delta * (1 - f_x)
        return delta, delta

    def _fit_and_plan(self, X_feat, known, in_u, t: Targets):
        U = np.flatnonzero(known == UNKNOWN)
        lab_rows = np.flatnonzero((known != UNKNOWN) & in_u)
        score, p_hat = self._fit_score(X_feat, lab_rows, known[lab_rows], len(U))
        if self.cfg.calibrate_uniform and len(lab_rows) > len(self._shape_rows):
            p_hat = self._recalibrate(p_hat, known)
        self._p_hat = p_hat
        order = U[np.argsort(score[U], kind="stable")]  # ascending score
        cum_p = np.concatenate([[0.0], np.cumsum(p_hat[order])])
        k_pos = float((known[in_u] == 1).sum())
        return order, cum_p, k_pos, self._plan(order, cum_p, k_pos, t)

    def _recalibrate(self, p_hat: np.ndarray, known: np.ndarray) -> np.ndarray:
        """Platt-recalibrate p_hat on the uniform shaping labels only. Refit labels are chosen near
        the decision boundary, so a model fitted on them ranks well but over-predicts positives in
        the tails, which the planner reads as expected counts. A ridge penalty pulls the map toward
        the identity (slope 1, intercept 0) when the shaping sample has few positives."""
        rows = self._shape_rows
        if len(rows) == 0:
            return p_hat
        z_all = _logit(p_hat)
        z, y = z_all[rows], known[rows].astype(float)
        a, b, lam = 1.0, 0.0, 2.0
        for _ in range(25):  # Newton steps on the penalized log-likelihood
            q = 1 / (1 + np.exp(-(a * z + b)))
            w = q * (1 - q)
            g = np.array([((q - y) * z).sum() + lam * (a - 1), (q - y).sum() + lam * b])
            H = np.array([[(w * z * z).sum() + lam, (w * z).sum()], [(w * z).sum(), w.sum() + lam]])
            step = np.linalg.solve(H, g)
            a, b = a - step[0], b - step[1]
            if np.abs(step).max() < 1e-8:
                break
        return 1 / (1 + np.exp(-(a * z_all + b)))

    def _pred_x(self, mean_rate: float, n: int, worse_up: bool) -> int:
        """Predicted positives among n draws; with plan_margin > 0, a pessimistic count."""
        m = mean_rate * n
        if self.cfg.plan_margin > 0:
            sd = np.sqrt(max(m * (1 - mean_rate), 0.0))
            m = m + self.cfg.plan_margin * sd if worse_up else m - self.cfg.plan_margin * sd
        return int(min(n, max(0, round(m))))

    def _fpc(self, bound_rate: float, x: int, n: int, size: int) -> float:
        """Planner's approximation of a without-replacement bound: shrink the CP margin by the
        finite-population factor. Only used to predict; certification uses exact bounds."""
        if not self.cfg.without_replacement or size <= 1:
            return bound_rate
        p = x / n
        return p + (bound_rate - p) * np.sqrt(max(size - n, 0) / (size - 1))

    def _plan(self, order, cum_p, k_pos, t: Targets):
        """Pick (i_x, j_a, n_x, n_a, f_x) minimizing predicted calls; all-O when nothing is predicted to pay."""
        cfg = self.cfg
        m = len(order)
        qs = np.unique(np.linspace(0, m, cfg.q_grid + 1).astype(int))
        best = (0, m, 0, 0, 0.5)
        best_cost = float(m)  # all-O
        total_p = cum_p[-1]
        for i in qs:
            px = cum_p[i]
            for j in qs[qs >= i]:
                sx, sa = i, m - j
                pa = total_p - cum_p[j]
                po = cum_p[j] - cum_p[i]
                o_cost = j - i
                if o_cost >= best_cost:
                    continue
                for f_x in (cfg.delta_splits if (sx and sa) else (0.5,)):
                    a_x, a_a = self._alphas(t.delta, sx, sa, f_x)
                    tx, ta = self._tables(a_x), self._tables(a_a)
                    for n_x in ([0] if sx == 0 else [v for v in cfg.n_grid if v < 0.7 * sx] or [0]):
                        if n_x == 0 and sx > 0:
                            continue
                        if sx:
                            xx = self._pred_x(px / sx, n_x, True)
                            ux = self._fpc(float(tx[n_x][0][xx]), xx, n_x, sx) * sx
                        else:
                            ux = 0.0
                        for n_a in ([0] if sa == 0 else [v for v in cfg.n_grid if v < 0.7 * sa] or [0]):
                            if n_a == 0 and sa > 0:
                                continue
                            cost = o_cost + n_x + n_a
                            if cost >= best_cost:
                                continue
                            if sa:
                                xa = self._pred_x(pa / sa, n_a, False)
                                la = self._fpc(float(ta[n_a][1][xa]), xa, n_a, sa) * sa
                            else:
                                la = 0.0
                            T = k_pos + po + la
                            missed = max(ux - px * n_x / max(sx, 1), 0.0)  # sampled positives are found
                            rec = 1.0 if T + missed == 0 else T / (T + missed)
                            ret = k_pos + po + sa
                            prec = 1.0 if ret == 0 else T / ret
                            if rec >= t.recall and prec >= t.precision:
                                best_cost, best = cost, (int(i), int(j), int(n_x), int(n_a), float(f_x))
        return best

    def _candidates(self, pid: str, given: list[str] | None) -> list[View]:
        if given is not None:
            return [self.catalog.get(v) for v in given if v in self.catalog]
        cands = self.catalog.others(pid)
        if self.embedder is not None and len(cands) > self.cfg.n_candidates and self.cfg.view_selection == "embed":
            q = self.embedder.embed([predicate_text(self.texts[pid])])[0]
            e = self.embedder.embed([predicate_text(v.text) for v in cands])
            idx = np.argsort(-(e @ q), kind="stable")[: self.cfg.n_candidates]
            cands = [cands[i] for i in idx]
        return cands

    def _features(self, pid: str, views: list[View]) -> np.ndarray:
        cols = []
        for v in views:
            cols += [(v.labels == 1).astype(float), (v.labels == UNKNOWN).astype(float)]
        if self.cfg.use_helper and self.helper is not None and self.helper.has(pid):
            h = self.helper.score(pid, np.arange(self.catalog.n_rows))
            cols.append(_logit(h) / 4.0)
        F = np.stack(cols, axis=1) if cols else np.zeros((self.catalog.n_rows, 0))
        if self.cfg.use_rowemb and self.row_features is not None:
            F = np.concatenate([F, self.row_features], axis=1)
        return F

    def _fit_score(self, F, rows, y, n_unknown):
        """Score = predicted P(q | features), fitted on paid labels (L2 logistic regression)."""
        n = F.shape[0]
        base = float(y.mean()) if len(y) else 0.5
        if F.shape[1] == 0 or len(y) < 10 or y.min() == y.max():
            p = np.full(n, (y.sum() + 0.5) / (len(y) + 1.0) if len(y) else 0.5)
            tie = F[:, 0] if F.shape[1] else np.zeros(n)
            return p + 1e-6 * tie, p
        from sklearn.linear_model import LogisticRegression

        w = None
        clf = LogisticRegression(C=self.cfg.l2, max_iter=500)
        clf.fit(F[rows], y, sample_weight=w)
        p = clf.predict_proba(F)[:, 1]
        # shrink toward the base rate to temper overconfident fits on small samples
        lam = len(y) / (len(y) + self.cfg.shrink)
        p = lam * p + (1 - lam) * base
        return p, p

    def _label(self, pid, rows, known, purpose):
        rows = np.asarray(rows, dtype=np.int64)
        need = np.unique(rows[known[rows] == UNKNOWN])
        if len(need):
            known[need] = self.oracle.label(pid, need, purpose=purpose)
        return known[rows].astype(np.int8)

    def _finish(self, pid, t, known, outside, saved, X, A, stats, ux=0.0, la=0.0) -> FilterResult:
        n = self.catalog.n_rows
        out = known == 1
        derived = np.full(n, UNKNOWN, dtype=np.int8)
        if len(A):
            out[A[known[A] != 0]] = True
            derived[A[known[A] == UNKNOWN]] = 1
        if len(X):
            derived[X[known[X] == UNKNOWN]] = 0
        known[outside] = saved
        out[outside] = False
        labels = np.where(known != UNKNOWN, known, derived).astype(np.int8)
        prov = np.where(known != UNKNOWN, ORACLE, np.where(derived != UNKNOWN, DERIVED, 0)).astype(np.int8)
        self.catalog.register(View(pid, self.texts.get(pid, pid), getattr(self.oracle, "model", ""), labels, prov,
                                   {"targets": [t.precision, t.recall, t.delta]}))
        stats.update(ux=ux, la=la)
        return FilterResult(out, stats)
