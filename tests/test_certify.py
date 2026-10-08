# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Coverage simulations and property tests for certify/."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from semviews.certify import (
    ACCEPT,
    ORACLE,
    REJECT,
    bonferroni_alpha,
    cp_lower,
    cp_upper,
    decide_exact,
    decide_greedy,
    plan_bounds,
    required_n_for_upper,
)

TRIALS = 10_000


@pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.3, 0.5, 0.9, 0.99, 1.0])
@pytest.mark.parametrize("n", [10, 50, 300])
def test_cp_one_sided_coverage(p, n):
    """Each one-sided bound covers p with frequency >= 1 - alpha over 10,000 trials."""
    rng = np.random.default_rng(int(p * 1000) + n)
    alpha = 0.05
    x = rng.binomial(n, p, size=TRIALS)
    lo = cp_lower(x, n, alpha)
    hi = cp_upper(x, n, alpha)
    # allow Monte Carlo slack of 3 standard errors
    se = np.sqrt(alpha * (1 - alpha) / TRIALS)
    assert (lo <= p).mean() >= 1 - alpha - 3 * se
    assert (hi >= p).mean() >= 1 - alpha - 3 * se


def test_simultaneous_coverage_bonferroni():
    """All 2m bounds hold jointly with frequency >= 1 - delta."""
    rng = np.random.default_rng(7)
    m, delta = 12, 0.1
    alpha = bonferroni_alpha(delta, m, 1)
    ps = rng.uniform(0, 1, size=m)
    ns = rng.integers(5, 200, size=m)
    ok = 0
    for _ in range(TRIALS // 2):
        x = rng.binomial(ns, ps)
        ok += bool(np.all(cp_lower(x, ns, alpha) <= ps) and np.all(cp_upper(x, ns, alpha) >= ps))
    assert ok / (TRIALS // 2) >= 1 - delta


def test_required_n_for_upper_matches_cp():
    alpha = 1e-3
    for ub in [0.2, 0.05, 0.01]:
        n = required_n_for_upper(ub, alpha)
        assert cp_upper(0, n, alpha) <= ub + 1e-12
        assert cp_upper(0, n - 1, alpha) > ub


def test_degenerate_cases():
    assert cp_lower(0, 10, 0.05) == 0.0
    assert cp_upper(10, 10, 0.05) == 1.0
    assert cp_lower(0, 0, 0.05) == 0.0 and cp_upper(0, 0, 0.05) == 1.0


def _brute(s, lo, hi, cost, gp, gr, k):
    best = None
    for combo in itertools.product([ACCEPT, REJECT, ORACLE], repeat=len(s)):
        a = np.array(combo)
        r, p = plan_bounds(a, s, lo, hi, k)
        if r >= gr - 1e-9 and p >= gp - 1e-9:
            c = cost[a == ORACLE].sum()
            if best is None or c < best - 1e-9:
                best = c
    return best


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 5), st.integers(0, 10_000), st.floats(0.5, 0.99), st.floats(0.5, 0.99))
def test_milp_matches_brute_force(m, seed, gp, gr):
    rng = np.random.default_rng(seed)
    s = rng.integers(1, 500, size=m).astype(float)
    pi = rng.uniform(0, 1, size=m)
    width = rng.uniform(0, 0.3, size=m)
    lo = np.clip(pi - width, 0, 1) * s
    hi = np.clip(pi + width, 0, 1) * s
    cost = s * rng.uniform(0.5, 1.0, size=m)
    k = float(rng.integers(0, 50))
    plan = decide_exact(s, lo, hi, cost, gp, gr, k)
    r, p = plan_bounds(plan.assign, s, lo, hi, k)
    assert r >= gr - 1e-6 and p >= gp - 1e-6
    assert plan.cost <= _brute(s, lo, hi, cost, gp, gr, k) + 1e-6
    g = decide_greedy(s, lo, hi, cost, gp, gr, k)
    assert g.cost >= plan.cost - 1e-6
    r, p = plan_bounds(g.assign, s, lo, hi, k)
    assert r >= gr - 1e-6 and p >= gp - 1e-6


def test_theorem1_end_to_end_simulation():
    """Sample, bound, decide, and check realized precision/recall against the truth.

    The realized miss rate over many seeds must be at most delta.
    """
    rng = np.random.default_rng(11)
    delta, gp, gr = 0.1, 0.9, 0.9
    sizes = np.array([3000, 800, 600, 400, 200])
    pis = np.array([0.002, 0.97, 0.5, 0.05, 0.9])
    labels = [rng.random(s) < p for s, p in zip(sizes, pis)]
    truth_pos = sum(lab.sum() for lab in labels)
    misses = 0
    trials = 1000
    alpha = bonferroni_alpha(delta, len(sizes), 1)
    for _ in range(trials):
        n = np.minimum(sizes, 300)
        x = np.array([lab[rng.integers(0, len(lab), size=k)].sum() for lab, k in zip(labels, n)])
        lo = cp_lower(x, n, alpha) * sizes
        hi = cp_upper(x, n, alpha) * sizes
        plan = decide_exact(sizes, lo, hi, sizes.astype(float), gp, gr)
        tp = ret = 0
        for lab, a in zip(labels, plan.assign):
            if a == ACCEPT:
                tp += lab.sum()
                ret += len(lab)
            elif a == ORACLE:
                tp += lab.sum()
                ret += lab.sum()
        rec = tp / truth_pos
        prec = 1.0 if ret == 0 else tp / ret
        misses += (rec < gr) or (prec < gp)
    assert misses / trials <= delta


@pytest.mark.parametrize("N,K,n", [(200, 0, 50), (500, 5, 100), (1000, 40, 300), (2000, 600, 800), (300, 299, 150)])
def test_hypergeometric_count_coverage(N, K, n):
    """Without-replacement count bounds cover the true K with frequency >= 1 - alpha, and are never
    looser than Clopper-Pearson scaled to the stratum."""
    from semviews.certify import hg_lower_count, hg_upper_count

    rng = np.random.default_rng(N + K + n)
    alpha, trials = 0.05, 2_000
    pop = np.zeros(N, dtype=int)
    pop[:K] = 1
    up_ok = lo_ok = 0
    for _ in range(trials):
        x = int(rng.choice(pop, size=n, replace=False).sum())
        u, lo = hg_upper_count(x, N, n, alpha), hg_lower_count(x, N, n, alpha)
        up_ok += u >= K
        lo_ok += lo <= K
        assert u <= float(cp_upper(x, n, alpha)) * N + 1e-9
        assert lo >= float(cp_lower(x, n, alpha)) * N - 1e-9
    se = np.sqrt(alpha * (1 - alpha) / trials)
    assert up_ok / trials >= 1 - alpha - 3 * se
    assert lo_ok / trials >= 1 - alpha - 3 * se
