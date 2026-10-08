# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Accept / reject / oracle decisions per stratum (Theorem 1).

Stratum S_i has s_i rows and c_i oracle positives, with simultaneous bounds lo_i <= c_i <= hi_i.
Of its rows, d_i are already known positive and e_i known negative (paid labels). An
exactly known block contributes k_pos positives. Decisions:

    A (accept)  return every row of S_i except its e_i known negatives
    X (reject)  return only its d_i known positives
    O (oracle)  label every row; return exactly the c_i positives

With T = k_pos + sum_{A,O} c_i + sum_X d_i, on the coverage event

    recall    = T / (T + sum_X (c_i - d_i))   >= T_lo / (T_lo + sum_X (hi_i - d_i))
    precision = T / (k_pos + sum_O c_i + sum_X d_i + sum_A (s_i - e_i))
              >= T_lo / (k_pos + sum_O lo_i + sum_X d_i + sum_A (s_i - e_i))

where T_lo replaces each c_i by lo_i. Both ratios are monotone in every c_i (the precision
ratio has numerator <= denominator), so plugging in the bounds is valid for *every*
assignment simultaneously. The planner minimizes oracle calls on O strata subject to both
lower bounds meeting the targets. All-O is always feasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACCEPT, REJECT, ORACLE = 0, 1, 2


@dataclass
class Plan:
    assign: np.ndarray  # ACCEPT / REJECT / ORACLE per stratum
    cost: float  # oracle calls on O strata
    recall_lb: float
    precision_lb: float


def _arr(v, m):
    return np.zeros(m) if v is None else np.asarray(v, dtype=float)


def plan_bounds(assign, s, lo, hi, k_pos: float = 0.0, d=None, e=None) -> tuple[float, float]:
    s, lo, hi = (np.asarray(v, dtype=float) for v in (s, lo, hi))
    d, e = _arr(d, len(s)), _arr(e, len(s))
    a, x, o = assign == ACCEPT, assign == REJECT, assign == ORACLE
    t = k_pos + lo[a | o].sum() + d[x].sum()
    missed = (hi[x] - d[x]).sum()
    rec = 1.0 if t + missed == 0 else t / (t + missed)
    den = k_pos + lo[o].sum() + d[x].sum() + (s[a] - e[a]).sum()
    prec = 1.0 if den == 0 else t / den
    return float(rec), float(prec)


def _rows(s, lo, hi, d, e, gp, gr):
    """Per-choice contributions to the linearized recall (r) and precision (p) constraints."""
    r = np.stack([(1 - gr) * lo, d - gr * hi, (1 - gr) * lo], axis=1)
    p = np.stack([lo - gp * (s - e), (1 - gp) * d, (1 - gp) * lo], axis=1)
    return r, p


def _feasible(assign, s, lo, hi, k_pos, gp, gr, d, e, tol=1e-9) -> bool:
    r, p = _rows(s, lo, hi, d, e, gp, gr)
    idx = np.arange(len(s))
    rr = (1 - gr) * k_pos + r[idx, assign].sum()
    pp = (1 - gp) * k_pos + p[idx, assign].sum()
    return bool(rr >= -tol and pp >= -tol)


def decide_greedy(s, lo, hi, cost_o, gp: float, gr: float, k_pos: float = 0.0, d=None, e=None) -> Plan:
    """Start from all-O and flip the most expensive strata to X or A while feasible."""
    s, lo, hi, cost_o = (np.asarray(v, dtype=float) for v in (s, lo, hi, cost_o))
    m = len(s)
    d, e = _arr(d, m), _arr(e, m)
    assign = np.full(m, ORACLE)
    for i in np.argsort(-cost_o, kind="stable"):
        if cost_o[i] <= 0:
            continue
        best = None
        for choice in (REJECT, ACCEPT):
            trial = assign.copy()
            trial[i] = choice
            if _feasible(trial, s, lo, hi, k_pos, gp, gr, d, e):
                rec, prec = plan_bounds(trial, s, lo, hi, k_pos, d, e)
                slack = min(rec - gr, prec - gp)
                if best is None or slack > best[0]:
                    best = (slack, choice)
        if best is not None:
            assign[i] = best[1]
    rec, prec = plan_bounds(assign, s, lo, hi, k_pos, d, e)
    return Plan(assign, float(cost_o[assign == ORACLE].sum()), rec, prec)


def decide_exact(s, lo, hi, cost_o, gp: float, gr: float, k_pos: float = 0.0, d=None, e=None,
                 node_limit: int = 200_000) -> Plan:
    """Exact minimum-cost assignment by branch and bound.

    The problem is a multiple-choice knapsack with two linear constraints. Strata are branched
    in decreasing cost order; a branch is pruned when even the most favorable completion
    cannot satisfy a constraint or cannot beat the incumbent. Past `node_limit` nodes the best
    feasible plan found so far is returned (always feasible, possibly not optimal).
    """
    s, lo, hi, cost_o = (np.asarray(v, dtype=float) for v in (s, lo, hi, cost_o))
    m = len(s)
    d, e = _arr(d, m), _arr(e, m)
    best = decide_greedy(s, lo, hi, cost_o, gp, gr, k_pos, d, e)
    if m == 0:
        return best
    order = np.argsort(-cost_o, kind="stable")
    r, p = _rows(s, lo, hi, d, e, gp, gr)
    r, p = r[order], p[order]
    c = np.stack([np.zeros(m), np.zeros(m), cost_o], axis=1)[order]
    r0, p0 = (1 - gr) * k_pos, (1 - gp) * k_pos
    r_best = np.concatenate([np.cumsum(r.max(1)[::-1])[::-1], [0.0]])
    p_best = np.concatenate([np.cumsum(p.max(1)[::-1])[::-1], [0.0]])
    best_cost = best.cost
    best_assign = best.assign.copy()
    choice = np.zeros(m, dtype=int)
    nodes = 0
    tol = 1e-9

    def dfs(i: int, rr: float, pp: float, cc: float) -> None:
        nonlocal best_cost, best_assign, nodes
        nodes += 1
        if nodes > node_limit or cc >= best_cost - tol:
            return
        if rr + r_best[i] < -tol or pp + p_best[i] < -tol:
            return
        if i == m:
            best_cost = cc
            a = np.empty(m, dtype=int)
            a[order] = choice
            best_assign = a
            return
        for k in (REJECT, ACCEPT, ORACLE):  # zero-cost choices first
            choice[i] = k
            dfs(i + 1, rr + r[i, k], pp + p[i, k], cc + c[i, k])

    dfs(0, r0, p0, 0.0)
    rec, prec = plan_bounds(best_assign, s, lo, hi, k_pos, d, e)
    return Plan(best_assign, float(cost_o[best_assign == ORACLE].sum()), rec, prec)
