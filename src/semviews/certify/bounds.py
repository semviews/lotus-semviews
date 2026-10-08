# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Finite-sample confidence bounds on a stratum's positive rate.

All bounds assume draws uniformly at random *with replacement* from a fixed, finite
stratum, so each draw is an independent Bernoulli(pi) with pi the stratum's oracle-positive
rate. Clopper-Pearson is then exact. Simultaneity across strata and looks comes from a
Bonferroni split of delta.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta


def cp_lower(x: np.ndarray | int, n: np.ndarray | int, alpha: float) -> np.ndarray:
    """One-sided (1 - alpha) Clopper-Pearson lower bound on p given x successes in n draws."""
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore"):
        lb = beta.ppf(alpha, x, n - x + 1)
    lb = np.where((x <= 0) | (n <= 0), 0.0, lb)
    return np.nan_to_num(lb, nan=0.0)


def cp_upper(x: np.ndarray | int, n: np.ndarray | int, alpha: float) -> np.ndarray:
    """One-sided (1 - alpha) Clopper-Pearson upper bound on p given x successes in n draws."""
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore"):
        ub = beta.ppf(1 - alpha, x + 1, n - x)
    ub = np.where((x >= n) | (n <= 0), 1.0, ub)
    return np.nan_to_num(ub, nan=1.0)


def bonferroni_alpha(delta: float, n_strata: int, n_looks: int) -> float:
    """Per-bound level so that 2 * n_strata * n_looks one-sided bounds hold jointly w.p. >= 1 - delta."""
    return delta / (2 * max(n_strata, 1) * max(n_looks, 1))


def simultaneous_bounds(x: np.ndarray, n: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-stratum (lower, upper) bounds, each at level alpha. Strata with n = 0 get [0, 1]."""
    return cp_lower(x, n, alpha), cp_upper(x, n, alpha)


def required_n_for_upper(target_ub: float, alpha: float) -> int:
    """Smallest n with zero positives whose CP upper bound is at most target_ub: (1 - ub)^n <= alpha."""
    if target_ub >= 1:
        return 0
    if target_ub <= 0:
        return np.iinfo(np.int64).max
    return int(np.ceil(np.log(alpha) / np.log1p(-target_ub)))


def _hg_counts(N: int) -> np.ndarray:
    return np.arange(int(N) + 1)


def hg_upper_count(x: int, N: int, n: int, alpha: float) -> float:
    """(1 - alpha) upper bound on the number K of positives in a finite stratum of N rows, given x
    positives among n rows drawn uniformly *without* replacement: the largest K whose hypergeometric
    P[X <= x] exceeds alpha. Exact; tighter than Clopper-Pearson by the finite-population factor."""
    from scipy.stats import hypergeom

    if n <= 0:
        return float(N)
    K = _hg_counts(N)
    ok = hypergeom.cdf(x, N, K, n) > alpha
    return float(K[ok].max())


def hg_lower_count(x: int, N: int, n: int, alpha: float) -> float:
    """(1 - alpha) lower bound on the number of positives K: the smallest K whose hypergeometric
    P[X >= x] exceeds alpha."""
    from scipy.stats import hypergeom

    if n <= 0 or x <= 0:
        return 0.0
    K = _hg_counts(N)
    ok = hypergeom.sf(x - 1, N, K, n) > alpha
    return float(K[ok].min())
