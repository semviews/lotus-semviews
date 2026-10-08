# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Compare a replayed result file with a shipped one, query by query.

    python experiments/compare_replay.py experiments/results/rerun/E2__pubmed__main.parquet \
        experiments/results/op2/E2__main.parquet

Rows are matched on (dataset, method, seed, query_idx, predicate_id); only rows present in the
rerun are compared. Oracle calls and the met flag must agree exactly; wall time and git commit
are expected to differ. Exits 1 on any mismatch.
"""

from __future__ import annotations

import sys

import pandas as pd

KEY = ["dataset", "method", "seed", "query_idx", "predicate_id"]
COLS = ["oracle_calls", "oracle_calls_cert", "met", "precision", "recall", "n_returned"]


def main(rerun: str, shipped: str) -> int:
    a, b = pd.read_parquet(rerun), pd.read_parquet(shipped)
    cols = [c for c in COLS if c in a and c in b]
    m = a.merge(b, on=KEY, how="left", suffixes=("_rerun", "_shipped"), indicator=True)
    unmatched = int((m._merge != "both").sum())
    m = m[m._merge == "both"]
    bad = pd.Series(False, index=m.index)
    for c in cols:
        eq = (m[f"{c}_rerun"] == m[f"{c}_shipped"]) | (m[f"{c}_rerun"].isna() & m[f"{c}_shipped"].isna())
        print(f"{c:18s} {eq.mean():.4%} of {len(m)} rows identical")
        bad |= ~eq
    print(f"{len(a)} rerun rows, {unmatched} without a shipped counterpart, {int(bad.sum())} differ")
    if bad.any():
        print(m.loc[bad, KEY + [f"{c}_{s}" for c in cols[:1] for s in ("rerun", "shipped")]].head(10).to_string())
    return 1 if bad.any() or unmatched else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
