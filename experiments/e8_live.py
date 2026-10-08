# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""E8-lite: one live end-to-end LOTUS run of the running example, with and without SemViews.

Three LazyFrame sem_filter queries over the first N reviews, executed (a) natively with LOTUS's
helper cascade and (b) with ViewOptimizer. All LM traffic goes through TapedLM (record mode), so
the run is replayable; the oracle-on-every-row reference is computed with the same LM.
Writes experiments/results/E8_live.parquet.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import lotus
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from lotus.ast import LazyFrame
from lotus.types import CascadeArgs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semviews.adapters.lotus.optimizer import ViewOptimizer  # noqa: E402
from semviews.adapters.lotus.taped_lm import TapedLM  # noqa: E402
from semviews.catalog import Catalog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES = ["{text} complains about the battery", "{text} reports a hardware defect", "{text} says the battery or device overheats"]


def main(n: int = 500) -> None:
    load_dotenv(os.path.join(ROOT, ".env"))
    logging.getLogger("lotus").setLevel(logging.ERROR)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "models.yaml")))
    tape = os.path.join(ROOT, "data", "tapes", "e8_live.sqlite")
    lm = TapedLM.for_alias("oracle-llama", cfg, tape_path=tape, max_batch_size=24)
    helper = TapedLM.for_alias("helper", cfg, tape_path=tape, max_batch_size=16)
    lotus.settings.configure(lm=lm, helper_lm=helper, enable_cache=False)
    df = pd.read_parquet(os.path.join(ROOT, "data", "datasets", "reviews.parquet")).iloc[:n][["text"]]
    args = CascadeArgs(recall_target=0.9, precision_target=0.9, failure_probability=0.05, cascade_IS_random_seed=0)
    rows = []
    # reference: oracle on every row (also warms the tape for the reference only)
    ref = {}
    for q in QUERIES:
        out = df.sem_filter(q, return_all=True)
        ref[q] = out["filter_label"].to_numpy().astype(bool) if "filter_label" in out else out.iloc[:, -1].to_numpy().astype(bool)
    catalog = Catalog("reviews_live", 0)
    for mode in ["lotus_cascade", "semviews"]:
        for q in QUERIES:
            lf = LazyFrame().sem_filter(q, cascade_args=args.model_copy(), return_all=True)
            if mode == "semviews":
                lf = lf.optimize([ViewOptimizer(catalog, seed=0)])
            lm.stats = type(lm.stats)()
            calls0 = lm.live_calls + lm.replayed_calls
            t0 = time.time()
            out = lf.execute(df)
            wall = time.time() - t0
            mask = out.iloc[:, -1].to_numpy().astype(bool)
            calls = lm.live_calls + lm.replayed_calls - calls0
            y = ref[q]
            tp = (mask & y).sum()
            rows.append(dict(mode=mode, query=q, oracle_calls=int(calls), wall_s=wall,
                             precision=tp / max(mask.sum(), 1), recall=tp / max(y.sum(), 1), n_rows=n, positives=int(y.sum())))
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, "experiments", "results", "op2")  # the paper's operator; the first-draft run stays in results/
    os.makedirs(out_dir, exist_ok=True)
    res.to_parquet(os.path.join(out_dir, f"E8_live__{n}.parquet"), index=False)
    print(res.groupby("mode")[["oracle_calls", "wall_s"]].sum(), "\nlive calls", lm.live_calls, "replayed", lm.replayed_calls)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
    _ = np
