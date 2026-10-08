# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Noise check: re-label random (row, predicate) pairs from the tape twice more, live.

Reports self-agreement with the taped label, which bounds what any guarantee relative to the
oracle can mean. Writes experiments/results/noise_check.parquet.

    python -m semviews.adapters.lotus.noise_check --dataset goemotions --pairs 2000
"""

from __future__ import annotations

import argparse
import logging
import os

import lotus
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from lotus.nl_expression import nle2str, parse_cols
from lotus.sem_ops.sem_filter import sem_filter
from lotus.templates import task_instructions

from semviews.adapters.lotus.build_tape import ROOT, parts_dir
from semviews.adapters.lotus.taped_lm import RecordingLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="oracle-llama")
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    lotus.settings.configure(enable_cache=False)
    logging.getLogger("lotus").setLevel(logging.ERROR)
    models_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "models.yaml")))
    pool = {f"{a.dataset}/{p['id']}": p for p in yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", f"{a.dataset}.yaml")))["predicates"]}
    rows = pd.read_parquet(os.path.join(ROOT, "data", "datasets", f"{a.dataset}.parquet")).set_index("row_id")
    import glob

    tape = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(parts_dir(a.dataset, a.model), "*.parquet"))])
    sample = tape.sample(n=a.pairs, random_state=7)
    lm = RecordingLM.for_alias(a.model, models_cfg, max_batch_size=24)
    out = []
    for pid, g in sample.groupby("predicate_id"):
        langex = pool[pid]["langex"]
        cols = parse_cols(langex)
        docs = task_instructions.df2multimodal_info(rows.loc[g.row_id].reset_index(), cols)
        reps = [sem_filter(docs, lm, nle2str(langex, cols), show_progress_bar=False).outputs for _ in range(a.repeats)]
        for j, (rid, lab) in enumerate(zip(g.row_id, g.label)):
            out.append({"dataset": a.dataset, "predicate_id": pid, "row_id": rid, "tape": int(lab),
                        **{f"rep{r}": int(reps[r][j]) for r in range(a.repeats)}})
    df = pd.DataFrame(out)
    agree_all = np.all([df[f"rep{r}"] == df.tape for r in range(a.repeats)], axis=0).mean()
    path = os.path.join(ROOT, "experiments", "results", f"noise_check_{a.dataset}.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"{a.dataset}: {len(df)} pairs, all {a.repeats} repeats agree with the tape on {100 * agree_all:.2f}%")


if __name__ == "__main__":
    main()
