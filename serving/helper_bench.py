# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""make helper-bench: rank candidate helper GGUFs by AUROC against oracle labels and throughput.

Restarts the local llama-server with each candidate, scores 500 rows for a few predicates
already on the oracle tape through LOTUS's sem_filter with logprobs, and writes
docs/helper_bench.json.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.request

import lotus
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from lotus.nl_expression import nle2str, parse_cols
from lotus.sem_ops.sem_filter import sem_filter
from lotus.templates import task_instructions
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = [
    "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
]
DATASET, PREDS, ROWS, PARALLEL = "goemotions", ["admiration", "amusement", "anger"], 500, 8


def restart_helper(gguf: str) -> subprocess.Popen:
    subprocess.run(["pkill", "-f", "llama-server"], check=False)
    time.sleep(2)
    p = subprocess.Popen(
        ["llama-server", "-m", os.path.join(ROOT, gguf), "--host", "127.0.0.1", "--port", "8081",
         "--parallel", str(PARALLEL), "-c", str(PARALLEL * 2048), "-ngl", "99", "--alias", "helper-local"],
        stdout=open(os.path.join(ROOT, "serving", "logs", "helper.log"), "w"), stderr=subprocess.STDOUT,
    )
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2)
            return p
        except Exception:
            time.sleep(1)
    raise RuntimeError("helper did not start")


def main() -> None:
    load_dotenv(os.path.join(ROOT, ".env"))
    lotus.settings.configure(enable_cache=False)
    logging.getLogger("lotus").setLevel(logging.ERROR)
    import sys

    sys.path.insert(0, os.path.join(ROOT, "src"))
    from semviews.adapters.lotus.taped_lm import proxy_lm

    models_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "models.yaml")))
    pool = {p["id"]: p for p in yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", f"{DATASET}.yaml")))["predicates"]}
    df = pd.read_parquet(os.path.join(ROOT, "data", "datasets", f"{DATASET}.parquet")).iloc[:ROWS]
    results = []
    for gguf in CANDIDATES:
        proc = restart_helper(gguf)
        lm = proxy_lm("helper", models_cfg, max_batch_size=PARALLEL)
        aucs, rates = {}, []
        for pid in PREDS:
            tape = pd.read_parquet(os.path.join(ROOT, "data", "tapes", "parts", f"{DATASET}__oracle-llama", f"{pid}.parquet"))
            y = df.merge(tape[["row_id", "label"]], on="row_id")["label"].to_numpy()
            cols = parse_cols(pool[pid]["langex"])
            docs = task_instructions.df2multimodal_info(df, cols)
            t0 = time.time()
            out = sem_filter(docs, lm, nle2str(pool[pid]["langex"], cols), logprobs=True, show_progress_bar=False)
            rates.append(len(docs) / (time.time() - t0))
            s = np.asarray(lm.format_logprobs_for_filter_cascade(out.logprobs).positive_probs)
            aucs[pid] = round(float(roc_auc_score(y, s)), 4)
        r = {"gguf": gguf, "auroc": aucs, "mean_auroc": round(float(np.mean(list(aucs.values()))), 4),
             "calls_per_s": round(float(np.median(rates)), 1)}
        print(json.dumps(r), flush=True)
        results.append(r)
        proc.terminate()
    json.dump(results, open(os.path.join(ROOT, "docs", "helper_bench.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
