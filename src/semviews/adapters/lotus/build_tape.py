# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Record the oracle and helper tapes with LOTUS's own sem_filter prompt.

Resumable: one part file per (dataset, model, predicate). Enforces the budget cap
from configs/budgets.yaml before every chunk. Usage:

    python -m semviews.adapters.lotus.build_tape --dataset goemotions --model oracle-llama
    python -m semviews.adapters.lotus.build_tape --dataset goemotions --model helper
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import os
import sys
import time

import lotus
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from lotus.nl_expression import nle2str, parse_cols
from lotus.sem_ops.sem_filter import sem_filter
from lotus.templates import task_instructions

from semviews.adapters.lotus.taped_lm import RecordingLM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
PROMPT_VERSION = f"lotus-{__import__('importlib.metadata').metadata.version('lotus-ai')}-sem_filter-default"
log = logging.getLogger("build_tape")


def _cfg(name: str) -> dict:
    with open(os.path.join(ROOT, "configs", name)) as f:
        return yaml.safe_load(f)


def parts_dir(dataset: str, model: str, tag: str = "") -> str:
    return os.path.join(ROOT, "data", "tapes", "parts", f"{dataset}__{model}" + (f"__{tag}" if tag else ""))


def spent_usd() -> float:
    """Total spend over every recorded oracle part, priced from budgets.yaml."""
    prices = _cfg("budgets.yaml")["price_per_mtok"]
    total = 0.0
    for d in glob.glob(os.path.join(ROOT, "data", "tapes", "parts", "*__*")):
        model = os.path.basename(d).split("__")[1]
        p = prices.get(model, {"input": 0, "output": 0})
        for f in glob.glob(os.path.join(d, "*.parquet")):
            for attempt in range(5):  # another builder may be replacing this part right now
                try:
                    t = pd.read_parquet(f, columns=["tokens_in", "tokens_out"])
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2)
            total += (t.tokens_in.sum() * p["input"] + t.tokens_out.sum() * p["output"]) / 1e6
    return float(total)


def run(dataset: str, model: str, predicates: list[str] | None, n_rows: int | None, chunk: int, batch: int,
        pool_name: str = "", tag: str = "") -> None:
    load_dotenv(os.path.join(ROOT, ".env"))
    lotus.settings.configure(enable_cache=False)
    logging.getLogger("lotus").setLevel(logging.ERROR)
    models_cfg = _cfg("models.yaml")
    cap = _cfg("budgets.yaml")["usd_cap_total"]
    pool = yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", f"{pool_name or dataset}.yaml")))
    df = pd.read_parquet(os.path.join(ROOT, "data", "datasets", f"{dataset}.parquet"))
    if n_rows:
        df = df.iloc[:n_rows]
    is_helper = model == models_cfg["helper"]
    lm = RecordingLM.for_alias(model, models_cfg, max_batch_size=batch)
    sampling = json.dumps(models_cfg["sampling"].get(model, {}), sort_keys=True)
    out_dir = parts_dir(dataset, model, tag)
    os.makedirs(out_dir, exist_ok=True)
    preds = [p for p in pool["predicates"] if predicates is None or p["id"] in predicates]
    for p in preds:
        pid = f"{dataset}/{p['id']}"
        part = os.path.join(out_dir, f"{p['id']}.parquet")
        done = pd.read_parquet(part) if os.path.exists(part) else None
        todo = df if done is None else df[~df.row_id.isin(done.row_id)]
        if len(todo) == 0:
            continue
        cols = parse_cols(p["langex"])
        instr = nle2str(p["langex"], cols)
        frames = [] if done is None else [done]
        t_start = time.time()
        for s in range(0, len(todo), chunk):
            if not is_helper and spent_usd() >= cap:
                log.error("budget cap %.2f reached; stopping", cap)
                sys.exit(2)
            sub = todo.iloc[s : s + chunk]
            docs = task_instructions.df2multimodal_info(sub, cols)
            for attempt in range(6):
                lm.records = []
                try:
                    out = sem_filter(docs, lm, instr, logprobs=is_helper, show_progress_bar=False)
                    break
                except Exception as e:  # transport errors and rate limits: back off and retry
                    wait = min(120, 5 * 2**attempt)
                    log.warning("%s chunk %d attempt %d failed: %s; retry in %ds", pid, s, attempt, str(e)[:160], wait)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"giving up on {pid} chunk {s}")
            rec = pd.DataFrame(lm.records)
            assert len(rec) == len(sub), (len(rec), len(sub))
            raw = [str(x)[:64] for x in out.raw_outputs]
            frame = pd.DataFrame(
                {
                    "dataset": dataset,
                    "row_id": sub.row_id.to_numpy(),
                    "predicate_id": pid,
                    "model": model,
                    "model_id": rec.model_id.to_numpy(),
                    "prompt_version": PROMPT_VERSION,
                    "raw_output": raw,
                    "parse_ok": [("true" in r.lower()) or ("false" in r.lower()) for r in raw],
                    "tokens_in": rec.tokens_in.astype("int32").to_numpy(),
                    "tokens_out": rec.tokens_out.astype("int32").to_numpy(),
                    "latency_ms": rec.latency_ms.astype("float32").to_numpy(),
                    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                    "sampling": sampling,
                }
            )
            if is_helper:
                f = lm.format_logprobs_for_filter_cascade(out.logprobs)
                frame["score"] = np.asarray(f.positive_probs, dtype=np.float32)
                frame["label"] = np.asarray(out.outputs, dtype=np.int8)
            else:
                frame["label"] = np.asarray(out.outputs, dtype=np.int8)
            frames.append(frame)
            pd.concat(frames, ignore_index=True).to_parquet(part + ".tmp", index=False)
            os.replace(part + ".tmp", part)  # atomic, so concurrent builders never read a partial part
        rate = len(todo) / max(time.time() - t_start, 1e-6)
        allp = pd.read_parquet(part)
        log.info("%s %s: %d rows, sel=%.3f, %.1f calls/s, spent so far $%.2f", model, pid, len(allp), allp.label.mean(), rate, spent_usd())


def consolidate(dataset: str, model: str, tag: str = "") -> str:
    files = sorted(glob.glob(os.path.join(parts_dir(dataset, model, tag), "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = os.path.join(ROOT, "data", "tapes", f"{dataset}__{model}" + (f"__{tag}" if tag else "") + ".parquet")
    df.to_parquet(out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--predicates", nargs="*")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--pool", default="", help="predicate pool file in configs/predicates/ (default: <dataset>)")
    ap.add_argument("--tag", default="", help="separate parts/tape suffix, e.g. rw for rephrased predicates")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not a.consolidate:
        run(a.dataset, a.model, a.predicates, a.rows, a.chunk, a.batch, a.pool, a.tag)
    print(consolidate(a.dataset, a.model, a.tag))


if __name__ == "__main__":
    main()
