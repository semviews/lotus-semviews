# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Fast helper-tape recorder with LOTUS's exact prompt and score semantics.

LOTUS's LM adds per-call Python overhead (token counting, cost lookup, logging) that capped the
helper tape at about 7 calls/s against a local server that serves 60. This recorder builds every
prompt with LOTUS's own `filter_formatter`, posts the same chat-completions request the LM would
(temperature 0, max_tokens from configs/models.yaml, logprobs with top_logprobs=10), and turns
the response into the positive-token probability with LOTUS's own
`LM.format_logprobs_for_filter_cascade`. Output rows match the helper tape schema.

    python -m semviews.adapters.lotus.fast_helper --dataset dbpedia
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import os
import time
import urllib.request

import lotus
import numpy as np
import pandas as pd
import yaml
from litellm.types.utils import ChatCompletionTokenLogprob
from lotus.models import LM
from lotus.nl_expression import nle2str, parse_cols
from lotus.sem_ops.postprocessors import filter_postprocess
from lotus.templates import task_instructions

from semviews.adapters.lotus.build_tape import PROMPT_VERSION, ROOT, parts_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--pool", default="", help="predicate pool file in configs/predicates/ (default: <dataset>)")
    ap.add_argument("--tag", default="", help="separate parts/tape suffix, e.g. rw")
    a = ap.parse_args()
    logging.getLogger("lotus").setLevel(logging.ERROR)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "models.yaml")))
    route = cfg["direct_routes"]["helper"]
    samp = dict(cfg["sampling"]["helper"])
    url = route["api_base"].rstrip("/") + "/chat/completions"
    model_name = route["model"].split("/", 1)[1]
    lm = LM(model=route["model"], api_base=route["api_base"], api_key=route["api_key"], max_tokens=samp["max_tokens"])
    pool = yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", f"{a.pool or a.dataset}.yaml")))["predicates"]
    df = pd.read_parquet(os.path.join(ROOT, "data", "datasets", f"{a.dataset}.parquet"))
    out_dir = parts_dir(a.dataset, "helper", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    sampling = json.dumps(samp, sort_keys=True)

    def call(messages):
        body = json.dumps({"model": model_name, "messages": messages, "temperature": 0, "max_tokens": samp["max_tokens"],
                           "logprobs": True, "top_logprobs": 10}).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=30) as r:
                    j = json.load(r)
                return j, time.time() - t0
            except Exception:
                time.sleep(min(30, 2**attempt))
        raise RuntimeError("helper request failed 6 times")

    with cf.ThreadPoolExecutor(a.threads) as ex:
        for p in pool:
            pid = f"{a.dataset}/{p['id']}"
            part = os.path.join(out_dir, f"{p['id']}.parquet")
            done = pd.read_parquet(part) if os.path.exists(part) else None
            todo = df if done is None else df[~df.row_id.isin(done.row_id)]
            if len(todo) == 0:
                continue
            cols = parse_cols(p["langex"])
            instr = nle2str(p["langex"], cols)
            frames = [] if done is None else [done]
            t_start = time.time()
            for s in range(0, len(todo), a.chunk):
                sub = todo.iloc[s : s + a.chunk]
                docs = task_instructions.df2multimodal_info(sub, cols)
                msgs = [task_instructions.filter_formatter(lm, d, instr) for d in docs]
                res = list(ex.map(call, msgs))
                contents = [r[0]["choices"][0]["message"]["content"] or "" for r in res]
                lps = [[ChatCompletionTokenLogprob(**tok) for tok in r[0]["choices"][0]["logprobs"]["content"]] for r in res]
                score = lm.format_logprobs_for_filter_cascade(lps).positive_probs
                labels = filter_postprocess(contents, lm, True).outputs
                usage = [r[0].get("usage", {}) for r in res]
                frames.append(pd.DataFrame({
                    "dataset": a.dataset, "row_id": sub.row_id.to_numpy(), "predicate_id": pid, "model": "helper",
                    "model_id": model_name, "prompt_version": PROMPT_VERSION, "raw_output": [c[:64] for c in contents],
                    "parse_ok": [("true" in c.lower()) or ("false" in c.lower()) for c in contents],
                    "tokens_in": np.array([u.get("prompt_tokens", 0) for u in usage], dtype="int32"),
                    "tokens_out": np.array([u.get("completion_tokens", 0) for u in usage], dtype="int32"),
                    "latency_ms": np.array([1000 * r[1] for r in res], dtype="float32"),
                    "timestamp": dt.datetime.now(dt.UTC).isoformat(), "sampling": sampling,
                    "score": np.asarray(score, dtype=np.float32), "label": np.asarray(labels, dtype=np.int8),
                }))
                pd.concat(frames, ignore_index=True).to_parquet(part, index=False)
            print(f"{pid}: {len(todo)} rows at {len(todo) / (time.time() - t_start):.1f}/s", flush=True)


if __name__ == "__main__":
    lotus.settings.configure(enable_cache=False)
    main()
