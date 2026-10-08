# Copyright 2026 The semviews authors. Licensed under the Apache License, Version 2.0.
"""make llm-check: smoke-test every alias through LOTUS's LM and the local proxy.

Reports: alias listing, one answer per alias, temperature-0 repeatability over 20
prompts, whether log-probabilities come back on the helper, latency, throughput.
Never prints secrets.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request

import yaml
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

import lotus  # noqa: E402
from lotus.models import LM  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "models.yaml")))
BASE = CFG["proxy_base"]
KEY = os.environ.get("LITELLM_MASTER_KEY", "")


def list_aliases() -> list[str]:
    req = urllib.request.Request(f"{BASE}/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return [m["id"] for m in json.load(r)["data"]]


def make_lm(alias: str, **kw) -> LM:
    samp = dict(CFG["sampling"].get(alias, {}))
    max_tokens = samp.pop("max_tokens", 8)
    samp.pop("temperature", None)
    return LM(
        model=f"openai/{alias}",
        api_base=BASE,
        api_key=KEY,
        temperature=0.0,
        max_tokens=max_tokens,
        max_batch_size=kw.pop("max_batch_size", 16),
        **samp,
        **kw,
    )


REVIEWS = [
    "The battery died after two days and the phone gets really hot when charging.",
    "Great sound quality, the headphones fit comfortably for hours.",
    "Screen cracked on its own within a week, clearly a manufacturing defect.",
    "Shipping was fast but the box was damaged.",
    "Battery life is amazing, lasts all weekend.",
]


def check_alias(alias: str, logprobs: bool = False) -> dict:
    lm = make_lm(alias)
    docs = [{"text": r} for r in REVIEWS * 4]  # 20 prompts
    from lotus.sem_ops.sem_filter import sem_filter

    t0 = time.time()
    out1 = sem_filter(docs, lm, "{text} complains about the battery", logprobs=logprobs, show_progress_bar=False)
    t1 = time.time()
    out2 = sem_filter(docs, lm, "{text} complains about the battery", show_progress_bar=False)
    t2 = time.time()
    agree = sum(a == b for a, b in zip(out1.outputs, out2.outputs)) / len(docs)
    # repeated identical prompts inside one run: positions i and i+5 are the same review
    self_agree = sum(out1.outputs[i] == out1.outputs[i + 5] for i in range(15)) / 15
    res = {
        "alias": alias,
        "labels_first5": out1.outputs[:5],
        "raw_first": out1.raw_outputs[0][:60],
        "repeat_agreement": agree,
        "within_run_agreement": self_agree,
        "throughput_calls_per_s": round(len(docs) / max(t1 - t0, 1e-6), 2),
        "throughput_run2": round(len(docs) / max(t2 - t1, 1e-6), 2),
    }
    if logprobs:
        ok = out1.logprobs is not None and all(len(lp) > 0 and lp[0].top_logprobs for lp in out1.logprobs)
        res["logprobs_ok"] = bool(ok)
        if ok:
            f = lm.format_logprobs_for_filter_cascade(out1.logprobs)
            res["helper_pos_probs_first5"] = [round(p, 3) for p in f.positive_probs[:5]]
    # single-call latency
    lats = []
    for r in REVIEWS:
        s = time.time()
        sem_filter([{"text": r}], lm, "{text} complains about the battery", show_progress_bar=False)
        lats.append(time.time() - s)
    res["latency_median_s"] = round(statistics.median(lats), 3)
    return res


def main() -> int:
    lotus.settings.configure(enable_cache=False)
    aliases = list_aliases()
    print("aliases:", aliases)
    ok = True
    for alias in [CFG["oracle"], CFG["oracle_secondary"], CFG["helper"]]:
        if alias not in aliases:
            print(f"MISSING alias {alias}")
            ok = False
            continue
        try:
            res = check_alias(alias, logprobs=(alias == CFG["helper"]))
            print(json.dumps(res))
            if res["repeat_agreement"] < 1.0:
                print(f"WARN {alias}: temperature-0 answers not repeatable")
            if alias == CFG["helper"] and not res.get("logprobs_ok"):
                print("FAIL helper: no log-probabilities")
                ok = False
        except Exception as e:  # report and continue with the other aliases
            print(f"FAIL {alias}: {type(e).__name__}: {str(e)[:300]}")
            ok = False
    print("llm-check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
