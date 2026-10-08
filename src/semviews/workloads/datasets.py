# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Prepare the experiment datasets: a fixed seeded sample of N rows each, with row ids.

Output: data/datasets/<name>.parquet with columns row_id, text, and human labels where
the source has them. Sources (opened 2026-10-01):
- goemotions: huggingface.co/datasets/google-research-datasets/go_emotions (simplified)
- dbpedia:    huggingface.co/datasets/DeveloperOats/DBPedia_Classes (test split)
- reviews:    huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023, raw Electronics reviews,
              first 60,000 lines of the JSONL file
- pubmed:     huggingface.co/datasets/ccdv/pubmed-summarization (section config, test split),
              abstracts only; paired with the ScaleDoc pubmed predicate pool (W1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from semviews.ids import row_id

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "data", "datasets")
MAX_CHARS = 700  # cap text length to bound tokens per call


def _finish(df: pd.DataFrame, n: int, seed: int, max_chars: int = MAX_CHARS) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, max_chars)
    df = df[df["text"].str.len() >= 20].drop_duplicates("text")
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    df.insert(0, "row_id", [row_id([t]) for t in df["text"]])
    assert df["row_id"].is_unique
    return df


def goemotions(n: int, seed: int) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    names = ds.features["labels"].feature.names
    df = ds.to_pandas()
    df["emotions"] = df["labels"].apply(lambda ls: json.dumps(sorted(names[i] for i in ls)))
    return _finish(df[["text", "emotions"]], n, seed)


def dbpedia(n: int, seed: int) -> pd.DataFrame:
    from datasets import load_dataset

    df = load_dataset("DeveloperOats/DBPedia_Classes", split="test").to_pandas()
    return _finish(df[["text", "l1", "l2", "l3"]], n, seed)


def reviews(n: int, seed: int) -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "raw", "electronics_head.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            body = (r.get("text") or "").strip()
            title = (r.get("title") or "").strip()
            if len(body) < 80:
                continue
            rows.append({"text": f"{title}. {body}" if title else body, "rating": r.get("rating")})
    return _finish(pd.DataFrame(rows), n, seed)


def pubmed(n: int, seed: int) -> pd.DataFrame:
    from datasets import load_dataset

    df = load_dataset("ccdv/pubmed-summarization", "section", split="test").to_pandas()
    df = df.rename(columns={"abstract": "text"})[["text"]]
    return _finish(df, n, seed, max_chars=1200)


BUILDERS = {"goemotions": goemotions, "dbpedia": dbpedia, "reviews": reviews, "pubmed": pubmed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--pubmed-pool", action="store_true",
                    help="only fetch the ScaleDoc PubMed predicate pool into configs/predicates/pubmed.yaml")
    ap.add_argument("names", nargs="*", default=list(BUILDERS))
    a = ap.parse_args()
    if a.pubmed_pool:
        scaledoc_pubmed_predicates()
        return
    os.makedirs(OUT, exist_ok=True)
    for name in a.names:
        df = BUILDERS[name](a.n, a.seed)
        df.to_parquet(os.path.join(OUT, f"{name}.parquet"), index=False)
        print(name, len(df), "mean chars", round(df["text"].str.len().mean(), 1))



def scaledoc_pubmed_predicates() -> None:
    """Write configs/predicates/pubmed.yaml from ScaleDoc's query.json, fetched at runtime.

    The source repository has no licence, so the pool is not committed or redistributed. The
    pool the experiments used hashes to POOL_SHA256; a different hash means the upstream file
    changed and the PubMed tapes no longer match it.
    """
    import urllib.request

    import yaml

    url = "https://raw.githubusercontent.com/Seurgul/ScaleDoc/main/dataset/query.json"
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    preds = [dict(id=f"q{q['q_id']:02d}", langex="Question about the document in {text}: " + q["query"], level=None, human=None)
             for q in d["pubmed"]]
    preds += [dict(id=f"ext{q['q_id']:02d}", langex="Question about the document in {text}: " + q["query"], level=None, human=None)
              for q in d["pubmed_ext"] if not q["query"].startswith("This is a random")]
    out = os.path.join(ROOT, "configs", "predicates", "pubmed.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(dict(dataset="pubmed", column="text", source=url, predicates=preds), f, sort_keys=False, width=200)
    h = hashlib.sha256(json.dumps([[p["id"], p["langex"]] for p in preds]).encode()).hexdigest()
    print(f"{out}: {len(preds)} predicates, sha256 {h}", "(matches)" if h == POOL_SHA256 else "(DIFFERS from the pool the experiments used)")


# SHA-256 over [[id, langex], ...] of the 25-predicate pool used in the experiments.
POOL_SHA256 = "5c2df285e922e631eae626a407e81fe5999b4078e9851ee8ca1219d6340d0ff0"


if __name__ == "__main__":
    np.random.seed(0)
    main()
