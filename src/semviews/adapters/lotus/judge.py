# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""LLM relation judge over predicate pairs, taped through TapedLM.

One call per unordered pair. Output: data/relations/<dataset>.json mapping "q||p" to the
relation of q with respect to p: equivalent, implies (q => p), implied_by (p => q),
exclusive, overlapping, unrelated.

    python -m semviews.adapters.lotus.judge --dataset goemotions
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os

import lotus
import yaml
from dotenv import load_dotenv

from semviews.adapters.lotus.taped_lm import TapedLM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

SYSTEM = (
    "You compare two yes/no conditions, A and B, that will each be checked against the same kind of text record. "
    "Decide which logical relationship holds between them for records of this kind.\n"
    "- equivalent: A holds exactly when B holds\n"
    "- A_implies_B: whenever A holds, B also holds, but not the other way around\n"
    "- B_implies_A: whenever B holds, A also holds, but not the other way around\n"
    "- exclusive: A and B can never both hold\n"
    "- overlapping: they often co-occur, but neither implies the other\n"
    "- unrelated: they are about different things and are largely independent\n"
    "Reply with exactly one label from the list and nothing else."
)
MAP = {"equivalent": "equivalent", "a_implies_b": "implies", "b_implies_a": "implied_by",
       "exclusive": "exclusive", "overlapping": "overlapping", "unrelated": "unrelated"}
FLIP = {"implies": "implied_by", "implied_by": "implies"}


def _cond(langex: str, record: str) -> str:
    return langex.replace("{text}", f"the {record}").replace("{", "").replace("}", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--record", default=None, help="noun for a record, e.g. 'product review'")
    ap.add_argument("--extra", default="", help="extra pool (e.g. reviews_rephrased): judge only pairs involving it, "
                    "written to data/relations/<extra>.json")
    a = ap.parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    lotus.settings.configure(enable_cache=False)
    logging.getLogger("lotus").setLevel(logging.ERROR)
    models_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "models.yaml")))
    record = a.record or {"goemotions": "social media comment", "dbpedia": "encyclopedia entry",
                          "reviews": "product review", "pubmed": "biomedical paper abstract"}.get(a.dataset, "text")
    pool = yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", f"{a.dataset}.yaml")))["predicates"]
    lm = TapedLM.for_alias(models_cfg["judge"], models_cfg, tape_path=os.path.join(ROOT, "data", "tapes", "judge.sqlite"),
                           max_batch_size=16)
    pairs = list(itertools.combinations(pool, 2))
    if a.extra:
        extra = yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", f"{a.extra}.yaml")))["predicates"]
        ids = {p["id"] for p in extra}
        pairs = [(p, q) for p, q in itertools.combinations(pool + extra, 2) if p["id"] in ids or q["id"] in ids]
    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"Records are {record}s.\nA: {_cond(p['langex'], record)}\nB: {_cond(q['langex'], record)}"}]
            for p, q in pairs]
    out = lm(msgs, show_progress_bar=False).outputs
    rel = {}
    bad = 0
    for (p, q), o in zip(pairs, out):
        lab = (o or "").strip().strip(".").lower().replace(" ", "_")
        r = MAP.get(lab)
        if r is None:
            r = next((v for k, v in MAP.items() if k in lab), "unrelated")
            bad += 1
        pa, qa = f"{a.dataset}/{p['id']}", f"{a.dataset}/{q['id']}"
        # r is the relation of A (=p) to B (=q)
        rel[f"{pa}||{qa}"] = r
        rel[f"{qa}||{pa}"] = FLIP.get(r, r)
    os.makedirs(os.path.join(ROOT, "data", "relations"), exist_ok=True)
    json.dump(rel, open(os.path.join(ROOT, "data", "relations", f"{a.extra or a.dataset}.json"), "w"), indent=0, sort_keys=True)
    print(a.dataset, len(pairs), "pairs; unparsed", bad, "; live", lm.live_calls, "replayed", lm.replayed_calls)


if __name__ == "__main__":
    main()
