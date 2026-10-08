# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""View matching signals: predicate embeddings, label co-occurrence, and the LLM relation judge."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence

import numpy as np

RELATIONS = ("equivalent", "implies", "implied_by", "exclusive", "overlapping", "unrelated")


class Embedder:
    """Local sentence-transformers embeddings with an on-disk cache. Zero oracle cost."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str | None = None):
        self.model_name = model
        self._model = None
        self.cache_dir = cache_dir
        self._mem: dict[str, np.ndarray] = {}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        texts = list(texts)
        missing = [t for t in texts if t not in self._mem]
        if missing:
            cached = self._disk_get(missing)
            todo = [t for t in missing if t not in cached]
            if todo:
                vecs = self._load().encode(todo, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
                cached.update(zip(todo, np.asarray(vecs, dtype=np.float32)))
                self._disk_put({t: cached[t] for t in todo})
            self._mem.update(cached)
        return np.stack([self._mem[t] for t in texts]) if texts else np.zeros((0, 384), dtype=np.float32)

    def _key(self, t: str) -> str:
        return hashlib.sha256(f"{self.model_name}\x00{t}".encode()).hexdigest()[:24]

    def _disk_get(self, texts: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if not self.cache_dir:
            return out
        for t in texts:
            p = os.path.join(self.cache_dir, self._key(t) + ".npy")
            if os.path.exists(p):
                out[t] = np.load(p)
        return out

    def _disk_put(self, items: dict[str, np.ndarray]) -> None:
        if not self.cache_dir:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        for t, v in items.items():
            np.save(os.path.join(self.cache_dir, self._key(t) + ".npy"), v)


def predicate_text(langex: str) -> str:
    """Strip the column placeholder so similarity compares the condition only."""
    import re

    return re.sub(r"\{[^}]+\}\s*", "", langex).strip()


def mutual_information(q: np.ndarray, sig: np.ndarray) -> float:
    """Plug-in MI (nats) between binary q and a discrete signature, with add-0.5 smoothing."""
    if len(q) == 0:
        return 0.0
    _, inv = np.unique(sig, return_inverse=True, axis=0)
    k = inv.max() + 1
    joint = np.full((k, 2), 0.5)
    np.add.at(joint, (inv, q.astype(int)), 1.0)
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    return float((joint * np.log(joint / (px * py))).sum())


def select_views(
    q_pilot: np.ndarray,
    cand_labels: list[np.ndarray],
    k: int,
    min_gain: float = 0.01,
) -> list[int]:
    """Greedy forward selection of up to k candidate views by conditional MI on pilot rows.

    cand_labels[j] holds candidate j's labels on the pilot rows (values -1/0/1). The pilot
    labels are spent on selection only and never enter a confidence bound.
    """
    chosen: list[int] = []
    base = np.zeros((len(q_pilot), 0), dtype=np.int8)
    cur = 0.0
    for _ in range(k):
        best, best_gain = None, min_gain
        for j, lab in enumerate(cand_labels):
            if j in chosen:
                continue
            sig = np.concatenate([base, lab[:, None]], axis=1)
            # penalize fragmentation: each extra signature value costs strata and Bonferroni budget
            gain = mutual_information(q_pilot, sig) - cur
            if gain > best_gain:
                best, best_gain = j, gain
        if best is None:
            break
        chosen.append(best)
        base = np.concatenate([base, cand_labels[best][:, None]], axis=1)
        cur = mutual_information(q_pilot, base)
    return chosen
