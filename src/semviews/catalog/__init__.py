# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Semantic view catalog: label columns bound to a predicate, table, and model, with provenance.

Label values: 1, 0, or UNKNOWN (-1). Provenance per label: ORACLE (paid oracle call),
DERIVED (accepted or rejected by a certified decision), or NONE. Only ORACLE labels ever
count as oracle samples; DERIVED labels may only shape strata.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

UNKNOWN = -1
NONE, ORACLE, DERIVED = 0, 1, 2


@dataclass
class View:
    predicate_id: str
    text: str
    model: str
    labels: np.ndarray  # int8 in {-1, 0, 1}
    provenance: np.ndarray  # int8 in {NONE, ORACLE, DERIVED}
    meta: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float((self.labels != UNKNOWN).mean())

    def oracle_mask(self) -> np.ndarray:
        return self.provenance == ORACLE


class Catalog:
    """Views over one table (a fixed canonical row order). In-memory, with DuckDB persistence."""

    def __init__(self, table: str, n_rows: int, path: str | None = None):
        self.table = table
        self.n_rows = n_rows
        self.path = path
        self.views: dict[str, View] = {}
        self.relations: dict[tuple[str, str], str] = {}  # cached LLM relation judgments
        self.row_ids: list[str] = []  # optional content-hash registry (live engines)
        self._row_pos: dict[str, int] = {}
        if path and os.path.exists(path):
            self.load(path)

    def ensure_rows(self, row_ids: list[str]) -> np.ndarray:
        """Map content-hash row ids to positions, appending unseen rows as UNKNOWN in every view."""
        pos = np.empty(len(row_ids), dtype=np.int64)
        new = 0
        for i, r in enumerate(row_ids):
            j = self._row_pos.get(r)
            if j is None:
                j = len(self.row_ids)
                self._row_pos[r] = j
                self.row_ids.append(r)
                new += 1
            pos[i] = j
        grow = len(self.row_ids) - self.n_rows
        if grow > 0:
            for v in self.views.values():
                v.labels = np.concatenate([v.labels, np.full(grow, UNKNOWN, dtype=np.int8)])
                v.provenance = np.concatenate([v.provenance, np.zeros(grow, dtype=np.int8)])
            self.n_rows = len(self.row_ids)
        return pos

    def __contains__(self, predicate_id: str) -> bool:
        return predicate_id in self.views

    def get(self, predicate_id: str) -> View | None:
        return self.views.get(predicate_id)

    def others(self, predicate_id: str) -> list[View]:
        return [v for k, v in self.views.items() if k != predicate_id]

    def register(self, view: View) -> None:
        """Insert or merge a view. Oracle labels always win over derived ones."""
        old = self.views.get(view.predicate_id)
        if old is None:
            self.views[view.predicate_id] = view
            return
        take = (view.provenance == ORACLE) | ((old.provenance != ORACLE) & (view.labels != UNKNOWN))
        old.labels = np.where(take, view.labels, old.labels).astype(np.int8)
        old.provenance = np.where(take, view.provenance, old.provenance).astype(np.int8)
        old.meta.update(view.meta)

    def known_oracle(self, predicate_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Row positions with oracle labels for this predicate, and those labels."""
        v = self.views.get(predicate_id)
        if v is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8)
        pos = np.flatnonzero(v.provenance == ORACLE)
        return pos, v.labels[pos]

    def storage_bytes(self) -> int:
        return sum(v.labels.nbytes + v.provenance.nbytes for v in self.views.values())

    # --- persistence ----------------------------------------------------------------------
    def save(self, path: str | None = None) -> None:
        import duckdb
        import pandas as pd

        path = path or self.path
        assert path, "no catalog path"
        rows = []
        for v in self.views.values():
            idx = np.flatnonzero(v.labels != UNKNOWN)
            rows.append(pd.DataFrame({"predicate_id": v.predicate_id, "row": idx, "label": v.labels[idx], "prov": v.provenance[idx]}))
        labels = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["predicate_id", "row", "label", "prov"])
        meta = pd.DataFrame(
            [{"predicate_id": v.predicate_id, "text": v.text, "model": v.model, "meta": json.dumps(v.meta)} for v in self.views.values()]
        )
        if os.path.exists(path):
            os.remove(path)
        con = duckdb.connect(path)
        con.execute("CREATE TABLE info AS SELECT ? AS tbl, ? AS n_rows", [self.table, self.n_rows])
        con.register("rows_df", pd.DataFrame({"row_id": self.row_ids}))
        con.execute("CREATE TABLE rows AS SELECT * FROM rows_df")
        con.register("labels_df", labels)
        con.register("meta_df", meta)
        con.execute("CREATE TABLE labels AS SELECT * FROM labels_df")
        con.execute("CREATE TABLE views AS SELECT * FROM meta_df")
        con.close()

    def load(self, path: str) -> None:
        import duckdb

        con = duckdb.connect(path, read_only=True)
        self.table, self.n_rows = con.execute("SELECT tbl, n_rows FROM info").fetchone()
        meta = con.execute("SELECT * FROM views").df()
        labels = con.execute("SELECT * FROM labels").df()
        self.row_ids = con.execute("SELECT row_id FROM rows").df()["row_id"].tolist()
        self._row_pos = {r: i for i, r in enumerate(self.row_ids)}
        con.close()
        for _, m in meta.iterrows():
            lab = np.full(self.n_rows, UNKNOWN, dtype=np.int8)
            prov = np.zeros(self.n_rows, dtype=np.int8)
            sub = labels[labels.predicate_id == m.predicate_id]
            lab[sub["row"].to_numpy()] = sub["label"].to_numpy()
            prov[sub["row"].to_numpy()] = sub["prov"].to_numpy()
            self.views[m.predicate_id] = View(m.predicate_id, m.text, m.model, lab, prov, json.loads(m.meta))
