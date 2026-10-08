# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Tape-backed oracle and helper scores. The only code that opens tape files."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd
import yaml

from semviews.oracle.meter import CostMeter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TapeMiss(KeyError):
    """A requested (predicate, row) pair is not on the tape."""


def load_prices(model: str, path: str | None = None) -> tuple[float, float]:
    path = path or os.path.join(ROOT, "configs", "budgets.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    p = cfg["price_per_mtok"].get(model, {"input": 0.0, "output": 0.0})
    return p["input"] / 1e6, p["output"] / 1e6


class Oracle(Protocol):
    model: str
    row_ids: np.ndarray

    def label(self, predicate_id: str, rows: Sequence[str] | np.ndarray, *, purpose: str) -> np.ndarray: ...

    @property
    def meter(self) -> CostMeter: ...


class _Matrix:
    """Dense (predicate x row) storage for one dataset's tape."""

    def __init__(self, df: pd.DataFrame, value_col: str, row_ids: np.ndarray | None = None):
        if row_ids is None:
            row_ids = np.array(sorted(df["row_id"].unique()))
        self.row_ids = np.asarray(row_ids)
        self.row_pos = {r: i for i, r in enumerate(self.row_ids)}
        self.pred_ids = sorted(df["predicate_id"].unique())
        self.pred_pos = {p: i for i, p in enumerate(self.pred_ids)}
        n_p, n_r = len(self.pred_ids), len(self.row_ids)
        fill = np.nan if value_col == "score" else -1
        dtype = np.float32 if value_col == "score" else np.int8
        self.values = np.full((n_p, n_r), fill, dtype=dtype)
        self.tin = np.zeros((n_p, n_r), dtype=np.int32)
        self.tout = np.zeros((n_p, n_r), dtype=np.int32)
        pi = df["predicate_id"].map(self.pred_pos).to_numpy()
        ri = df["row_id"].map(self.row_pos)
        keep = ri.notna().to_numpy()
        pi, ri = pi[keep], ri[keep].astype(int).to_numpy()
        self.values[pi, ri] = df[value_col].to_numpy()[keep]
        if "tokens_in" in df:
            self.tin[pi, ri] = df["tokens_in"].to_numpy()[keep]
            self.tout[pi, ri] = df["tokens_out"].to_numpy()[keep]

    def positions(self, rows: Sequence[str] | np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        if rows.dtype.kind in "iu":
            return rows.astype(np.int64)
        try:
            return np.fromiter((self.row_pos[r] for r in rows), dtype=np.int64, count=len(rows))
        except KeyError as e:
            raise TapeMiss(str(e)) from e


class TapeOracle:
    """Replay-only oracle over a recorded tape. Charges every distinct pair once per scope."""

    def __init__(self, tape: pd.DataFrame | str, model: str, row_ids: np.ndarray | None = None):
        df = pd.read_parquet(tape) if isinstance(tape, str) else tape
        df = df[df["model"] == model] if "model" in df.columns else df
        self.model = model
        self._m = _Matrix(df, "label", row_ids)
        self.row_ids = self._m.row_ids
        pin, pout = load_prices(model)
        self._meter = CostMeter(price_in=pin, price_out=pout)
        self._charged: dict[str, np.ndarray] = {}
        self._scope_token = -1

    def clone(self) -> TapeOracle:
        """Same tape, fresh meter and charging state (cheap: shares the label matrix)."""
        o = object.__new__(TapeOracle)
        o.model, o._m, o.row_ids = self.model, self._m, self.row_ids
        o._meter = CostMeter(price_in=self._meter.price_in, price_out=self._meter.price_out)
        o._charged, o._scope_token = {}, -1
        return o

    @property
    def meter(self) -> CostMeter:
        return self._meter

    @property
    def predicate_ids(self) -> list[str]:
        return list(self._m.pred_ids)

    def n_rows(self) -> int:
        return len(self.row_ids)

    def label(self, predicate_id: str, rows: Sequence[str] | np.ndarray, *, purpose: str) -> np.ndarray:
        if predicate_id not in self._m.pred_pos:
            raise TapeMiss(predicate_id)
        p = self._m.pred_pos[predicate_id]
        pos = self._m.positions(rows)
        vals = self._m.values[p, pos]
        if (vals < 0).any():
            raise TapeMiss(f"{predicate_id}: {int((vals < 0).sum())} rows missing")
        # charge distinct, not-yet-charged pairs in the current scope
        scope = self._meter.scope_id
        if self._scope_token != scope:
            self._charged = {}
            self._scope_token = scope
        charged = self._charged.setdefault(predicate_id, np.zeros(len(self.row_ids), dtype=bool))
        upos = np.unique(pos)
        new = upos[~charged[upos]]
        if len(new):
            charged[new] = True
            m = self._meter
            m.calls += len(new)
            m.tokens_in += int(self._m.tin[p, new].sum())
            m.tokens_out += int(self._m.tout[p, new].sum())
            m.calls_by_purpose[purpose] += len(new)
        return vals.astype(np.int8)

    # --- evaluation-only access -------------------------------------------------------------
    def ground_truth(self, predicate_id: str) -> np.ndarray:
        """Full label column for scoring results. Experiment harness only; never passed to methods."""
        return self._m.values[self._m.pred_pos[predicate_id]].astype(np.int8)


class HelperScores:
    """Taped helper-LM scores. Free for oracle accounting; counted as helper calls."""

    def __init__(self, tape: pd.DataFrame | str, row_ids: np.ndarray, meter: CostMeter | None = None):
        df = pd.read_parquet(tape) if isinstance(tape, str) else tape
        self._m = _Matrix(df, "score", row_ids)
        self.meter = meter

    def clone(self, meter: CostMeter | None = None) -> HelperScores:
        h = object.__new__(HelperScores)
        h._m, h.meter = self._m, meter
        return h

    def has(self, predicate_id: str) -> bool:
        return predicate_id in self._m.pred_pos

    def score(self, predicate_id: str, rows: Sequence[str] | np.ndarray | None = None) -> np.ndarray:
        p = self._m.pred_pos[predicate_id]
        pos = np.arange(len(self._m.row_ids)) if rows is None else self._m.positions(rows)
        if self.meter is not None:
            self.meter.helper_calls += len(pos)
        return self._m.values[p, pos].astype(np.float64)


def tape_path(dataset: str, model: str) -> str:
    """Consolidated tape for (dataset, model), falling back to the in-progress part files."""
    return os.path.join(ROOT, "data", "tapes", f"{dataset}__{model}.parquet")


def load_tape(dataset: str, model: str, consolidated_only: bool = False) -> pd.DataFrame:
    import glob

    parts = sorted(glob.glob(os.path.join(ROOT, "data", "tapes", "parts", f"{dataset}__{model}", "*.parquet")))
    p = tape_path(dataset, model)
    if parts and not consolidated_only:
        df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    else:
        df = pd.read_parquet(p)
    # keep only predicates recorded for every row
    full = df.groupby("predicate_id").row_id.transform("size") == df.row_id.nunique()
    return df[full].reset_index(drop=True)
