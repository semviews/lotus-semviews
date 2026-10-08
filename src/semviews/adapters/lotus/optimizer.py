# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""LazyFrame optimizers: a no-op contract optimizer and the ViewOptimizer.

ViewOptimizer rewrites each SemFilterNode into a ViewFilterNode bound to a Catalog. The
rewritten node keeps every SemFilterNode field, so LOTUS's prompts, LM, and cascade
arguments are unchanged; only the execution strategy differs. No LOTUS source is edited.

The node runs CertifiedFilter, the paper's operator, with the operator configuration frozen in
configs/experiments/sprint.yaml (`paper_config`). Row features are a local sentence embedding of
the columns the langex reads, reduced to at most 32 dimensions, as in the experiments. Passing a
ReuseConfig instead selects the earlier per-signature strata operator (ReuseFilter).
"""

from __future__ import annotations

from typing import Any

import lotus
import numpy as np
import pandas as pd
from lotus.ast.nodes import SemFilterNode
from lotus.ast.optimizer.base import BaseOptimizer
from lotus.nl_expression import nle2str, parse_cols
from lotus.sem_ops.sem_filter import sem_filter
from lotus.templates import task_instructions
from lotus.types import ProxyModel
from pydantic import ConfigDict

from semviews.catalog import Catalog
from semviews.ids import PROMPT_VERSION_TEMPLATE, row_id, view_key
from semviews.operators.region_filter import CertConfig, CertifiedFilter
from semviews.operators.reuse_filter import ReuseConfig, ReuseFilter, Targets
from semviews.oracle.meter import CostMeter
from semviews.relate import Embedder

PROMPT_VERSION = PROMPT_VERSION_TEMPLATE.format(version=__import__("importlib.metadata").metadata.version("lotus-ai"))


def paper_config(**overrides: Any) -> CertConfig:
    """CertConfig with the paper's operator settings (configs/experiments/sprint.yaml: operator)."""
    from semviews.workloads.runner import OPERATOR

    return CertConfig(**{**OPERATOR, **overrides})


def row_features(embedder: Embedder, df: pd.DataFrame, cols: list[str], positions: np.ndarray, n_rows: int,
                 dims: int = 32) -> np.ndarray:
    """Embed the langex columns of each row, PCA-reduce and scale as in the experiments, and place
    the result at the rows' catalog positions (other catalog rows get zeros; they are outside the
    filtered universe)."""
    texts = [" ".join(str(v) for v in vals) for vals in df[cols].itertuples(index=False, name=None)]
    emb = embedder.embed(texts)
    k = min(dims, emb.shape[0] - 1, emb.shape[1])
    out = np.zeros((n_rows, max(k, 0)))
    if k <= 0:
        return out
    from sklearn.decomposition import PCA

    z = PCA(n_components=k, random_state=0).fit_transform(emb)
    out[positions] = z / np.maximum(z.std(axis=0, keepdims=True), 1e-9) * 0.5
    return out


class NoOpOptimizer(BaseOptimizer):
    """Returns the node list unchanged. Proves the extension point needs no LOTUS edits."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def optimize(self, nodes, train_data=None):
        self.seen = [type(n).__name__ for n in nodes]
        return nodes


class LotusOracle:
    """Oracle over the rows of one DataFrame, labeling with LOTUS's own sem_filter prompt."""

    def __init__(self, lm: lotus.models.LM, df: pd.DataFrame, langex: str, positions: np.ndarray, pid: str, **filter_kw: Any):
        self.lm = lm
        self.model = lm.model
        self.df = df
        self.cols = parse_cols(langex)
        self.instr = nle2str(langex, self.cols)
        self.pid = pid
        self._pos_to_iloc = {int(p): i for i, p in enumerate(positions)}
        self._kw = filter_kw
        self._meter = CostMeter()

    @property
    def meter(self) -> CostMeter:
        return self._meter

    def label(self, predicate_id: str, rows, *, purpose: str) -> np.ndarray:
        assert predicate_id == self.pid
        rows = np.asarray(rows, dtype=np.int64)
        ilocs = [self._pos_to_iloc[int(r)] for r in rows]
        docs = task_instructions.df2multimodal_info(self.df.iloc[ilocs], self.cols)
        out = sem_filter(docs, self.lm, self.instr, show_progress_bar=False, **self._kw)
        self._meter.calls += len(rows)
        self._meter.calls_by_purpose[purpose] += len(rows)
        return np.asarray(out.outputs, dtype=np.int8)


class LotusHelperScores:
    """Helper-LM scores for one predicate over one DataFrame, computed once with LOTUS."""

    def __init__(self, helper_lm: lotus.models.LM, df: pd.DataFrame, langex: str, positions: np.ndarray, pid: str):
        cols = parse_cols(langex)
        docs = task_instructions.df2multimodal_info(df, cols)
        out = sem_filter(docs, helper_lm, nle2str(langex, cols), logprobs=True, show_progress_bar=False)
        probs = helper_lm.format_logprobs_for_filter_cascade(out.logprobs).positive_probs
        self.pid = pid
        self._score = {int(p): float(s) for p, s in zip(positions, probs)}
        self.calls = len(docs)

    def has(self, pid: str) -> bool:
        return pid == self.pid

    def score(self, pid: str, rows) -> np.ndarray:
        return np.array([self._score[int(r)] for r in rows], dtype=float)


class ViewFilterNode(SemFilterNode):
    """A SemFilterNode executed by certified reuse over a semantic-view catalog."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    catalog: Any = None
    reuse_cfg: Any = None  # CertConfig (default: paper_config()) or ReuseConfig (strata operator)
    embedder: Any = None
    seed: int = 0
    last_stats: dict | None = None

    def __call__(self, df: pd.DataFrame, resolver=None, **context: Any):  # type: ignore[override]
        if self.return_explanations or self.return_raw_outputs or self.examples is not None or self.strategy is not None:
            return super().__call__(df, resolver=resolver, **context)  # unsupported options: native path
        cat: Catalog = self.catalog
        cols = parse_cols(self.user_instruction)
        rids = [row_id(list(vals)) for vals in df[cols].itertuples(index=False, name=None)]
        positions = cat.ensure_rows(rids)
        lm = lotus.settings.lm
        pid = view_key(self.user_instruction, PROMPT_VERSION, lm.model)
        ca = self.cascade_args
        targets = Targets()
        if ca is not None and ca.recall_target is not None:
            targets = Targets(precision=ca.precision_target, recall=ca.recall_target, delta=ca.failure_probability)
        helper = None
        if ca is not None and ca.proxy_model == ProxyModel.HELPER_LM and lotus.settings.helper_lm is not None:
            helper = LotusHelperScores(lotus.settings.helper_lm, df, self.user_instruction, positions, pid)
        oracle = LotusOracle(lm, df, self.user_instruction, positions, pid, default=self.default,
                             system_prompt=self.system_prompt, output_tokens=self.output_tokens)
        texts = {v.predicate_id: v.text for v in cat.views.values()}
        texts[pid] = self.user_instruction
        if isinstance(self.reuse_cfg, ReuseConfig):
            rf = ReuseFilter(oracle, cat, texts, helper=helper, cfg=self.reuse_cfg)
        else:
            cfg = self.reuse_cfg or paper_config()
            emb = self.embedder or Embedder()
            feats = row_features(emb, df, cols, positions, cat.n_rows) if cfg.use_rowemb else None
            rf = CertifiedFilter(oracle, cat, texts, helper=helper, embedder=emb, cfg=cfg, row_features=feats)
        res = rf.run(pid, targets, np.random.default_rng(self.seed), universe=positions)
        keep = res.mask[positions]
        self.last_stats = {**res.stats, "oracle_calls": oracle.meter.calls, "by_purpose": dict(oracle.meter.calls_by_purpose),
                           "helper_calls": helper.calls if helper else 0}
        if self.return_all:
            out = df.copy()
            out[f"{self.suffix}"] = keep
            return out
        return df[keep]


class ViewOptimizer(BaseOptimizer):
    """Rewrite every SemFilterNode into a ViewFilterNode bound to `catalog`."""

    def __init__(self, catalog: Catalog, cfg: CertConfig | ReuseConfig | None = None, seed: int = 0,
                 embedder: Embedder | None = None):
        self.catalog = catalog
        self.cfg = cfg
        self.seed = seed
        self.embedder = embedder or Embedder()  # shared across queries; the model loads lazily

    def optimize(self, nodes, train_data=None):
        out = []
        for n in nodes:
            if type(n) is SemFilterNode:
                n = ViewFilterNode(**{k: getattr(n, k) for k in SemFilterNode.model_fields}, catalog=self.catalog,
                                   reuse_cfg=self.cfg, seed=self.seed, embedder=self.embedder)
            out.append(n)
        return out
