# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""LOTUS contract tests: the extension points we rely on work on the pinned lotus-ai.

A fake LM (subclass of lotus.models.LM, overriding only the network call) answers by
keyword, so these tests run offline.
"""

from __future__ import annotations

import importlib.metadata

import lotus
import pandas as pd
from litellm.types.utils import Choices, Message, ModelResponse
from lotus.ast import LazyFrame
from lotus.models import LM

from semviews.adapters.lotus.optimizer import (
    NoOpOptimizer,
    ViewFilterNode,
    ViewOptimizer,
    paper_config,
)
from semviews.catalog import Catalog
from semviews.operators.reuse_filter import ReuseConfig


class KeywordLM(LM):
    """Answers True iff the keyword for the claim appears in the user message."""

    RULES = {"battery": "battery", "overheat": "hot", "defect": "broke"}

    def __init__(self):
        super().__init__(model="openai/fake", max_batch_size=4)
        self.calls = 0

    def _process_uncached_messages(self, uncached_data, all_kwargs, show_progress_bar, progress_bar_desc):
        out = []
        for msgs, _ in uncached_data:
            user = " ".join(m["content"] for m in msgs if m["role"] == "user").lower()
            claim = user.split("claim:")[-1] if "claim:" in user else user
            key = next((v for k, v in self.RULES.items() if k in claim), None)
            ctx = user.split("claim:")[0]
            ans = "True" if key and key in ctx else "False"
            out.append(ModelResponse(choices=[Choices(message=Message(content=f"Answer: {ans}"), finish_reason="stop")]))
            self.calls += 1
        return out


def _df(n=400):
    texts = []
    for i in range(n):
        t = f"review {i}: "
        if i % 4 == 0:
            t += "the battery gets hot. "
        if i % 4 == 1:
            t += "battery is weak. "
        if i % 7 == 0:
            t += "screen broke. "
        texts.append(t + "fine otherwise")
    return pd.DataFrame({"review": texts})


def test_pinned_version():
    assert importlib.metadata.version("lotus-ai") == "1.2.4"


def test_noop_optimizer_plugs_in():
    lm = KeywordLM()
    lotus.settings.configure(lm=lm, enable_cache=False)
    df = _df(40)
    lf = LazyFrame().sem_filter("{review} complains about the battery")
    opt = NoOpOptimizer()
    ref = lf.execute(df)
    got = lf.optimize([opt]).execute(df)
    assert "SemFilterNode" in opt.seen
    pd.testing.assert_frame_equal(ref.reset_index(drop=True), got.reset_index(drop=True))


def test_view_optimizer_runs_certified_filter():
    """Default path: the paper's operator (CertifiedFilter, paper_config) behind ViewOptimizer."""
    lm = KeywordLM()
    lotus.settings.configure(lm=lm, enable_cache=False)
    df = _df(2000)
    cat = Catalog("reviews", 0)
    opt = ViewOptimizer(cat)
    assert paper_config().active_rounds > 0 and paper_config().without_replacement
    for langex, key in [("{review} complains about the battery", "battery"), ("{review} says the device overheats", "hot")]:
        lm.calls = 0
        node = next(n for n in LazyFrame().sem_filter(langex).optimize([opt])._nodes if isinstance(n, ViewFilterNode))
        q = LazyFrame().sem_filter(langex).optimize([opt])
        got = set(q.execute(df).review)
        truth = set(df[df.review.str.contains(key)].review)
        assert len(got & truth) / len(truth) >= 0.9 and len(got & truth) / max(1, len(got)) >= 0.9
        assert node.reuse_cfg is None  # default configuration is paper_config()
        assert lm.calls < len(df)  # certified regions spare part of the table
    assert len(cat.views) == 2


def test_view_optimizer_rewrites_and_reuses():
    lm = KeywordLM()
    lotus.settings.configure(lm=lm, enable_cache=False)
    df = _df(400)
    cat = Catalog("reviews", 0)
    cfg = ReuseConfig(use_helper=False, pilot_n=30, min_stratum=20)
    q1 = LazyFrame().sem_filter("{review} complains about the battery").optimize([ViewOptimizer(cat, cfg)])
    assert any(isinstance(n, ViewFilterNode) for n in q1._nodes)
    r1 = q1.execute(df)
    truth1 = df[df.review.str.contains("battery")]
    assert set(r1.review) == set(truth1.review)  # first query: no views, everything labeled
    calls_q1 = lm.calls
    lm.calls = 0
    q3 = LazyFrame().sem_filter("{review} says the battery overheats").optimize([ViewOptimizer(cat, cfg)])
    r3 = q3.execute(df)
    truth3 = df[df.review.str.contains("hot")]
    tp = len(set(r3.review) & set(truth3.review))
    assert tp / len(truth3) >= 0.9
    assert lm.calls < calls_q1  # the battery view lets most rows be rejected without the oracle
    assert len(cat.views) == 2
