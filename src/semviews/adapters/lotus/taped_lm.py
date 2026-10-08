# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""LOTUS LM wrappers for the tape: a recording LM and a replaying LM.

Both subclass lotus.models.LM and override only the uncached-call path, so LOTUS's
prompts, parsing, and usage accounting stay native. Nothing in LOTUS is patched.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any

from litellm.types.utils import ModelResponse
from lotus.models import LM


def proxy_lm(alias: str, models_cfg: dict, **kw: Any) -> LM:
    """Build a LOTUS LM that reaches `alias` through the local proxy."""
    return _make(LM, alias, models_cfg, **kw)


def _make(cls, alias: str, models_cfg: dict, **kw: Any):
    samp = dict(models_cfg["sampling"].get(alias, {}))
    max_tokens = samp.pop("max_tokens", 8)
    samp.pop("temperature", None)
    samp.update(kw.pop("extra", {}))
    route = (models_cfg.get("direct_routes") or {}).get(alias) if kw.pop("direct", True) else None
    return cls(
        model=route["model"] if route else f"openai/{alias}",
        api_base=route["api_base"] if route else models_cfg["proxy_base"],
        api_key=route["api_key"] if route else os.environ.get("LITELLM_MASTER_KEY", ""),
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=kw.pop("timeout", float(os.environ.get("LM_TIMEOUT", "180"))),  # a hung request must not stall a batch
        **samp,
        **kw,
    )


class RecordingLM(LM):
    """Records usage, model id, and latency for every uncached call, in call order."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.records: list[dict[str, Any]] = []

    def _process_uncached_messages(self, uncached_data, all_kwargs, show_progress_bar, progress_bar_desc):  # type: ignore[override]
        t0 = time.time()
        resps = super()._process_uncached_messages(uncached_data, all_kwargs, show_progress_bar, progress_bar_desc)
        per_call_ms = 1000 * (time.time() - t0) / max(len(resps), 1)
        for r in resps:
            if isinstance(r, Exception):
                raise r
            u = getattr(r, "usage", None)
            self.records.append(
                {
                    "model_id": str(getattr(r, "model", "")),
                    "tokens_in": int(getattr(u, "prompt_tokens", 0) or 0),
                    "tokens_out": int(getattr(u, "completion_tokens", 0) or 0),
                    "latency_ms": per_call_ms,
                }
            )
        return resps

    @classmethod
    def for_alias(cls, alias: str, models_cfg: dict, **kw: Any) -> RecordingLM:
        return _make(cls, alias, models_cfg, **kw)


class TapedLM(LM):
    """Record-and-replay LM keyed by the exact messages and sampling kwargs.

    In replay mode a miss raises; in record mode a miss goes live through the proxy and
    is appended to the SQLite tape. Counts live and replayed calls separately.
    """

    def __init__(self, *args: Any, tape_path: str, replay_only: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.tape_path = tape_path
        self.replay_only = replay_only
        self.live_calls = 0
        self.replayed_calls = 0
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(tape_path) or ".", exist_ok=True)
        with sqlite3.connect(tape_path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS tape (k TEXT PRIMARY KEY, response TEXT)")

    def _key(self, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
        payload = json.dumps([self.model, messages, {k: kwargs[k] for k in sorted(kwargs)}], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _process_uncached_messages(self, uncached_data, all_kwargs, show_progress_bar, progress_bar_desc):  # type: ignore[override]
        keys = [self._key(m, all_kwargs) for m, _ in uncached_data]
        with sqlite3.connect(self.tape_path) as c:
            found = dict(c.execute(f"SELECT k, response FROM tape WHERE k IN ({','.join('?' * len(keys))})", keys).fetchall()) if keys else {}
        out: list[Any] = [None] * len(keys)
        missing = []
        for i, k in enumerate(keys):
            if k in found:
                out[i] = ModelResponse(**json.loads(found[k]))
            else:
                missing.append(i)
        self.replayed_calls += len(keys) - len(missing)
        if missing:
            if self.replay_only:
                raise KeyError(f"TapedLM replay miss for {len(missing)} calls")
            live = super()._process_uncached_messages([uncached_data[i] for i in missing], all_kwargs, show_progress_bar, progress_bar_desc)
            rows = []
            for i, r in zip(missing, live):
                if isinstance(r, Exception):
                    raise r
                out[i] = r
                rows.append((keys[i], r.model_dump_json()))
            with self._lock, sqlite3.connect(self.tape_path) as c:
                c.executemany("INSERT OR REPLACE INTO tape VALUES (?, ?)", rows)
            self.live_calls += len(missing)
        return out

    @classmethod
    def for_alias(cls, alias: str, models_cfg: dict, tape_path: str, replay_only: bool = False, **kw: Any) -> TapedLM:
        return _make(cls, alias, models_cfg, tape_path=tape_path, replay_only=replay_only, **kw)
