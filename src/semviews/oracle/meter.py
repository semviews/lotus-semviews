# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Cost meter: the only source of cost numbers in experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class CostMeter:
    price_in: float = 0.0  # USD per token
    price_out: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    helper_calls: int = 0
    calls_by_purpose: Counter = field(default_factory=Counter)
    scope_id: int = 0
    _scope_seen: set = field(default_factory=set)

    @property
    def usd(self) -> float:
        return self.tokens_in * self.price_in + self.tokens_out * self.price_out

    @contextmanager
    def scope(self, name: str) -> Iterator[CostMeter]:
        """Open a charging scope: a pair is charged at most once inside one scope.

        Scopes do not nest; entering or leaving one starts a fresh scope.
        """
        self.scope_id += 1
        self._scope_seen = set()
        try:
            yield self
        finally:
            self.scope_id += 1
            self._scope_seen = set()

    def charge(self, key: tuple[str, str], purpose: str, tokens_in: int, tokens_out: int) -> bool:
        """Charge one oracle call unless already charged in this scope. Returns True if charged."""
        if key in self._scope_seen:
            return False
        self._scope_seen.add(key)
        self.calls += 1
        self.tokens_in += int(tokens_in)
        self.tokens_out += int(tokens_out)
        self.calls_by_purpose[purpose] += 1
        return True

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "helper_calls": self.helper_calls,
            "usd": self.usd,
            "by_purpose": dict(self.calls_by_purpose),
        }
