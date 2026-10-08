# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Compare a regenerated claims.yaml with a reference copy, number by number.

    python experiments/check_claims.py outputs/claims.yaml reference/claims.yaml

Prints every claim whose value differs, is missing, or is new, and exits 1 if any differ.
Values compare after the claim's own display format, so a rounding difference that the paper
would not show is not reported.
"""

from __future__ import annotations

import sys

import yaml


def _load(path: str) -> dict:
    return (yaml.safe_load(open(path)) or {}).get("claims") or {}


def _shown(c: dict) -> str:
    return c.get("fmt", "{}").format(c["value"])


def main(new_path: str, ref_path: str) -> int:
    new, ref = _load(new_path), _load(ref_path)
    differ = [k for k in sorted(set(new) & set(ref)) if _shown(new[k]) != _shown(ref[k])]
    missing = sorted(set(ref) - set(new))
    extra = sorted(set(new) - set(ref))
    for k in differ:
        print(f"DIFFER  {k}: {_shown(new[k])} (reference {_shown(ref[k])}; source {ref[k].get('source', '?')})")
    for k in missing:
        print(f"MISSING {k} (source {ref[k].get('source', '?')})")
    for k in extra:
        print(f"NEW     {k}")
    same = len(set(new) & set(ref)) - len(differ)
    print(f"{same} of {len(ref)} reference claims match; {len(differ)} differ, {len(missing)} missing, {len(extra)} new")
    return 1 if differ or missing else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
