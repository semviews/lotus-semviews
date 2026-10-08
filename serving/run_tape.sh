#!/usr/bin/env bash
# Record the full oracle tape, dataset by dataset. Resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-oracle-llama}"; shift || true
for ds in "${@:-goemotions dbpedia reviews}"; do
  for d in $ds; do
    uv run --quiet python -m semviews.adapters.lotus.build_tape --dataset "$d" --model "$MODEL" --chunk "${CHUNK:-500}" --batch "${BATCH:-32}"
  done
done
