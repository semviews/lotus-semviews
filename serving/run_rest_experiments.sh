#!/usr/bin/env bash
# Wait for each remaining helper tape, consolidate, then run that dataset's experiments.
cd "$(dirname "$0")/.."
for d in dbpedia reviews; do
  n=$(grep -c "^- id:" configs/predicates/$d.yaml)
  until [ "$(ls data/tapes/parts/${d}__helper 2>/dev/null | wc -l)" -ge "$n" ] && ! pgrep -f "fast_helper --dataset $d" >/dev/null; do sleep 60; done
  uv run --quiet python -m semviews.adapters.lotus.build_tape --dataset $d --model helper --consolidate
  uv run --quiet python experiments/run_all.py --datasets $d --only E2 E9 E4 E5 E3 --workers 15
  echo "DONE $d $(date -u +%T)"
done
