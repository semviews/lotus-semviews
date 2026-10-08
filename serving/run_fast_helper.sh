#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for d in goemotions pubmed dbpedia reviews; do
  uv run --quiet python -m semviews.adapters.lotus.fast_helper --dataset $d --threads 24
done
