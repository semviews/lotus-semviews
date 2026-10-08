#!/usr/bin/env bash
# After goemotions and pubmed helper tapes finish, run dbpedia and reviews.
cd "$(dirname "$0")/.."
until [ "$(ls data/tapes/parts/goemotions__helper | wc -l)" -ge 40 ] && [ "$(ls data/tapes/parts/pubmed__helper | wc -l)" -ge 25 ]; do sleep 60; done
for d in dbpedia reviews; do LM_TIMEOUT=30 CHUNK=200 BATCH=8 ./serving/run_tape.sh helper $d > serving/logs/tape_helper_$d.log 2>&1 & done
wait
