#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
for f in proxy.pid helper.pid; do
  if [ -f "$f" ]; then
    pid="$(cat "$f")"; kill "$pid" 2>/dev/null
    for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    rm -f "$f"
  fi
done
echo "llm-down: stopped"
