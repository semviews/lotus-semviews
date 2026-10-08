#!/usr/bin/env bash
# Start the local helper model server and the LiteLLM proxy as background processes.
# Secrets are loaded from ../.env into the proxy's environment only; nothing is echoed.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
[ -f litellm.yaml ] || { echo "llm-up: copy serving/litellm.example.yaml to serving/litellm.yaml and set your endpoints" >&2; exit 1; }
HELPER_GGUF="${HELPER_GGUF:-$(grep -E '^helper_gguf:' ../configs/models.yaml | awk '{print $2}')}"
HELPER_PARALLEL="${HELPER_PARALLEL:-16}"
if ! curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1; then
  nohup llama-server -m "../$HELPER_GGUF" --host 127.0.0.1 --port 8081 \
    --parallel "$HELPER_PARALLEL" -c $((HELPER_PARALLEL * 2048)) --device MTL0 -ngl 99 --alias helper-local \
    > logs/helper.log 2>&1 &
  echo $! > helper.pid
fi
if ! curl -sf http://127.0.0.1:4000/health/liveliness >/dev/null 2>&1; then
  set -a; source ../.env; set +a
  nohup .venv/bin/litellm --config litellm.yaml --host 127.0.0.1 --port 4000 --num_workers 4 \
    > logs/proxy.log 2>&1 &
  echo $! > proxy.pid
fi
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:4000/health/liveliness >/dev/null 2>&1 && \
     curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1; then echo "llm-up: proxy and helper ready"; exit 0; fi
  sleep 1
done
echo "llm-up: timed out; see serving/logs/" >&2; exit 1
