#!/usr/bin/env bash
# llama.cpp throughput degrades over hours of continuous load; restart it every 40 minutes
# while the helper tape is being recorded. The recorder retries requests across a restart.
cd "$(dirname "$0")"
while pgrep -f run_fast_helper.sh >/dev/null; do
  sleep 2400
  pkill -f llama-server; sleep 3; rm -f helper.pid
  HELPER_PARALLEL=16 ./llm_up.sh >> logs/watchdog.log 2>&1
done
