#!/usr/bin/env bash
# Reads the test split from /data and writes /result/submission.jsonl plus per-question logs.
set -euo pipefail
DATA_DIR="${1:-/data}"
RESULT_DIR="${2:-/result}"
mkdir -p "$RESULT_DIR/logs"
export DATAMATE_CACHE="${DATAMATE_CACHE:-/tmp/datamate_cache}"
python "$(dirname "$0")/solver/run.py" --data "$DATA_DIR" --out "$RESULT_DIR/submission.jsonl" --logs "$RESULT_DIR/logs"
