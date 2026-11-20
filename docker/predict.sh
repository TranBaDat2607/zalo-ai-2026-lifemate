#!/usr/bin/env bash
# LifeMate inference inside the image: reads the test split mounted at /data and writes /result/submission.jsonl
# (one line per sample) and /result/timing.jsonl. Fully offline; both models are in /code/models.
#   bash /code/predict.sh [DATA_DIR] [RESULT_DIR]
set -euo pipefail
DATA_DIR="${1:-/data}"
RESULT_DIR="${2:-/result}"
# The final agent flags, as validated (see README.md). --lookup 10: prompt-lookup decoding, the same greedy tokens
# up to fp16 near-ties, 37 % faster on validation (8 Nov 2026, log.md).
SOLVE_FLAGS="--denoise --lookup 10"
# The learned chooser over the five readings (lifemate/stack_view.py, 16 Nov 2026), when the package holds one.
if [ -f /code/models/stack_model.json ]; then SOLVE_FLAGS="$SOLVE_FLAGS --stack-model /code/models/stack_model.json"; fi
cd /code
if [ -d /code/models/asr_view ]; then
    # The readings view: the base speech model and its adapters (export_models.py --asr-view-adapters), in the
    # training order asr_view/0, 1, ...; the main reading is their mean, merged at load time.
    VIEW=$(ls -d /code/models/asr_view/* | sort -V | paste -sd+)
    python -m lifemate.solve --data "$DATA_DIR" --out "$RESULT_DIR" --llm /code/models/llm --asr /code/models/asr_base \
        --view-adapters "$VIEW" $SOLVE_FLAGS
else
    # One speech model, or two when the package holds a second reading (models/asr2); the order is the training order.
    ASR_DIRS=/code/models/asr
    if [ -d /code/models/asr2 ]; then ASR_DIRS="$ASR_DIRS,/code/models/asr2"; fi
    python -m lifemate.solve --data "$DATA_DIR" --out "$RESULT_DIR" --llm /code/models/llm --asr "$ASR_DIRS" $SOLVE_FLAGS
fi
# The rules name the output "result/submission.jsonl": also put it there, relative to the working directory.
if [ "$(realpath -m "$RESULT_DIR")" != "/code/result" ]; then
    mkdir -p /code/result
    cp "$RESULT_DIR/submission.jsonl" "$RESULT_DIR/timing.jsonl" /code/result/
fi
