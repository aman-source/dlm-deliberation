#!/usr/bin/env bash
# v1.3 exploratory GPU runs (pre-approved cap 12 instance-hours for B + C). Resumable.
#   C: Dream-v0-Instruct-7B in a separate checkout (~/dlm-dream): template check, dev fits, test (original items)
#   B: LLaDA C1/C2 S=32 T=32 on the original test items (main checkout)
set -uo pipefail
MAIN="$(cd "$(dirname "$0")/.." && pwd)"
DREAM="$HOME/dlm-dream"
export HF_HOME="$MAIN/.cache/huggingface" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$MAIN/.venv/bin/python"
log() { echo "[v13 $(date -u +%H:%M:%S)] $*"; }

if [ ! -d "$DREAM/.git" ]; then
  log "creating Dream checkout"
  git clone -q "$MAIN" "$DREAM"
  # Dream gets its own frozen files; keep only the shared items/splits.
  rm -f "$DREAM"/results/{template.json,protocol_v1_1.json,dev_fits.json,timing.json,gate2.json,resmoke.json}
  find "$DREAM/results/raw" -maxdepth 1 -name '*.jsonl' -delete
else
  git -C "$DREAM" fetch -q "$MAIN" main && git -C "$DREAM" checkout -q -f FETCH_HEAD -- dd configs scripts tests
fi

cd "$DREAM"
if [ ! -f results/protocol_v1_1.json ]; then
  log "C: Dream template check (resmoke)"
  "$PY" -m dd.run --config configs/dream_resmoke.yaml || { log "Dream resmoke FAILED or gate not met"; DREAM_OK=0; }
fi
DREAM_OK=${DREAM_OK:-1}
if [ "$DREAM_OK" = 1 ] && [ -f results/protocol_v1_1.json ]; then
  log "C: Dream dev (temperatures)"
  "$PY" -m dd.run --config configs/dream_dev.yaml || { log "Dream dev FAILED"; exit 3; }
  log "C: Dream test (original items)"
  "$PY" -m dd.run --config configs/dream_test.yaml || { log "Dream test FAILED"; exit 4; }
else
  log "C skipped: Dream template gate not met (see $DREAM/results/resmoke.json)"
fi

cd "$MAIN"
log "B: LLaDA C1/C2 S=32 T=32 (original test items)"
"$PY" -m dd.run --config configs/v13_t32.yaml || { log "B FAILED"; exit 5; }
log "v13 done"
