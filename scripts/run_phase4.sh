#!/usr/bin/env bash
# Phase 4 on the instance (protocol v1.1 + v1.2 registration). Pre-approved; cap 18 instance-hours.
# Order: LLaDA 13-cell test grid on the expanded test set, Qwen3-8B C0 on the expanded test set,
# then the noeos ablation on the original test items. Resumable: rerunning skips finished rows.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[phase4 $(date -u +%H:%M:%S)] $*"; }

. .venv/bin/activate
log "LLaDA test grid (13 cells, expanded test set)"
python -m dd.run --config configs/full.yaml || { log "LLaDA test FAILED"; exit 3; }
deactivate

log "Qwen3-8B C0 test baseline (.venv-qwen)"
. .venv-qwen/bin/activate
python -m dd.run --config configs/full_qwen.yaml || { log "Qwen test FAILED"; exit 4; }
deactivate

. .venv/bin/activate
log "noeos ablation (S=128, suppress ON, original test items)"
python -m dd.run --config configs/full_noeos.yaml || { log "noeos ablation FAILED"; exit 5; }
log "Phase 4 done"
