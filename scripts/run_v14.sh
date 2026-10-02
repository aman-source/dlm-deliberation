#!/usr/bin/env bash
# v1.4 exploratory GPU: Qwen2.5-7B-Instruct (requested) and Qwen2.5-7B base (Dream's init) C0 baselines.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[v14 $(date -u +%H:%M:%S)] $*"; }
for m in qwen2.5-7b-instruct qwen2.5-7b-base; do
  log "C0 baseline $m"
  .venv/bin/python -m dd.causal_baseline --model "$m" || log "$m FAILED"
done
log "v14 done"
