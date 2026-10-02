#!/usr/bin/env bash
# Phases 2 and 3 on the instance (protocol v1.1). Pre-approved by the lead; stops before Phase 4.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[phase23 $(date -u +%H:%M:%S)] $*"; }

. .venv/bin/activate
log "Phase 2: anytime validation"
python -m dd.run --config configs/dev.yaml --validate-anytime || { log "Phase 2 FAILED"; exit 2; }
cat results/gate2.json

log "Phase 3: LLaDA dev sweep (13 cells) + dev fits"
python -m dd.run --config configs/dev.yaml || { log "Phase 3 LLaDA FAILED"; exit 3; }
deactivate

log "Phase 3: Qwen3-8B C0 baseline in .venv-qwen"
if [ ! -d .venv-qwen ]; then
  /opt/pytorch/bin/python3 -m venv .venv-qwen
  . .venv-qwen/bin/activate
  echo "/opt/pytorch/lib/python3.12/site-packages" > "$(python -c 'import site; print(site.getsitepackages()[0])')/ami_torch.pth"
  echo "torch==$(python -c 'import torch; print(torch.__version__)')" > .torch-constraint.txt
  pip install -q "transformers==4.56.2" accelerate datasets numpy scipy pandas matplotlib pyyaml tqdm psutil -c .torch-constraint.txt
else
  . .venv-qwen/bin/activate
fi
python -c "import torch, transformers; print('qwen venv: torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
python -m dd.run --config configs/dev_qwen.yaml || { log "Phase 3 Qwen FAILED"; exit 4; }
log "Phase 3 done"
