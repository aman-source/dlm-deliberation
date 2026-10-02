#!/usr/bin/env bash
# Bootstrap the deliberation-dial harness on EC2.
# Target: g6.2xlarge (1x NVIDIA L4 24GB), us-east-1, Deep Learning AMI (PyTorch, Ubuntu), 100GB gp3.
#
# Run ON the instance, as the default user:   bash bootstrap.sh
# It does not launch, stop, or change any AWS resource, IAM setting, or other app in the account.
# It does not run any experiment. It stops after checks and prints the Phase 1 command.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/aman-source/dlm-deliberation.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/dlm-deliberation}"
MODEL_ID="GSAI-ML/LLaDA-8B-Instruct"
MODEL_REV="08b83a6feb34df1a6011b80c3c00c7563e963b07"   # pinned; configs/*.yaml use the same revision

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

log "1/7 GPU visible to the driver"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv

log "2/7 clone or update $REPO_URL ($BRANCH) into $WORKDIR"
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" fetch origin "$BRANCH"
  git -C "$WORKDIR" checkout "$BRANCH"
  git -C "$WORKDIR" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
log "repo at commit $(git rev-parse --short HEAD)"

log "3/7 venv on top of the AMI's CUDA PyTorch"
BASE_PY="python3"
for candidate in /opt/pytorch/bin/python3 /opt/conda/envs/pytorch/bin/python3; do
  if [ -x "$candidate" ]; then BASE_PY="$candidate"; break; fi
done
log "base interpreter: $BASE_PY"
# The DLAMI's /opt/pytorch is itself a venv without system site-packages, so --system-site-packages
# would NOT expose its torch. Link its site-packages with a .pth file instead. The link is appended
# after the venv's own site-packages, so packages pinned in .venv (transformers 4.46.3) win.
"$BASE_PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
BASE_SITE="$("$BASE_PY" -c 'import site; print(site.getsitepackages()[0])')"
VENV_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
if [ "$BASE_SITE" != "$VENV_SITE" ]; then
  echo "$BASE_SITE" > "$VENV_SITE/ami_torch.pth"
  log "linked AMI site-packages: $BASE_SITE"
fi
python -m pip install --upgrade pip
# requirements.txt has an unpinned torch>=2.4. Pin pip to the AMI's CUDA torch so no dependency can
# swap it for a different (possibly CPU-only) build, and record the exact version used.
AMI_TORCH="$(python -c 'import torch; print(torch.__version__)')"
log "AMI torch ${AMI_TORCH}; constraining pip to it"
echo "torch==${AMI_TORCH}" > .torch-constraint.txt
pip install -r requirements.txt -c .torch-constraint.txt
python - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), "torch cannot see CUDA; stop and check the AMI / driver"
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| transformers", transformers.__version__)
assert transformers.__version__ == "4.46.3", "transformers must be the pinned 4.46.3"
PY

export HF_HOME="$WORKDIR/.cache/huggingface"   # same default dd/run.py uses
export TOKENIZERS_PARALLELISM=false

log "4/7 download $MODEL_ID @ $MODEL_REV (~16.0 GB of safetensors)"
python - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download("$MODEL_ID", revision="$MODEL_REV")
print("snapshot:", path)
PY

log "5/7 unit tests (no weights, no results/ writes)"
python -m unittest discover -s tests

log "6/7 GPU name and free memory (torch view, before load)"
python - <<'PY'
import torch
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)} | free {free/2**30:.2f} GiB of {total/2**30:.2f} GiB")
PY

log "7/7 load check: device_map='cuda', low_cpu_mem_usage=True, bf16, one short forward"
python - <<PY
import torch
from dd.llada import load_masked_model
bundle, mask_report = load_masked_model(
    "$MODEL_ID", "llada-8b-instruct", torch.device("cuda"), torch.bfloat16, revision="$MODEL_REV"
)
print("mask token ok:", mask_report["matches_expected_string"], "| end tokens:", bundle.end_tokens)
ids = torch.full((1, 32), bundle.pad_id, dtype=torch.long)
ids[0, -1] = bundle.mask_id
logits = bundle.forward(ids, torch.ones_like(ids))   # raises on NaN/Inf
print("forward ok, logits", tuple(logits.shape), logits.dtype)
free, total = torch.cuda.mem_get_info()
print(f"after load: free {free/2**30:.2f} GiB of {total/2**30:.2f} GiB")
PY

log "done. Nothing was run. Phase 1 (smoke) when approved:"
echo "  cd $WORKDIR && . .venv/bin/activate && export HF_HOME=$HF_HOME"
echo "  nohup python -m dd.run --config configs/smoke.yaml > results/smoke.out 2>&1 &"
