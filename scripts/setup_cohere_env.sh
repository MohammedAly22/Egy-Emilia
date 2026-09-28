#!/usr/bin/env bash
# One-time setup for the "cohere" ASR backend (CohereLabs/cohere-transcribe-arabic-07-2026).
#
# That model needs transformers>=5.4, but NeMo / QwenCleo need transformers 4.57.x,
# so it gets its own venv. The venv is created from the egy env's Python with
# --system-site-packages: it REUSES egy's torch/CUDA (no second multi-GB torch
# download) and only installs transformers 5.x (+ its hub/tokenizers) on top,
# which shadow egy's copies inside this venv only. The egy env is not modified.
#
#   conda activate egy
#   bash scripts/setup_cohere_env.sh
#
# Then set  asr.backend: "cohere"  in config.yaml.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/.venvs/cohere"

python -c "import torch" 2>/dev/null || {
    echo "torch not found — run this with the egy env active:  conda activate egy"; exit 1; }

echo "==> venv at $VENV (inherits torch from $(python -c 'import sys; print(sys.prefix)'))"
python -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -U "transformers>=5.4.0" sentencepiece protobuf

echo "==> check"
"$VENV/bin/python" - <<'PY'
import torch, transformers
from transformers import CohereAsrForConditionalGeneration  # noqa: F401
print(f"torch {torch.__version__} | CUDA ok: {torch.cuda.is_available()} | transformers {transformers.__version__}")
PY

echo "==> pre-downloading the model (~4 GB)"
"$VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('CohereLabs/cohere-transcribe-arabic-07-2026'))"

echo
echo "done. In config.yaml set:  asr.backend: \"cohere\""
