#!/usr/bin/env bash
# One-shot setup: virtual environment, dependencies, PixOOD.
# Usage:  bash install.sh   |   bash install.sh --skip-pixood
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_PIXOOD=0
[[ "${1:-}" == "--skip-pixood" ]] && SKIP_PIXOOD=1

echo "=== OoD-ROI-Benchmark Setup ==="

# 1. virtual environment -----------------------------------------------------
if [[ ! -d "$ROOT/.venv" ]]; then
    echo "[1/4] Creating virtual environment (.venv)..."
    python3 -m venv "$ROOT/.venv"
else
    echo "[1/4] .venv already exists - reusing."
fi
PIP="$ROOT/.venv/bin/pip"

# 2. PyTorch with CUDA -------------------------------------------------------
echo "[2/4] Installing PyTorch (CUDA 12.1)..."
"$PIP" install --upgrade pip
"$PIP" install "torch==2.2.2" "torchvision==0.17.2" --index-url https://download.pytorch.org/whl/cu121

# 3. remaining dependencies --------------------------------------------------
echo "[3/4] Installing requirements.txt..."
"$PIP" install -r "$ROOT/requirements.txt"

# 4. PixOOD as sibling repo (compute_score_maps.py expects ../PixOOD) --------
if [[ $SKIP_PIXOOD -eq 0 ]]; then
    PIXOOD="$(dirname "$ROOT")/PixOOD"
    if [[ ! -d "$PIXOOD" ]]; then
        echo "[4/4] Cloning PixOOD to $PIXOOD ..."
        git clone https://github.com/vojirt/PixOOD.git "$PIXOOD"
        echo "      IMPORTANT: download the official PixOOD Cityscapes checkpoints"
        echo "      as described in the PixOOD README and place them in $PIXOOD."
    else
        echo "[4/4] PixOOD already exists: $PIXOOD"
    fi
else
    echo "[4/4] PixOOD skipped (--skip-pixood)."
fi

echo
echo "=== Setup complete ==="
echo "Activate with:  source .venv/bin/activate"
echo
echo "Notes:"
echo " - SegFormer-B2 & DINOv2 weights download automatically on first run."
echo " - RbA is NOT installed locally (Detectron2 + CUDA kernel, see README §2)."
echo "   Download precomputed maps:  python scripts/download_score_maps.py"
echo "   or recompute on Colab:      colab/README.md"
echo " - Datasets: see data/README.md"
