#!/usr/bin/env bash
# Training script for plasma arc detection model.
#
# Usage:
#   bash scripts/train.sh                  # Default config
#   bash scripts/train.sh training.epochs=50 training.batch_size=128

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "============================================"
echo "  Plasma Arc Detection - Training"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Python:       $(python --version 2>&1)"
echo "PyTorch:      $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "Device:       $(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo 'unknown')"
echo "============================================"

python -m src.training.trainer \
    --config-path=../configs \
    --config-name=train \
    "$@"
