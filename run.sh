#!/usr/bin/env bash

set -euo pipefail

PARTITION="gpu"
GRES="gpu:1"
MEM="16G"
TIME="02:00:00"

srun --pty \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --mem="$MEM" \
  --time="$TIME" \
  bash -c "cd '$(pwd)' && python main.py --load-models"
