#!/usr/bin/env bash
# Usage: bash run.sh

set -eo pipefail

PARTITION="general-gpu"
GRES="gpu:1"
MEM="16G"
TIME="02:00:00"

export PATH=$HOME/miniconda3/bin:$PATH
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate sf-mvp
cd /home/sps22006/semantic-forgetfulness/

srun --pty \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --mem="$MEM" \
  --time="$TIME" \
  bash -c "cd '$(pwd)' && python main.py --load-models"
