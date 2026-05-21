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
export WANDB_API_KEY="c06a2700c1c0f937aaa9a1279556b44558366c4c"
python -m spacy download en_core_web_sm

srun --pty \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --mem="$MEM" \
  --time="$TIME" \
  bash -c "cd '$(pwd)' && python main.py --load-models"
