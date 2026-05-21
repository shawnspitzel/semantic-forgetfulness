#!/usr/bin/env bash
#SBATCH --job-name=sf-eval-baseline
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=observability/logs/eval_%j.out
#SBATCH --error=observability/logs/eval_%j.err

set -eo pipefail

export PATH=$HOME/miniconda3/bin:$PATH
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate sf-mvp

cd /home/sps22006/semantic-forgetfulness/
export WANDB_API_KEY="c06a2700c1c0f937aaa9a1279556b44558366c4c"

python eval.py \
  --benchmark all \
  --tasks narrativeqa,qasper,hotpotqa,2wikimqa,gov_report \
  --device cuda
