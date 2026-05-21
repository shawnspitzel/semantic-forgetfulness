#!/usr/bin/env bash
# Usage: bash pretrain.sh
#SBATCH --job-name=pretrain
#SBATCH --output=observability/logs/pretrain.out
#SBATCH --error=observability/logs/pretrain.err
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -n 1 

export PATH=$HOME/miniconda3/bin:$PATH
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate sf-mvp
cd /home/sps22006/semantic-forgetfulness/
export WANDB_API_KEY="c06a2700c1c0f937aaa9a1279556b44558366c4c"
python -m spacy download en_core_web_sm

PYTHONPATH=src python -m training.pretrain --data-path data/train.txt --steps 10000 --device cuda

