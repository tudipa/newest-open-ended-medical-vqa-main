#!/bin/bash
#SBATCH --job-name=slake_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/slake_train_%j.out
#SBATCH --error=logs/slurm/slake_train_%j.err

set -eo pipefail

source ~/.bashrc
set +u
conda activate open1
set -u

cd /home/s225507154/projects/newest-open-ended-medical-vqa-main
mkdir -p logs/slurm

python main.py \
  --dataset slake \
  --dataset_path /home/s225507154/datasets/ \
  --model_type gpt2-xl \
  --setting frozen \
  --prefix_length 8 \
  --batch_size 32 \
  --epochs 30 \
  --lr 5e-3
