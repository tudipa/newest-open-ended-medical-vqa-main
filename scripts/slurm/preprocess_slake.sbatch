#!/bin/bash
#SBATCH --job-name=slake_preprocess
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/slake_preprocess_%j.out
#SBATCH --error=logs/slurm/slake_preprocess_%j.err

set -eo pipefail

source ~/.bashrc
set +u
conda activate open1
set -u

cd /home/s225507154/projects/newest-open-ended-medical-vqa-main
mkdir -p logs/slurm

python data_preprocessing/preprocess_vqa_datasets.py \
  --dataset slake \
  --slake_root /home/s225507154/datasets/slake \
  --device cuda:0
