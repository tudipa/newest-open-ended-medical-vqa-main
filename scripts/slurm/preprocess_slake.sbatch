#!/bin/bash
#SBATCH --job-name=slake_preprocess
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slake_preprocess_%j.out
#SBATCH --error=slake_preprocess_%j.err

set -euo pipefail

source ~/.bashrc
# conda activate <your_env>

cd /home/s225507154/projects/newest-open-ended-medical-vqa-main

python data_preprocessing/preprocess_vqa_datasets.py \
  --dataset slake \
  --slake_root /home/s225507154/datasets/slake \
  --device cuda:0
