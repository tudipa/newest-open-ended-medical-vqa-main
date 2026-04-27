#!/bin/bash
#SBATCH --job-name=slake_train_dae
#SBATCH --exclude=g16-2gpu-[1-3,6]
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/slake_train_dae_%j.out
#SBATCH --error=logs/slurm/slake_train_dae_%j.err

set -eo pipefail

source ~/.bashrc
set +u
conda activate open1
set -u

cd /home/s225507154/projects/newest-open-ended-medical-vqa-main
mkdir -p logs/slurm

RUN_DIR="${RUN_DIR:-/home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_${SLURM_JOB_ID}}"
DAE_DIR="${DAE_DIR:-$RUN_DIR/dae_stage}"
mkdir -p "$DAE_DIR"

echo "DAE directory: $DAE_DIR"

DATASET="${DATASET:-slake}"
DATASET_PATH="${DATASET_PATH:-/home/s225507154/datasets/}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
SEED="${SEED:-0}"
DAE_NOISE_STD="${DAE_NOISE_STD:-0.05}"
DAE_BOTTLENECK_DIM="${DAE_BOTTLENECK_DIM:-256}"
DAE_PRETRAIN_RECON_LOSS="${DAE_PRETRAIN_RECON_LOSS:-mse}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"

set -- \
  --dataset "$DATASET" \
  --dataset_path "$DATASET_PATH" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --dae_noise_std "$DAE_NOISE_STD" \
  --dae_bottleneck_dim "$DAE_BOTTLENECK_DIM" \
  --dae_recon_loss "$DAE_PRETRAIN_RECON_LOSS" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --out_dir "$DAE_DIR"

CONFIG_FILE="$DAE_DIR/run_config_dae_${SLURM_JOB_ID}.txt"
{
  echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "job_id=${SLURM_JOB_ID}"
  echo "job_name=${SLURM_JOB_NAME:-slake_train_dae}"
  echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-4}"
  echo "mem_per_node=${SLURM_MEM_PER_NODE:-unknown}"
  echo "conda_env=${CONDA_DEFAULT_ENV:-unknown}"
  echo "python=$(python --version 2>&1)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
  echo "run_dir=$RUN_DIR"
  echo "dae_dir=$DAE_DIR"
  echo ""
  echo "dae_train_args=$*"
} > "$CONFIG_FILE"

python train_dae.py "$@"
