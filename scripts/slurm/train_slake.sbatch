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

RUN_DIR="/home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

echo "Checkpoint directory: $RUN_DIR"

# Persist full run metadata for reproducibility
CONFIG_FILE="$RUN_DIR/run_config.txt"
{
  echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "job_id=${SLURM_JOB_ID}"
  echo "job_name=${SLURM_JOB_NAME:-slake_train}"
  echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "gpus=${SLURM_GPUS:-unknown}"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-6}"
  echo "mem_per_node=${SLURM_MEM_PER_NODE:-unknown}"
  echo "conda_env=${CONDA_DEFAULT_ENV:-unknown}"
  echo "python=$(python --version 2>&1)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
  echo "project_dir=/home/s225507154/projects/newest-open-ended-medical-vqa-main"
  echo ""
  echo "train_args=--dataset slake --dataset_path /home/s225507154/datasets/ --model_type gpt2-xl --setting frozen --prefix_length 8 --batch_size 32 --epochs 30 --lr 5e-3 --out_dir $RUN_DIR"
} > "$CONFIG_FILE"

python main.py \
  --dataset slake \
  --dataset_path /home/s225507154/datasets/ \
  --model_type gpt2-xl \
  --setting frozen \
  --prefix_length 8 \
  --batch_size 32 \
  --epochs 30 \
  --lr 5e-3 \
  --out_dir "$RUN_DIR"
