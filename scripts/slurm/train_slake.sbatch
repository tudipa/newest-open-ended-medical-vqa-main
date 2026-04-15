#!/bin/bash
#SBATCH --job-name=slake_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g16-2gpu-1
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

# Defaults (override via sbatch --export)
DATASET="${DATASET:-slake}"
DATASET_PATH="${DATASET_PATH:-/home/s225507154/datasets/}"
MODEL_TYPE="${MODEL_TYPE:-gpt2-xl}"
SETTING="${SETTING:-frozen}"
PREFIX_LENGTH="${PREFIX_LENGTH:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-3}"

set -- \
  --dataset "$DATASET" \
  --dataset_path "$DATASET_PATH" \
  --model_type "$MODEL_TYPE" \
  --setting "$SETTING" \
  --prefix_length "$PREFIX_LENGTH" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --out_dir "$RUN_DIR"

# Persist full run metadata for reproducibility
CONFIG_FILE="$RUN_DIR/run_config.txt"
START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
{
  echo "timestamp=$START_HUMAN"
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
  echo "train_args=$*"
} > "$CONFIG_FILE"

# GPU details from PyTorch (portable fallback when nvidia-smi is unavailable)
python - <<'PY' >> "$CONFIG_FILE"
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu_name={p.name}")
    print(f"gpu_total_memory_gb={p.total_memory/1024**3:.2f}")
    print(f"gpu_compute_capability={p.major}.{p.minor}")
    print(f"gpu_device_count={torch.cuda.device_count()}")
else:
    print("gpu_name=no_cuda")
    print("gpu_total_memory_gb=0")
    print("gpu_compute_capability=none")
    print("gpu_device_count=0")
PY

python main.py "$@"

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SEC=$((END_EPOCH - START_EPOCH))
ELAPSED_H=$((ELAPSED_SEC / 3600))
ELAPSED_M=$(((ELAPSED_SEC % 3600) / 60))
ELAPSED_S=$((ELAPSED_SEC % 60))

{
  echo ""
  echo "run_start=$START_HUMAN"
  echo "run_end=$END_HUMAN"
  echo "elapsed_seconds=$ELAPSED_SEC"
  echo "elapsed_hms=${ELAPSED_H}h ${ELAPSED_M}m ${ELAPSED_S}s"
  echo "elapsed_minutes=$(awk \"BEGIN {printf \\\"%.2f\\\", $ELAPSED_SEC/60}\")"
  echo "elapsed_hours=$(awk \"BEGIN {printf \\\"%.2f\\\", $ELAPSED_SEC/3600}\")"
} >> "$CONFIG_FILE"