#!/bin/bash
#SBATCH --job-name=slake_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g16-2gpu-[1-3,6]
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/slake_eval_%j.out
#SBATCH --error=logs/slurm/slake_eval_%j.err

set -eo pipefail

source ~/.bashrc
set +u
conda activate open1
set -u

cd /home/s225507154/projects/newest-open-ended-medical-vqa-main
mkdir -p logs/slurm

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%a %b %d %I:%M:%S %p %Z %Y')
echo "=== Job started: $START_HUMAN ==="
echo "=== Host: $(hostname) ==="

DATASET="${DATASET:-slake}"
DATASET_PATH="${DATASET_PATH:-/home/s225507154/datasets/}"
MODEL_TYPE="${MODEL_TYPE:-gpt2-xl}"
SETTING="${SETTING:-frozen}"
PREFIX_LENGTH="${PREFIX_LENGTH:-8}"
USE_DAE="${USE_DAE:-0}"
DAE_NOISE_STD="${DAE_NOISE_STD:-0.05}"
DAE_BOTTLENECK_DIM="${DAE_BOTTLENECK_DIM:-256}"
DAE_LOSS_WEIGHT="${DAE_LOSS_WEIGHT:-0.1}"
DAE_RECON_LOSS="${DAE_RECON_LOSS:-mse}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-8}"
OUT_DIR="${OUT_DIR:-}"

if [ -z "$OUT_DIR" ]; then
  echo "ERROR: OUT_DIR is required (directory containing open_ended_latest.pt)" >&2
  echo "Example: sbatch --export=ALL,OUT_DIR=/home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_7912 scripts/slurm/eval_slake.sbatch" >&2
  exit 2
fi

set -- \
  --dataset "$DATASET" \
  --dataset_path "$DATASET_PATH" \
  --model_type "$MODEL_TYPE" \
  --setting "$SETTING" \
  --prefix_length "$PREFIX_LENGTH" \
  --dae_noise_std "$DAE_NOISE_STD" \
  --dae_bottleneck_dim "$DAE_BOTTLENECK_DIM" \
  --dae_loss_weight "$DAE_LOSS_WEIGHT" \
  --dae_recon_loss "$DAE_RECON_LOSS" \
  --out_dir "$OUT_DIR" \
  --max_test_samples "$MAX_TEST_SAMPLES" \
  --eval \
  --verbose

if [ "$USE_DAE" = "1" ]; then
  set -- "$@" --use_dae
fi

echo "Evaluation args: $*"

set +e
python main.py "$@"
RC=$?
set -e

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%a %b %d %I:%M:%S %p %Z %Y')
ELAPSED=$((END_EPOCH - START_EPOCH))
ELAPSED_M=$((ELAPSED / 60))
ELAPSED_S=$((ELAPSED % 60))
ELAPSED_H=$((ELAPSED / 3600))

echo "=== Job finished: $END_HUMAN ==="
echo "=== Elapsed: ${ELAPSED}s (${ELAPSED_H}h ${ELAPSED_M}m ${ELAPSED_S}s) ==="

exit $RC
