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

# Robust conda initialization for non-interactive Slurm shells
if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
  source /opt/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2
  exit 127
fi

conda activate open1 || { echo "ERROR: conda activate failed" >&2; exit 127; }

cd /home/s225507154/projects/newest-open-ended-medical-vqa-main
mkdir -p logs/slurm

DATASET="${DATASET:-slake}"
DATASET_PATH="${DATASET_PATH:-/home/s225507154/datasets/}"
MODEL_TYPE="${MODEL_TYPE:-gpt2-xl}"
SETTING="${SETTING:-frozen}"
PREFIX_LENGTH="${PREFIX_LENGTH:-8}"
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
  --out_dir "$OUT_DIR" \
  --max_test_samples "$MAX_TEST_SAMPLES" \
  --eval \
  --verbose

echo "Evaluation args: $*"
python main.py "$@"
