#!/bin/bash
#SBATCH --job-name=slake_train_two_stage
#SBATCH --exclude=g16-2gpu-[1-3,6]
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/slake_train_two_stage_%j.out
#SBATCH --error=logs/slurm/slake_train_two_stage_%j.err

set -eo pipefail

source ~/.bashrc
set +u
conda activate open1
set -u

cd /home/s225507154/projects/newest-open-ended-medical-vqa-main
mkdir -p logs/slurm

RUN_DIR="/home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_${SLURM_JOB_ID}"
DAE_DIR="$RUN_DIR/dae_stage"
mkdir -p "$DAE_DIR"

echo "Two-stage run directory: $RUN_DIR"
echo "DAE pretrain directory: $DAE_DIR"

# Shared defaults (override via sbatch --export)
DATASET="${DATASET:-slake}"
DATASET_PATH="${DATASET_PATH:-/home/s225507154/datasets/}"
MODEL_TYPE="${MODEL_TYPE:-gpt2-xl}"
SETTING="${SETTING:-frozen}"
PREFIX_LENGTH="${PREFIX_LENGTH:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
MLP_DROPOUT="${MLP_DROPOUT:-0.5}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}"
DAE_BOTTLENECK_DIM="${DAE_BOTTLENECK_DIM:-256}"
DAE_NOISE_STD="${DAE_NOISE_STD:-0.05}"
DAE_LOSS_WEIGHT="${DAE_LOSS_WEIGHT:-0.1}"
DAE_RECON_LOSS="${DAE_RECON_LOSS:-mse}"
DAE_FREEZE_ENCODER="${DAE_FREEZE_ENCODER:-1}"
DAE_FREEZE_DECODER="${DAE_FREEZE_DECODER:-0}"

# Stage-1 (DAE pretraining) defaults
DAE_PRETRAIN_EPOCHS="${DAE_PRETRAIN_EPOCHS:-20}"
DAE_PRETRAIN_BATCH_SIZE="${DAE_PRETRAIN_BATCH_SIZE:-256}"
DAE_PRETRAIN_LR="${DAE_PRETRAIN_LR:-1e-3}"
DAE_PRETRAIN_WEIGHT_DECAY="${DAE_PRETRAIN_WEIGHT_DECAY:-1e-5}"
DAE_PRETRAIN_RECON_LOSS="${DAE_PRETRAIN_RECON_LOSS:-mse}"

echo "[stage1] pretraining DAE..."
python train_dae.py \
  --dataset "$DATASET" \
  --dataset_path "$DATASET_PATH" \
  --epochs "$DAE_PRETRAIN_EPOCHS" \
  --batch_size "$DAE_PRETRAIN_BATCH_SIZE" \
  --lr "$DAE_PRETRAIN_LR" \
  --weight_decay "$DAE_PRETRAIN_WEIGHT_DECAY" \
  --seed "${SEED:-0}" \
  --dae_noise_std "$DAE_NOISE_STD" \
  --dae_bottleneck_dim "$DAE_BOTTLENECK_DIM" \
  --dae_recon_loss "$DAE_PRETRAIN_RECON_LOSS" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --out_dir "$DAE_DIR"

DAE_CKPT="$DAE_DIR/dae_best.pt"
if [ ! -f "$DAE_CKPT" ]; then
  echo "ERROR: Stage-1 failed, missing checkpoint: $DAE_CKPT" >&2
  exit 2
fi
echo "[stage1] DAE checkpoint ready: $DAE_CKPT"

echo "[stage2] training VQA with pretrained DAE..."
set -- \
  --dataset "$DATASET" \
  --dataset_path "$DATASET_PATH" \
  --model_type "$MODEL_TYPE" \
  --setting "$SETTING" \
  --prefix_length "$PREFIX_LENGTH" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --mlp_dropout "$MLP_DROPOUT" \
  --dae_noise_std "$DAE_NOISE_STD" \
  --dae_bottleneck_dim "$DAE_BOTTLENECK_DIM" \
  --dae_loss_weight "$DAE_LOSS_WEIGHT" \
  --dae_recon_loss "$DAE_RECON_LOSS" \
  --dae_checkpoint "$DAE_CKPT" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --max_test_samples "$MAX_TEST_SAMPLES" \
  --out_dir "$RUN_DIR" \
  --use_dae

if [ "$DAE_FREEZE_ENCODER" = "1" ]; then
  set -- "$@" --dae_freeze_encoder
fi
if [ "$DAE_FREEZE_DECODER" = "1" ]; then
  set -- "$@" --dae_freeze_decoder
fi

python main.py "$@"
