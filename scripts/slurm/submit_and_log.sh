#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${1:-scripts/slurm/preprocess_slake.sbatch}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Missing script: $SCRIPT_PATH" >&2
  exit 1
fi

mkdir -p logs/runs
LOG_FILE="logs/slurm_submissions.log"
RUN_NOTE="logs/runs/preprocess_slake_$(date +%F_%H%M%S).md"

JOB_ID=$(sbatch "$SCRIPT_PATH" | awk '{print $4}')
COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
NOW=$(date '+%Y-%m-%d %H:%M:%S %Z')

{
  echo "$NOW | job=$JOB_ID | script=$SCRIPT_PATH | commit=$COMMIT_HASH"
} >> "$LOG_FILE"

{
  echo "# Slurm Run Note"
  echo
  echo "- Timestamp: $NOW"
  echo "- Job ID: $JOB_ID"
  echo "- Script: $SCRIPT_PATH"
  echo "- Git commit: $COMMIT_HASH"
  echo "- Stdout: slake_preprocess_${JOB_ID}.out"
  echo "- Stderr: slake_preprocess_${JOB_ID}.err"
} > "$RUN_NOTE"

echo "Submitted job $JOB_ID"
echo "Submission log: $LOG_FILE"
echo "Run note: $RUN_NOTE"
