# Agent Change Log

This file tracks changes made by the coding assistant so project updates are auditable.

## Logging Rules

- Every code or documentation change made by the assistant must be appended here.
- Each entry should include: date, files changed, and a short summary.
- If no file was changed, no entry is added.

## Entries

### 2026-04-15
- Files changed: `docs/RUNBOOK.md`
- Summary: Added an end-to-end runbook covering setup, data layout, preprocessing, training, evaluation, reproducibility, quick-start checklist, and known code issues.

### 2026-04-15
- Files changed: `docs/AGENT_CHANGE_LOG.md`
- Summary: Added persistent assistant change-tracking policy and initial entries.

### 2026-04-15
- Files changed: `.gitignore`
- Summary: Added a repository-level Python/ML `.gitignore` to exclude caches, virtual envs, logs, checkpoints, and dataset artifacts (`*.pkl`, `vqa_datasets`).

### 2026-04-15
- Files changed: `main.py`, `models.py`, `data_preprocessing/preprocess_vqa_datasets.py`
- Summary: Fixed must-do blockers: corrected dataloader import path, removed undefined `args.data_partition` usage, allowed `ablation=none`, aligned model compatibility field (`model_type`) for generation, fixed preprocessing dictionary index bugs, and replaced invalid `update_classes()` calls with explicit train/val/test pickle paths per dataset.
- Validation: Python syntax check passed for core modules using in-memory `compile(...)` (no pycache writes).

### 2026-04-15
- Files changed: `train.py`
- Summary: Updated optimizer import to `torch.optim.AdamW` (instead of deprecated/removed `transformers.AdamW`) while keeping `get_linear_schedule_with_warmup` from `transformers` for compatibility with current package versions.
- Validation: `python main.py --help` now runs successfully.

### 2026-04-15
- Files changed: `data_preprocessing/preprocess_vqa_datasets.py`
- Summary: Adjusted hardcoded SLaKE preprocessing paths to use the real dataset root (`/home/s225507154/datasets/slake`) directly, mapped `val` split to `validation.json`, and added CLI args (`--dataset`, `--slake_root`, `--device`) with SLaKE-only default behavior.
- Validation: Python syntax check passed for `data_preprocessing/preprocess_vqa_datasets.py`.

### 2026-04-15
- Files changed: `scripts/slurm/preprocess_slake.sbatch`, `scripts/slurm/submit_and_log.sh`, `logs/.gitkeep`, `logs/runs/.gitkeep`
- Summary: Added a reusable Slurm preprocessing job script for SLaKE and a submission helper that records job id, commit hash, and run notes under `logs/`.

### 2026-04-15
- Files changed: `scripts/slurm/preprocess_slake.sbatch`, `scripts/slurm/submit_and_log.sh`, `logs/slurm/.gitkeep`
- Summary: Updated Slurm workflow to write stdout/stderr under `logs/slurm/`, added safe conda activation guard (`set +u` around `conda activate`) to avoid `ADDR2LINE` unbound variable failures, and kept scripts encoded as UTF-8 without BOM and LF line endings for Slurm compatibility.### 2026-04-15
- Files changed: `scripts/slurm/train_slake.sbatch`, `docs/AGENT_CHANGE_LOG.md`
- Summary: Added Slurm training job script for SLaKE using `--setting frozen` (no LoRA), with cluster paths and logs under `logs/slurm/`.
### 2026-04-15
- Files changed: `scripts/slurm/train_slake.sbatch`, `docs/AGENT_CHANGE_LOG.md`
- Summary: Updated training Slurm command formatting to one parameter per line and changed `--batch_size` from `4` to `32` to match requested default-scale run.
### 2026-04-15
- Files changed: `train.py`, `scripts/slurm/train_slake.sbatch`, `docs/AGENT_CHANGE_LOG.md`
- Summary: Updated SLaKE training config to `--lr 5e-3` and removed early stopping termination logic so training runs the full configured number of epochs.
- Validation: `train.py` syntax check passed.
### 2026-04-15
- Files changed: `scripts/slurm/train_slake.sbatch`, `docs/AGENT_CHANGE_LOG.md`
- Summary: Updated training Slurm job to save checkpoints under a job-specific directory using `${SLURM_JOB_ID}` and pass that path via `--out_dir`.
### 2026-04-15
- Files changed: `scripts/slurm/train_slake.sbatch`, `docs/AGENT_CHANGE_LOG.md`
- Summary: Added automatic run metadata logging for training jobs to `${RUN_DIR}/run_config.txt`, including Slurm job details, conda env, Python version, git branch/commit, and full `main.py` argument string.
### 2026-04-15
- Files changed: `scripts/slurm/train_slake.sbatch`, `docs/AGENT_CHANGE_LOG.md`
- Summary: Added end-of-job runtime tracking to `run_config.txt`, including start/end timestamps, elapsed seconds, elapsed HMS, elapsed minutes, and elapsed hours.
### 2026-04-15
- Files changed: `train.py`, `docs/AGENT_CHANGE_LOG.md`
- Summary: Added per-epoch runtime logging to `${out_dir}/epoch_log.csv` with epoch number, train/val losses, epoch duration in seconds, minutes, and hours.
- Validation: `train.py` syntax check passed.
### 2026-04-15
- Files changed: `scripts/slurm/preprocess_slake.sbatch`, `scripts/slurm/train_slake.sbatch`, `docs/AGENT_CHANGE_LOG.md`
- Summary: Added Slurm node exclusion directive `#SBATCH --exclude=g16-2gpu-1` to avoid scheduling on the specified GPU node for preprocessing and training jobs.
### 2026-04-15
- Files changed: `scripts/slurm/train_slake.sbatch`, `docs/AGENT_CHANGE_LOG.md`
- Summary: Added GPU metadata logging via PyTorch to `run_config.txt` (`gpu_name`, total memory GB, compute capability, and visible device count), with graceful no-CUDA fallback.
