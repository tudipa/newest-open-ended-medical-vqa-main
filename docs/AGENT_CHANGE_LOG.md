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
- Validation: Python syntax check passed for `data_preprocessing/preprocess_vqa_datasets.py`.### 2026-04-15
- Files changed: `scripts/slurm/preprocess_slake.sbatch`, `scripts/slurm/submit_and_log.sh`, `logs/.gitkeep`, `logs/runs/.gitkeep`
- Summary: Added a reusable Slurm preprocessing job script for SLaKE and a submission helper that records job id, commit hash, and run notes under `logs/`.
