# Medical VQA Runbook

This runbook explains how to run preprocessing, training, and evaluation for this repository.

## 1. Pipeline Overview

The code follows this flow:

1. Preprocess raw dataset files into `.pkl` files with CLIP image embeddings.
2. Load preprocessed data with `medvqaDataset`.
3. Train `VQAmedModel` (or ablation variant).
4. Evaluate with beam generation and report BLEU/BERTScore/F1/Accuracy.

Main files:

- `main.py`: training/eval entry point
- `train.py`: training loop + validation + checkpoint save
- `predict.py`: test-time generation + metrics
- `models.py`: model and tuning strategy setup
- `data_preprocessing/preprocess_vqa_datasets.py`: preprocessing scripts
- `data_preprocessing/dataloader.py`: dataset class

## 2. Environment Setup

Recommended:

- Python 3.9 or 3.10
- CUDA-enabled GPU
- PyTorch with matching CUDA build

Install core dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers peft accelerate evaluate nltk scikit-learn pandas tqdm pillow scikit-image
pip install git+https://github.com/openai/CLIP.git
```

Optional first-time NLTK setup:

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 3. Expected Data Layout

By default, code expects datasets under:

```text
../vqa_datasets/
```

Expected subfolders (as used in preprocessing code):

- `../vqa_datasets/ovqa/`
- `../vqa_datasets/slake/Slake1.0/`
- `../vqa_datasets/pathvqa/pathVQAprocessed/`

After preprocessing, each dataset should contain split files:

- `train.pkl`
- `val.pkl` (if available)
- `test.pkl`

## 4. Preprocessing

Run preprocessing script (edit it first; see Known Issues section):

```bash
python data_preprocessing/preprocess_vqa_datasets.py
```

Goal: create pickle files containing:

- `img_prefix` (CLIP image features)
- `img_ids`
- `questions`
- `answers`
- `img_path`
- `class_ids`, `class_names`
- `max_seqs_len`

## 5. Training

Example:

```bash
python main.py --dataset ovqa --model_type gpt2-xl --setting lora --prefix_length 8 --batch_size 8 --epochs 10 --lr 1e-4 --dataset_path ../vqa_datasets/
```

Common arguments:

- `--dataset`: `pathvqa | ovqa | slake`
- `--model_type`: `gpt2-xl | microsoft/biogpt | stanford-crfm/BioMedLM`
- `--setting`: `lora | frozen | prefixtuning | p_tuning | prompttuning | unfrozen`
- `--prefix_length`: visual prefix length
- `--dataset_path`: root dataset dir
- `--out_dir`: checkpoint output directory

## 6. Evaluation

Evaluation mode example:

```bash
python main.py --eval --dataset ovqa --model_type gpt2-xl --setting lora --prefix_length 8 --dataset_path ../vqa_datasets/
```

Expected checkpoint path:

- `../checkpoints/<run_suffix>/open_ended_latest.pt`

Reported metrics:

- BLEU
- BERTScore
- F1
- Accuracy (overall, yes/no, open-ended)

## 7. Checkpoints and Outputs

Training saves:

- `open_ended_latest.pt` (best validation loss model)

Default output root:

- `../checkpoints/`

## 8. Known Issues (Important)

Before first run, fix these code issues:

1. `main.py` uses `args.data_partition` in run suffix, but no such argument exists.
2. `main.py` imports dataloader from `data_loaders.dataloader`, but repository folder is `data_preprocessing/dataloader.py`.
3. `predict.py` expects `model.model_type`, while `VQAmedModel` defines `self.gpttype`.
4. `preprocess_vqa_datasets.py` calls `update_classes()` without required arguments.
5. In `preprocess_pathvqa`, dictionary indices used while converting `img_dict` appear inconsistent (`[3]`/`[4]`).

Recommendation: patch these issues first, then run preprocessing/training.

## 9. Reproducibility

`main.py` sets random seeds for:

- `torch`
- `numpy`
- `random`

Use a fixed `--seed` for repeatable runs:

```bash
python main.py --seed 0 ...
```

## 10. Quick Start Checklist

1. Install dependencies.
2. Verify dataset directories and raw files.
3. Patch Known Issues.
4. Run preprocessing and check generated `.pkl`.
5. Start a short training run (`1-2` epochs).
6. Run `--eval` and verify metric outputs.
