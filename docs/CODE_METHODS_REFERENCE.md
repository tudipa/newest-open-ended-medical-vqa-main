# Code Methods Reference (Curated)

This guide explains what each function/method does and where it is used in the current project pipeline.

## `main.py`

### `set_random_seeds(random_seed=0)`
- Function: Sets deterministic/random seeds for PyTorch, NumPy, and Python `random`.
- Used in: `parse_argument()` before any dataset/model creation.

### `parse_argument()`
- Function: Defines and parses CLI arguments (model/data/train/eval options).
- Used in: script entrypoint (`if __name__ == "__main__"`).

### `maybe_subset(dataset, max_samples)`
- Function: Truncates a dataset to first `max_samples` items for smoke tests.
- Used in: main flow to optionally limit train/val/test datasets.

## `train.py`

### `pytorch_model_run(train_loader, valid_loader, model_obj, args)`
- Function: Full training loop with Accelerate, optimizer/scheduler, train+val loss, checkpoint save, and epoch CSV logging.
- Used in: `main.py` when `--eval` is not set.

## `predict.py`

### `_unwrap_subset(dataset)`
- Function: Resolves a `torch.utils.data.Subset` back to base dataset + index mapping.
- Used in: `eval_gpt_open_ended()` to recover raw sample indexing.

### `_dataset_raw_index(local_idx, indices)`
- Function: Converts subset-local index to raw base-dataset index.
- Used in: `eval_gpt_open_ended()` loop.

### `_get_gold_answer(base_dataset, raw_idx)`
- Function: Reads gold answer from dataset, supporting both `answers_raw` and `answers` field names.
- Used in: `eval_gpt_open_ended()`.

### `normalize_answer(s)`
- Function: Strict text normalization for exact-match accuracy (lowercase, remove special tokens/punctuation, collapse spaces).
- Used in: strict `Accuracy`, `Accuracy YN`, `Accuracy OE` in eval.

### `normalize_text(s)`
- Function: Soft-normalization for robust comparison (strips prefixes like `answer:`, punctuation, special tokens).
- Used in: soft matching, mismatch debugging, BLEU tokenization.

### `yes_no_value(s)`
- Function: Converts normalized answer into `yes`/`no` bucket or `None`.
- Used in: category split metrics (`yn_exact`).

### `soft_match(pred, gold)`
- Function: Soft comparison (`exact` OR `substring containment`) after normalization.
- Used in: `evaluate_predictions()` and `debug_mismatches()`.

### `evaluate_predictions(pred_texts, gold_texts)`
- Function: Computes normalized metrics: `exact_match`, `soft_match`, `yn_exact`, `oe_soft`.
- Used in: `eval_gpt_open_ended()` summary reporting.

### `debug_mismatches(pred_texts, gold_texts, k=20)`
- Function: Prints first `k` soft-match failures with raw+normalized text for diagnosis.
- Used in: `eval_gpt_open_ended()`.

### `eval_gpt_open_ended(model, dataset, args, print_vis_token_meaning=True)`
- Function: End-to-end evaluation loop: generation, BLEU/F1/BERTScore, strict accuracy metrics, normalized metrics, mismatch debug.
- Used in: `main.py` when `--eval` is set.

### `print_nearest_text_token(vis_token, model)`
- Function: Debug helper that maps a visual embedding vector to nearest text token embedding.
- Used in: optional debug branch in `eval_gpt_open_ended()`.

### `compute_f1(gold_toks, pred_toks)`
- Function: Token-level F1 between two token-id sequences.
- Used in: `eval_gpt_open_ended()`.

## `models.py`

### `VQAmedModel.__init__(...)`
- Function: Builds base VQA model (LLM + PEFT/frozen strategy + visual-prefix mapper).
- Used in: `main.py` when `--ablation none`.

### `VQAmedModel.forward(prefix, labels, tokens, mask, q_len, batch_size)`
- Function: Training forward pass; projects image features and inserts visual prefix embeddings into text embeddings.
- Used in: `train.py` loop (`model(...)`).

### `VQAmedModel.generate(prefix, labels, tokens, mask, q_len)`
- Function: Builds embedding sequence for decoding/generation time.
- Used in: `predict.py` during eval generation.

### `VQAmedModel_abl.__init__(...)`
- Function: Builds ablation variant model, including learnable replacement visual tokens for specific ablation modes.
- Used in: `main.py` when `--ablation` is not `none`.

### `VQAmedModel_abl.forward(prefix, labels, tokens, mask, q_len, batch_size, abl)`
- Function: Ablation-aware training forward pass (`replace_visual`, `remove_question`, `swap`).
- Used in: ablation training path.

### `VQAmedModel_abl.generate(prefix, labels, tokens, mask, q_len, abl)`
- Function: Ablation-aware generation embedding builder.
- Used in: ablation eval path.

## `prefix_mappers.py`

### `MLP.__init__(sizes, bias=True, act=nn.Tanh)`
- Function: Builds multilayer projection network for CLIP image feature -> language prefix embeddings.
- Used in: `VQAmedModel` / `VQAmedModel_abl` when `mapping_type=MLP`.

### `MLP.forward(x)`
- Function: Runs MLP projection.
- Used in: model forward/generate via `self.clip_project(...)`.

### `MlpTransformer.__init__(...)`
- Function: Initializes feed-forward block used inside transformer layers.
- Used in: `TransformerLayer`.

### `MlpTransformer.forward(x)`
- Function: Applies two-layer feed-forward transform with activation/dropout.
- Used in: `TransformerLayer`.

### `MultiHeadAttention.__init__(...)`
- Function: Initializes multi-head attention projections and output projection.
- Used in: `TransformerLayer`.

### `MultiHeadAttention.forward(x, y=None, mask=None)`
- Function: Computes attention outputs and attention maps.
- Used in: `TransformerLayer.forward*`.

### `TransformerLayer.__init__(...)`
- Function: Builds one transformer block (norm + attention + MLP with residuals).
- Used in: `Transformer`.

### `TransformerLayer.forward_with_attention(x, y=None, mask=None)`
- Function: Forward pass that also returns attention maps.
- Used in: `Transformer.forward_with_attention`.

### `TransformerLayer.forward(x, y=None, mask=None)`
- Function: Standard transformer block forward.
- Used in: `Transformer.forward`.

### `Transformer.__init__(...)`
- Function: Stacks transformer layers for self-attention or encoder-decoder pattern.
- Used in: `TransformerMapper`.

### `Transformer.forward_with_attention(x, y=None, mask=None)`
- Function: Runs full stack and collects all attention maps.
- Used in: debugging/analysis paths.

### `Transformer.forward(x, y=None, mask=None)`
- Function: Runs full stack forward.
- Used in: `TransformerMapper.forward`.

### `TransformerMapper.__init__(dim_clip, dim_embedding, prefix_length, clip_length, num_layers=8)`
- Function: Builds transformer-based visual prefix mapper with learnable prefix constants.
- Used in: `VQAmedModel` / `VQAmedModel_abl` when `mapping_type=Transformer`.

### `TransformerMapper.forward(x)`
- Function: Projects CLIP embedding, concatenates learnable prefix slots, and returns mapped prefix embeddings.
- Used in: model forward/generate via `self.clip_project(...)`.

## `utils.py`

### `treebank_tokenize(s)`
- Function: Tokenizes text with NLTK Treebank tokenizer.
- Used in: utility/NLP experiments (not in active train/eval path).

### `generate_beam(model, tokenizer, beam_size=5, generated=None, entry_length=65, temperature=1.0, stop_token='<|endoftext|>')`
- Function: Beam-search decoder over language model embeddings.
- Used in: `predict.py` for open-ended answer generation.

## `data_preprocessing/dataloader.py`

### `medvqaDataset.__init__(...)`
- Function: Loads preprocessed `.pkl`, tokenizer, split mode, and sequence limits.
- Used in: `main.py` for train/val/test datasets.

### `medvqaDataset.__len__()`
- Function: Returns number of QA samples.
- Used in: PyTorch DataLoader internals.

### `medvqaDataset.pad_sequences(index)`
- Function: Builds token sequence template (`question`, `context`, visual placeholder, `answer`) and attention mask for train/test modes.
- Used in: `__getitem__()`.

### `medvqaDataset.make_padding(max_len, tokens, question=False, leftover_tokens=0)`
- Function: Pads/truncates question or answer segment and builds matching masks.
- Used in: `pad_sequences()` train mode.

### `medvqaDataset.make_padding_test_setting(max_len, tokens, do_padding=False)`
- Function: Test-time question padding/truncation helper.
- Used in: `pad_sequences()` test mode.

### `medvqaDataset.__getitem__(index)`
- Function: Returns `(image_prefix, class_id, tokens, mask, q_len)` sample tuple.
- Used in: train/eval DataLoaders.

## `data_preprocessing/dataloader_ablations.py`

### `medvqaDataset.__init__(...)`
- Function: Loads preprocessed data for ablation experiments with additional mode flags.
- Used in: ablation experiments.

### `medvqaDataset.__len__()`
- Function: Returns number of ablation samples.
- Used in: PyTorch DataLoader internals.

### `medvqaDataset.pad_sequences(index)`
- Function: Builds ablation-specific token layouts and masks (`replace_visual`, `remove_visual`, `remove_question`, `swap`).
- Used in: `__getitem__()`.

### `medvqaDataset.make_padding(max_len, tokens, question=False, leftover_tokens=0)`
- Function: Pads/truncates segments for ablation training sequence assembly.
- Used in: `pad_sequences()`.

### `medvqaDataset.make_padding_only_question(max_len, tokens, do_padding=False)`
- Function: Question-only padding helper for test/inference ablation mode.
- Used in: `pad_sequences()` test branch.

### `medvqaDataset.__getitem__(index)`
- Function: Returns ablation tuple `(prefix, class_id, tokens, mask, q_len, q_len2)`.
- Used in: ablation train/eval loops.

## `data_preprocessing/preprocess_vqa_datasets.py`

### `isEglish(s)`
- Function: ASCII check used as quick English filter.
- Used in: OVQA/SLaKE preprocessing filters.

### `punc(s)`
- Function: Strips punctuation and lowercases text.
- Used in: OVQA question/answer normalization before serialization.

### `update_classes(pkl_train, pkl_val, pkl_test)`
- Function: Builds consistent class-id mapping and computes sequence length stats (`max_seqs_len`) across splits.
- Used in: post-processing step after dataset split preprocessing.

### `preprocess_pathvqa(split, out_path)`
- Function: Loads PathVQA split, extracts CLIP image embeddings, stores QA/image metadata to pickle.
- Used in: CLI `--dataset pathvqa|all`.

### `preprocess_ovqa(split, out_path)`
- Function: Loads OVQA split, filters/normalizes text, extracts CLIP embeddings, stores pickle.
- Used in: CLI `--dataset ovqa|all`.

### `preprocess_slake(split, out_path, slake_root=..., device_str='cuda:0')`
- Function: Loads SLaKE JSON, filters English QA pairs, extracts CLIP embeddings, stores pickle.
- Used in: CLI `--dataset slake|all`.

### `parse_args()`
- Function: CLI parser for preprocessing script (`dataset`, `slake_root`, `device`).
- Used in: preprocessing script entrypoint.
