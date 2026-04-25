import collections
import csv
import json
import os
import random
import re
import string
from contextlib import nullcontext
from datetime import datetime

import torch
from evaluate import load
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm
from transformers import GPT2Tokenizer

from utils import generate_beam


def _unwrap_subset(dataset):
    """Return base dataset and index mapping when dataset is a torch.utils.data.Subset."""
    indices = None
    while hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        current_indices = list(dataset.indices)
        if indices is None:
            indices = current_indices
        else:
            indices = [current_indices[i] for i in indices]
        dataset = dataset.dataset
    return dataset, indices


def _dataset_raw_index(local_idx, indices):
    if indices is None:
        return local_idx
    return indices[local_idx]


def _get_gold_answer(base_dataset, raw_idx):
    if hasattr(base_dataset, 'answers_raw'):
        return base_dataset.answers_raw[raw_idx]
    if hasattr(base_dataset, 'answers'):
        return base_dataset.answers[raw_idx]
    raise AttributeError('Dataset has neither answers_raw nor answers')


def normalize_answer(s: str) -> str:
    if s is None:
        return ''
    s = str(s).lower()
    s = s.replace('<|endoftext|>', ' ')
    s = s.replace('</s>', ' ')
    s = s.replace('<pad>', ' ')
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_text(s: str) -> str:
    if s is None:
        return ''

    s = str(s).lower().strip()
    s = re.sub(r'^(answer\s*:?\s*)', '', s)
    s = re.sub(r'^(ans\s*:?\s*)', '', s)
    s = s.replace('<|endoftext|>', ' ')
    s = s.replace('</s>', ' ')
    s = s.replace('<pad>', ' ')
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def yes_no_value(s: str):
    s = normalize_text(s)
    if s in {'yes', 'no'}:
        return s
    return None


def soft_match(pred: str, gold: str) -> bool:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    return (p in g) or (g in p)


def evaluate_predictions(pred_texts, gold_texts):
    assert len(pred_texts) == len(gold_texts)

    n = len(pred_texts)
    em = 0
    soft = 0

    yn_total = 0
    yn_em = 0

    oe_total = 0
    oe_em = 0

    for pred, gold in zip(pred_texts, gold_texts):
        pred_n = normalize_text(pred)
        gold_n = normalize_text(gold)

        if pred_n == gold_n:
            em += 1
        if soft_match(pred, gold):
            soft += 1

        gold_yn = yes_no_value(gold)
        if gold_yn is not None:
            yn_total += 1
            if yes_no_value(pred) == gold_yn:
                yn_em += 1
        else:
            oe_total += 1
            if soft_match(pred, gold):
                oe_em += 1

    return {
        'N': n,
        'exact_match': em / n if n else 0.0,
        'soft_match': soft / n if n else 0.0,
        'yn_exact': yn_em / yn_total if yn_total else 0.0,
        'oe_soft': oe_em / oe_total if oe_total else 0.0,
    }


def debug_mismatches(pred_texts, gold_texts, k=20):
    shown = 0
    for i, (pred, gold) in enumerate(zip(pred_texts, gold_texts)):
        pred_n = normalize_text(pred)
        gold_n = normalize_text(gold)

        ok_em = pred_n == gold_n
        ok_soft = soft_match(pred, gold)

        if not ok_soft:
            print('=' * 60)
            print(f'idx={i}')
            print(f'PRED raw : {repr(pred)}')
            print(f'GOLD raw : {repr(gold)}')
            print(f'PRED norm: {repr(pred_n)}')
            print(f'GOLD norm: {repr(gold_n)}')
            print(f'EM={ok_em} SOFT={ok_soft}')
            shown += 1
            if shown >= k:
                break


def eval_gpt_open_ended(model, dataset, args, print_vis_token_meaning=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluation device={device}')
    print('BERTScore Config')
    print('bertscore_model_type=bert-base-uncased')
    print('bertscore_lang=en')
    print(f'bertscore_device={device.type}')

    checkpoint_path = getattr(args, 'checkpoint', None)
    if checkpoint_path:
        print('Checkpoint Debug')
        print(f'checkpoint_path={checkpoint_path}')
        if os.path.exists(checkpoint_path):
            ckpt_bytes = os.path.getsize(checkpoint_path)
            ckpt_mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_path)).isoformat(timespec='seconds')
            print(f'checkpoint_size_bytes={ckpt_bytes}')
            print(f'checkpoint_modified={ckpt_mtime}')

    run_config_path = _find_run_config_file(getattr(args, 'out_dir', ''))
    print(f'linked_run_config={run_config_path if run_config_path else "not_found"}')

    pred_texts = []
    gold_texts = []
    smooth = SmoothingFunction().method1

    model.eval()
    model = model.to(device)

    base_dataset, subset_indices = _unwrap_subset(dataset)

    bert_score = load('bertscore')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    tokenizer_legacy = GPT2Tokenizer.from_pretrained('gpt2')

    # Original metrics (first-commit style)
    orig_bleu_avg1 = 0.0
    orig_bert_avg3 = 0.0
    orig_f1_avg = 0.0
    orig_acc = 0.0
    orig_acc_oe = 0.0
    orig_acc_yn = 0.0
    orig_c_oe = 1e-9
    orig_c_yn = 1e-9

    # Normalized/soft diagnostics
    bleu_avg1 = 0.0
    bert_avg3 = 0.0
    f1_avg = 0.0
    acc = 0.0
    acc_oe = 0.0
    acc_yn = 0.0
    c_oe = 1e-9
    c_yn = 1e-9
    per_sample_rows = []
    bert_f1_scores = []

    with tqdm(total=len(dataset)) as epoch_pbar:
        epoch_pbar.set_description('Testing')
        for item in range(len(dataset)):
            raw_idx = _dataset_raw_index(item, subset_indices)
            prefix, labels, tokens, mask, q_len = dataset[item]
            prefix = prefix.type(torch.float32).to(device)
            tokens = tokens.type(torch.long).to(device)
            mask = mask.to(device)

            amp_ctx = (
                torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                if device.type == 'cuda'
                else nullcontext()
            )

            with amp_ctx:
                with torch.no_grad():
                    embed = model.generate(prefix, labels, tokens, mask, q_len).view(1, tokens.size(0), -1)
                    if print_vis_token_meaning:
                        prefix_projections = embed[:, q_len:q_len + model.prefix_length, :]
                        for i in range(prefix_projections.size(1)):
                            print_nearest_text_token(prefix_projections[0, i], model)

                    out_text = generate_beam(
                        model,
                        model.tokenizer,
                        generated=embed,
                        entry_length=base_dataset.max_seqs_len[1],
                        temperature=1,
                    )[0]

            gold_answer = _get_gold_answer(base_dataset, raw_idx)
            question_text = base_dataset.questions[raw_idx] if hasattr(base_dataset, 'questions') else ''
            pred_texts.append(out_text)
            gold_texts.append(str(gold_answer))

            # Original metrics (first-commit style)
            ref_text = str(gold_answer)
            pred_text = out_text
            reference = [ref_text]
            candidate = [pred_text]

            if pred_text.lower() == ref_text.lower():
                orig_acc += 1
            if ref_text.lower() == 'yes' or ref_text.lower() == 'no':
                if pred_text.lower() == ref_text.lower():
                    orig_acc_yn += 1
                orig_c_yn += 1
            else:
                if pred_text.lower() == ref_text.lower():
                    orig_acc_oe += 1
                orig_c_oe += 1

            orig_bleu_1 = sentence_bleu(reference[0], candidate[0], weights=(1, 0, 0, 0))
            a_orig = _bertscore_compute(
                bert_score,
                references=reference,
                predictions=candidate,
                model_type='bert-base-uncased',
                device_type=device.type,
            )
            orig_bert_avg3 += a_orig['f1'][0]
            orig_f1_avg += compute_f1(tokenizer_legacy.encode(reference[0]), tokenizer_legacy.encode(candidate[0]))
            orig_bleu_avg1 += orig_bleu_1

            # Normalized/soft diagnostics
            pred = normalize_answer(out_text)
            gold = normalize_answer(gold_answer)

            if pred == gold:
                acc += 1

            if gold in {'yes', 'no'}:
                if pred == gold:
                    acc_yn += 1
                c_yn += 1
            else:
                if pred == gold:
                    acc_oe += 1
                c_oe += 1

            ref_tokens = normalize_text(ref_text).split()
            cand_tokens = normalize_text(pred_text).split()

            bleu_1 = sentence_bleu(
                [ref_tokens],
                cand_tokens,
                weights=(1, 0, 0, 0),
                smoothing_function=smooth,
            ) if len(cand_tokens) > 0 else 0.0

            a = _bertscore_compute(
                bert_score,
                references=[ref_text],
                predictions=[pred_text],
                model_type='bert-base-uncased',
                device_type=device.type,
            )
            sample_bert = float(a['f1'][0])
            bert_avg3 += sample_bert
            bert_f1_scores.append(sample_bert)

            f1_avg += compute_f1(tokenizer.encode(ref_text), tokenizer.encode(pred_text))
            bleu_avg1 += bleu_1

            pred_norm = normalize_text(pred_text)
            gold_norm = normalize_text(ref_text)
            per_sample_rows.append(
                {
                    'idx': item,
                    'raw_idx': raw_idx,
                    'question': str(question_text),
                    'prediction': str(pred_text),
                    'reference': str(ref_text),
                    'bertscore_f1': sample_bert,
                    'bleu1': float(bleu_1),
                    'exact_match': int(pred_norm == gold_norm),
                    'soft_match': int(soft_match(pred_text, ref_text)),
                    'yn_gold': yes_no_value(ref_text),
                }
            )
            epoch_pbar.update(1)

    results = evaluate_predictions(pred_texts, gold_texts)
    print('NORMALISED RESULTS:', results)
    debug_mismatches(pred_texts, gold_texts, k=10)

    print('------------')
    print('ORIGINAL METRICS (first-commit style)')
    print('BLEU {}'.format(round(orig_bleu_avg1 / len(dataset), 3)))
    print('BERTScore {}'.format(round(orig_bert_avg3 / len(dataset), 3)))
    print('F1 {}'.format(round(orig_f1_avg / len(dataset), 3)))
    print('Accuracy {}'.format(round(orig_acc / len(dataset), 3)))
    print('Accuracy YN{}'.format(round(orig_acc_yn / orig_c_yn, 3)))
    print('Accuracy OE{}'.format(round(orig_acc_oe / orig_c_oe, 3)))

    print('------------')
    print('NORMALIZED/SOFT DIAGNOSTICS')
    print('BLEU {}'.format(round(bleu_avg1 / len(dataset), 3)))
    print('BERTScore {}'.format(round(bert_avg3 / len(dataset), 3)))
    print('F1 {}'.format(round(f1_avg / len(dataset), 3)))
    print('Accuracy {}'.format(round(acc / len(dataset), 3)))
    print('Accuracy YN{}'.format(round(acc_yn / c_yn, 3)))
    print('Accuracy OE{}'.format(round(acc_oe / c_oe, 3)))

    if bert_f1_scores:
        print('------------')
        print('BERTScore Distribution')
        mean_v = sum(bert_f1_scores) / len(bert_f1_scores)
        var_v = sum((x - mean_v) ** 2 for x in bert_f1_scores) / len(bert_f1_scores)
        std_v = var_v ** 0.5
        print(f'count={len(bert_f1_scores)}')
        print(f'mean={mean_v:.6f}')
        print(f'std={std_v:.6f}')
        print(f'min={min(bert_f1_scores):.6f}')
        print(f'p10={_percentile(bert_f1_scores, 10):.6f}')
        print(f'p25={_percentile(bert_f1_scores, 25):.6f}')
        print(f'p50={_percentile(bert_f1_scores, 50):.6f}')
        print(f'p75={_percentile(bert_f1_scores, 75):.6f}')
        print(f'p90={_percentile(bert_f1_scores, 90):.6f}')
        print(f'max={max(bert_f1_scores):.6f}')

    if pred_texts and gold_texts:
        print('------------')
        print('BERTScore Sanity Checks')

        rr = _bertscore_compute(
            bert_score,
            references=gold_texts,
            predictions=gold_texts,
            model_type='bert-base-uncased',
            device_type=device.type,
        )
        rr_mean = sum(rr['f1']) / len(rr['f1'])
        print(f'ref_vs_ref_mean_f1={rr_mean:.6f}')

        shuffled_refs = list(gold_texts)
        random.Random(0).shuffle(shuffled_refs)
        pr = _bertscore_compute(
            bert_score,
            references=shuffled_refs,
            predictions=pred_texts,
            model_type='bert-base-uncased',
            device_type=device.type,
        )
        pr_mean = sum(pr['f1']) / len(pr['f1'])
        print(f'pred_vs_shuffled_ref_mean_f1={pr_mean:.6f}')

        empty_preds = ['' for _ in gold_texts]
        er = _bertscore_compute(
            bert_score,
            references=gold_texts,
            predictions=empty_preds,
            model_type='bert-base-uncased',
            device_type=device.type,
        )
        er_mean = sum(er['f1']) / len(er['f1'])
        print(f'empty_pred_baseline_mean_f1={er_mean:.6f}')

    if per_sample_rows:
        _write_eval_debug_files(args, per_sample_rows)

        print('------------')
        print('Worst Samples By BERTScore')
        worst_rows = sorted(per_sample_rows, key=lambda x: x['bertscore_f1'])[:10]
        for row in worst_rows:
            print('=' * 60)
            print(f"idx={row['idx']} raw_idx={row['raw_idx']} bert_f1={row['bertscore_f1']:.6f}")
            print(f"Q: {row['question']}")
            print(f"PRED: {row['prediction']}")
            print(f"GOLD: {row['reference']}")


def print_nearest_text_token(vis_token, model):
    """Print the nearest token in vocabulary to the given visual token."""
    embeddings = model.gpt.transformer.wte.weight
    distances = torch.norm(embeddings - vis_token, dim=1)
    nearest_token_idx = torch.argmin(distances)
    print(model.tokenizer.decode([nearest_token_idx.item()]))


def compute_f1(gold_toks, pred_toks):
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _percentile(values, pct):
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(round((pct / 100.0) * (len(xs) - 1)))
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


def _bertscore_compute(metric, references, predictions, model_type, device_type):
    try:
        result = metric.compute(
            references=references,
            predictions=predictions,
            model_type=model_type,
            lang='en',
            device=device_type,
        )
    except TypeError:
        result = metric.compute(
            references=references,
            predictions=predictions,
            model_type=model_type,
            lang='en',
        )
    return result


def _find_run_config_file(out_dir):
    if not out_dir or not os.path.isdir(out_dir):
        return None

    candidates = []
    for name in os.listdir(out_dir):
        if name.startswith('run_config') and name.endswith('.txt'):
            p = os.path.join(out_dir, name)
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                continue
            candidates.append((mtime, p))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _write_eval_debug_files(args, rows):
    out_dir = getattr(args, 'out_dir', './checkpoints')
    debug_dir = os.path.join(out_dir, 'eval_debug')
    os.makedirs(debug_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    stem = f'eval_debug_{job_id}_{ts}'

    jsonl_path = os.path.join(debug_dir, f'{stem}.jsonl')
    csv_path = os.path.join(debug_dir, f'{stem}.csv')

    with open(jsonl_path, 'w', encoding='utf-8') as f_jsonl:
        for row in rows:
            f_jsonl.write(json.dumps(row, ensure_ascii=False) + '\n')

    fieldnames = [
        'idx',
        'raw_idx',
        'question',
        'prediction',
        'reference',
        'bertscore_f1',
        'bleu1',
        'exact_match',
        'soft_match',
        'yn_gold',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    print(f'eval_debug_jsonl={jsonl_path}')
    print(f'eval_debug_csv={csv_path}')
