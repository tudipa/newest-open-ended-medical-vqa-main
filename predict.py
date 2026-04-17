import collections
import re
import string
from contextlib import nullcontext

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
            a_orig = bert_score.compute(
                references=reference,
                predictions=candidate,
                model_type='bert-base-uncased',
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

            a = bert_score.compute(
                references=[ref_text],
                predictions=[pred_text],
                model_type='bert-base-uncased',
            )
            bert_avg3 += a['f1'][0]

            f1_avg += compute_f1(tokenizer.encode(ref_text), tokenizer.encode(pred_text))
            bleu_avg1 += bleu_1
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
