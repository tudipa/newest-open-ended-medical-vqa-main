import collections
from contextlib import nullcontext

import torch
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from transformers import GPT2Tokenizer

from utils import generate_beam


def _unwrap_subset(dataset):
    """Return base dataset and index mapping when dataset is a torch.utils.data.Subset."""
    indices = None
    while hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
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
    if hasattr(base_dataset, "answers_raw"):
        return base_dataset.answers_raw[raw_idx]
    if hasattr(base_dataset, "answers"):
        return base_dataset.answers[raw_idx]
    raise AttributeError("Dataset has neither answers_raw nor answers")
def eval_gpt_open_ended(model, dataset, args, print_vis_token_meaning=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device={device}")

    model.eval()
    model = model.to(device)

    base_dataset, subset_indices = _unwrap_subset(dataset)

    bert_score = load("bertscore")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    bleu_avg1 = 0.0
    bert_avg3 = 0.0
    f1_avg = 0.0
    acc = 0.0
    acc_oe = 0.0
    acc_yn = 0.0
    c_oe = 1e-9
    c_yn = 1e-9

    with tqdm(total=len(dataset)) as epoch_pbar:
        epoch_pbar.set_description("Testing")
        for item in range(len(dataset)):
            raw_idx = _dataset_raw_index(item, subset_indices)

            prefix, labels, tokens, mask, q_len = dataset[item]
            prefix = prefix.type(torch.float32).to(device)
            tokens = tokens.type(torch.long).to(device)
            mask = mask.to(device)

            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                with torch.no_grad():
                    embed = model.generate(prefix, labels, tokens, mask, q_len).view(1, tokens.size(0), -1)
                    if print_vis_token_meaning:
                        prefix_projections = embed[:, q_len : q_len + model.prefix_length, :]
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

            if out_text.lower() == gold_answer.lower():
                acc += 1

            if gold_answer.lower() in ("yes", "no"):
                if out_text.lower() == gold_answer.lower():
                    acc_yn += 1
                c_yn += 1
            else:
                if out_text.lower() == gold_answer.lower():
                    acc_oe += 1
                c_oe += 1

            reference = [str(gold_answer)]
            candidate = [out_text]

            bleu_1 = sentence_bleu(reference[0], candidate[0], weights=(1, 0, 0, 0))
            a = bert_score.compute(
                references=reference,
                predictions=candidate,
                model_type="bert-base-uncased",
            )
            bert_avg3 += a["f1"][0]

            f1_avg += compute_f1(tokenizer.encode(reference[0]), tokenizer.encode(candidate[0]))
            bleu_avg1 += bleu_1
            epoch_pbar.update(1)

    print("------------")
    print("BLEU {}".format(round(bleu_avg1 / len(dataset), 3)))
    print("BERTScore {}".format(round(bert_avg3 / len(dataset), 3)))
    print("F1 {}".format(round(f1_avg / len(dataset), 3)))
    print("Accuracy {}".format(round(acc / len(dataset), 3)))
    print("Accuracy YN{}".format(round(acc_yn / c_yn, 3)))
    print("Accuracy OE{}".format(round(acc_oe / c_oe, 3)))


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