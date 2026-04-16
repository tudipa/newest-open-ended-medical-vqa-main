from train import pytorch_model_run
import torch
from predict import eval_gpt_open_ended
from models import VQAmedModel, VQAmedModel_abl
from data_preprocessing.dataloader import medvqaDataset
from torch.utils.data import DataLoader, Subset
import argparse
import numpy as np
import random
import os


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt2-xl", choices=("gpt2-xl", "microsoft/biogpt", "stanford-crfm/BioMedLM"))
    parser.add_argument("--setting", type=str, default="frozen", choices=("lora", "frozen", "prefixtuning", "p_tuning", "prompttuning", "unfrozen"))
    parser.add_argument("--ablation", type=str, default="none", choices=("none", "remove_question", "remove_visual", "replace_visual", "swap"))
    parser.add_argument("--mapping_type", type=str, default="MLP")
    parser.add_argument("--prefix_length", type=int, default=8)
    parser.add_argument("--dataset_path", type=str, default="../vqa_datasets/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dataset", type=str, default="pathvqa", choices=("pathvqa", "ovqa", "slake"))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters_to_accumulate", type=int, default=4)
    parser.add_argument("--validation_step", type=int, default=1000)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--out_dir", default="./checkpoints")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")

    args = parser.parse_args()
    set_random_seeds(args.seed)
    return args


def maybe_subset(dataset, max_samples):
    if max_samples is None or max_samples <= 0:
        return dataset
    subset_size = min(max_samples, len(dataset))
    return Subset(dataset, list(range(subset_size)))


if __name__ == "__main__":
    args = parse_argument()
    suffix = (
        f"v5_prefixlength_{args.prefix_length}_mapping_{args.mapping_type}_seed_{args.seed}_"
        f"gpttype_{args.model_type.replace('/', '')}_setting_{args.setting}_dataset_{args.dataset}"
    )

    # Respect explicit --out_dir from CLI/Slurm; only build default path when not provided.
    if args.out_dir == "./checkpoints":
        args.out_dir = os.path.join("../checkpoints", suffix)
    dataset_root = os.path.join(args.dataset_path, args.dataset)

    train_dataset = medvqaDataset(
        dataset_root + "/",
        split="train",
        prefix_length=args.prefix_length,
        model_type=args.model_type,
    )
    val_dataset = medvqaDataset(
        dataset_root + "/",
        split="val",
        prefix_length=args.prefix_length,
        model_type=args.model_type,
    )
    test_dataset = medvqaDataset(
        dataset_root + "/",
        split="test",
        prefix_length=args.prefix_length,
        model_type=args.model_type,
        like_test=True,
    )

    train_dataset = maybe_subset(train_dataset, args.max_train_samples)
    val_dataset = maybe_subset(val_dataset, args.max_val_samples)
    test_dataset = maybe_subset(test_dataset, args.max_test_samples)

    if args.verbose:
        print(f"Dataset sizes => train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    if args.ablation != "none":
        model = VQAmedModel_abl(
            prefix_length=args.prefix_length,
            clip_length=4,
            setting=args.setting,
            mapping_type=args.mapping_type,
            args=args,
        )
    else:
        model = VQAmedModel(
            prefix_length=args.prefix_length,
            clip_length=4,
            setting=args.setting,
            mapping_type=args.mapping_type,
            args=args,
        )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    if not args.eval:
        model = pytorch_model_run(train_dataloader, val_dataloader, model, args)
    else:
        checkpoint = os.path.join(args.out_dir, "open_ended_latest.pt")
        if args.verbose:
            print(f">> Loading pre-trained model {checkpoint}!")
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")), strict=False)
        else:
            raise ValueError("Please provide valid path for loading checkpoint")
        eval_gpt_open_ended(model, test_dataset, args)