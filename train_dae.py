import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_preprocessing.dataloader import medvqaDataset
from models import DenoisingAutoencoder


def set_random_seeds(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 DAE pretraining on cached visual embeddings")
    parser.add_argument("--dataset", type=str, default="slake", choices=("pathvqa", "ovqa", "slake"))
    parser.add_argument("--dataset_path", type=str, default="../vqa_datasets/")
    parser.add_argument("--out_dir", type=str, default="./checkpoints/dae_pretrain")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dae_noise_std", type=float, default=0.05)
    parser.add_argument("--dae_bottleneck_dim", type=int, default=256)
    parser.add_argument("--dae_recon_loss", type=str, default="mse", choices=("mse", "smooth_l1"))
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    return parser.parse_args()


def maybe_take_prefixes(prefixes, max_samples):
    if max_samples is None or max_samples <= 0:
        return prefixes
    return prefixes[: min(max_samples, len(prefixes))]


def extract_prefix_embeddings(dataset, max_samples=0):
    # Dataset stores visual embeddings in img_prefixes; gather one per QA item via img_ids.
    gathered = []
    for img_id in dataset.img_ids:
        prefix = dataset.img_prefixes[img_id]
        if not torch.is_tensor(prefix):
            prefix = torch.tensor(prefix)
        gathered.append(prefix.float().view(-1))
    gathered = maybe_take_prefixes(gathered, max_samples)
    return torch.stack(gathered, dim=0).contiguous()


def reconstruction_loss(loss_name, recon, target):
    if loss_name == "smooth_l1":
        return nnf.smooth_l1_loss(recon, target)
    return nnf.mse_loss(recon, target)


def run_epoch(model, loader, device, optimizer, noise_std, recon_loss_name):
    is_train = optimizer is not None
    model.train(is_train)
    running = 0.0
    n_batches = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for (clean_x,) in loader:
            clean_x = clean_x.to(device)
            noisy_x = clean_x
            if noise_std > 0:
                noisy_x = clean_x + torch.randn_like(clean_x) * noise_std
            _, recon_x = model(noisy_x)
            loss = reconstruction_loss(recon_loss_name, recon_x, clean_x)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            running += float(loss.item())
            n_batches += 1
    return running / max(1, n_batches)


def main():
    args = parse_args()
    set_random_seeds(args.seed)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dataset_root = os.path.join(args.dataset_path, args.dataset)
    train_dataset = medvqaDataset(dataset_root + "/", split="train", prefix_length=8, model_type="gpt2-xl")
    val_dataset = medvqaDataset(dataset_root + "/", split="val", prefix_length=8, model_type="gpt2-xl")

    train_prefixes = extract_prefix_embeddings(train_dataset, args.max_train_samples)
    val_prefixes = extract_prefix_embeddings(val_dataset, args.max_val_samples)

    input_dim = int(train_prefixes.shape[1])
    model = DenoisingAutoencoder(input_dim=input_dim, bottleneck_dim=args.dae_bottleneck_dim, dropout=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(TensorDataset(train_prefixes), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_prefixes), batch_size=args.batch_size, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = os.path.join(args.out_dir, "dae_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_recon_loss", "val_recon_loss", "saved_best", "epoch_seconds"])

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            noise_std=args.dae_noise_std,
            recon_loss_name=args.dae_recon_loss,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            noise_std=args.dae_noise_std,
            recon_loss_name=args.dae_recon_loss,
        )
        elapsed = time.time() - t0

        torch.save(
            {
                "dae_state": model.state_dict(),
                "epoch": epoch,
                "dae_recon_loss": args.dae_recon_loss,
                "dae_bottleneck_dim": args.dae_bottleneck_dim,
                "dae_noise_std": args.dae_noise_std,
                "input_dim": input_dim,
            },
            os.path.join(args.out_dir, "dae_latest.pt"),
        )

        saved_best = 0
        if val_loss < best_val:
            best_val = val_loss
            saved_best = 1
            torch.save(
                {
                    "dae_state": model.state_dict(),
                    "epoch": epoch,
                    "dae_recon_loss": args.dae_recon_loss,
                    "dae_bottleneck_dim": args.dae_bottleneck_dim,
                    "dae_noise_std": args.dae_noise_std,
                    "input_dim": input_dim,
                },
                os.path.join(args.out_dir, "dae_best.pt"),
            )

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.10f}",
                    f"{val_loss:.10f}",
                    saved_best,
                    f"{elapsed:.4f}",
                ]
            )
        print(
            f"DAE epoch {epoch}/{args.epochs} - train_recon_loss={train_loss:.6f} val_recon_loss={val_loss:.6f} "
            f"time={elapsed:.2f}s"
        )

    print(f"DAE pretraining completed. Best val recon loss={best_val:.6f}. Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
