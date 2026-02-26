"""Single training job for modular arithmetic grokking."""

import argparse
import glob as globmod
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn

from model import GrokTransformer


def make_dataset(p):
    """Generate all (a, b, (a+b) mod p) triples."""
    pairs = []
    labels = []
    for a in range(p):
        for b in range(p):
            pairs.append([a, b])
            labels.append((a + b) % p)
    return torch.tensor(pairs), torch.tensor(labels)


def split_dataset(inputs, labels, frac_train=0.5, seed=0):
    """Deterministic train/val split."""
    n = len(inputs)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * frac_train)
    train_idx = indices[:split]
    val_idx = indices[split:]
    return (inputs[train_idx], labels[train_idx],
            inputs[val_idx], labels[val_idx])


def detect_grokking(metrics, threshold, window=10):
    """Return the epoch where val_acc first crosses threshold and stays.

    Scans metrics for the first run of `window` consecutive eval steps
    at or above `threshold`. Returns the epoch of the first step in
    that run, or None if grokking was not detected.
    """
    above = 0
    for i, m in enumerate(metrics):
        if m["val_acc"] >= threshold:
            above += 1
            if above >= window:
                return metrics[i - window + 1]["epoch"]
        else:
            above = 0
    return None


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in output_dir. Returns (path, epoch) or (None, 0)."""
    files = globmod.glob(os.path.join(output_dir, "checkpoint_*.pt"))
    if not files:
        return None, 0
    best_path, best_epoch = None, -1
    for f in files:
        m = re.search(r"checkpoint_(\d+)\.pt$", f)
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_path = f
    if best_path is None:
        return None, 0
    return best_path, best_epoch


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    inputs, labels = make_dataset(args.prime)
    train_x, train_y, val_x, val_y = split_dataset(
        inputs, labels, frac_train=args.frac_train, seed=args.seed
    )
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)

    model = GrokTransformer(
        p=args.prime,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    loss_fn = nn.CrossEntropyLoss()

    metrics = []
    start_epoch = 0
    checkpoint_interval = max(1, args.epochs // args.n_checkpoints)

    # --- resume from checkpoint ---
    if args.resume:
        ckpt_path, ckpt_epoch = find_latest_checkpoint(args.output_dir)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                metrics = ckpt.get("metrics", [])
                print(f"Resumed from {ckpt_path} (epoch {start_epoch}, optimizer restored)")
            else:
                # Old format: bare state_dict, no optimizer state
                model.load_state_dict(ckpt)
                start_epoch = ckpt_epoch + 1
                print(f"Resumed from {ckpt_path} (epoch {start_epoch}, optimizer reset)")
        else:
            print("No checkpoint found, starting from scratch")

    for epoch in range(start_epoch, args.epochs):

        # --- train step (full batch) ---
        model.train()
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- eval ---
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                t_logits = model(train_x)
                t_loss = loss_fn(t_logits, train_y).item()
                t_acc = (t_logits.argmax(-1) == train_y).float().mean().item()

                v_logits = model(val_x)
                v_loss = loss_fn(v_logits, val_y).item()
                v_acc = (v_logits.argmax(-1) == val_y).float().mean().item()

            metrics.append({
                "epoch": epoch,
                "train_loss": t_loss,
                "train_acc": t_acc,
                "val_loss": v_loss,
                "val_acc": v_acc,
            })

        # --- checkpoint ---
        if epoch % checkpoint_interval == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.output_dir, f"checkpoint_{epoch:06d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            }, path)

    grok_epoch = detect_grokking(metrics, args.grok_threshold)

    result = {
        "args": vars(args),
        "metrics": metrics,
        "grok_epoch": grok_epoch,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"grok_epoch={grok_epoch}")
    return grok_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--frac_train", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--n_checkpoints", type=int, default=50)
    parser.add_argument("--grok_threshold", type=float, default=0.95)
    parser.add_argument("--output_dir", type=str, default="runs/default")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output_dir")
    args = parser.parse_args()
    train(args)
