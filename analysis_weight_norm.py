"""Analyze weight norm evolution across checkpoints for a grokking run.

Loads each checkpoint from a single run, computes the total L2 norm of all
model parameters, and cross-references with val_acc from the metrics JSON.
"""

import json
import math
import os
import re
import sys

import torch

# Ensure project root is on the path so we can import model.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model import GrokTransformer


def load_metrics(metrics_path):
    """Load the metrics JSON and return (args_dict, list of metric dicts)."""
    with open(metrics_path) as f:
        data = json.load(f)
    return data["args"], data["metrics"]


def get_val_acc_at_epoch(metrics_list, target_epoch):
    """Return val_acc at the nearest eval step <= target_epoch.

    Metrics are recorded every eval_interval epochs. We find the closest
    entry whose epoch is <= target_epoch.
    """
    best = None
    for m in metrics_list:
        if m["epoch"] <= target_epoch:
            best = m
        else:
            break
    return best["val_acc"] if best else None


def compute_weight_norm(state_dict):
    """Compute total L2 norm: sqrt(sum of squared values across all params)."""
    total_sq = 0.0
    for name, param in state_dict.items():
        total_sq += param.float().pow(2).sum().item()
    return math.sqrt(total_sq)


def main():
    run_dir = os.path.join(
        PROJECT_ROOT, "sweep_results", "exp1_weight_decay", "wd_1.0"
    )
    metrics_path = os.path.join(run_dir, "metrics_wd_1.0.json")

    # Load metrics
    args, metrics_list = load_metrics(metrics_path)
    print(f"Run config: prime={args['prime']}, d_model={args['d_model']}, "
          f"n_heads={args['n_heads']}, n_layers={args['n_layers']}, "
          f"weight_decay={args['weight_decay']}")
    print(f"Grok threshold: {args['grok_threshold']}")
    print()

    # Instantiate model (needed to validate state dict loading)
    model = GrokTransformer(
        p=args["prime"],
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
    )

    # Discover and sort checkpoint files
    ckpt_files = sorted([
        f for f in os.listdir(run_dir)
        if re.match(r"checkpoint_\d+\.pt", f)
    ])

    # Print header
    print(f"{'Epoch':>8s}  {'Weight Norm':>12s}  {'Val Acc':>8s}")
    print(f"{'-----':>8s}  {'-----------':>12s}  {'-------':>8s}")

    results = []
    for ckpt_file in ckpt_files:
        # Extract epoch from filename
        epoch = int(re.search(r"checkpoint_(\d+)\.pt", ckpt_file).group(1))

        # Load checkpoint
        ckpt_path = os.path.join(run_dir, ckpt_file)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Compute weight norm directly from state dict
        w_norm = compute_weight_norm(state_dict)

        # Get val_acc at nearest eval step
        val_acc = get_val_acc_at_epoch(metrics_list, epoch)

        results.append((epoch, w_norm, val_acc))
        print(f"{epoch:>8d}  {w_norm:>12.4f}  {val_acc:>8.4f}")

    # Summary statistics
    print()
    norms = [r[1] for r in results]
    print(f"Weight norm range: {min(norms):.4f} - {max(norms):.4f}")
    print(f"Initial norm (epoch 0): {results[0][1]:.4f}")
    print(f"Final norm (epoch {results[-1][0]}): {results[-1][1]:.4f}")
    print(f"Ratio final/initial: {results[-1][1] / results[0][1]:.4f}")

    # Find the epoch of peak norm
    peak_idx = max(range(len(results)), key=lambda i: results[i][1])
    print(f"Peak norm: {results[peak_idx][1]:.4f} at epoch {results[peak_idx][0]}")

    # Identify grokking transition
    grok_results = [r for r in results if r[2] is not None and r[2] >= 0.95]
    if grok_results:
        grok_epoch = grok_results[0][0]
        grok_norm = grok_results[0][1]
        print(f"First checkpoint with val_acc >= 0.95: epoch {grok_epoch}, "
              f"norm={grok_norm:.4f}")


if __name__ == "__main__":
    main()
