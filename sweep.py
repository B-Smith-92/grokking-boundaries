"""Parameter sweep orchestration for grokking experiments."""

import argparse
import csv
import json
import os
import subprocess
import sys


def run_train(base_args, overrides, output_dir):
    """Run train.py as a subprocess with the given overrides."""
    cmd = [sys.executable, "train.py", "--output_dir", output_dir]
    merged = {**base_args, **overrides}
    for k, v in merged.items():
        cmd.extend([f"--{k}", str(v)])

    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:500]}")
        return None

    with open(os.path.join(output_dir, "metrics.json")) as f:
        return json.load(f)


def sweep_weight_decay(base_args, root):
    values = [0.1, 0.5, 1.0, 3.0, 10.0]
    rows = []
    for wd in values:
        out = os.path.join(root, f"wd_{wd}")
        data = run_train(base_args, {"weight_decay": wd}, out)
        if data:
            rows.append({"weight_decay": wd, "grok_epoch": data["grok_epoch"]})
    return rows


def sweep_hidden_dim(base_args, root):
    values = [32, 64, 128, 256]
    rows = []
    for d in values:
        n_heads = min(4, d)
        out = os.path.join(root, f"dim_{d}")
        data = run_train(base_args, {"d_model": d, "n_heads": n_heads}, out)
        if data:
            rows.append({"d_model": d, "grok_epoch": data["grok_epoch"]})
    return rows


def sweep_prime(base_args, root):
    values = [23, 47, 59, 97]
    rows = []
    for p in values:
        out = os.path.join(root, f"prime_{p}")
        data = run_train(base_args, {"prime": p}, out)
        if data:
            rows.append({"prime": p, "grok_epoch": data["grok_epoch"]})
    return rows


def sweep_optimal_ratio(base_args, root):
    prime = base_args["prime"]
    dims = [
        int(prime * 0.75),
        prime - 1,
        int(prime * 1.25),
        int(prime * 1.5),
        prime * 2,
        int(prime * 2.5),
        prime * 3,
        prime * 4,
    ]
    # round each to nearest multiple of 4 for n_heads=4
    dims = [max(4, (d // 4) * 4) for d in dims]
    seeds = [42, 123, 456, 789, 1337]
    rows = []
    for d in dims:
        grok_epochs = []
        for seed in seeds:
            out = os.path.join(root, f"dim_{d}_seed_{seed}")
            existing = os.path.join(out, "metrics.json")
            if os.path.exists(existing):
                with open(existing) as f:
                    data = json.load(f)
                print(f"  Skipping {out} (already complete)")
            else:
                data = run_train(base_args, {"d_model": d, "n_heads": 4, "seed": seed}, out)
            if data and data["grok_epoch"] is not None:
                grok_epochs.append(data["grok_epoch"])
        if grok_epochs:
            rows.append({
                "d_model": d,
                "ratio": round(d / prime, 2),
                "grok_epoch_mean": round(sum(grok_epochs) / len(grok_epochs)),
                "grok_epoch_min": min(grok_epochs),
                "grok_epoch_max": max(grok_epochs),
                "n_grokked": len(grok_epochs),
                "seeds_run": len(seeds),
            })
    return rows


def sweep_matched(base_args, root):
    configs = [
        {"prime": 23, "d_model": 32,  "n_heads": 4},
        {"prime": 47, "d_model": 64,  "n_heads": 4},
        {"prime": 59, "d_model": 64,  "n_heads": 4},
        {"prime": 97, "d_model": 128, "n_heads": 4},
    ]
    rows = []
    for cfg in configs:
        out = os.path.join(root, f"prime_{cfg['prime']}_dim_{cfg['d_model']}")
        data = run_train(base_args, cfg, out)
        if data:
            rows.append({"prime": cfg["prime"], "d_model": cfg["d_model"],
                         "grok_epoch": data["grok_epoch"]})
    return rows


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    base_args = {
        "prime": 97,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "lr": 1e-3,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "frac_train": 0.5,
        "seed": args.seed,
        "eval_interval": 10,
        "n_checkpoints": 50,
        "grok_threshold": 0.95,
    }

    all_results = {}

    if args.experiment in ("all", "weight_decay"):
        print("Experiment 1: Weight decay sweep")
        all_results["weight_decay"] = sweep_weight_decay(
            base_args, os.path.join(args.output_dir, "exp1_weight_decay"))

    if args.experiment in ("all", "hidden_dim"):
        print("Experiment 2: Hidden dimension sweep")
        all_results["hidden_dim"] = sweep_hidden_dim(
            base_args, os.path.join(args.output_dir, "exp2_hidden_dim"))

    if args.experiment in ("all", "prime"):
        print("Experiment 3: Prime sweep")
        all_results["prime"] = sweep_prime(
            base_args, os.path.join(args.output_dir, "exp3_prime"))

    if args.experiment in ("all", "optimal_ratio"):
        print("Experiment 5: Optimal dim/prime ratio sweep")
        wd_tag = f"wd{args.weight_decay}".replace(".", "p")
        all_results["optimal_ratio"] = sweep_optimal_ratio(
            base_args, os.path.join(args.output_dir, f"exp5_optimal_ratio_{wd_tag}"))

    if args.experiment in ("all", "matched"):
        print("Experiment 4: Matched capacity sweep")
        all_results["matched"] = sweep_matched(
            base_args, os.path.join(args.output_dir, "exp4_matched"))

    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "parameter", "value", "grok_epoch"])
        for exp_name, rows in all_results.items():
            for row in rows:
                param_key = [k for k in row if k != "grok_epoch"][0]
                writer.writerow([
                    exp_name, param_key, row[param_key], row["grok_epoch"]
                ])

    print(f"\nSummary: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "weight_decay", "hidden_dim", "prime", "matched", "optimal_ratio"])
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="sweep_results")
    args = parser.parse_args()
    main(args)
