"""Parameter sweep orchestration for grokking experiments."""

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def _run_train_job(args):
    """Wrapper for parallel execution. Returns (key, data)."""
    base_args, overrides, output_dir, job_key = args
    existing = os.path.join(output_dir, "metrics.json")
    if os.path.exists(existing):
        with open(existing) as f:
            data = json.load(f)
        print(f"  Skipping {output_dir} (already complete)")
        return job_key, data
    data = run_train(base_args, overrides, output_dir)
    return job_key, data


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


def sweep_optimal_ratio(base_args, root, n_workers=1, dims=None):
    prime = base_args["prime"]
    if dims is None:
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

    # Build all jobs
    jobs = []
    for d in dims:
        for seed in seeds:
            out = os.path.join(root, f"dim_{d}_seed_{seed}")
            jobs.append((base_args, {"d_model": d, "n_heads": 4, "seed": seed}, out, (d, seed)))

    # Run jobs
    results_map = {}
    if n_workers > 1:
        print(f"  Running {len(jobs)} jobs with {n_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_train_job, job): job for job in jobs}
            for future in as_completed(futures):
                key, data = future.result()
                if data and data.get("grok_epoch") is not None:
                    results_map[key] = data["grok_epoch"]
                d, seed = key
                print(f"  Done: dim={d} seed={seed} grok={data.get('grok_epoch') if data else 'FAIL'}")
    else:
        for job in jobs:
            key, data = _run_train_job(job)
            if data and data.get("grok_epoch") is not None:
                results_map[key] = data["grok_epoch"]

    # Aggregate by dim
    rows = []
    for d in dims:
        grok_epochs = [results_map[(d, s)] for s in seeds if (d, s) in results_map]
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
            base_args, os.path.join(args.output_dir, f"exp5_optimal_ratio_{wd_tag}"),
            n_workers=args.workers)

    if args.experiment == "extended_ratio":
        prime = base_args["prime"]
        ext_dims = [prime * 8, prime * 16]  # 776, 1552 for prime=97
        ext_dims = [(d // 4) * 4 for d in ext_dims]

        # Epochs per WD must match the existing experiments
        wd_epochs = {1.0: 20000, 0.1: 100000, 0.01: 200000}
        wd = base_args["weight_decay"]
        if wd in wd_epochs:
            base_args["epochs"] = wd_epochs[wd]
        wd_tag = f"wd{wd}".replace(".", "p")

        print(f"Experiment 5 (extended): 8x and 16x ratio sweep at WD={wd}, {base_args['epochs']} epochs")
        all_results["extended_ratio"] = sweep_optimal_ratio(
            base_args, os.path.join(args.output_dir, f"exp5_optimal_ratio_{wd_tag}"),
            n_workers=args.workers, dims=ext_dims)

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
                        choices=["all", "weight_decay", "hidden_dim", "prime", "matched", "optimal_ratio", "extended_ratio"])
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="sweep_results")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for optimal_ratio sweep")
    args = parser.parse_args()
    main(args)
