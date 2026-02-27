#!/usr/bin/env python3
"""
Analyze the gap between memorization and generalization across grokking experiments.

For each run, finds:
  - Memorization epoch: first epoch where train_acc > 0.99
  - Generalization epoch: grok_epoch from the JSON (val_acc crosses grok_threshold)
  - Gap: generalization epoch - memorization epoch
"""

import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent / "sweep_results"

EXPERIMENTS = [
    "exp1_weight_decay",
    "exp2_hidden_dim",
    "exp3_prime",
]

# Display-friendly labels for the varying parameter in each experiment
PARAM_LABELS = {
    "exp1_weight_decay": "weight_decay",
    "exp2_hidden_dim": "d_model",
    "exp3_prime": "prime",
}


def find_memorization_epoch(metrics_list, threshold=0.99):
    """Return the first epoch where train_acc exceeds the threshold, or None."""
    for entry in metrics_list:
        if entry["train_acc"] > threshold:
            return entry["epoch"]
    return None


def extract_param_value(exp_name, args):
    """Pull the swept parameter value from the run's args dict."""
    key = PARAM_LABELS[exp_name]
    return args.get(key)


def sort_key_for_param(value):
    """Return a sort key that works for both numeric and string values."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def load_run(json_path):
    """Load a single metrics JSON and return parsed data."""
    with open(json_path) as f:
        return json.load(f)


def analyze_experiment(exp_name):
    """Scan all runs in an experiment directory and return a list of result dicts."""
    exp_dir = BASE / exp_name
    if not exp_dir.is_dir():
        return []

    results = []
    # Look for metrics.json or metrics_*.json in each subdirectory
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        # Find any metrics JSON file in the run directory
        json_files = list(run_dir.glob("metrics*.json"))
        if not json_files:
            continue

        data = load_run(json_files[0])
        args = data.get("args", {})
        metrics = data.get("metrics", [])
        grok_epoch = data.get("grok_epoch")

        param_value = extract_param_value(exp_name, args)
        mem_epoch = find_memorization_epoch(metrics)

        if grok_epoch is not None and mem_epoch is not None:
            gap = grok_epoch - mem_epoch
        else:
            gap = None

        results.append({
            "param": param_value,
            "mem_epoch": mem_epoch,
            "grok_epoch": grok_epoch,
            "gap": gap,
            "run_dir": run_dir.name,
        })

    # Sort by parameter value
    results.sort(key=lambda r: sort_key_for_param(r["param"]))
    return results


def fmt(val):
    """Format a value for the table: right-align, dash if None."""
    if val is None:
        return "---"
    if isinstance(val, float):
        return f"{val:g}"
    return str(val)


def print_table(title, param_label, results):
    """Print a neatly formatted table for one experiment."""
    col_widths = {
        "param": max(len(param_label), max((len(fmt(r["param"])) for r in results), default=5)),
        "mem": max(len("Mem Epoch"), 10),
        "grok": max(len("Grok Epoch"), 10),
        "gap": max(len("Gap"), 10),
    }

    header = (
        f"  {param_label:<{col_widths['param']}}  "
        f"{'Mem Epoch':>{col_widths['mem']}}  "
        f"{'Grok Epoch':>{col_widths['grok']}}  "
        f"{'Gap':>{col_widths['gap']}}"
    )
    sep = "  " + "-" * (col_widths["param"] + col_widths["mem"] + col_widths["grok"] + col_widths["gap"] + 6)

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(header)
    print(sep)

    for r in results:
        param_str = fmt(r["param"])
        mem_str = fmt(r["mem_epoch"])
        grok_str = fmt(r["grok_epoch"])
        gap_str = fmt(r["gap"])
        print(
            f"  {param_str:<{col_widths['param']}}  "
            f"{mem_str:>{col_widths['mem']}}  "
            f"{grok_str:>{col_widths['grok']}}  "
            f"{gap_str:>{col_widths['gap']}}"
        )

    print()


def main():
    print("Grokking Gap Analysis: Memorization vs. Generalization")
    print("(Memorization = first epoch with train_acc > 0.99)")
    print("(Generalization = grok_epoch from JSON, val_acc > grok_threshold)")

    for exp_name in EXPERIMENTS:
        results = analyze_experiment(exp_name)
        if not results:
            print(f"\n[!] No results found for {exp_name}")
            continue

        param_label = PARAM_LABELS[exp_name]
        title = exp_name.replace("_", " ").upper()
        print_table(title, param_label, results)


if __name__ == "__main__":
    main()
