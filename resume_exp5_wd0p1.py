"""Resume interrupted exp5_optimal_ratio_wd0p1 runs for dim_1552.

Resumes seed_42 from checkpoint_024000 (last checkpoint with full optimizer state),
then runs the remaining 4 seeds from scratch.
"""

import subprocess
import sys
import os

PYTHON = sys.executable
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")
OUTPUT_ROOT = os.path.join(
    os.path.dirname(__file__),
    "sweep_results", "exp5_optimal_ratio_wd0p1",
)

COMMON_ARGS = {
    "prime": 97,
    "d_model": 1552,
    "n_heads": 4,
    "n_layers": 2,
    "lr": 1e-3,
    "weight_decay": 0.1,
    "epochs": 100000,
    "frac_train": 0.5,
    "eval_interval": 10,
    "n_checkpoints": 50,
    "grok_threshold": 0.95,
}

SEEDS = [42, 123, 456, 789, 1337]


def build_cmd(seed, resume=False):
    output_dir = os.path.join(OUTPUT_ROOT, f"dim_1552_seed_{seed}")
    cmd = [PYTHON, TRAIN_SCRIPT]
    for k, v in COMMON_ARGS.items():
        cmd.extend([f"--{k}", str(v)])
    cmd.extend(["--seed", str(seed)])
    cmd.extend(["--output_dir", output_dir])
    if resume:
        cmd.append("--resume")
    return cmd, output_dir


def main():
    # --- Step 1: Clean up old bare-state-dict checkpoints from seed_42 ---
    # Checkpoints 026000-058000 are old format (no optimizer state).
    # checkpoint_024000 is the latest with full optimizer+metrics.
    # Remove the old ones so --resume picks up 024000 cleanly.
    seed42_dir = os.path.join(OUTPUT_ROOT, "dim_1552_seed_42")
    old_checkpoints = []
    for f in sorted(os.listdir(seed42_dir)):
        if f.startswith("checkpoint_") and f.endswith(".pt"):
            epoch_str = f.replace("checkpoint_", "").replace(".pt", "")
            epoch = int(epoch_str)
            if epoch > 24000:
                old_checkpoints.append(os.path.join(seed42_dir, f))

    if old_checkpoints:
        print(f"Removing {len(old_checkpoints)} old-format checkpoints (026000-058000)...")
        for path in old_checkpoints:
            os.remove(path)
            print(f"  Removed {os.path.basename(path)}")

    # --- Step 2: Resume seed_42 from checkpoint_024000 ---
    cmd, out_dir = build_cmd(42, resume=True)
    print(f"\n{'='*60}")
    print(f"RESUMING: dim_1552_seed_42 from epoch 24001")
    print(f"Output:   {out_dir}")
    print(f"Command:  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: seed_42 returned {result.returncode}")
        sys.exit(1)
    print("seed_42 COMPLETE\n")

    # --- Step 3: Run remaining seeds from scratch ---
    for seed in [123, 456, 789, 1337]:
        cmd, out_dir = build_cmd(seed, resume=False)
        print(f"\n{'='*60}")
        print(f"STARTING: dim_1552_seed_{seed}")
        print(f"Output:   {out_dir}")
        print(f"Command:  {' '.join(cmd)}")
        print(f"{'='*60}\n")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: seed_{seed} returned {result.returncode}")
            sys.exit(1)
        print(f"seed_{seed} COMPLETE\n")

    print("\nAll dim_1552 runs finished!")


if __name__ == "__main__":
    main()
