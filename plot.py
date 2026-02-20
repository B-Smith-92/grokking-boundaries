"""Generate figures from sweep results and probe data."""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_summary(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def plot_grok_vs_param(rows, experiment, xlabel, title, output_path):
    """Scatter + line for grokking epoch vs one swept parameter."""
    filtered = [r for r in rows if r["experiment"] == experiment]
    if not filtered:
        print(f"No data for '{experiment}', skipping")
        return

    pairs = []
    for r in filtered:
        ge = r["grok_epoch"]
        if ge == "None" or ge is None:
            continue
        pairs.append((float(r["value"]), int(ge)))

    if not pairs:
        print(f"No grokking detected for '{experiment}', skipping")
        return

    pairs.sort()
    values, grok = zip(*pairs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values, grok, "o-", color="black", markersize=6)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Grokking Epoch", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_etf_timelapse(probe_path, output_path, n_panels=8):
    """Grid of cosine-similarity heatmaps across training."""
    data = np.load(probe_path)
    epochs = data["epochs"]
    cos_sims = data["cosine_similarities"]

    n_total = len(epochs)
    indices = np.linspace(0, n_total - 1, n_panels, dtype=int)

    ncols = n_panels // 2
    fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        im = ax.imshow(cos_sims[idx], cmap="RdBu_r", vmin=-1, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"Epoch {epochs[idx]}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")

    fig.suptitle("Class-Mean Cosine Similarity Across Training",
                 fontsize=13, y=1.02)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.summary_csv:
        rows = load_summary(args.summary_csv)

        plot_grok_vs_param(
            rows, "weight_decay",
            xlabel="Weight Decay",
            title="Grokking Epoch vs Weight Decay",
            output_path=os.path.join(args.output_dir, "fig1_weight_decay.png"),
        )

        plot_grok_vs_param(
            rows, "hidden_dim",
            xlabel="Hidden Dimension",
            title="Grokking Epoch vs Hidden Dimension",
            output_path=os.path.join(args.output_dir, "fig2_hidden_dim.png"),
        )

        plot_grok_vs_param(
            rows, "prime",
            xlabel="Modulo Prime (N)",
            title="Grokking Epoch vs Prime",
            output_path=os.path.join(args.output_dir, "fig3_prime.png"),
        )

    if args.probe_npz:
        plot_etf_timelapse(
            args.probe_npz,
            output_path=os.path.join(args.output_dir, "fig4_etf_timelapse.png"),
            n_panels=args.n_panels,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="Path to sweep summary.csv for figures 1-3")
    parser.add_argument("--probe_npz", type=str, default=None,
                        help="Path to probe_results.npz for figure 4")
    parser.add_argument("--n_panels", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()
    main(args)
