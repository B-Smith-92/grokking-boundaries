"""Generate figures from sweep results, probe data, and experiments.db."""

import argparse
import csv
import os
import sqlite3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLORS = {1.0: "#1b9e77", 0.1: "#d95f02", 0.01: "#7570b3"}
WD_LABELS = {1.0: "WD = 1.0", 0.1: "WD = 0.1", 0.01: "WD = 0.01"}


def _style():
    """Apply consistent plot styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def query_ratio_sweep(conn):
    """Get per-seed grok data for all exp5 ratio sweeps."""
    rows = conn.execute("""
        SELECT e.name, r.weight_decay, r.d_model, r.ratio, r.seed, r.grok_epoch,
               e.epochs AS max_epochs
        FROM runs r
        JOIN experiments e ON e.id = r.experiment_id
        WHERE e.name LIKE 'exp5_ratio%'
        ORDER BY r.weight_decay DESC, r.ratio, r.seed
    """).fetchall()
    return rows


def query_ratio_stats(conn):
    """Aggregated stats per (weight_decay, ratio)."""
    rows = conn.execute("""
        SELECT r.weight_decay, r.ratio, r.d_model,
               COUNT(*) AS n_runs,
               SUM(r.grok_epoch IS NOT NULL) AS n_grokked,
               AVG(r.grok_epoch) AS mean_grok,
               MIN(r.grok_epoch) AS min_grok,
               MAX(r.grok_epoch) AS max_grok
        FROM runs r
        JOIN experiments e ON e.id = r.experiment_id
        WHERE e.name LIKE 'exp5_ratio%'
        GROUP BY r.weight_decay, r.ratio
        ORDER BY r.weight_decay DESC, r.ratio
    """).fetchall()
    return rows


def query_training_curves(conn, experiment, d_model, seed):
    """Get full training curve for a specific run."""
    rows = conn.execute("""
        SELECT m.epoch, m.train_loss, m.train_acc, m.val_loss, m.val_acc
        FROM metrics m
        JOIN runs r ON r.id = m.run_id
        JOIN experiments e ON e.id = r.experiment_id
        WHERE e.name = ? AND r.d_model = ? AND r.seed = ?
        ORDER BY m.epoch
    """, (experiment, d_model, seed)).fetchall()
    return rows


# ---------------------------------------------------------------------------
# Figure 1: The U-Shape Triptych
# ---------------------------------------------------------------------------

def plot_u_shape(conn, output_dir):
    """Grok epoch vs ratio for all three WDs with error bands."""
    _style()
    stats = query_ratio_stats(conn)

    fig, ax = plt.subplots(figsize=(10, 6))

    for wd in [1.0, 0.1, 0.01]:
        subset = [r for r in stats if r["weight_decay"] == wd and r["mean_grok"] is not None]
        if not subset:
            continue
        ratios = [r["ratio"] for r in subset]
        means = [r["mean_grok"] for r in subset]
        mins = [r["min_grok"] for r in subset]
        maxs = [r["max_grok"] for r in subset]

        ax.plot(ratios, means, "o-", color=COLORS[wd], label=WD_LABELS[wd],
                markersize=6, linewidth=2)
        ax.fill_between(ratios, mins, maxs, alpha=0.15, color=COLORS[wd])

    ax.set_xlabel("Capacity Ratio (d_model / prime)", fontsize=13)
    ax.set_ylabel("Grokking Epoch", fontsize=13)
    ax.set_title("The U-Shape: Grokking Speed vs Model Capacity", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2, which="both")

    path = os.path.join(output_dir, "fig_u_shape.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Seed Spread Strip Plot
# ---------------------------------------------------------------------------

def plot_seed_spread(conn, output_dir):
    """Individual seed grok epochs as strips, showing variance structure."""
    _style()
    rows = query_ratio_sweep(conn)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for i, wd in enumerate([1.0, 0.1, 0.01]):
        ax = axes[i]
        subset = [r for r in rows if r["weight_decay"] == wd]

        ratios_seen = sorted(set(r["ratio"] for r in subset))
        ratio_to_x = {r: j for j, r in enumerate(ratios_seen)}

        # Plot individual seeds
        for r in subset:
            if r["grok_epoch"] is not None:
                ax.scatter(ratio_to_x[r["ratio"]], r["grok_epoch"],
                           color=COLORS[wd], alpha=0.5, s=30, zorder=3)
            else:
                # Failed to grok -- mark at top
                ax.scatter(ratio_to_x[r["ratio"]], r["max_epochs"] * 1.05,
                           color="red", marker="x", s=40, alpha=0.7, zorder=3)

        # Plot means
        stats = query_ratio_stats(conn)
        wd_stats = [s for s in stats if s["weight_decay"] == wd and s["mean_grok"] is not None]
        xs = [ratio_to_x[s["ratio"]] for s in wd_stats if s["ratio"] in ratio_to_x]
        ys = [s["mean_grok"] for s in wd_stats if s["ratio"] in ratio_to_x]
        ax.plot(xs, ys, "-", color=COLORS[wd], alpha=0.4, linewidth=1.5)

        ax.set_xticks(range(len(ratios_seen)))
        ax.set_xticklabels([f"{r:.1f}" for r in ratios_seen], rotation=45, fontsize=8)
        ax.set_title(WD_LABELS[wd], fontsize=12, color=COLORS[wd])
        ax.set_xlabel("Ratio (d/p)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Grokking Epoch", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_yscale("log")

    fig.suptitle("Per-Seed Grokking Epochs Across the Capacity Ratio",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_seed_spread.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Variance (CV) vs Ratio
# ---------------------------------------------------------------------------

def plot_variance(conn, output_dir):
    """Coefficient of variation of grok_epoch across seeds."""
    _style()
    rows = query_ratio_sweep(conn)

    fig, ax = plt.subplots(figsize=(10, 5))

    for wd in [1.0, 0.1, 0.01]:
        subset = [r for r in rows if r["weight_decay"] == wd and r["grok_epoch"] is not None]
        ratios = sorted(set(r["ratio"] for r in subset))

        cvs = []
        valid_ratios = []
        for ratio in ratios:
            vals = [r["grok_epoch"] for r in subset if r["ratio"] == ratio]
            if len(vals) >= 3:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                cvs.append(std / mean if mean > 0 else 0)
                valid_ratios.append(ratio)

        if valid_ratios:
            ax.plot(valid_ratios, cvs, "o-", color=COLORS[wd], label=WD_LABELS[wd],
                    markersize=6, linewidth=2)

    ax.set_xlabel("Capacity Ratio (d_model / prime)", fontsize=13)
    ax.set_ylabel("Coefficient of Variation (σ/μ)", fontsize=13)
    ax.set_title("Grokking Variance Across Seeds: Turbulence at the Extremes", fontsize=14)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, which="both")

    path = os.path.join(output_dir, "fig_variance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Phase Boundary Heatmap
# ---------------------------------------------------------------------------

def plot_phase_boundary(conn, output_dir):
    """Heatmap of grok success rate by (ratio, weight_decay)."""
    _style()
    stats = query_ratio_stats(conn)

    wds = sorted(set(r["weight_decay"] for r in stats))
    ratios = sorted(set(r["ratio"] for r in stats))

    grid = np.full((len(wds), len(ratios)), np.nan)
    for r in stats:
        wi = wds.index(r["weight_decay"])
        ri = ratios.index(r["ratio"])
        grid[wi, ri] = r["n_grokked"] / r["n_runs"] * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                   interpolation="nearest")

    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{r:.1f}" for r in ratios], rotation=45, fontsize=9)
    ax.set_yticks(range(len(wds)))
    ax.set_yticklabels([str(w) for w in wds], fontsize=11)
    ax.set_xlabel("Capacity Ratio (d_model / prime)", fontsize=12)
    ax.set_ylabel("Weight Decay", fontsize=12)
    ax.set_title("Phase Boundary: Grokking Success Rate (%)", fontsize=14)

    # Annotate cells
    for wi in range(len(wds)):
        for ri in range(len(ratios)):
            val = grid[wi, ri]
            if not np.isnan(val):
                color = "white" if val < 50 else "black"
                ax.text(ri, wi, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="% Grokked", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_phase_boundary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 5: Grok Epoch Heatmap (the full surface)
# ---------------------------------------------------------------------------

def plot_grok_surface(conn, output_dir):
    """Heatmap of mean grok_epoch by (ratio, weight_decay) -- the 3D surface flattened."""
    _style()
    stats = query_ratio_stats(conn)

    wds = sorted(set(r["weight_decay"] for r in stats))
    ratios = sorted(set(r["ratio"] for r in stats))

    grid = np.full((len(wds), len(ratios)), np.nan)
    for r in stats:
        if r["mean_grok"] is not None:
            wi = wds.index(r["weight_decay"])
            ri = ratios.index(r["ratio"])
            grid[wi, ri] = r["mean_grok"]

    fig, ax = plt.subplots(figsize=(12, 4))
    masked = np.ma.masked_invalid(grid)
    vmin = np.nanmin(grid[grid > 0]) if np.any(grid > 0) else 1
    im = ax.imshow(masked, aspect="auto", cmap="magma_r",
                   norm=LogNorm(vmin=max(1, vmin), vmax=np.nanmax(grid)),
                   interpolation="nearest")

    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{r:.1f}" for r in ratios], rotation=45, fontsize=9)
    ax.set_yticks(range(len(wds)))
    ax.set_yticklabels([str(w) for w in wds], fontsize=11)
    ax.set_xlabel("Capacity Ratio (d_model / prime)", fontsize=12)
    ax.set_ylabel("Weight Decay", fontsize=12)
    ax.set_title("Grokking Epoch Surface (log scale)", fontsize=14)

    # Annotate
    for wi in range(len(wds)):
        for ri in range(len(ratios)):
            val = grid[wi, ri]
            if not np.isnan(val):
                ax.text(ri, wi, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if val > 5000 else "black")

    fig.colorbar(im, ax=ax, label="Mean Grokking Epoch", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_grok_surface.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 6: Reynolds Number Collapse
# ---------------------------------------------------------------------------

def plot_reynolds_collapse(conn, output_dir):
    """Test if Re = d_model / (weight_decay * prime) collapses the curves."""
    _style()
    stats = query_ratio_stats(conn)

    fig, ax = plt.subplots(figsize=(10, 6))

    for wd in [1.0, 0.1, 0.01]:
        subset = [r for r in stats if r["weight_decay"] == wd and r["mean_grok"] is not None]
        if not subset:
            continue
        re_numbers = [r["d_model"] / (wd * 97) for r in subset]
        means = [r["mean_grok"] for r in subset]
        mins = [r["min_grok"] for r in subset]
        maxs = [r["max_grok"] for r in subset]

        ax.plot(re_numbers, means, "o-", color=COLORS[wd], label=WD_LABELS[wd],
                markersize=6, linewidth=2)
        ax.fill_between(re_numbers, mins, maxs, alpha=0.15, color=COLORS[wd])

    ax.set_xlabel("Re = d_model / (weight_decay × prime)", fontsize=13)
    ax.set_ylabel("Grokking Epoch", fontsize=13)
    ax.set_title("Reynolds Number Collapse Test", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, which="both")

    path = os.path.join(output_dir, "fig_reynolds.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 7: Training Dynamics -- Sweet Spot vs Oversized
# ---------------------------------------------------------------------------

def _smooth(values, window=50):
    """Rolling mean with edge handling."""
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="same")
    # Fix edge effects by using smaller windows at boundaries
    for i in range(window // 2):
        w = i + 1
        smoothed[i] = np.mean(arr[:w * 2])
        smoothed[-(i + 1)] = np.mean(arr[-(w * 2):])
    return smoothed


def plot_training_dynamics(conn, output_dir):
    """Val accuracy curves comparing sweet spot vs oversized at each WD."""
    _style()

    configs = [
        ("exp5_ratio_wd1", 1.0, [(776, "8x (sweet spot)"), (1552, "16x (oversized)")]),
        ("exp5_ratio_wd0p1", 0.1, [(776, "8x (sweet spot)"), (1552, "16x (oversized)")]),
        ("exp5_ratio_wd0p01", 0.01, [(776, "8x (sweet spot)"), (1552, "16x (oversized)")]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    line_styles = ["-", "--"]
    dim_colors = ["#2166ac", "#b2182b"]

    for i, (exp, wd, dims) in enumerate(configs):
        ax = axes[i]
        for j, (d_model, label) in enumerate(dims):
            curve = query_training_curves(conn, exp, d_model, seed=42)
            if not curve:
                continue
            epochs = np.array([r["epoch"] for r in curve])
            val_acc = np.array([r["val_acc"] for r in curve])

            # Smooth the curve
            smoothed = _smooth(val_acc, window=80)

            # Subsample for performance
            step = max(1, len(epochs) // 2000)

            # Raw data as faint background
            ax.plot(epochs[::step], val_acc[::step],
                    color=dim_colors[j], alpha=0.08, linewidth=0.5)
            # Smoothed line on top
            ax.plot(epochs[::step], smoothed[::step],
                    linestyle=line_styles[j], color=dim_colors[j],
                    label=f"d={d_model} ({label})", linewidth=2, alpha=0.9)

        ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5, label="Grok threshold")
        ax.set_xlabel("Epoch", fontsize=11)
        if i == 0:
            ax.set_ylabel("Validation Accuracy", fontsize=11)
        ax.set_title(WD_LABELS[wd], fontsize=12, color=COLORS[wd])
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Training Dynamics: Sweet Spot vs Oversized Model (seed=42)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_dynamics.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure A: Memorization-Generalization Gap
# ---------------------------------------------------------------------------

def plot_memorization_gap(conn, output_dir):
    """Train acc vs val acc gap over time for key runs."""
    _style()

    runs = [
        ("exp5_ratio_wd0p1", 388, "d=388 (4x)"),
        ("exp5_ratio_wd0p1", 776, "d=776 (8x)"),
        ("exp5_ratio_wd0p1", 1552, "d=1552 (16x)"),
    ]
    colors = ["#66a61e", "#2166ac", "#b2182b"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for (exp, dim, label), color in zip(runs, colors):
        curve = query_training_curves(conn, exp, dim, seed=42)
        if not curve:
            continue
        epochs = np.array([r["epoch"] for r in curve])
        gap = np.array([r["train_acc"] - r["val_acc"] for r in curve])

        # Smooth
        smoothed = _smooth(gap, window=80)
        step = max(1, len(epochs) // 2000)

        ax.plot(epochs[::step], smoothed[::step], color=color, label=label,
                linewidth=2, alpha=0.9)

    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train Acc - Val Acc (Generalization Gap)", fontsize=12)
    ax.set_title("Memorization-Generalization Gap (WD=0.1, seed=42)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    path = os.path.join(output_dir, "fig_mem_gap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure B: Grok Transition Zoom -- Multiple seeds overlaid
# ---------------------------------------------------------------------------

def plot_grok_transition_zoom(conn, output_dir):
    """Zoom into the grokking transition for dim_1552 across seeds at WD=0.1."""
    _style()
    seeds = [42, 123, 456, 789, 1337]
    seed_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for seed, color in zip(seeds, seed_colors):
        curve = query_training_curves(conn, "exp5_ratio_wd0p1", 1552, seed)
        if not curve:
            continue
        epochs = np.array([r["epoch"] for r in curve])
        val_acc = np.array([r["val_acc"] for r in curve])
        smoothed = _smooth(val_acc, window=40)
        step = max(1, len(epochs) // 2000)
        ax.plot(epochs[::step], smoothed[::step], color=color,
                label=f"seed={seed}", linewidth=1.5, alpha=0.85)

    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5, label="Grok threshold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("Grokking Transition Variability: d=1552 Across Seeds (WD=0.1)", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.05, 1.05)

    path = os.path.join(output_dir, "fig_grok_zoom.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 8: Eddy Comparison -- Representation rank and structure over time
# ---------------------------------------------------------------------------

def plot_eddy_comparison(sweet_npz, oversized_npz, output_dir):
    """Effective rank + eigenvalue spectra: sweet spot vs oversized."""
    _style()
    sweet = np.load(sweet_npz)
    over = np.load(oversized_npz)

    def compute_rank_metrics(data):
        epochs = data["epochs"]
        cos_sims = data["cosine_similarities"]
        eff_ranks = []
        top_eigenratios = []
        for cs in cos_sims:
            eigvals = np.linalg.eigvalsh(cs)
            eigvals = np.maximum(eigvals, 0)  # numerical stability
            eigvals = eigvals / (eigvals.sum() + 1e-12)
            # Effective rank (exponential of entropy)
            nonzero = eigvals[eigvals > 1e-10]
            entropy = -np.sum(nonzero * np.log(nonzero))
            eff_ranks.append(np.exp(entropy))
            # Top eigenvalue ratio (how dominated by rank-1)
            sorted_eig = np.sort(eigvals)[::-1]
            top_eigenratios.append(sorted_eig[0])
        return epochs, eff_ranks, top_eigenratios

    s_epochs, s_ranks, s_top = compute_rank_metrics(sweet)
    o_epochs, o_ranks, o_top = compute_rank_metrics(over)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Effective rank over time
    ax1.plot(s_epochs, s_ranks, "o-", color="#2166ac", label="d=776 (sweet spot)",
             markersize=3, linewidth=1.5)
    ax1.plot(o_epochs, o_ranks, "s--", color="#b2182b", label="d=1552 (oversized)",
             markersize=3, linewidth=1.5)
    ax1.axhline(97, color="gray", linestyle=":", alpha=0.4, label="p=97 (max classes)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Effective Rank", fontsize=12)
    ax1.set_title("Representation Dimensionality Over Training", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Top eigenvalue fraction
    ax2.plot(s_epochs, s_top, "o-", color="#2166ac", label="d=776 (sweet spot)",
             markersize=3, linewidth=1.5)
    ax2.plot(o_epochs, o_top, "s--", color="#b2182b", label="d=1552 (oversized)",
             markersize=3, linewidth=1.5)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Top Eigenvalue Fraction", fontsize=12)
    ax2.set_title("Representation Collapse (1.0 = all classes identical)", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("Eddy Signature: Representation Structure (WD=0.1, seed=42)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_eddy_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 9: Eddy Divergence -- Representation quality over time
# ---------------------------------------------------------------------------

def plot_eddy_divergence(sweet_npz, oversized_npz, output_dir):
    """Track representation structure metrics over training for both models."""
    _style()
    sweet = np.load(sweet_npz)
    over = np.load(oversized_npz)

    def compute_metrics(data):
        epochs = data["epochs"]
        cos_sims = data["cosine_similarities"]
        p = cos_sims.shape[1]

        off_diag_means = []
        diag_stds = []
        etf_scores = []

        for cs in cos_sims:
            # Off-diagonal mean (ideal: -1/(p-1) for ETF)
            mask = ~np.eye(p, dtype=bool)
            off_diag_means.append(np.mean(cs[mask]))

            # Diagonal std (should be 0 for perfect ETF -- all 1s)
            diag_stds.append(np.std(np.diag(cs)))

            # ETF score: how close to equiangular tight frame
            # Perfect ETF has off-diag = -1/(p-1)
            ideal_off = -1.0 / (p - 1)
            etf_scores.append(1.0 - np.mean(np.abs(cs[mask] - ideal_off)))

        return epochs, off_diag_means, etf_scores

    s_epochs, s_offdiag, s_etf = compute_metrics(sweet)
    o_epochs, o_offdiag, o_etf = compute_metrics(over)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Off-diagonal mean
    ax1.plot(s_epochs, s_offdiag, "o-", color="#2166ac", label="d=776 (sweet spot)",
             markersize=3, linewidth=1.5)
    ax1.plot(o_epochs, o_offdiag, "s--", color="#b2182b", label="d=1552 (oversized)",
             markersize=3, linewidth=1.5)
    ax1.axhline(-1/96, color="gray", linestyle=":", alpha=0.5, label="Ideal ETF (-1/96)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Mean Off-Diagonal Cosine Similarity", fontsize=12)
    ax1.set_title("Class Separation Over Training", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # ETF score
    ax2.plot(s_epochs, s_etf, "o-", color="#2166ac", label="d=776 (sweet spot)",
             markersize=3, linewidth=1.5)
    ax2.plot(o_epochs, o_etf, "s--", color="#b2182b", label="d=1552 (oversized)",
             markersize=3, linewidth=1.5)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("ETF Score (1 = perfect)", fontsize=12)
    ax2.set_title("Equiangular Tight Frame Convergence", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Eddy Visualization: Representation Quality (WD=0.1, seed=42)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_eddy_divergence.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Legacy: CSV-based plots
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # --- DB-based figures ---
    if args.db and os.path.exists(args.db):
        conn = get_db(args.db)

        plot_u_shape(conn, args.output_dir)
        plot_seed_spread(conn, args.output_dir)
        plot_variance(conn, args.output_dir)
        plot_phase_boundary(conn, args.output_dir)
        plot_grok_surface(conn, args.output_dir)
        plot_reynolds_collapse(conn, args.output_dir)
        plot_training_dynamics(conn, args.output_dir)

        conn.close()

    # --- Legacy CSV-based figures ---
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

    if args.db and os.path.exists(args.db):
        conn2 = get_db(args.db)
        plot_memorization_gap(conn2, args.output_dir)
        plot_grok_transition_zoom(conn2, args.output_dir)
        conn2.close()

    if args.probe_npz:
        plot_etf_timelapse(
            args.probe_npz,
            output_path=os.path.join(args.output_dir, "fig4_etf_timelapse.png"),
            n_panels=args.n_panels,
        )

    # --- Eddy comparison ---
    if args.probe_sweet and args.probe_oversized:
        plot_eddy_comparison(args.probe_sweet, args.probe_oversized, args.output_dir)
        plot_eddy_divergence(args.probe_sweet, args.probe_oversized, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="experiments.db",
                        help="Path to experiments.db for DB-based figures")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="Path to sweep summary.csv for legacy figures 1-3")
    parser.add_argument("--probe_npz", type=str, default=None,
                        help="Path to probe_results.npz for figure 4")
    parser.add_argument("--n_panels", type=int, default=8)
    parser.add_argument("--probe_sweet", type=str, default=None,
                        help="Path to probe_results.npz for sweet-spot model")
    parser.add_argument("--probe_oversized", type=str, default=None,
                        help="Path to probe_results.npz for oversized model")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()
    main(args)
