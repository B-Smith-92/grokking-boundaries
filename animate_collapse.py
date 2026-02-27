"""Animate the 97-class representation collapsing into structure over training.

Produces a rotating 3D animation (GIF) showing class centroids evolving
from random -> collapsed -> equiangular structure. Uses eigendecomposition
of the cosine similarity matrix to project to 3D.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def embed_3d(cos_sim):
    """Project cosine similarity matrix to 3D via top-3 eigenvectors."""
    eigvals, eigvecs = np.linalg.eigh(cos_sim)
    # Take top 3 (largest eigenvalues are at the end for eigh)
    idx = np.argsort(eigvals)[::-1][:3]
    coords = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))
    return coords


def main(args):
    data = np.load(args.probe_npz)
    epochs = data["epochs"]
    cos_sims = data["cosine_similarities"]
    p = cos_sims.shape[1]

    # Precompute all 3D embeddings
    all_coords = []
    for cs in cos_sims:
        all_coords.append(embed_3d(cs))
    all_coords = np.array(all_coords)  # (n_checkpoints, p, 3)

    # Align embeddings across time (Procrustes-like sign/permutation fix)
    # Just fix sign flips relative to previous frame
    for i in range(1, len(all_coords)):
        for dim in range(3):
            if np.dot(all_coords[i, :, dim], all_coords[i - 1, :, dim]) < 0:
                all_coords[i, :, dim] *= -1

    # Color by class label (cyclic colormap)
    colors = plt.cm.hsv(np.linspace(0, 1, p, endpoint=False))

    # Global axis limits
    pad = 1.1
    max_range = np.abs(all_coords).max() * pad

    # Set up figure
    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    # Style the 3D axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")
    ax.tick_params(colors="gray", labelsize=6)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color("gray")

    scatter = ax.scatter([], [], [], c=[], s=30, alpha=0.85, depthshade=True)
    title = ax.set_title("", color="white", fontsize=14, pad=20)

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # Subsample frames for reasonable GIF size
    n_frames = min(len(epochs), args.max_frames)
    frame_indices = np.linspace(0, len(epochs) - 1, n_frames, dtype=int)

    # Add slow rotation
    base_elev = 25

    def update(frame_num):
        idx = frame_indices[frame_num]
        coords = all_coords[idx]

        ax.cla()
        ax.set_facecolor("black")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333333")
        ax.yaxis.pane.set_edgecolor("#333333")
        ax.zaxis.pane.set_edgecolor("#333333")
        ax.tick_params(colors="gray", labelsize=6)
        ax.grid(True, alpha=0.1)

        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=colors, s=35, alpha=0.85, depthshade=True, edgecolors="none")

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        ax.view_init(elev=base_elev, azim=30)

        ax.set_title(f"Epoch {epochs[idx]:,}", color="white", fontsize=14, pad=20)

        return []

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=args.interval, blit=False)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.output.endswith(".gif"):
        ani.save(args.output, writer="pillow", dpi=args.dpi)
    elif args.output.endswith(".mp4"):
        ani.save(args.output, writer="ffmpeg", dpi=args.dpi)
    else:
        ani.save(args.output, dpi=args.dpi)

    plt.close(fig)
    print(f"Saved {args.output} ({n_frames} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_npz", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/collapse.gif")
    parser.add_argument("--max_frames", type=int, default=100,
                        help="Max frames in animation")
    parser.add_argument("--interval", type=int, default=120,
                        help="Milliseconds per frame")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    main(args)
