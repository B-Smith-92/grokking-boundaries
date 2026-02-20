"""Extract per-class feature representations and cosine similarity matrices."""

import argparse
import json
import os

import numpy as np
import torch

from model import GrokTransformer


def probe_checkpoint(model, checkpoint_path, inputs, device):
    """Load a checkpoint and return the K x K class-mean cosine similarity."""
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                     weights_only=True))
    model.eval()

    p = model.p
    labels = ((inputs[:, 0] + inputs[:, 1]) % p)

    with torch.no_grad():
        _, features = model(inputs.to(device), return_features=True)

    # per-class mean features
    class_means = torch.zeros(p, features.shape[1], device=device)
    for k in range(p):
        mask = labels == k
        class_means[k] = features[mask.to(device)].mean(dim=0)

    # normalise and compute cosine similarity
    norms = class_means.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = class_means / norms
    cos_sim = (normed @ normed.T).cpu().numpy()

    return cos_sim


def main(args):
    output_dir = args.output_dir or os.path.join(args.run_dir, "probes")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(args.run_dir, "metrics.json")) as f:
        result = json.load(f)

    ra = result["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GrokTransformer(
        p=ra["prime"],
        d_model=ra["d_model"],
        n_heads=ra["n_heads"],
        n_layers=ra["n_layers"],
    ).to(device)

    # full dataset
    pairs = []
    for a in range(ra["prime"]):
        for b in range(ra["prime"]):
            pairs.append([a, b])
    inputs = torch.tensor(pairs)

    checkpoints = sorted([
        f for f in os.listdir(args.run_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ])
    print(f"Found {len(checkpoints)} checkpoints")

    epochs = []
    cos_sims = []
    for name in checkpoints:
        epoch = int(name.replace("checkpoint_", "").replace(".pt", ""))
        path = os.path.join(args.run_dir, name)
        cs = probe_checkpoint(model, path, inputs, device)
        epochs.append(epoch)
        cos_sims.append(cs)
        print(f"  epoch {epoch}")

    np.savez(
        os.path.join(output_dir, "probe_results.npz"),
        epochs=np.array(epochs),
        cosine_similarities=np.stack(cos_sims),
    )
    print(f"Saved to {output_dir}/probe_results.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directory with checkpoints and metrics.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save (default: <run_dir>/probes)")
    args = parser.parse_args()
    main(args)
