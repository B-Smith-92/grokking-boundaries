# Grokking Boundaries

Mapping the phase transition between memorization and generalization in neural networks trained on modular arithmetic.

## What is grokking?

A small transformer learns `(a + b) mod p`. Early in training it memorizes the training set (100% train accuracy, chance-level validation). Then, long after memorization, validation accuracy suddenly jumps to ~100%. This delayed generalization is **grokking**.

This project systematically maps **where and why** that transition happens by sweeping model capacity, task complexity, and regularization strength.

## The core question

How does the ratio of model capacity to task complexity (`d_model / prime`) interact with regularization (`weight_decay`) to determine the grokking boundary?

We're building a 3D surface: **ratio x weight_decay -> grok_epoch** to find:
- The critical ratio below which models fail to grok
- Whether excess capacity without regularization *delays* grokking (the U-shape hypothesis)
- How the phase boundary shifts with regularization strength

## Model

- 2-layer transformer, 4 attention heads, GELU activations
- Token + positional embeddings, mean-pooled features, linear classification head
- Full-batch training with AdamW (lr=0.001, betas=0.9/0.98)
- 50% train/val split on all `p^2` input pairs

## Experiments

| Experiment | Sweeps | Fixed | Epochs | Runs |
|---|---|---|---|---|
| `exp1_weight_decay` | WD: 0.1, 0.5, 1.0, 3.0, 10.0 | p=97, d=128 | 50k | 5 |
| `exp2_hidden_dim` | d: 32, 64, 128, 256 | p=97, WD=1.0 | 50k | 4 |
| `exp3_prime` | p: 23, 47, 59, 97 | d=128, WD=1.0 | 50k | 4 |
| `exp4_matched` | (p,d) pairs | WD=1.0 | 20k | 4 |
| `exp5_ratio_wd1` | d: 8 ratios x 5 seeds | p=97, WD=1.0 | 20k | 40 |
| `exp5_ratio_wd0p1` | d: 8 ratios x 5 seeds | p=97, WD=0.1 | 100k | 40 |
| `exp5_ratio_wd0p01` | d: 8 ratios x 5 seeds | p=97, WD=0.01 | 200k | 40 |
| `prime53_ratio` | d: 11 dims | p=53, WD=1.0 | 20k | 11 |

Extended ratio sweeps (8x, 16x) are in progress across all three weight decay levels.

## Grokking detection

Grokking is detected when validation accuracy stays at or above 95% for 10 consecutive evaluation steps (every 10 epochs). The epoch of the first step in that window is recorded as `grok_epoch`.

## Project structure

```
train.py        # Single training run
model.py        # GrokTransformer architecture
sweep.py        # Experiment orchestration (parallel seed support)
build_db.py     # SQLite database builder (3 tables, 4 views)
plot.py         # Figures from sweep results
probe.py        # Per-class cosine similarity analysis of learned representations
```

## Database

`build_db.py` ingests all experiment results into `experiments.db`:

- **`experiments`** — one row per experiment group, encodes what's fixed vs swept
- **`runs`** — one row per training execution, fully denormalized (no joins needed for filtering)
- **`metrics`** — per-epoch timeseries (~1M+ rows), clustered on `(run_id, epoch)`

Four views for common queries:
- `run_summary` — runs joined with experiment metadata, includes `grokked` flag
- `ratio_sweep_stats` — mean/min/max grok_epoch aggregated across seeds
- `phase_boundary` — grok success rate grouped by (weight_decay, ratio)
- `experiment_overview` — run counts and fastest/slowest grok per experiment

```bash
python build_db.py            # clean build
python build_db.py --update   # incremental (re-ingest changed runs)
```

## Usage

```bash
pip install -r requirements.txt

# Single training run
python train.py --prime 97 --d_model 128 --weight_decay 1.0 --epochs 50000

# Run a full experiment sweep
python sweep.py --experiment optimal_ratio --weight_decay 0.1 --epochs 100000 --workers 2

# Extended ratios (8x, 16x) — won't re-run existing dims
python sweep.py --experiment extended_ratio --weight_decay 1.0 --epochs 20000

# Build analysis database
python build_db.py

# Probe learned representations
python probe.py --run_dir sweep_results/exp5_optimal_ratio/dim_192_seed_42
```

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- numpy, scipy, matplotlib
