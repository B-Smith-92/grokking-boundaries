"""
Build SQLite database from grokking-boundaries experiment results.

Schema: 3 tables (experiments, runs, metrics) + 4 views.
Classification uses args.output_dir from each metrics.json, not path parsing.
"""

import sqlite3
import json
import glob
import os
import sys
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Experiment definitions: keyed by canonical name
# NULL fields in the experiments table mean that parameter is swept.
# ---------------------------------------------------------------------------

EXPERIMENT_DEFS = {
    "exp1_weight_decay": {
        "description": "Weight decay sweep (p=97, d=128)",
        "sweep_params": "weight_decay",
        "prime": 97,
        "d_model": 128,
        "weight_decay": None,  # swept
        "epochs": 50000,
        "seeds": [42],
    },
    "exp2_hidden_dim": {
        "description": "Hidden dimension sweep (p=97)",
        "sweep_params": "d_model",
        "prime": 97,
        "d_model": None,  # swept
        "weight_decay": 1.0,
        "epochs": 50000,
        "seeds": [42],
    },
    "exp3_prime": {
        "description": "Prime sweep (d=128)",
        "sweep_params": "prime",
        "prime": None,  # swept
        "d_model": 128,
        "weight_decay": 1.0,
        "epochs": 50000,
        "seeds": [42],
    },
    "exp4_matched": {
        "description": "Matched capacity across primes",
        "sweep_params": "prime,d_model",
        "prime": None,  # swept
        "d_model": None,  # swept
        "weight_decay": 1.0,
        "epochs": 20000,
        "seeds": [42],
    },
    "exp5_ratio_wd1": {
        "description": "Ratio sweep at WD=1.0 (p=97)",
        "sweep_params": "d_model",
        "prime": 97,
        "d_model": None,  # swept
        "weight_decay": 1.0,
        "epochs": 20000,
        "seeds": [42, 123, 456, 789, 1337],
    },
    "exp5_ratio_wd0p01": {
        "description": "Ratio sweep at WD=0.01 (p=97)",
        "sweep_params": "d_model",
        "prime": 97,
        "d_model": None,  # swept
        "weight_decay": 0.01,
        "epochs": 200000,
        "seeds": [42, 123, 456, 789, 1337],
    },
    "exp5_ratio_wd0p1": {
        "description": "Ratio sweep at WD=0.1 (p=97)",
        "sweep_params": "d_model",
        "prime": 97,
        "d_model": None,  # swept
        "weight_decay": 0.1,
        "epochs": 100000,
        "seeds": [42, 123, 456, 789, 1337],
    },
    "prime53_ratio": {
        "description": "Ratio sweep for p=53",
        "sweep_params": "d_model",
        "prime": 53,
        "d_model": None,  # swept
        "weight_decay": 1.0,
        "epochs": 20000,
        "seeds": [42],
    },
}

# Fixed across all experiments
FIXED_PARAMS = {
    "n_heads": 4,
    "n_layers": 2,
    "lr": 0.001,
    "frac_train": 0.5,
}


def classify_experiment(output_dir):
    """Map args.output_dir to an experiment name. Returns None if unrecognized."""
    # Normalize to forward slashes
    odir = output_dir.replace("\\", "/")

    # Order matters: check wd0p01/wd0p1 before the bare exp5_optimal_ratio
    if odir.startswith("sweep_results/exp5_optimal_ratio_wd0p01"):
        return "exp5_ratio_wd0p01"
    if odir.startswith("sweep_results/exp5_optimal_ratio_wd0p1"):
        return "exp5_ratio_wd0p1"
    if odir.startswith("sweep_results/exp5_optimal_ratio"):
        return "exp5_ratio_wd1"
    if odir.startswith("sweep_results/exp1_weight_decay"):
        return "exp1_weight_decay"
    if odir.startswith("sweep_results/exp2_hidden_dim"):
        return "exp2_hidden_dim"
    if odir.startswith("sweep_results/exp3_prime"):
        return "exp3_prime"
    if odir.startswith("sweep_results/exp4_matched"):
        return "exp4_matched"
    if odir.startswith("runs/prime53"):
        return "prime53_ratio"
    return None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    description TEXT,
    sweep_params TEXT,
    prime       INTEGER,
    d_model     INTEGER,
    weight_decay REAL,
    n_heads     INTEGER NOT NULL DEFAULT 4,
    n_layers    INTEGER NOT NULL DEFAULT 2,
    lr          REAL    NOT NULL DEFAULT 0.001,
    frac_train  REAL    NOT NULL DEFAULT 0.5,
    epochs      INTEGER NOT NULL,
    seeds       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    prime           INTEGER NOT NULL,
    d_model         INTEGER NOT NULL,
    weight_decay    REAL    NOT NULL,
    seed            INTEGER NOT NULL,
    ratio           REAL    NOT NULL,
    grok_epoch      INTEGER,
    final_train_loss REAL,
    final_train_acc  REAL,
    final_val_loss   REAL,
    final_val_acc    REAL,
    source_file     TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    epoch       INTEGER NOT NULL,
    train_loss  REAL,
    train_acc   REAL,
    val_loss    REAL,
    val_acc     REAL,
    PRIMARY KEY (run_id, epoch)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_runs_experiment   ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_ratio        ON runs(ratio);
CREATE INDEX IF NOT EXISTS idx_runs_weight_decay ON runs(weight_decay);
CREATE INDEX IF NOT EXISTS idx_runs_grok_epoch   ON runs(grok_epoch);
CREATE INDEX IF NOT EXISTS idx_runs_prime_dmodel ON runs(prime, d_model);
"""

VIEWS_SQL = """
-- View 1: run_summary — joins runs + experiments, adds grokked flag and max_epochs
CREATE VIEW IF NOT EXISTS run_summary AS
SELECT
    r.id            AS run_id,
    e.name          AS experiment,
    r.prime,
    r.d_model,
    r.weight_decay,
    r.seed,
    r.ratio,
    r.grok_epoch,
    r.grok_epoch IS NOT NULL AS grokked,
    e.epochs        AS max_epochs,
    r.final_train_loss,
    r.final_train_acc,
    r.final_val_loss,
    r.final_val_acc
FROM runs r
JOIN experiments e ON e.id = r.experiment_id;

-- View 2: ratio_sweep_stats — aggregates across seeds per (experiment, weight_decay, ratio)
CREATE VIEW IF NOT EXISTS ratio_sweep_stats AS
SELECT
    e.name          AS experiment,
    r.weight_decay,
    r.prime,
    r.d_model,
    r.ratio,
    COUNT(*)        AS n_runs,
    SUM(r.grok_epoch IS NOT NULL)   AS n_grokked,
    AVG(r.grok_epoch)               AS mean_grok_epoch,
    MIN(r.grok_epoch)               AS min_grok_epoch,
    MAX(r.grok_epoch)               AS max_grok_epoch
FROM runs r
JOIN experiments e ON e.id = r.experiment_id
GROUP BY e.name, r.weight_decay, r.prime, r.d_model;

-- View 3: phase_boundary — groups by (weight_decay, ratio): n_grokked vs n_failed
CREATE VIEW IF NOT EXISTS phase_boundary AS
SELECT
    r.weight_decay,
    r.ratio,
    r.prime,
    COUNT(*)                         AS n_total,
    SUM(r.grok_epoch IS NOT NULL)    AS n_grokked,
    SUM(r.grok_epoch IS NULL)        AS n_failed,
    ROUND(100.0 * SUM(r.grok_epoch IS NOT NULL) / COUNT(*), 1) AS pct_grokked,
    AVG(r.grok_epoch)               AS mean_grok_epoch
FROM runs r
GROUP BY r.weight_decay, r.ratio, r.prime;

-- View 4: experiment_overview — dashboard per experiment
CREATE VIEW IF NOT EXISTS experiment_overview AS
SELECT
    e.name          AS experiment,
    e.description,
    e.sweep_params,
    e.epochs        AS max_epochs,
    COUNT(r.id)     AS n_runs,
    SUM(r.grok_epoch IS NOT NULL) AS n_grokked,
    SUM(r.grok_epoch IS NULL)     AS n_failed,
    MIN(r.grok_epoch)             AS fastest_grok,
    MAX(r.grok_epoch)             AS slowest_grok
FROM experiments e
LEFT JOIN runs r ON r.experiment_id = e.id
GROUP BY e.id
ORDER BY e.id;
"""


def create_schema(conn):
    """Create tables, indices, and views."""
    conn.executescript(SCHEMA_SQL)
    # Drop and recreate views so they stay current with code changes
    for view in ("run_summary", "ratio_sweep_stats", "phase_boundary", "experiment_overview"):
        conn.execute(f"DROP VIEW IF EXISTS {view}")
    conn.executescript(VIEWS_SQL)
    conn.commit()


def seed_experiments(conn):
    """Insert experiment rows from EXPERIMENT_DEFS if they don't exist."""
    for name, defn in EXPERIMENT_DEFS.items():
        conn.execute("""
            INSERT OR IGNORE INTO experiments
                (name, description, sweep_params, prime, d_model, weight_decay,
                 n_heads, n_layers, lr, frac_train, epochs, seeds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            defn["description"],
            defn["sweep_params"],
            defn["prime"],
            defn["d_model"],
            defn["weight_decay"],
            FIXED_PARAMS["n_heads"],
            FIXED_PARAMS["n_layers"],
            FIXED_PARAMS["lr"],
            FIXED_PARAMS["frac_train"],
            defn["epochs"],
            json.dumps(defn["seeds"]),
        ))
    conn.commit()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def find_metrics_files(base_path):
    """Find all metrics*.json files under base_path."""
    return sorted(glob.glob(os.path.join(base_path, "**", "metrics*.json"), recursive=True))


def load_metrics_file(filepath):
    """Load and validate a metrics JSON file. Returns (data, error_msg)."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return None, str(e)

    if "args" not in data:
        return None, "missing 'args' key"
    if "metrics" not in data:
        return None, "missing 'metrics' key"
    return data, None


def ingest(conn, base_path, update=False):
    """Ingest all metrics files. Returns (new_runs, updated_runs, skipped, errors)."""
    c = conn.cursor()

    # Build experiment name → id lookup
    exp_ids = {}
    for row in c.execute("SELECT id, name FROM experiments"):
        exp_ids[row[1]] = row[0]

    # Pre-load existing source_files for skip/update logic
    existing = {}  # source_file → (run_id, grok_epoch, metric_count)
    for row in c.execute("""
        SELECT r.id, r.source_file, r.grok_epoch,
               (SELECT COUNT(*) FROM metrics m WHERE m.run_id = r.id)
        FROM runs r
    """):
        existing[row[1]] = (row[0], row[2], row[3])

    files = find_metrics_files(base_path)
    new_runs = 0
    updated_runs = 0
    skipped = 0
    errors = []

    for filepath in files:
        # Normalize the source_file path for consistent dedup
        source_file = os.path.relpath(filepath, base_path)

        # Check if already ingested
        if source_file in existing:
            if not update:
                skipped += 1
                continue
            # Update mode: check if data changed
            run_id, old_grok, old_metric_count = existing[source_file]
            data, err = load_metrics_file(filepath)
            if err:
                errors.append((filepath, err))
                continue

            new_grok = data.get("grok_epoch")
            new_metric_count = len(data.get("metrics", []))

            if new_grok == old_grok and new_metric_count == old_metric_count:
                skipped += 1
                continue

            # Re-ingest: delete old run + metrics, then fall through to insert
            c.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            c.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            del existing[source_file]
            updated_runs += 1
        else:
            data, err = load_metrics_file(filepath)
            if err:
                errors.append((filepath, err))
                continue

        args = data["args"]
        metrics = data["metrics"]

        # Classify by output_dir
        output_dir = args.get("output_dir", "")
        exp_name = classify_experiment(output_dir)
        if exp_name is None:
            errors.append((filepath, f"unrecognized output_dir: {output_dir}"))
            continue

        exp_id = exp_ids.get(exp_name)
        if exp_id is None:
            errors.append((filepath, f"no experiment row for: {exp_name}"))
            continue

        prime = args["prime"]
        d_model = args["d_model"]
        ratio = round(d_model / prime, 4) if prime else 0.0

        final = metrics[-1] if metrics else {}

        c.execute("""
            INSERT INTO runs
                (experiment_id, prime, d_model, weight_decay, seed, ratio,
                 grok_epoch, final_train_loss, final_train_acc,
                 final_val_loss, final_val_acc, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exp_id,
            prime,
            d_model,
            args["weight_decay"],
            args["seed"],
            ratio,
            data.get("grok_epoch"),
            final.get("train_loss"),
            final.get("train_acc"),
            final.get("val_loss"),
            final.get("val_acc"),
            source_file,
        ))
        run_id = c.lastrowid

        if source_file not in existing:
            new_runs += 1

        # Bulk insert metrics
        metric_rows = [
            (run_id, m["epoch"], m.get("train_loss"), m.get("train_acc"),
             m.get("val_loss"), m.get("val_acc"))
            for m in metrics
        ]
        c.executemany("""
            INSERT OR IGNORE INTO metrics
                (run_id, epoch, train_loss, train_acc, val_loss, val_acc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, metric_rows)

    conn.commit()
    return new_runs, updated_runs, skipped, errors


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(conn):
    c = conn.cursor()

    print("\n" + "=" * 70)
    print("DATABASE SUMMARY")
    print("=" * 70)

    print("\n  Experiments:")
    print("  " + "-" * 66)
    rows = c.execute("SELECT * FROM experiment_overview").fetchall()
    cols = [d[0] for d in c.description]
    for row in rows:
        r = dict(zip(cols, row))
        grok_info = ""
        if r["n_grokked"]:
            grok_info = f", grokked={r['n_grokked']}"
            if r["fastest_grok"] is not None:
                grok_info += f" (fastest={r['fastest_grok']}, slowest={r['slowest_grok']})"
        print(f"    {r['experiment']:25s}  {r['n_runs']:3d} runs{grok_info}")
        print(f"      {r['description']}  [sweep: {r['sweep_params']}, epochs: {r['max_epochs']}]")

    total_runs = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    total_grokked = c.execute("SELECT COUNT(*) FROM runs WHERE grok_epoch IS NOT NULL").fetchone()[0]
    total_metrics = c.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]

    print(f"\n  Totals: {total_runs} runs ({total_grokked} grokked), "
          f"{total_metrics:,} metric points")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build SQLite database from grokking-boundaries experiment results.")
    parser.add_argument("--db", type=str, default="experiments.db",
                        help="Output database filename (default: experiments.db)")
    parser.add_argument("--update", action="store_true",
                        help="Re-ingest runs where grok_epoch or metric count changed")
    args = parser.parse_args()

    base_path = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_path, args.db)

    # Remove existing DB for a clean build (unless --update)
    if not args.update and os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing {args.db}")

    print(f"Building database: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    create_schema(conn)
    seed_experiments(conn)

    # Ingest from all result directories
    print(f"\nScanning for metrics files...")
    new, updated, skipped, errors = ingest(conn, base_path, update=args.update)

    print(f"  New runs:     {new}")
    if args.update:
        print(f"  Updated runs: {updated}")
    print(f"  Skipped:      {skipped}")
    if errors:
        print(f"  Errors:       {len(errors)}")
        for filepath, err in errors:
            print(f"    {filepath}: {err}")

    print_summary(conn)
    conn.close()
    print(f"\nDone: {db_path}")


if __name__ == "__main__":
    main()
