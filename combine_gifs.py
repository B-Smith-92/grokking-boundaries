"""Combine two GIFs side-by-side with metric overlays from experiments.db."""

import os
import sqlite3

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_metrics_from_db(db_path, run_id, probe_epochs):
    """Pull train_acc, val_acc, val_loss at probe epochs from the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    train_acc, val_acc, val_loss = [], [], []
    for ep in probe_epochs:
        # Metrics are every 10 epochs; find the nearest
        cur.execute("""
            SELECT train_acc, val_acc, val_loss FROM metrics
            WHERE run_id = ? ORDER BY ABS(epoch - ?) LIMIT 1
        """, (run_id, int(ep)))
        row = cur.fetchone()
        train_acc.append(row[0])
        val_acc.append(row[1])
        val_loss.append(row[2])

    conn.close()
    return np.array(train_acc), np.array(val_acc), np.array(val_loss)


def draw_metrics(draw, x_offset, y_offset, train_acc, val_acc, val_loss, label, panel_w):
    """Draw metric text overlay on one panel."""
    try:
        font_label = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
        font_metric = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
    except OSError:
        font_label = ImageFont.load_default()
        font_metric = font_label

    # Panel label at top
    draw.text((x_offset + 8, y_offset + 4), label, fill="#AAAAAA", font=font_label)

    # Metrics at bottom-left of panel
    y = y_offset + 295
    spacing = 16

    # Color-code accuracy: green >= 0.95, yellow >= 0.5, red otherwise
    if val_acc >= 0.95:
        val_color = "#4ADE80"
    elif val_acc >= 0.5:
        val_color = "#FACC15"
    else:
        val_color = "#F87171"

    train_color = "#4ADE80" if train_acc >= 0.95 else "#FACC15" if train_acc >= 0.5 else "#F87171"

    lines = [
        (f"Train Acc  {train_acc:6.1%}", train_color),
        (f"Val Acc    {val_acc:6.1%}", val_color),
        (f"Val Loss   {val_loss:6.3f}", "#93C5FD"),
    ]

    for text, color in lines:
        draw.text((x_offset + 8, y), text, fill=color, font=font_metric)
        y += spacing


def combine_gifs(left_path, right_path, output_path,
                 db_path=None, left_run_id=None, right_run_id=None,
                 left_probe_path=None, right_probe_path=None,
                 left_label="Left", right_label="Right",
                 target_width=800):
    left = Image.open(left_path)
    right = Image.open(right_path)
    n_frames = min(left.n_frames, right.n_frames)

    # Load probe epochs and metrics from DB
    left_data = right_data = None
    if db_path and left_run_id and left_probe_path:
        epochs = np.load(left_probe_path)["epochs"]
        left_data = load_metrics_from_db(db_path, left_run_id, epochs)

    if db_path and right_run_id and right_probe_path:
        epochs = np.load(right_probe_path)["epochs"]
        right_data = load_metrics_from_db(db_path, right_run_id, epochs)

    # Scale panels
    panel_w = target_width // 2
    scale = panel_w / left.size[0]
    panel_h = int(left.size[1] * scale)
    canvas_size = (panel_w * 2, panel_h)

    frames = []
    durations = []

    for i in range(n_frames):
        left.seek(i)
        right.seek(i)

        l_frame = left.convert("RGBA").resize((panel_w, panel_h), Image.LANCZOS)
        r_frame = right.convert("RGBA").resize((panel_w, panel_h), Image.LANCZOS)

        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
        canvas.paste(l_frame, (0, 0))
        canvas.paste(r_frame, (panel_w, 0))

        draw = ImageDraw.Draw(canvas)

        if left_data:
            draw_metrics(draw, 0, 0,
                         left_data[0][i], left_data[1][i], left_data[2][i],
                         left_label, panel_w)
        if right_data:
            draw_metrics(draw, panel_w, 0,
                         right_data[0][i], right_data[1][i], right_data[2][i],
                         right_label, panel_w)

        # Thin separator line
        draw.line([(panel_w, 0), (panel_w, panel_h)], fill="#444444", width=1)

        frames.append(canvas.convert("RGB"))
        durations.append(left.info.get("duration", 120))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved {output_path}: {canvas_size[0]}x{canvas_size[1]}, {n_frames} frames, {size_kb:.0f} KB")


if __name__ == "__main__":
    base = "/Users/brendansmith/Documents/GitHub/grokking-boundaries"
    combine_gifs(
        left_path=f"{base}/figures/collapse_oversized.gif",
        right_path=f"{base}/figures/collapse_sweetspot.gif",
        output_path=f"{base}/figures/collapse_combined.gif",
        db_path=f"{base}/experiments.db",
        left_run_id=131,   # dim_1552, wd=0.1, seed=42, ratio=16x
        right_run_id=161,  # dim_776,  wd=0.1, seed=42, ratio=8x
        left_probe_path=f"{base}/figures/probes/wd0p1_dim1552/probe_results.npz",
        right_probe_path=f"{base}/figures/probes/wd0p1_dim776/probe_results.npz",
        left_label="Oversized (d=1552, ratio=16x)",
        right_label="Sweet Spot (d=776, ratio=8x)",
    )
