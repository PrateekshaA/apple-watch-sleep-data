"""Plot NN training curves and per-k test performance from saved model outputs.

Reads validation_search.json and metrics.csv from a compare_sleep_models.py
output directory and writes dependency-free SVGs. No test data influences
these plots' training curves; test-vs-history uses final test rows only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {"(64, 32)": "#2563eb", "(128, 64)": "#f97316"}
MODEL_COLORS = {"random_forest": "#16a34a", "neural_network": "#7c3aed",
                "original_rf": "#9ca3af", "majority_stage": "#6b7280"}
MODEL_LABELS = {"random_forest": "Random Forest", "neural_network": "Neural Network",
                "original_rf": "Original RF", "majority_stage": "Majority Stage"}
FONT_FAMILY = "Roboto, Arial, sans-serif"


def svg_frame(width, height, title, y_label, x_label, y_min, y_max, x_ticks, y_ticks=None):
    left, right, top, bottom = 95, 35, 70, 85
    # Use one consistent title size across anchored and walk-forward figures.
    title_font_size = 26
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="{title_font_size}" font-weight="bold">{title}</text>',
    ]
    y_ticks = y_ticks if y_ticks is not None else [y_min + (y_max - y_min) * tick / 5 for tick in range(6)]
    for value in y_ticks:
        y = top + (y_max - value) / (y_max - y_min) * (height - top - bottom)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        # Keep numeric tick labels clear of both the axis line and the vertical title.
        elements.append(f'<text x="{left-18}" y="{y+6:.1f}" text-anchor="end" font-family="{FONT_FAMILY}" font-size="20">{value:.3f}</text>')
    elements += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>',
        f'<text x="{left+(width-left-right)/2}" y="{height-16}" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="23">{x_label}</text>',
        f'<text x="20" y="{top+(height-top-bottom)/2}" text-anchor="middle" transform="rotate(-90 20 {top+(height-top-bottom)/2})" font-family="{FONT_FAMILY}" font-size="23">{y_label}</text>',
    ]
    for x, label in x_ticks:
        elements.append(f'<text x="{x:.1f}" y="{height-bottom+27}" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="20">{label}</text>')
    return elements, left, right, top, bottom


def training_curves(search, output, stage, mode):
    """Per-fold validation macro-F1 by epoch, plus per-batch training loss."""
    nn = [r for r in search if r["model"] == "neural_network"]
    if not nn:
        print("No neural-network search data found; skipping training curves.")
        return
    for hidden in ["(64, 32)", "(128, 64)"]:
        rows = [r for r in nn if tuple(r["hidden"]) == tuple(eval(hidden))]
        if not rows:
            continue
        width, height = 940, 590
        all_f1 = [r["macro_f1"] for r in rows if r.get("macro_f1") is not None]
        y_min = max(0.0, min(all_f1) - 0.03)
        y_max = max(all_f1) + 0.03
        max_epoch = max(r["epoch"] for r in rows)
        folds = sorted({r["fold"] for r in rows})
        ticks = [(95 + (width - 130) * e / max(max_epoch, 1), str(e))
                 for e in range(1, max_epoch + 1)]
        elements, left, right, top, bottom = svg_frame(
            width, height,
            f"MLP {hidden}: source-validation macro-F1 by epoch ({mode}, {stage})",
            "Macro-F1 (higher is better)", "Epoch", y_min, y_max, ticks)
        palette = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2"]
        legend = []
        for i, fold in enumerate(folds):
            fold_rows = sorted((r for r in rows if r["fold"] == fold and r.get("macro_f1") is not None),
                               key=lambda r: r["epoch"])
            xs = [left + (width - left - right) * r["epoch"] / max(max_epoch, 1) for r in fold_rows]
            ys = [top + (y_max - r["macro_f1"]) / (y_max - y_min) * (height - top - bottom) for r in fold_rows]
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
            color = palette[i % len(palette)]
            elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            best = max(fold_rows, key=lambda r: r["macro_f1"])
            bx = left + (width - left - right) * best["epoch"] / max(max_epoch, 1)
            by = top + (y_max - best["macro_f1"]) / (y_max - y_min) * (height - top - bottom)
            elements.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="5" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
            legend.append((f"fold {fold} (best ep {best['epoch']})", color))
        for i, (label, color) in enumerate(legend):
            y = 80 + i * 24
            elements.append(f'<line x1="{width-250}" y1="{y}" x2="{width-215}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            elements.append(f'<text x="{width-207}" y="{y+7}" font-family="{FONT_FAMILY}" font-size="18">{label}</text>')
        elements.append("</svg>")
        path = output / f"mlp_training_curve_{hidden.replace('(', '').replace(')', '').replace(', ', 'x')}.svg"
        path.write_text("\n".join(elements), encoding="utf-8")
        print(f"wrote {path}")


def batch_loss_curves(search, output, stage, mode):
    """Per-batch training loss: steep drop then plateau, log x-axis, per fold."""
    nn = [r for r in search if r["model"] == "neural_network" and r.get("train_loss") is not None]
    if not nn:
        print("No per-batch loss data found; skipping batch loss curves.")
        return
    for hidden in ["(64, 32)", "(128, 64)"]:
        rows = [r for r in nn if tuple(r["hidden"]) == tuple(eval(hidden))]
        if not rows:
            continue
        width, height = 940, 590
        losses = [r["train_loss"] for r in rows]
        y_min = max(0.0, min(losses) * 0.85)
        y_max = max(losses[:10]) * 1.1
        max_batch = max(r["global_batch"] for r in rows)
        folds = sorted({r["fold"] for r in rows})
        # log-scale x ticks
        import math
        decade = 10 ** math.floor(math.log10(max(max_batch, 2)))
        powers = [1]
        p = 1
        while p * 2 <= max_batch and p < decade:
            p *= 2
            powers.append(p)
        if decade > 1:
            powers += [decade] if decade <= max_batch else []
        seen = sorted(set(powers))
        x_pos = {}
        for value in seen:
            x_pos[value] = 95 + (math.log10(value) if value > 1 else 0) / max(math.log10(max(max_batch, 2)), 0.001) * (width - 130)
        ticks = [(x_pos[v], str(v)) for v in seen if v <= max_batch]
        elements, left, right, top, bottom = svg_frame(
            width, height,
            f"MLP {hidden}: per-batch training loss ({mode}, {stage})",
            "Weighted cross-entropy (lower is better)", "Mini-batch (log scale)",
            y_min, y_max, ticks)
        palette = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2"]
        legend = []
        max_log = max(math.log10(max(max_batch, 2)), 0.001)
        for i, fold in enumerate(folds):
            fold_rows = sorted((r for r in rows if r["fold"] == fold), key=lambda r: r["global_batch"])
            if not fold_rows:
                continue
            # light per-batch points
            color = palette[i % len(palette)]
            for r in fold_rows:
                x = left + math.log10(max(r["global_batch"], 2)) / max_log * (width - left - right)
                y = top + (y_max - min(r["train_loss"], y_max)) / (y_max - y_min) * (height - top - bottom)
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.6" fill="{color}" opacity="0.45"/>')
            # smoothed running mean over window 20
            vals = [r["train_loss"] for r in fold_rows]
            window = 20
            smooth = [np.mean(vals[max(0, j - window):j + 1]) for j in range(len(vals))]
            points = " ".join(
                f"{left + math.log10(max(r['global_batch'], 2)) / max_log * (width - left - right):.1f},"
                f"{top + (y_max - min(s, y_max)) / (y_max - y_min) * (height - top - bottom):.1f}"
                for r, s in zip(fold_rows, smooth))
            elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            legend.append((f"fold {fold} ({fold_rows[-1]['global_batch']} batches)", color))
        for i, (label, color) in enumerate(legend):
            y = 80 + i * 24
            elements.append(f'<line x1="{width-265}" y1="{y}" x2="{width-230}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            elements.append(f'<text x="{width-222}" y="{y+7}" font-family="{FONT_FAMILY}" font-size="18">{label}</text>')
        elements.append("</svg>")
        path = output / f"mlp_batch_loss_{hidden.replace('(', '').replace(')', '').replace(', ', 'x')}.svg"
        path.write_text("\n".join(elements), encoding="utf-8")
        print(f"wrote {path}")


def history_curves(metrics, output, stage, mode):
    """Final test accuracy and macro-F1 by personalization history k."""
    results = metrics[metrics.split == "test"]
    models = [m for m in ["random_forest", "neural_network", "majority_stage"]
              if m in set(results.model)]
    metric_labels = {"accuracy": "Accuracy", "macro_f1": "Macro-F1"}
    mode_label = mode.replace("_", "-").title()
    for metric in ["accuracy", "macro_f1"]:
        metric_label = metric_labels[metric]
        width, height = 940, 590
        # Keep every history curve on the same scale for direct visual comparison.
        y_min, y_max = 0.0, 0.8
        ks = sorted(results.history.unique())
        k_max = max(ks)
        ticks = [(95 + (width - 130) * k / k_max, str(k)) for k in ks]
        elements, left, right, top, bottom = svg_frame(
            width, height,
            f"Test {metric_label} by Personalization History ({mode_label}, {stage.title()})",
            metric_label if metric == "accuracy" else f"{metric_label} (Higher Is Better)",
            "Personalization History, k (Nights)",
            y_min, y_max, ticks, y_ticks=[0.0, 0.2, 0.4, 0.6, 0.8])
        legend = []
        label_positions = {}
        for model in models:
            group = results[results.model == model]
            per_k = group.groupby("history")[metric].agg(["mean", "count"])
            xs = [left + (width - left - right) * k / k_max for k in per_k.index]
            ys_ = [top + (y_max - v) / (y_max - y_min) * (height - top - bottom) for v in per_k["mean"]]
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys_))
            color = MODEL_COLORS[model]
            elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
            for k, x, y in zip(per_k.index, xs, ys_):
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
                label_positions[k] = min(label_positions.get(k, float("inf")), y)
            legend.append((MODEL_LABELS[model], color))
        subject_counts = results.groupby("history")["subject"].nunique()
        for k, n in subject_counts.items():
            x = left + (width - left - right) * k / k_max
            y = max(top + 14, label_positions[k] - 12)
            elements.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
                            f'font-family="{FONT_FAMILY}" font-size="17" fill="#4b5563">n={n}</text>')
        for i, (label, color) in enumerate(legend):
            y = 80 + i * 24
            elements.append(f'<line x1="{width-260}" y1="{y}" x2="{width-225}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            elements.append(f'<text x="{width-217}" y="{y+7}" font-family="{FONT_FAMILY}" font-size="18">{label}</text>')
        elements.append("</svg>")
        path = output / f"test_{metric}_by_history.svg"
        path.write_text("\n".join(elements), encoding="utf-8")
        print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("outputs/mode_anchored"))
    args = parser.parse_args()
    search = json.loads((args.directory / "validation_search.json").read_text())
    metrics = pd.read_csv(args.directory / "metrics.csv")
    config = json.loads((args.directory / "configuration.json").read_text())
    mode = config.get("mode", "anchored")
    stage = "k >= 1 personalization" if False else f"{metrics.subject.nunique()} subjects"
    training_curves(search, args.directory, stage, mode)
    batch_loss_curves(search, args.directory, stage, mode)
    history_curves(metrics, args.directory, stage, mode)


if __name__ == "__main__":
    main()
