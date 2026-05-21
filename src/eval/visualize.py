from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def niah_heatmap(
    grid: dict[float, dict[int, float]],
    output_path: Path,
    title: str = "Needle-in-a-Haystack: Retrieval Rate",
) -> None:
    """2-D heatmap: X = context length, Y = needle depth, color = hit rate."""
    depths = sorted(grid.keys())
    lengths = sorted({l for d in grid.values() for l in d.keys()})

    data = np.array([[grid[d].get(l, float("nan")) for l in lengths] for d in depths])

    fig, ax = plt.subplots(figsize=(len(lengths) * 1.6 + 1.5, len(depths) * 1.2 + 1.5))
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([f"{l // 1000}K" if l >= 1000 else str(l) for l in lengths], fontsize=10)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"{int(d * 100)}%" for d in depths], fontsize=10)
    ax.set_xlabel("Context Length (tokens)", fontsize=11)
    ax.set_ylabel("Needle Depth", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(len(depths)):
        for j in range(len(lengths)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "black" if 0.35 < val < 0.75 else "white"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Hit Rate", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def score_distribution(
    scores: list[float],
    metric_name: str,
    output_path: Path,
    title: str = "Score Distribution",
) -> None:
    """Histogram + box-plot of per-sample scores."""
    arr = np.array(scores)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(arr, bins=25, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].axvline(arr.mean(), color="crimson", linestyle="--", linewidth=1.5,
                    label=f"Mean {arr.mean():.3f}")
    axes[0].axvline(np.median(arr), color="darkorange", linestyle=":", linewidth=1.5,
                    label=f"Median {np.median(arr):.3f}")
    axes[0].set_xlabel(metric_name, fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Distribution", fontsize=12)
    axes[0].legend(fontsize=9)

    bp = axes[1].boxplot(arr, patch_artist=True, widths=0.4)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["medians"][0].set_color("white")
    axes[1].set_ylabel(metric_name, fontsize=11)
    axes[1].set_title("Box Plot", fontsize=12)
    axes[1].set_xticks([])

    fig.suptitle(f"{title}  (n={len(scores)})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def longbench_bar(
    task_scores: dict[str, float],
    output_path: Path,
    title: str = "LongBench Scores by Task (Full Context Baseline)",
) -> None:
    """Horizontal bar chart of per-task scores."""
    tasks = list(task_scores.keys())
    scores = [task_scores[t] for t in tasks]

    fig, ax = plt.subplots(figsize=(8, max(3, len(tasks) * 0.55 + 1)))
    colors = ["#4c8cbf" if s >= 0.4 else "#e07b54" for s in scores]
    bars = ax.barh(tasks, scores, color=colors, edgecolor="white", height=0.6)

    for bar, score in zip(bars, scores):
        ax.text(
            min(score + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=9,
        )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score (F1 / ROUGE-L)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(0.4, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def overview_summary(summaries: dict, output_path: Path) -> None:
    """Single-figure overview across all benchmarks run in this session."""
    panels = []

    if "niah" in summaries:
        panels.append(("NIAH\nHit Rate", summaries["niah"].get("overall_hit_rate", 0.0), "#5ab4ac"))

    if "narrativeqa" in summaries:
        panels.append(("NarrativeQA\nToken F1", summaries["narrativeqa"].get("mean_f1", 0.0), "#8da0cb"))

    if "longbench" in summaries:
        lb = summaries["longbench"]
        for task, score in lb.items():
            if isinstance(score, (int, float)):
                panels.append((f"LB\n{task}", score, "#fc8d62"))

    if not panels:
        return

    labels, values, colors = zip(*panels)
    fig, ax = plt.subplots(figsize=(max(6, len(panels) * 1.2), 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Baseline (Full Context) — All Benchmarks", fontsize=13, fontweight="bold")
    ax.axhline(0.4, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
