"""Matplotlib figures for Phase 3, returned as Figure objects so they can be shown in the
notebook AND logged with `mlflow.log_figure` (no files written by hand).

matplotlib is imported inside each function: it's on Databricks serverless, but it isn't a
local dev dependency, and the rest of the package must import without it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def correlation_heatmap(corr: Mapping[tuple[str, str], float], columns: Sequence[str], title: str):
    import matplotlib.pyplot as plt

    n = len(columns)
    grid = [[corr[(a, b)] for b in columns] for a in columns]
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.0 * n + 1.5))
    im = ax.imshow(grid, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n), labels=columns, rotation=45, ha="right")
    ax.set_yticks(range(n), labels=columns)
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if abs(v) > 0.6 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def metric_by_k(rows, metric: str, ylabel: str, title: str, marks: dict | None = None):
    """One line per algorithm (rows = flat result dicts). `marks` = {algorithm: k} to circle."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for algo in sorted({r["algorithm"] for r in rows}):
        pts = sorted(
            (r["k"], r[metric])
            for r in rows
            if r["algorithm"] == algo and r.get(metric) is not None and not r.get("error")
        )
        if not pts:
            continue
        ks, vs = zip(*pts, strict=True)
        (line,) = ax.plot(ks, vs, marker="o", label=algo)
        if marks and marks.get(algo) in ks:
            k = marks[algo]
            ax.scatter(
                [k],
                [dict(pts)[k]],
                s=180,
                facecolors="none",
                edgecolors=line.get_color(),
                linewidths=2,
            )
    ax.set_xlabel("k (number of segments)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def share_bars(shares_by_label: dict, title: str):
    """Bar chart of segment shares, {label: share}."""
    import matplotlib.pyplot as plt

    labels = [str(k) for k in shares_by_label]
    values = [100 * v for v in shares_by_label.values()]
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(labels) + 2), 3.5))
    bars = ax.bar(labels, values, color="#4C72B0")
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], fontsize=8)
    ax.set_ylabel("% of customers")
    ax.set_title(title)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def ari_bars(rows, threshold: float, title: str):
    """Grouped bars: mean seed ARI and mean bootstrap ARI per (algorithm, k); whiskers to min."""
    import matplotlib.pyplot as plt

    rows = [r for r in rows if not r.get("error")]
    labels = [f"{r['algorithm']}\nk={r['k']}" for r in rows]
    x = list(range(len(rows)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(rows) + 2), 4))
    for offset, key, name in [(-w / 2, "seed", "seeds"), (w / 2, "bootstrap", "bootstrap")]:
        means = [r[f"{key}_ari_mean"] for r in rows]
        mins = [r[f"{key}_ari_min"] for r in rows]
        err = [[m - lo for m, lo in zip(means, mins, strict=True)], [0] * len(rows)]
        ax.bar(
            [i + offset for i in x],
            means,
            w,
            yerr=err,
            capsize=3,
            label=f"{name} (mean, whisker = min)",
        )
    ax.axhline(
        threshold, color="black", linestyle="--", linewidth=1, label=f"threshold {threshold}"
    )
    ax.set_xticks(x, labels=labels, fontsize=8)
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.05)
    ax.set_ylabel("adjusted Rand index")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig


def centroid_heatmap(centers, features, segment_labels, title: str):
    """Segment centres on the standardised scale: 0 = average customer, +1 = one sd above."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.3 * len(features) + 2.5, 0.7 * len(centers) + 1.8))
    lim = max(1.0, max(abs(v) for c in centers for v in c))
    im = ax.imshow(centers, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(features)), labels=features, rotation=35, ha="right")
    ax.set_yticks(range(len(centers)), labels=segment_labels)
    for i, row in enumerate(centers):
        for j, v in enumerate(row):
            ax.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(v) > 0.6 * lim else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="sd from average (after log/clip)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
