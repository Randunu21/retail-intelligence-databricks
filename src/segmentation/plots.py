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
