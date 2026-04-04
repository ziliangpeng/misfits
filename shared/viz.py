"""Shared visualization style and chart utilities."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ────────────────────────────────────────
# Theme constants
# ────────────────────────────────────────

DARK_STYLE: dict = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.alpha": 0.3,
    "font.size": 10,
}

# Standard color palette keyed by architectural component
COMPONENT_COLORS: dict[str, str] = {
    "attention": "#4fc3f7",
    "mlp_dense": "#ff8a65",
    "mlp_expert": "#ce93d8",
    "mlp_shared": "#81c784",
    "lm_head": "#fff176",
    "other": "#90a4ae",
}


# ────────────────────────────────────────
# Style helpers
# ────────────────────────────────────────

def init_style() -> None:
    """Apply the dark theme to all subsequent matplotlib figures.

    Call this at the start of every chart-generation function to ensure
    consistent styling. It updates ``plt.rcParams`` in-place.
    """
    plt.rcParams.update(DARK_STYLE)


def save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk and close it.

    Creates any missing parent directories automatically. Always passes
    ``bbox_inches="tight"`` and preserves the figure's own facecolor so
    that dark-background figures are not cropped or recolored.

    Args:
        fig:  The figure to save.
        path: Destination file path (e.g. ``"output/01_kurtosis.png"``).
        dpi:  Output resolution in dots per inch. Defaults to 150.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
