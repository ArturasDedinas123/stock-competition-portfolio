"""Shared chart style: a colorblind-safe palette, quiet axes and helpers for saving figures."""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from .paths import IMAGES_DIR

COLORS = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "red": "#e34948"}
INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
CORR_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8c8b", "#b52a2a"]
)


def set_chart_style() -> None:
    """Apply the project's matplotlib style."""
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK["primary"],
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK["axis"],
        "ytick.color": INK["axis"],
        "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"],
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=list(COLORS.values())),
        "figure.dpi": 110,
    })


def save_figure(fig, name: str) -> None:
    """Save a figure to ``docs/images/<name>.png`` for the README."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
