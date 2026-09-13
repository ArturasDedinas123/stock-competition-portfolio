"""Charts for the notebooks: a colorblind-safe palette, quiet axes and one function per chart type."""

from collections.abc import Mapping

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcsetup
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, PercentFormatter

from .paths import IMAGES_DIR

COLORS = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "red": "#e34948"}
SERIES_COLORS = [COLORS["blue"], COLORS["orange"], COLORS["aqua"]]  # fixed order for up to three series
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
        "axes.prop_cycle": rcsetup.cycler(color=list(COLORS.values())),
        "figure.dpi": 110,
    })


def _as_series(values: pd.Series | pd.DataFrame) -> pd.Series:
    """A Series from a Series or a one-column DataFrame (``df["column"]`` is typed as either)."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("Expected a single column of values")
        return values.iloc[:, 0]
    return values


def save_figure(fig: Figure, name: str) -> None:
    """Save a figure to ``docs/images/<name>.png`` for the README."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def price_history_grid(prices: pd.DataFrame, tickers, window_start, window_end, columns: int = 5) -> Figure:
    """Each stock's full price history on a log scale, with the simulation window shaded."""
    rows = int(np.ceil(len(tickers) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 2.8 * rows), constrained_layout=True, squeeze=False)
    dollars = FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:.2f}")
    for ax, ticker in zip(axes.flat, tickers):
        series = prices[ticker].dropna()
        ax.axvspan(float(mdates.date2num(window_start)), float(mdates.date2num(window_end)),
                   color=INK["grid"], alpha=0.7, linewidth=0, zorder=0)
        ax.plot(series.index, series, color=COLORS["blue"], linewidth=1.2)
        ax.set_yscale("log")
        decades = np.log10(series.max() / series.min())
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,) if decades > 2 else (1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(dollars)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=8)
        ax.set_title(f"{ticker} · since {series.index[0]:%Y}", fontsize=10)
    for ax in list(axes.flat)[len(tickers):]:
        ax.set_visible(False)
    fig.suptitle("Adjusted price, full history (log scale). Shaded: simulation window",
                 x=0.005, ha="left", fontsize=11, color=INK["secondary"])
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str) -> Figure:
    """Correlation matrix with the value printed in every cell."""
    values = corr.to_numpy()
    labels = list(corr.columns)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    image = ax.imshow(values, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, j in np.ndindex(values.shape):
        value = float(values[i, j])
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(value) > 0.6 else INK["primary"])
    fig.colorbar(image, ax=ax, shrink=0.8, label="Correlation of daily returns")
    ax.set_title(title)
    return fig


def distributions(series: Mapping[str, np.ndarray], xlabel: str, title: str, bins: int = 80) -> Figure:
    """Up to three return distributions as outline histograms on shared bins."""
    values = list(series.values())
    edges = np.linspace(min(np.percentile(v, 0.2) for v in values), max(np.percentile(v, 99.8) for v in values), bins)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for (label, data), color in zip(series.items(), SERIES_COLORS):
        ax.hist(data, bins=edges, histtype="step", linewidth=1.5, color=color, label=label, density=True)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.legend(loc="upper right")
    ax.set_title(title)
    return fig


def win_loss_scatter(win: np.ndarray, loss: np.ndarray, highlights: Mapping[str, int], n_rivals: int,
                     sample: int = 40_000, seed: int = 0) -> Figure:
    """P(win) against P(loss) for a random sample of portfolios, with up to three portfolios highlighted."""
    rows = np.random.default_rng(seed).choice(len(win), min(sample, len(win)), replace=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(win[rows], loss[rows], s=4, color=INK["axis"], alpha=0.35, linewidths=0,
               label="Allowed portfolios (sample)")
    for (label, position), color in zip(highlights.items(), SERIES_COLORS):
        ax.scatter(win[position], loss[position], s=70, color=color, edgecolors=INK["surface"], linewidths=1.5,
                   zorder=3, label=label)
        ax.annotate(label, (float(win[position]), float(loss[position])), xytext=(8, 4), textcoords="offset points",
                    fontsize=9, color=INK["primary"])
    ax.margins(x=0.12)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(f"P(win): beat all {n_rivals} rivals")
    ax.set_ylabel("P(loss): end below the purchase value")
    ax.legend(loc="lower right")
    ax.set_title("Chance of winning vs. chance of losing money")
    return fig


def grouped_bars(frame: pd.DataFrame, ylabel: str, title: str) -> Figure:
    """One group of bars per row (dates), one bar per column (up to three series)."""
    x = np.arange(len(frame))
    width = 0.8 / len(frame.columns)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k, (column, color) in enumerate(zip(frame.columns, SERIES_COLORS)):
        ax.bar(x + (k - (len(frame.columns) - 1) / 2) * width, frame[column], width=width - 0.03, color=color,
               label=str(column))
    ax.axhline(0, color=INK["axis"], linewidth=0.8)
    ax.set_xticks(x, [f"{d:%b %y}" for d in frame.index], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=len(frame.columns))
    ax.set_title(title)
    return fig


def ranking_bars(values: pd.Series | pd.DataFrame, title: str, xlabel: str, reference: float | None = None,
                 reference_label: str = "", percent: bool = False) -> Figure:
    """Horizontal bars, largest at the top, with an optional reference line."""
    ordered = _as_series(values).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(ordered) + 1.2))
    ax.barh([str(i) for i in ordered.index], ordered.to_numpy(), height=0.7, color=COLORS["blue"])
    if reference is not None:
        ax.axvline(reference, color=INK["secondary"], linewidth=1)
        ax.annotate(reference_label, (reference, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -2),
                    textcoords="offset points", va="top", fontsize=9, color=INK["secondary"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    if percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return fig


def return_lines(series: Mapping[str, pd.Series], end, title: str) -> Figure:
    """Cumulative returns over time (up to three series), with the axis running to ``end``."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    first = next(iter(series.values()))
    for (label, returns), color in zip(series.items(), SERIES_COLORS):
        ax.plot(returns.index, returns, color=color, label=label, marker="o" if len(returns) == 1 else None)
    latest = float(first.iloc[-1])
    ax.annotate(f"{latest:+.1%}", (float(mdates.date2num(first.index[-1])), latest), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK["primary"])
    ax.axhline(0, color=INK["axis"], linewidth=0.8)
    ax.set_xlim(float(mdates.date2num(first.index[0])), float(mdates.date2num(pd.Timestamp(end))))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=len(series))
    ax.set_title(title)
    return fig


def contribution_bars(contribution: pd.Series | pd.DataFrame, title: str) -> Figure:
    """Horizontal bars: blue for positive contributions, red for negative ones."""
    ordered = _as_series(contribution).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([str(i) for i in ordered.index], ordered.to_numpy(), height=0.7,
            color=[COLORS["blue"] if v >= 0 else COLORS["red"] for v in ordered.to_numpy()])
    ax.axvline(0, color=INK["axis"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2%}"))
    ax.set_title(title)
    return fig
