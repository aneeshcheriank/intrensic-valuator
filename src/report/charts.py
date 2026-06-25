"""
Chart generation for the PDF report.

Generates matplotlib charts rendered to PNG bytes for embedding in the
reportlab PDF.  Two charts are produced:

1. Monte Carlo distribution histogram — shows the distribution of
   5,000 intrinsic values with percentile markers.
2. Scenario comparison bar chart — Bull / Base / Bear intrinsic values.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#CCCCCC",
}


def _apply_style() -> None:
    for k, v in _STYLE.items():
        plt.rcParams[k] = v


# ---------------------------------------------------------------------------
# Monte Carlo histogram
# ---------------------------------------------------------------------------


def monte_carlo_histogram(
    intrinsic_values: np.ndarray | list[float],
    current_price: float = 0.0,
    base_value: float = 0.0,
    figsize: tuple[int, int] = (8, 4.5),
    dpi: int = 150,
) -> io.BytesIO:
    """Generate a Monte Carlo distribution histogram.

    Parameters
    ----------
    intrinsic_values : array-like
        The 5,000+ simulated intrinsic values.
    current_price : float
        Current market price (vertical reference line).
    base_value : float
        Base case intrinsic value (vertical reference line).
    figsize : tuple
        Figure dimensions in inches.
    dpi : int
        Resolution.

    Returns
    -------
    io.BytesIO
        PNG image as a byte buffer (ready for reportlab Image).
    """
    _apply_style()

    values = np.asarray(intrinsic_values, dtype=float)
    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No finite simulation results", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Histogram
    bins = max(30, min(100, len(finite) // 50))
    ax.hist(finite, bins=bins, color="#1565C0", alpha=0.75, edgecolor="white", linewidth=0.3)

    # Percentile lines
    p10 = np.percentile(finite, 10)
    p50 = np.percentile(finite, 50)
    p90 = np.percentile(finite, 90)
    mean_val = np.mean(finite)

    ylim = ax.get_ylim()

    ax.axvline(p10, color="#EF5350", linestyle="--", linewidth=1.2, label=f"10th pctl: ${p10:.2f}")
    ax.axvline(p90, color="#43A047", linestyle="--", linewidth=1.2, label=f"90th pctl: ${p90:.2f}")
    ax.axvline(mean_val, color="#1565C0", linestyle="-", linewidth=1.5, label=f"Mean: ${mean_val:.2f}")

    if current_price > 0:
        ax.axvline(current_price, color="#FF6F00", linestyle=":", linewidth=1.5, label=f"Current: ${current_price:.2f}")

    ax.set_xlabel("Intrinsic Value per Share ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo Simulation — Distribution of Intrinsic Values")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(left=max(0, p10 * 0.7))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Scenario comparison bar chart
# ---------------------------------------------------------------------------


def scenario_comparison_chart(
    scenarios: dict,
    current_price: float = 0.0,
    figsize: tuple[int, int] = (6, 3.5),
    dpi: int = 150,
) -> io.BytesIO:
    """Generate a horizontal bar chart comparing Bull / Base / Bear scenarios.

    Parameters
    ----------
    scenarios : dict
        Dict with keys "Bull", "Base", "Bear", each containing
        ``intrinsic_value_per_share``.
    current_price : float
        Current market price.
    figsize : tuple
    dpi : int

    Returns
    -------
    io.BytesIO
        PNG image buffer.
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    labels = []
    values = []
    colors = []

    color_map = {"Bull": "#43A047", "Base": "#1565C0", "Bear": "#EF5350"}

    for name in ["Bull", "Base", "Bear"]:
        if name in scenarios:
            labels.append(name)
            values.append(scenarios[name].get("intrinsic_value_per_share", 0))
            colors.append(color_map.get(name, "#999999"))

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.5, edgecolor="white")

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"${val:.2f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    if current_price > 0:
        ax.axvline(current_price, color="#FF6F00", linestyle=":", linewidth=1.5, label=f"Current: ${current_price:.2f}")
        ax.legend(fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Intrinsic Value per Share ($)")
    ax.set_title("Scenario Analysis")
    ax.set_xlim(left=min(values + [current_price]) * 0.80 if values else 0)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf
