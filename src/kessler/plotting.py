"""Shared matplotlib style (validated colour-blind-safe categorical order)."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e0"
SEQ = "Blues"

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.titleweight": "semibold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=COLORS),
        "xtick.color": TEXT2, "ytick.color": TEXT2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": TEXT2,
        "lines.linewidth": 2.0, "font.family": "sans-serif", "figure.dpi": 110,
        "text.color": TEXT,
    })


def save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.relative_to(FIG_DIR.parent)}")
    return path
