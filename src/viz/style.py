"""Project-wide visual style constants and helpers.
Import COLORS and call apply_house_style() at the top of any script
that produces figures. This ensures every chart in the project uses
the same palette, fonts, and layout so the deck looks cohesive.
Color meanings are fixed across the entire project:
  teal-blue  = approval-friendly, safe, stable
  brick-red  = denial-friendly, risky, drifted
  slate-gray = neutral context
  navy       = baseline model (WOE logistic regression)
  gold       = advanced model (FT-Transformer)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS: dict[str, str] = {
    "favorable":   "#2E86AB",   # teal-blue: approval-friendly / stable
    "unfavorable": "#C73E1D",   # brick-red: denial-friendly / drifted / risky
    "neutral":     "#5C6B73",   # slate-gray: context
    "baseline":    "#1F3A5F",   # navy: logistic regression series
    "advanced":    "#D4A017",   # gold: FT-Transformer series
    "grid":        "#E5E7EB",   # light gray: background grid lines
}

def apply_house_style() -> None:
    """Set matplotlib defaults for a consistent, clean look.
    Call once at the top of any script that makes figures.
    """
    mpl.rcParams.update({
        "figure.dpi":           120,
        "savefig.dpi":          200,
        "savefig.bbox":         "tight",
        "font.family":          "DejaVu Sans",
        "font.size":            11,
        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "axes.labelsize":       11,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "grid.color":           COLORS["grid"],
        "grid.linestyle":       "-",
        "grid.linewidth":       0.6,
        "legend.frameon":       False,
    })

def annotate_takeaway(ax, text: str, loc: str = "upper right") -> None:
    """Place a yellow callout box with the chart's main takeaway.
    Every headline figure should include a 1-2 sentence plain-English
    annotation so a non-technical reader can get the point at a glance.
    """
    bbox = dict(
        boxstyle="round,pad=0.4",
        fc="#FFF8DC",
        ec=COLORS["neutral"],
        lw=0.8,
    )
    locs = {
        "upper right":  (0.98, 0.97, "right",  "top"),
        "upper left":   (0.02, 0.97, "left",   "top"),
        "upper center": (0.50, 0.97, "center", "top"),
        "center right": (0.98, 0.50, "right",  "center"),
        "lower right":  (0.98, 0.03, "right",  "bottom"),
        "lower left":   (0.02, 0.03, "left",   "bottom"),
    }
    x, y, ha, va = locs[loc]
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=10,
        bbox=bbox,
        zorder=10,
    )
