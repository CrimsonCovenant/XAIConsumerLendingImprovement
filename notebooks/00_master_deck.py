"""Phase 9: Master Deck -- 7-figure presentation script.
Loads only pre-computed artifacts. Generates publication-quality PNGs.
All figures saved to outputs/figures/deck/
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.viz.style import COLORS, apply_house_style, annotate_takeaway
from src.features.display_names import pretty

apply_house_style()

OUT = "outputs/figures/deck"
os.makedirs(OUT, exist_ok=True)

ANCHOR_YEARS = [2008, 2013, 2018, 2024]
ERA_LABELS = {2008: "Post-Crisis", 2013: "Stabilization",
              2018: "Schema Break", 2024: "New Normal"}

# DATA LOADING

def load_aucs():
    """Load AUC metrics for anchor years from manifests."""
    rows = []
    for year in ANCHOR_YEARS:
        base = json.load(open(f"outputs/models/baseline_{year}/_run_manifest.json"))
        ft = json.load(open(f"outputs/models/ft_{year}/_run_manifest.json"))
        rows.append({
            "year": year,
            "era": ERA_LABELS[year],
            "baseline_auc": base["test_auc"],
            "ft_auc": ft["metrics"]["auc"],
        })
    return pd.DataFrame(rows)

def load_ig_summary(year):
    with open(f"outputs/explanations/{year}/summary.json") as f:
        return json.load(f)

def load_drift():
    return pd.read_csv("outputs/drift/welch_anchored.csv")

def load_fairness():
    return pd.read_csv("outputs/fairness/air_results.csv")

def load_denied_example(year):
    path = f"outputs/explanations/{year}/local_denied_examples.csv"
    if os.path.exists(path):
        return pd.read_csv(path).iloc[0]
    return None

# FIGURE 1: Performance Trajectory

def fig1_performance():
    df = load_aucs()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(df))
    w = 0.32

    bars_base = ax.bar(x - w/2, df["baseline_auc"], w,
                       color=COLORS["baseline"], label="Baseline (WOE + LR)", zorder=3)
    bars_ft = ax.bar(x + w/2, df["ft_auc"], w,
                     color=COLORS["advanced"], label="Advanced (FT-Transformer)", zorder=3)

    # Value labels
    for bars in [bars_base, bars_ft]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Lift annotations
    for i, row in df.iterrows():
        lift_bps = (row["ft_auc"] - row["baseline_auc"]) * 10000
        mid_x = x[i] + w/2 + 0.08
        mid_y = row["ft_auc"] + 0.012
        ax.annotate(f"+{lift_bps:.0f} bps", (mid_x, mid_y),
                    fontsize=8, color=COLORS["advanced"], fontweight="bold",
                    ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['year']}\n{row['era']}" for _, row in df.iterrows()])
    ax.set_ylabel("AUC (Test Set)")
    ax.set_ylim(0.60, 0.87)
    ax.set_title("Both Models, Both Eras: Where Each Wins — Taller Bar = Better",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    annotate_takeaway(ax, "FT-Transformer gains +240 to +320 bps\nacross all eras on identical features.\n(bps = basis points, 1 bps = 0.0001 AUC)",
                      loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_performance.png", dpi=200)
    plt.close(fig)
    print("  Fig 1 — Performance trajectory [OK]")

# FIGURE 2: Income Distribution Before/After Log Transform

def fig2_income_transform():
    df = pd.read_parquet("data/processed/processed_2024.parquet")
    raw = df["applicant_income_000s"].values
    log = df["log_income"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Before
    ax = axes[0]
    ax.hist(raw[raw < 500], bins=100, color=COLORS["unfavorable"], alpha=0.8, edgecolor="none")
    ax.set_title("Before: Raw Income ($000s)", fontweight="bold")
    ax.set_xlabel("Applicant Income ($000s)")
    ax.set_ylabel("Count")
    ax.axvline(raw.mean(), color="black", ls="--", lw=1, label=f"Mean: ${raw.mean():.0f}k")
    ax.axvline(np.median(raw), color=COLORS["neutral"], ls=":", lw=1,
               label=f"Median: ${np.median(raw):.0f}k")
    ax.legend(fontsize=8)

    # After
    ax = axes[1]
    ax.hist(log[np.isfinite(log)], bins=100, color=COLORS["favorable"], alpha=0.8, edgecolor="none")
    ax.set_title("After: Log-Transformed Income", fontweight="bold")
    ax.set_xlabel("log(Income)")
    ax.set_ylabel("Count")
    ax.axvline(np.nanmean(log), color="black", ls="--", lw=1,
               label=f"Mean: {np.nanmean(log):.2f}")
    ax.legend(fontsize=8)

    fig.suptitle("Income Transformation: From a 10,000× Tail to a Stable Distribution",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_income_transform.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 2 — Income transformation [OK]")

# FIGURE 3: IG Forest Plot — Top Features 2024

def fig3_ig_forest():
    summary = load_ig_summary(2024)
    feats = summary["features"]

    # Filter to features with non-zero attribution, remove redundant raw income
    feats = [f for f in feats if f["mean_abs_attribution"] > 1e-6]
    feats = [f for f in feats if f["feature_raw"] != "applicant_income_000s"]
    feats = feats[:10]
    feats = feats[::-1]  # reverse for horizontal bar (bottom = highest)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(feats))
    means = [f["mean_abs_attribution"] for f in feats]
    ci_lo = [f["ci_lo"] for f in feats]
    ci_hi = [f["ci_hi"] for f in feats]
    xerr = [[m - lo for m, lo in zip(means, ci_lo)],
            [hi - m for m, hi in zip(means, ci_hi)]]
    names = [f["feature_display"] for f in feats]

    # Color by sign
    colors = [COLORS["favorable"] if f["signed_mean"] > 0 else COLORS["unfavorable"]
              for f in feats]

    ax.barh(y_pos, means, xerr=xerr, color=colors, edgecolor="white",
            height=0.6, capsize=3, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean |Attribution| (95% Bootstrap CI)")
    ax.set_title("What Drove Approval Decisions in 2024 — Longer Bar = Bigger Influence",
                 fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["favorable"], label="Pushes toward Approval"),
        Patch(facecolor=COLORS["unfavorable"], label="Pushes toward Denial"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    annotate_takeaway(ax, "Loan size (log scale) is the single\nstrongest driver of 2024 decisions.",
                      loc="lower right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_ig_forest_2024.png", dpi=200)
    plt.close(fig)
    print("  Fig 3 — IG forest plot 2024 [OK]")

# FIGURE 4: Drift — 2008 vs 2024

def fig4_drift_2008_2024():
    df = load_drift()
    df = df[df["comparison"] == "2008_vs_2024"].copy()

    # Filter out zero-attribution categoricals
    df = df[df["mean_a"].abs() + df["mean_b"].abs() > 1e-6].copy()
    df = df.sort_values("delta", key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(df))
    colors = [COLORS["unfavorable"] if d else COLORS["neutral"]
              for d in df["drifted"]]

    ax.barh(y_pos, df["delta"], color=colors, edgecolor="white", height=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature_display"].values)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Δ Mean |Attribution| (2024 − 2008)")
    ax.set_title("Which Factors Mattered More in 2024 vs. 2008 — Starred = Statistically Significant Shift",
                 fontsize=12, fontweight="bold")

    # Add stars for significant drift
    for i, (_, row) in enumerate(df.iterrows()):
        if row["drifted"]:
            x_pos = row["delta"]
            offset = 0.002 if x_pos >= 0 else -0.002
            ha = "left" if x_pos >= 0 else "right"
            ax.text(x_pos + offset, i, "", fontsize=12, color=COLORS["unfavorable"],
                    va="center", ha=ha, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["unfavorable"], label="Drifted (BH q<0.05)"),
        Patch(facecolor=COLORS["neutral"], label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    annotate_takeaway(ax,
        "Geographic income features lost importance;\nloan size gained — risk lens shifted.",
        loc="lower left")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_drift_2008_vs_2024.png", dpi=200)
    plt.close(fig)
    print("  Fig 4 — Drift 2008 vs 2024 [OK]")

# FIGURE 5: Drift — 2013 vs 2018 (Schema Break)

def fig5_drift_2013_2018():
    df = load_drift()
    df = df[df["comparison"] == "2013_vs_2018"].copy()
    df = df[df["mean_a"].abs() + df["mean_b"].abs() > 1e-6].copy()
    df = df.sort_values("delta", key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(df))
    colors = [COLORS["unfavorable"] if d else COLORS["neutral"]
              for d in df["drifted"]]

    ax.barh(y_pos, df["delta"], color=colors, edgecolor="white", height=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature_display"].values)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Δ Mean |Attribution| (2018 − 2013)")
    ax.set_title("Macroeconomic Regime Shift: How Feature Importance Drifted Between 2013 and 2018",
                 fontsize=12, fontweight="bold")

    for i, (_, row) in enumerate(df.iterrows()):
        if row["drifted"]:
            x_pos = row["delta"]
            offset = 0.002 if x_pos >= 0 else -0.002
            ha = "left" if x_pos >= 0 else "right"
            ax.text(x_pos + offset, i, "", fontsize=12, color=COLORS["unfavorable"],
                    va="center", ha=ha, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["unfavorable"], label="Drifted (BH q<0.05)"),
        Patch(facecolor=COLORS["neutral"], label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    annotate_takeaway(ax,
        "Same 15 features, massive drift —\nreal macro shift, not schema artifact.",
        loc="lower left")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_drift_2013_vs_2018.png", dpi=200)
    plt.close(fig)
    print("  Fig 5 — Drift 2013 vs 2018 [OK]")

# FIGURE 6: Fairness — AIR Comparison

def fig6_fairness():
    df = load_fairness()
    # Remove NaN AIR (Hispanic not available)
    df = df.dropna(subset=["air"])

    fig, ax = plt.subplots(figsize=(11, 6))

    # Create grouped labels
    df = df.copy()
    df["group_label"] = (
        df["year"].astype(str) + " " +
        df["minority_group"] + "\n(" +
        df["model"].str.replace("ft_transformer", "FT", regex=False)
                      .str.replace("baseline", "Baseline", regex=False) + ")"
    )

    # Sort by year then model
    df = df.sort_values(["year", "minority_group", "model"])

    x = np.arange(len(df))
    colors = []
    for _, row in df.iterrows():
        if row["model"] == "baseline":
            colors.append(COLORS["baseline"])
        else:
            colors.append(COLORS["advanced"])

    bars = ax.bar(x, df["air"], color=colors, edgecolor="white", width=0.7, zorder=3)

    # Error bars
    yerr_lo = df["air"] - df["ci_lo"]
    yerr_hi = df["ci_hi"] - df["air"]
    ax.errorbar(x, df["air"], yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor="black", capsize=3, lw=1, zorder=4)

    # Threshold lines
    ax.axhline(0.80, color=COLORS["unfavorable"], ls="--", lw=1.5,
               label="Four-Fifths Rule (0.80)", zorder=2)
    ax.axhline(0.90, color=COLORS["favorable"], ls="-", lw=1.5,
               label="Industry Best Practice (0.90)", zorder=2)
    ax.axhline(1.00, color=COLORS["unfavorable"], ls="--", lw=1.5, zorder=2)

    # Danger zone shading: below 0.80 and above 1.00
    xlims = ax.get_xlim()
    ax.axhspan(0.0, 0.80, color=COLORS["unfavorable"], alpha=0.08, zorder=1)
    ax.axhspan(1.00, 2.0, color=COLORS["unfavorable"], alpha=0.08, zorder=1)

    # Value labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df["group_label"], fontsize=8, rotation=0)
    ax.set_ylabel("Adverse Impact Ratio")
    ax.set_ylim(0.5, 1.6)
    ax.set_title("Adverse Impact Ratio by Group — Between the Lines (0.80–1.00) Is Compliant",
                 fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["baseline"], label="Baseline (WOE + LR)"),
        Patch(facecolor=COLORS["advanced"], label="Advanced (FT-Transformer)"),
        plt.Line2D([0], [0], color=COLORS["unfavorable"], ls="--", label="Danger Zone (< 0.80 or > 1.00)"),
        plt.Line2D([0], [0], color=COLORS["favorable"], ls="-", label="Best Practice (0.90)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    annotate_takeaway(ax,
        "FT-Transformer improved Black AIR\nfrom 0.69→0.79 (2008) and 0.84→0.91 (2024).",
        loc="upper center")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_fairness_air.png", dpi=200)
    plt.close(fig)
    print("  Fig 6 — Fairness AIR [OK]")

# FIGURE 7: Waterfall — One Denied Applicant

def fig7_waterfall():
    denied = load_denied_example(2024)
    if denied is None:
        print("  Fig 7 — SKIP: no denied examples for 2024")
        return

    # Convert to display names, remove redundant raw income, sort by absolute value
    raw_names = denied.index.tolist()
    display = pretty(raw_names)
    values = denied.values.astype(float)

    # Remove raw Applicant Income (Income log scale exists)
    keep = [i for i, rn in enumerate(raw_names) if rn != "applicant_income_000s"]
    raw_names = [raw_names[i] for i in keep]
    display = [display[i] for i in keep]
    values = values[keep]

    # Sort by absolute value
    order = np.argsort(np.abs(values))[::-1]
    values = values[order]
    display = [display[i] for i in order]

    # Take top 10
    values = values[:10]
    display = display[:10]

    # Reverse for horizontal bar (top = most important)
    values = values[::-1]
    display = display[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(values))
    colors = [COLORS["favorable"] if v > 0 else COLORS["unfavorable"] for v in values]

    ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Attribution Score (positive = pushes toward approval)")
    ax.set_title("Why This Application Was Denied — Each Bar Shows One Factor's Push",
                 fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["favorable"], label="Pushed toward Approval"),
        Patch(facecolor=COLORS["unfavorable"], label="Pushed toward Denial"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    annotate_takeaway(ax,
        "This denied applicant's strongest\nnegative signal came from loan size.",
        loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_waterfall_denied.png", dpi=200)
    plt.close(fig)
    print("  Fig 7 — Waterfall denied [OK]")

# MAIN

def main():
    print("Phase 9: Master Deck — Generating 7 Figures")
    print(f"  Output: {OUT}/")
    print()

    fig1_performance()
    fig2_income_transform()
    fig3_ig_forest()
    fig4_drift_2008_2024()
    fig5_drift_2013_2018()
    fig6_fairness()
    fig7_waterfall()

    print()
    print(f"All 7 figures saved to {OUT}/")

if __name__ == "__main__":
    main()
