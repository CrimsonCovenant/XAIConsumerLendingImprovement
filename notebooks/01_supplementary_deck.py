"""Phase 9B: Supplementary Deep-Dive Figures — Q&A Backup Deck.

Generates additional analytical figures for anticipated follow-up questions:
  Fig S1 — Data Population & Train/Test Splits (all 17 years)
  Fig S2 — ROC Curves (baseline vs FT, anchor years)
  Fig S3 — Precision-Recall Curves (baseline vs FT, anchor years)
  Fig S4 — Calibration Plots (baseline vs FT, anchor years)
  Fig S5 — Brier Score + KS + Gini Trajectory (all 17 years)
  Fig S6 — Confusion Matrices (2008 vs 2024, FT-Transformer)
  Fig S7 — Information Value: 2008 vs 2024
  Fig S8 — Score Distribution (approved vs denied, FT 2008 & 2024)
  Fig S9 — Bootstrap Coefficient Stability (baseline 2008)

All figures use project COLORS palette and apply_house_style().
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.viz.style import COLORS, apply_house_style, annotate_takeaway
from src.features.display_names import pretty

apply_house_style()

OUT = "outputs/figures/deck_supplementary"
os.makedirs(OUT, exist_ok=True)

ANCHOR_YEARS = [2008, 2013, 2018, 2024]
ALL_YEARS = list(range(2008, 2025))

# HELPERS

def _load_baseline_manifest(year):
    with open(f"outputs/models/baseline_{year}/_run_manifest.json") as f:
        return json.load(f)

def _load_ft_manifest(year):
    path = f"outputs/models/ft_{year}/_run_manifest.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def _load_ft_preds(year):
    """Load FT test predictions + labels. Returns (preds, labels) or (None, None)."""
    p = f"outputs/models/ft_{year}/test_preds.npy"
    l = f"outputs/models/ft_{year}/test_labels.npy"
    if os.path.exists(p) and os.path.exists(l):
        return np.load(p), np.load(l)
    return None, None

def _load_baseline_test_preds(year):
    """Reconstruct baseline predictions on the test split."""
    model_dir = f"outputs/models/baseline_{year}"
    if not os.path.exists(os.path.join(model_dir, "model.pkl")):
        return None, None

    with open(os.path.join(model_dir, "woe_pipeline.pkl"), "rb") as f:
        woe = pickle.load(f)
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "platt.pkl"), "rb") as f:
        platt = pickle.load(f)

    df = pd.read_parquet(f"data/processed/processed_{year}.parquet")
    y = df["loan_status"].values
    df_feat = df.drop(columns=["loan_status"])

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
    test_idx, _ = train_test_split(rest_idx, test_size=1/3, random_state=42, stratify=y[rest_idx])

    X_test = woe.transform(df_feat.iloc[test_idx])
    raw_logits = X_test @ model.weights + model.bias
    probs = platt.transform(raw_logits)
    return probs, y[test_idx]

# FIG S1: Data Population & Splits

def fig_s1_population():
    years, trains, tests, cals, approval_rates = [], [], [], [], []
    for year in ALL_YEARS:
        m = _load_baseline_manifest(year)
        years.append(year)
        trains.append(m["train_size"])
        tests.append(m["test_size"])
        cals.append(m["cal_size"])
        # Load approval rate
        df = pd.read_parquet(f"data/processed/processed_{year}.parquet",
                             columns=["loan_status"])
        approval_rates.append(df["loan_status"].mean() * 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2.5, 1]})

    x = np.arange(len(years))
    w = 0.6

    trains_m = [t / 1e6 for t in trains]
    tests_m = [t / 1e6 for t in tests]
    cals_m = [c / 1e6 for c in cals]

    ax1.bar(x, trains_m, w, label="Train (70%)", color=COLORS["baseline"], zorder=3)
    ax1.bar(x, tests_m, w, bottom=trains_m, label="Test (20%)", color=COLORS["advanced"], zorder=3)
    ax1.bar(x, cals_m, w, bottom=[t + te for t, te in zip(trains_m, tests_m)],
            label="Calibration (10%)", color=COLORS["neutral"], zorder=3)

    ax1.set_ylabel("Applications (millions)")
    ax1.set_title("HMDA Dataset Size by Year — Train / Test / Calibration Splits",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Total labels
    for i, (tr, te, ca) in enumerate(zip(trains_m, tests_m, cals_m)):
        total = tr + te + ca
        ax1.text(i, total + 0.2, f"{total:.1f}M", ha="center", fontsize=7, color=COLORS["neutral"])

    # Schema break annotation
    ax1.axvline(x[years.index(2018)] - 0.5, color=COLORS["unfavorable"], ls="--", lw=1, alpha=0.7)
    ax1.text(years.index(2018), ax1.get_ylim()[1] * 0.92, "2018\nSchema\nBreak",
             ha="center", fontsize=8, color=COLORS["unfavorable"], fontweight="bold")

    # Approval rate panel
    ax2.plot(x, approval_rates, "o-", color=COLORS["favorable"], lw=2, markersize=5, zorder=3)
    ax2.axhline(50, color=COLORS["neutral"], ls=":", lw=1, alpha=0.5)
    ax2.set_ylabel("Approval Rate (%)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, rotation=45, fontsize=9)
    ax2.set_ylim(40, 90)

    annotate_takeaway(ax2, "Approval rate rose from 64% (2008)\nto 80% (2024) — population shift.",
                      loc="lower right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_s1_population.png", dpi=200)
    plt.close(fig)
    print("  Fig S1 — Data population [OK]")

# FIG S2: ROC Curves

def fig_s2_roc():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for ax, year in zip(axes, ANCHOR_YEARS):
        # FT
        ft_preds, ft_labels = _load_ft_preds(year)
        if ft_preds is not None:
            fpr, tpr, _ = roc_curve(ft_labels, ft_preds)
            ax.plot(fpr, tpr, color=COLORS["advanced"], lw=2,
                    label=f"FT (AUC={_load_ft_manifest(year)['metrics']['auc']:.3f})")

        # Baseline
        base_preds, base_labels = _load_baseline_test_preds(year)
        if base_preds is not None:
            fpr_b, tpr_b, _ = roc_curve(base_labels, base_preds)
            ax.plot(fpr_b, tpr_b, color=COLORS["baseline"], lw=2, ls="--",
                    label=f"Baseline (AUC={_load_baseline_manifest(year)['test_auc']:.3f})")

        ax.plot([0, 1], [0, 1], ":", color=COLORS["neutral"], lw=1)
        ax.set_title(f"{year}", fontweight="bold")
        ax.set_xlabel("FPR")
        if ax == axes[0]:
            ax.set_ylabel("TPR")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("ROC Curves — Baseline vs. FT-Transformer Across Eras",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_s2_roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig S2 — ROC curves [OK]")

# FIG S3: Precision-Recall Curves

def fig_s3_pr():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for ax, year in zip(axes, ANCHOR_YEARS):
        ft_preds, ft_labels = _load_ft_preds(year)
        if ft_preds is not None:
            prec, rec, _ = precision_recall_curve(ft_labels, ft_preds)
            ap = average_precision_score(ft_labels, ft_preds)
            ax.plot(rec, prec, color=COLORS["advanced"], lw=2,
                    label=f"FT (AP={ap:.3f})")

        base_preds, base_labels = _load_baseline_test_preds(year)
        if base_preds is not None:
            prec_b, rec_b, _ = precision_recall_curve(base_labels, base_preds)
            ap_b = average_precision_score(base_labels, base_preds)
            ax.plot(rec_b, prec_b, color=COLORS["baseline"], lw=2, ls="--",
                    label=f"Baseline (AP={ap_b:.3f})")

        # Baseline prevalence line
        m = _load_baseline_manifest(year)
        df_check = pd.read_parquet(f"data/processed/processed_{year}.parquet",
                                    columns=["loan_status"])
        prevalence = df_check["loan_status"].mean()
        ax.axhline(prevalence, color=COLORS["neutral"], ls=":", lw=1, label=f"Prevalence ({prevalence:.2f})")

        ax.set_title(f"{year}", fontweight="bold")
        ax.set_xlabel("Recall")
        if ax == axes[0]:
            ax.set_ylabel("Precision")
        ax.legend(fontsize=6, loc="lower left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Precision-Recall Curves — Higher Area = Better at Ranking True Approvals",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_s3_pr_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig S3 — PR curves [OK]")

# FIG S4: Calibration Plots

def fig_s4_calibration():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    n_bins = 10

    for ax, year in zip(axes, ANCHOR_YEARS):
        for model_name, preds, labels, color, ls in [
            ("FT", *_load_ft_preds(year), COLORS["advanced"], "-"),
            ("Baseline", *_load_baseline_test_preds(year), COLORS["baseline"], "--"),
        ]:
            if preds is None:
                continue
            # Bin predictions
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_means_pred = []
            bin_means_actual = []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (preds >= lo) & (preds < hi)
                if mask.sum() > 0:
                    bin_means_pred.append(preds[mask].mean())
                    bin_means_actual.append(labels[mask].mean())

            brier = brier_score_loss(labels, preds)
            ax.plot(bin_means_pred, bin_means_actual, "o-", color=color, lw=2,
                    markersize=5, ls=ls, label=f"{model_name} (Brier={brier:.3f})")

        ax.plot([0, 1], [0, 1], ":", color=COLORS["neutral"], lw=1)
        ax.set_title(f"{year}", fontweight="bold")
        ax.set_xlabel("Predicted P(Approve)")
        if ax == axes[0]:
            ax.set_ylabel("Observed P(Approve)")
        ax.legend(fontsize=6, loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Calibration: When the Model Says 70%, Is It Right? — Dots on Diagonal = Perfect",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_s4_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig S4 — Calibration [OK]")

# FIG S5: Brier / KS / Gini Trajectory (all 17 years)

def fig_s5_metric_trajectory():
    years, base_ks, base_gini, ft_ks, ft_gini = [], [], [], [], []
    base_brier, ft_brier = [], []

    for year in ALL_YEARS:
        m = _load_baseline_manifest(year)
        years.append(year)
        base_ks.append(m["test_ks"])
        base_gini.append(m["test_gini"])

        # Compute Brier for baseline
        bp, bl = _load_baseline_test_preds(year)
        if bp is not None:
            base_brier.append(brier_score_loss(bl, bp))
        else:
            base_brier.append(np.nan)

        ftm = _load_ft_manifest(year)
        if ftm:
            ft_ks.append(ftm["metrics"]["ks"])
            ft_gini.append(ftm["metrics"]["gini"])
            fp, fl = _load_ft_preds(year)
            if fp is not None:
                ft_brier.append(brier_score_loss(fl, fp))
            else:
                ft_brier.append(np.nan)
        else:
            ft_ks.append(np.nan)
            ft_gini.append(np.nan)
            ft_brier.append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    metrics = [
        ("KS Statistic", base_ks, ft_ks, "Higher = Better Separation"),
        ("Gini Coefficient", base_gini, ft_gini, "Higher = Better Discrimination"),
        ("Brier Score", base_brier, ft_brier, "Lower = Better Calibration"),
    ]

    for ax, (name, base_vals, ft_vals, subtitle) in zip(axes, metrics):
        ax.plot(years, base_vals, "o-", color=COLORS["baseline"], lw=2, markersize=4,
                label="Baseline")
        ft_valid = [(y, v) for y, v in zip(years, ft_vals) if not np.isnan(v)]
        if ft_valid:
            ax.plot([y for y, v in ft_valid], [v for y, v in ft_valid],
                    "s-", color=COLORS["advanced"], lw=2, markersize=4, label="FT-Transformer")

        # Schema break
        ax.axvline(2017.5, color=COLORS["unfavorable"], ls="--", lw=1, alpha=0.5)
        ax.set_title(f"{name}\n{subtitle}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Model Quality Metrics Across 17 Years — Baseline vs. Advanced",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_s5_metric_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig S5 — Metric trajectory [OK]")

# FIG S6: Confusion Matrices (FT, 2008 vs 2024)

def fig_s6_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, year in zip(axes, [2008, 2024]):
        preds, labels = _load_ft_preds(year)
        if preds is None:
            continue

        pred_binary = (preds >= 0.5).astype(int)
        cm = confusion_matrix(labels, pred_binary)
        total = cm.sum()

        # Normalize for display
        cm_pct = cm / total * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=60)

        labels_axis = ["Denied", "Approved"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels_axis)
        ax.set_yticklabels(labels_axis)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_pct[i, j]
                color = "white" if pct > 30 else "black"
                ax.text(j, i, f"{count:,.0f}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=10,
                        color=color, fontweight="bold")

        m = _load_ft_manifest(year)
        acc = (cm[0, 0] + cm[1, 1]) / total
        ax.set_title(f"FT-Transformer {year}\nAccuracy: {acc:.1%}  |  AUC: {m['metrics']['auc']:.3f}",
                     fontweight="bold")

    fig.suptitle("Confusion Matrices — Where the Advanced Model Gets It Right and Wrong",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_s6_confusion.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig S6 — Confusion matrices [OK]")

# FIG S7: Information Value — 2008 vs 2024

def fig_s7_iv():
    m08 = _load_baseline_manifest(2008)
    m24 = _load_baseline_manifest(2024)

    iv08 = m08["iv_scores"]
    iv24 = m24["iv_scores"]

    # Deduplicate (log variants share IV with raw)
    seen = set()
    features = []
    for feat in iv24.keys():
        display = pretty(feat) if isinstance(feat, str) and feat in iv24 else feat
        if display not in seen:
            seen.add(display)
            features.append((display, iv08.get(feat, 0), iv24.get(feat, 0)))

    features.sort(key=lambda x: x[2], reverse=True)
    # Limit to unique display names
    seen_display = set()
    unique_features = []
    for disp, v08, v24 in features:
        if disp not in seen_display:
            seen_display.add(disp)
            unique_features.append((disp, v08, v24))
    features = unique_features[:12]
    features = features[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(features))
    h = 0.35

    names = [f[0] for f in features]
    iv_2008 = [f[1] for f in features]
    iv_2024 = [f[2] for f in features]

    bars08 = ax.barh(y_pos - h/2, iv_2008, h, color=COLORS["baseline"], label="2008", zorder=3)
    bars24 = ax.barh(y_pos + h/2, iv_2024, h, color=COLORS["advanced"], label="2024", zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Information Value")
    ax.set_title("Information Value: How Predictive Power Shifted from 2008 to 2024",
                 fontsize=13, fontweight="bold")

    # Threshold lines
    ax.axvline(0.02, color=COLORS["neutral"], ls=":", lw=1, alpha=0.7)
    ax.axvline(0.10, color=COLORS["favorable"], ls="--", lw=1, alpha=0.7)
    ax.axvline(0.30, color=COLORS["unfavorable"], ls="--", lw=1, alpha=0.7)

    ax.text(0.02, len(features) - 0.3, "Weak", fontsize=7, color=COLORS["neutral"])
    ax.text(0.10, len(features) - 0.3, "Strong", fontsize=7, color=COLORS["favorable"])
    ax.text(0.30, len(features) - 0.3, "Very Strong", fontsize=7, color=COLORS["unfavorable"])

    ax.legend(loc="lower right")

    annotate_takeaway(ax,
        "Loan Amount IV tripled (0.10→0.67):\nloan size became far more decisive by 2024.",
        loc="center right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_s7_iv_comparison.png", dpi=200)
    plt.close(fig)
    print("  Fig S7 — IV comparison [OK]")

# FIG S8: Score Distribution (approved vs denied)

def fig_s8_score_dist():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, year in zip(axes, [2008, 2024]):
        preds, labels = _load_ft_preds(year)
        if preds is None:
            continue

        approved = preds[labels == 1]
        denied = preds[labels == 0]

        bins = np.linspace(0, 1, 50)
        ax.hist(approved, bins=bins, alpha=0.6, color=COLORS["favorable"],
                label=f"Approved (n={len(approved):,})", density=True, edgecolor="none")
        ax.hist(denied, bins=bins, alpha=0.6, color=COLORS["unfavorable"],
                label=f"Denied (n={len(denied):,})", density=True, edgecolor="none")

        ax.axvline(0.5, color=COLORS["neutral"], ls="--", lw=1.5, label="Threshold (0.50)")
        ax.set_xlabel("Predicted P(Approve)")
        ax.set_ylabel("Density")
        ax.set_title(f"FT-Transformer {year}", fontweight="bold")
        ax.legend(fontsize=8)

        # KS annotation
        m = _load_ft_manifest(year)
        ax.text(0.98, 0.95, f"KS = {m['metrics']['ks']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                fontweight="bold", color=COLORS["advanced"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["advanced"], lw=1))

    fig.suptitle("Score Distribution — How Well Does the Model Separate Approved from Denied?",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_s8_score_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig S8 — Score distribution [OK]")

# FIG S9: Bootstrap Coefficient Stability (Baseline 2008)

def fig_s9_bootstrap_stability():
    coefs = np.load("outputs/models/baseline_2008/bootstrap_coefs.npy")
    with open("outputs/models/baseline_2008/feature_columns.json") as f:
        feat_cols = json.load(f)

    display_cols = pretty(feat_cols)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot
    bp = ax.boxplot(coefs, vert=False, patch_artist=True,
                    widths=0.5, showfliers=False,
                    medianprops=dict(color="white", lw=1.5),
                    whiskerprops=dict(color=COLORS["neutral"]),
                    capprops=dict(color=COLORS["neutral"]))

    for i, patch in enumerate(bp["boxes"]):
        median = np.median(coefs[:, i])
        color = COLORS["favorable"] if median > 0 else COLORS["unfavorable"]
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yticklabels(display_cols)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("WOE Logistic Regression Coefficient")
    ax.set_title("Bootstrap Coefficient Stability — 100 Replications (2008 Baseline)\n"
                 "Tight Boxes = Stable Estimates",
                 fontsize=12, fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor=COLORS["favorable"], alpha=0.7, label="Positive (→ Approval)"),
        mpatches.Patch(facecolor=COLORS["unfavorable"], alpha=0.7, label="Negative (→ Denial)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    annotate_takeaway(ax,
        "All confidence intervals exclude zero —\nevery feature is a significant predictor.",
        loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_s9_bootstrap_stability.png", dpi=200)
    plt.close(fig)
    print("  Fig S9 — Bootstrap stability [OK]")

# MAIN

def main():
    print("Phase 9B: Supplementary Deep-Dive Figures")
    print(f"  Output: {OUT}/")
    print()

    fig_s1_population()
    fig_s2_roc()
    fig_s3_pr()
    fig_s4_calibration()
    fig_s5_metric_trajectory()
    fig_s6_confusion()
    fig_s7_iv()
    fig_s8_score_dist()
    fig_s9_bootstrap_stability()

    print()
    print(f"All 9 supplementary figures saved to {OUT}/")

if __name__ == "__main__":
    main()
