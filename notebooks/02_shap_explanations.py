"""Phase 9C: SHAP Explanations — Global & Local Feature Importance.

Uses KernelSHAP on the FT-Transformer ensemble for 2024 to produce:
  Fig SHAP-1 — Global: Mean |SHAP| bar chart (top features)
  Fig SHAP-2 — Local: Beeswarm plot (per-sample SHAP distribution)
  Fig SHAP-3 — Local: Waterfall for a single denied applicant

KernelSHAP is model-agnostic (treats the ensemble as a black box),
so it works correctly with the FT-Transformer's categorical embeddings.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ft_ensemble import FTEnsemble
from src.features.ft_preprocessor import FTPreprocessor
from src.features.display_names import pretty
from src.viz.style import COLORS, apply_house_style, annotate_takeaway

apply_house_style()

OUT = "outputs/figures/deck_shap"
os.makedirs(OUT, exist_ok=True)

YEAR = 2024
DEVICE = "mps"
BACKGROUND_N = 100    # KernelSHAP background sample size
EXPLAIN_N = 500       # Number of test samples to explain
SEED = 42

# LOAD MODEL + DATA

def load_model_and_data():
    """Load ensemble, preprocessor, and subsampled test data."""
    print(f"  Loading FT-Transformer ensemble for {YEAR}...")
    dev = torch.device(DEVICE)
    model_dir = f"outputs/models/ft_{YEAR}"

    ens = FTEnsemble.load(model_dir, dev)
    prep = FTPreprocessor.load(os.path.join(model_dir, "ft_preprocessor.pkl"))

    # Load full processed data
    df = pd.read_parquet(f"data/processed/processed_{YEAR}.parquet")
    y = df["loan_status"].values
    df_feat = df.drop(columns=["loan_status"])

    # Get feature names with display names
    feature_names_raw = prep.numeric_cols_ + prep.categorical_cols_
    feature_names_display = pretty(feature_names_raw)

    # Remove redundant "Applicant Income" (raw), keep "Income (log scale)"
    exclude_raw = {"applicant_income_000s"}

    # Transform to model inputs
    x_num, x_cat = prep.transform(df_feat)
    x_combined = np.hstack([x_num, x_cat.astype(np.float32)])

    # Build a clean feature name list (with display names, excluding redundant)
    keep_idx = [i for i, rn in enumerate(feature_names_raw) if rn not in exclude_raw]
    x_combined = x_combined[:, keep_idx]
    feature_names_display = [feature_names_display[i] for i in keep_idx]
    feature_names_raw_kept = [feature_names_raw[i] for i in keep_idx]
    n_num_kept = sum(1 for i in keep_idx if i < len(prep.numeric_cols_))

    print(f"  Features: {len(feature_names_display)} ({n_num_kept} numeric, "
          f"{len(feature_names_display) - n_num_kept} categorical)")

    return ens, prep, x_combined, y, feature_names_display, feature_names_raw_kept, keep_idx, n_num_kept

def make_predict_fn(ens, prep, keep_idx, n_num_original):
    """Create a prediction function that takes combined numpy array → P(approve)."""
    dev = torch.device(DEVICE)
    n_num_kept = sum(1 for i in keep_idx if i < n_num_original)

    def predict(x_combined):
        # Reconstruct full feature array (insert zeros for excluded columns)
        n_total = n_num_original + len(prep.categorical_cols_)
        x_full = np.zeros((len(x_combined), n_total), dtype=np.float32)
        for j, orig_idx in enumerate(keep_idx):
            x_full[:, orig_idx] = x_combined[:, j]

        x_num_t = torch.tensor(x_full[:, :n_num_original], dtype=torch.float32, device=dev)
        x_cat_t = torch.tensor(x_full[:, n_num_original:], dtype=torch.int64, device=dev)

        # Average predictions across ensemble members
        all_probs = []
        for member in ens.members_:
            member.eval()
            with torch.no_grad():
                logit = member(x_num_t, x_cat_t)
                prob = torch.sigmoid(logit).squeeze(1).cpu().numpy()
            all_probs.append(prob)

        return np.mean(all_probs, axis=0)

    return predict

# COMPUTE SHAP VALUES

def compute_shap(ens, prep, x_combined, y, feature_names_display, keep_idx):
    """Run KernelSHAP on the ensemble."""
    n_num_original = len(prep.numeric_cols_)
    predict_fn = make_predict_fn(ens, prep, keep_idx, n_num_original)

    np.random.seed(SEED)

    # Background: stratified subsample
    bg_idx = np.random.choice(len(x_combined), BACKGROUND_N, replace=False)
    background = x_combined[bg_idx]

    # Explain: subsample of test points (balanced)
    denied_idx = np.where(y == 0)[0]
    approved_idx = np.where(y == 1)[0]
    n_half = EXPLAIN_N // 2
    explain_denied = np.random.choice(denied_idx, min(n_half, len(denied_idx)), replace=False)
    explain_approved = np.random.choice(approved_idx, min(n_half, len(approved_idx)), replace=False)
    explain_idx = np.concatenate([explain_denied, explain_approved])
    np.random.shuffle(explain_idx)

    x_explain = x_combined[explain_idx]
    y_explain = y[explain_idx]

    print(f"  KernelSHAP: {BACKGROUND_N} background, {len(x_explain)} explain samples...")
    t0 = time.time()

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(x_explain, nsamples=200, silent=True)

    wall = time.time() - t0
    print(f"  SHAP computed in {wall:.0f}s")

    return shap_values, x_explain, y_explain, explain_idx

# FIG SHAP-1: Global Bar Chart

def fig_shap1_global(shap_values, feature_names_display):
    """Mean |SHAP| bar chart — global feature importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    # Also compute signed mean for coloring by direction
    signed_mean = shap_values.mean(axis=0)

    # Sort descending by absolute
    order = np.argsort(mean_abs)[::-1]
    sorted_names = [feature_names_display[i] for i in order]
    sorted_vals = mean_abs[order]
    sorted_signs = signed_mean[order]

    # Take top features (non-zero)
    mask = sorted_vals > 1e-6
    sorted_names = [n for n, m in zip(sorted_names, mask) if m]
    sorted_vals = sorted_vals[mask]
    sorted_signs = sorted_signs[mask]

    # Reverse for horizontal bar (bottom = highest)
    sorted_names = sorted_names[::-1]
    sorted_vals = sorted_vals[::-1]
    sorted_signs = sorted_signs[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_names))

    # Color by direction: positive SHAP = pushes toward approval, negative = denial
    colors = [COLORS["favorable"] if s > 0 else COLORS["unfavorable"] for s in sorted_signs]

    ax.barh(y_pos, sorted_vals, color=colors, edgecolor="white", height=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Model Output)")
    ax.set_title(f"Global Feature Importance — SHAP ({YEAR})\nLonger Bar = Stronger Influence on Approval Probability",
                 fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["favorable"], label="On Average Pushes Toward Approval"),
        Patch(facecolor=COLORS["unfavorable"], label="On Average Pushes Toward Denial"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
             framealpha=0.9, edgecolor=COLORS["neutral"])

    annotate_takeaway(ax,
        "SHAP confirms IG findings: loan size\nand income ratios dominate decisions.",
        loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap1_global.png", dpi=200)
    plt.close(fig)
    print("  Fig SHAP-1 — Global importance [OK]")

# FIG SHAP-2: Beeswarm (Local Explanations)

def fig_shap2_beeswarm(shap_values, x_explain, feature_names_display):
    """SHAP beeswarm plot — local per-sample explanations."""
    # Create SHAP Explanation object
    explanation = shap.Explanation(
        values=shap_values,
        data=x_explain,
        feature_names=feature_names_display,
    )

    # Sort by mean |SHAP|
    fig = plt.figure(figsize=(10, 7))

    # Use SHAP's beeswarm with project colors
    shap.plots.beeswarm(
        explanation,
        max_display=14,
        show=False,
        color=plt.cm.RdBu_r,
    )

    ax = plt.gca()
    ax.set_title(f"Local Explanations — Every Dot Is One Application ({YEAR})\n"
                 f"Red = High Feature Value, Blue = Low Feature Value",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("SHAP Value (Impact on Approval Probability)")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap2_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig SHAP-2 — Beeswarm (local) [OK]")

# FIG SHAP-3: Waterfall — Single Denied Applicant

def fig_shap3_waterfall(shap_values, x_explain, y_explain, feature_names_display, predict_fn, x_combined):
    """Custom SHAP waterfall for a single denied applicant with clear legend."""
    # Find denied applicants in the explained set
    denied_mask = y_explain == 0
    if not denied_mask.any():
        print("  Fig SHAP-3 — SKIP: no denied applicants in sample")
        return

    # Pick the denied applicant with the strongest total negative SHAP
    denied_indices = np.where(denied_mask)[0]
    shap_sums = shap_values[denied_indices].sum(axis=1)
    pick = denied_indices[np.argmin(shap_sums)]  # most strongly denied

    # Expected value (base rate)
    base_value = predict_fn(x_combined[:BACKGROUND_N]).mean()
    shap_row = shap_values[pick]

    # Sort by absolute SHAP value, take top 12
    order = np.argsort(np.abs(shap_row))[::-1][:12]
    feat_names = [feature_names_display[i] for i in order]
    feat_vals = shap_row[order]

    # Reverse for horizontal bar (bottom = highest)
    feat_names = feat_names[::-1]
    feat_vals = feat_vals[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(feat_names))

    colors = [COLORS["favorable"] if v > 0 else COLORS["unfavorable"] for v in feat_vals]

    ax.barh(y_pos, feat_vals, color=colors, edgecolor="white", height=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value (Positive = Pushes Toward Approval)")

    final_pred = base_value + shap_row.sum()
    ax.set_title(
        f"Why This Applicant Was Denied — SHAP Waterfall ({YEAR})\n"
        f"Base Rate: {base_value:.1%} → Final Prediction: {final_pred:.1%}",
        fontsize=12, fontweight="bold"
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["favorable"], label="Pushes Toward Approval (+)"),
        Patch(facecolor=COLORS["unfavorable"], label="Pushes Toward Denial (−)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
             framealpha=0.9, edgecolor=COLORS["neutral"])

    annotate_takeaway(ax,
        "Each bar shows one feature's SHAP\ncontribution to this individual denial.",
        loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap3_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig SHAP-3 — Waterfall (denied) [OK]")

# MAIN

def main():
    print(f"Phase 9C: SHAP Explanations ({YEAR})")
    print(f"  Output: {OUT}/")
    print(f"  Background: {BACKGROUND_N}  |  Explain: {EXPLAIN_N}  |  Device: {DEVICE}")
    print()

    ens, prep, x_combined, y, feature_names_display, feature_names_raw, keep_idx, n_num_kept = \
        load_model_and_data()

    predict_fn = make_predict_fn(ens, prep, keep_idx, len(prep.numeric_cols_))

    shap_values, x_explain, y_explain, explain_idx = \
        compute_shap(ens, prep, x_combined, y, feature_names_display, keep_idx)

    # Save raw SHAP values
    os.makedirs(f"outputs/explanations/{YEAR}", exist_ok=True)
    np.save(f"outputs/explanations/{YEAR}/shap_values.npy", shap_values.astype(np.float32))
    np.save(f"outputs/explanations/{YEAR}/shap_x_explain.npy", x_explain.astype(np.float32))
    with open(f"outputs/explanations/{YEAR}/shap_meta.json", "w") as f:
        json.dump({
            "year": YEAR,
            "background_n": BACKGROUND_N,
            "explain_n": len(x_explain),
            "feature_names": feature_names_display,
        }, f, indent=2)

    fig_shap1_global(shap_values, feature_names_display)
    fig_shap2_beeswarm(shap_values, x_explain, feature_names_display)
    fig_shap3_waterfall(shap_values, x_explain, y_explain, feature_names_display,
                        predict_fn, x_combined)

    print()
    print(f"All SHAP figures saved to {OUT}/")

if __name__ == "__main__":
    main()
