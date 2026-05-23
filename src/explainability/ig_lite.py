"""Phase 6-LITE: Integrated Gradients on 4 anchor years.
Computes per-sample IG attributions for the FT-Transformer ensemble on a
subsampled test set, then builds global importance with B=200 bootstrap CIs.
Integrated Gradients works by interpolating between a baseline (median input)
and each actual input, accumulating the model's gradient at each step. The
result tells us how much each feature contributed to the model's prediction
for that specific applicant.
Known limitation: categorical embeddings are treated as continuous inputs
for IG. A production build would use Captum LayerIntegratedGradients on
the embedding layer for rigorous discrete attribution.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from captum.attr import IntegratedGradients  # noqa: E402
from src.models.ft_ensemble import FTEnsemble  # noqa: E402
from src.features.ft_preprocessor import FTPreprocessor  # noqa: E402
from src.features.display_names import pretty  # noqa: E402

ANCHOR_YEARS = [2008, 2013, 2018, 2024]
B = 200               # bootstrap iterations for confidence intervals
TEST_SUBSAMPLE = 20_000  # cap test set for speed
IG_STEPS = 30          # Riemann steps (30 is safe for LITE; 50 is textbook)
BATCH_SIZE = 2048      # IG forward-pass batch size

def _make_forward_fn(member, n_num, device):
    """Build a differentiable forward function for Captum.
    Captum expects a single tensor input, but our model expects separate
    (x_num, x_cat) tensors. This wrapper splits the concatenated input
    back into numeric and categorical parts.
    """
    def fn(x_combined):
        x_num_part = x_combined[:, :n_num]
        x_cat_part = x_combined[:, n_num:].long()
        logit = member(x_num_part, x_cat_part)
        return torch.sigmoid(logit).squeeze(1)
    return fn

def run_year(year: int, device: str = "mps") -> dict:
    """Compute IG attributions for a single anchor year."""
    t0 = time.time()
    out_dir = f"outputs/explanations/{year}"
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device)
    print(f"  Year {year}  |  device={device}  |  B={B}  |  subsample={TEST_SUBSAMPLE:,}")
    # Load the trained ensemble and its matching preprocessor
    model_dir = f"outputs/models/ft_{year}"
    ens = FTEnsemble.load(model_dir, dev)
    prep = FTPreprocessor.load(os.path.join(model_dir, "ft_preprocessor.pkl"))
    K = len(ens.members_)
    n_num = len(prep.numeric_cols_)
    feature_names = prep.numeric_cols_ + prep.categorical_cols_
    display_names = pretty(feature_names)
    print(f"  Loaded {K}-member ensemble, {len(feature_names)} features")
    # Load test data and subsample for speed
    df = pd.read_parquet(f"data/processed/processed_{year}.parquet")
    n_total = len(df)
    if n_total > TEST_SUBSAMPLE:
        df = df.sample(n=TEST_SUBSAMPLE, random_state=42)
    y = df["loan_status"].values
    df_feat = df.drop(columns=["loan_status"])
    x_num, x_cat = prep.transform(df_feat)
    print(f"  Test subset: {len(df):,} rows (from {n_total:,})")
    # Concatenate numeric and categorical into one tensor for Captum.
    # The baseline is the median of each feature (a neutral reference point).
    x_combined = np.hstack([x_num, x_cat.astype(np.float32)])
    baseline = np.median(x_combined, axis=0, keepdims=True)
    # Run IG for each ensemble member separately, then average.
    # This gives us the ensemble-level attribution instead of just one model's.
    all_attr = []
    for k, member in enumerate(ens.members_):
        member.eval()
        forward_fn = _make_forward_fn(member, n_num, dev)
        ig = IntegratedGradients(forward_fn)
        x_t = torch.tensor(x_combined, dtype=torch.float32, device=dev)
        b_t = torch.tensor(baseline, dtype=torch.float32, device=dev).expand_as(x_t)
        # Process in batches to avoid out-of-memory
        attr_chunks = []
        for start in range(0, len(x_t), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(x_t))
            attr_chunk = ig.attribute(
                x_t[start:end], baselines=b_t[start:end], n_steps=IG_STEPS,
            )
            attr_chunks.append(attr_chunk.detach().cpu().numpy())
        attr = np.vstack(attr_chunks)
        all_attr.append(attr)
        print(f"    Member {k}/{K} IG done ({attr.shape})")
    mean_attr = np.mean(all_attr, axis=0)
    print(f"  Ensemble-averaged attributions: {mean_attr.shape}")
    # Bootstrap global importance: resample attributions B times to get
    # confidence intervals for each feature's average importance.
    rng = np.random.RandomState(42)
    bootstrap_global = np.zeros((B, mean_attr.shape[1]))
    for b_idx in range(B):
        idx = rng.choice(len(mean_attr), len(mean_attr), replace=True)
        bootstrap_global[b_idx] = np.abs(mean_attr[idx]).mean(axis=0)
    # Save all artifacts
    np.save(f"{out_dir}/local_attr.npy", mean_attr.astype(np.float32))
    np.save(f"{out_dir}/global_bootstrap.npy", bootstrap_global.astype(np.float32))
    # Build summary JSON with per-feature stats
    summary_features = []
    for i, (raw_name, disp_name) in enumerate(zip(feature_names, display_names)):
        col = bootstrap_global[:, i]
        summary_features.append({
            "feature_raw": raw_name,
            "feature_display": disp_name,
            "mean_abs_attribution": float(col.mean()),
            "ci_lo": float(np.percentile(col, 2.5)),
            "ci_hi": float(np.percentile(col, 97.5)),
            "signed_mean": float(mean_attr[:, i].mean()),
        })
    summary = {
        "year": year,
        "n_test_subsample": len(mean_attr),
        "n_test_total": n_total,
        "B": B,
        "ig_steps": IG_STEPS,
        "device": device,
        "feature_names_raw": feature_names,
        "feature_names_display": display_names,
        "features": sorted(summary_features, key=lambda x: -x["mean_abs_attribution"]),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    # Save local attributions for a few denied and approved examples
    # (used for waterfall plots in the deck)
    denied_idx = np.where(y == 0)[0][:5]
    if len(denied_idx) > 0:
        pd.DataFrame(
            mean_attr[denied_idx], columns=feature_names
        ).to_csv(f"{out_dir}/local_denied_examples.csv", index=False)
    approved_idx = np.where(y == 1)[0][:5]
    if len(approved_idx) > 0:
        pd.DataFrame(
            mean_attr[approved_idx], columns=feature_names
        ).to_csv(f"{out_dir}/local_approved_examples.csv", index=False)
    wall = time.time() - t0
    print(f"  Year {year} done in {wall:.0f}s -- saved to {out_dir}/")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Phase 6-LITE: IG on anchor years")
    parser.add_argument("--years", type=int, nargs="*", default=ANCHOR_YEARS,
                        help=f"Years to compute (default: {ANCHOR_YEARS})")
    parser.add_argument("--device", type=str, default="mps",
                        help="Torch device (default: mps)")
    args = parser.parse_args()
    print("Phase 6-LITE: Integrated Gradients")
    print(f"  Years: {args.years}  |  B={B}  |  subsample={TEST_SUBSAMPLE:,}")
    print(f"  Device: {args.device}")
    results = {}
    for year in args.years:
        results[year] = run_year(year, device=args.device)
    # Save a global manifest linking all year results
    os.makedirs("outputs/explanations", exist_ok=True)
    with open("outputs/explanations/_ig_lite_manifest.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"All {len(args.years)} years complete.")

if __name__ == "__main__":
    main()
