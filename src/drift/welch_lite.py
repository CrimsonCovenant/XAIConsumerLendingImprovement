"""Phase 7-LITE: Welch's t-test on bootstrap attribution distributions.
Compares IG global importance distributions between anchor year pairs:
  2008 vs 2024: full-spectrum drift across 16 years
  2013 vs 2018: isolates the schema break / macro regime shift
Uses Welch's t-test (unequal variance) because bootstrap sample sizes
and variances differ across years. Applies Benjamini-Hochberg FDR
correction at q=0.05 to control false discovery rate when testing
multiple features simultaneously.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.features.display_names import pretty  # noqa: E402

# Year pairs to compare for drift
COMPARISONS = [
    ("2008", "2024", "2008_vs_2024"),
    ("2013", "2018", "2013_vs_2018"),
]

def welch_t(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Welch's t-test (unequal variance). Returns (t_stat, df, p_value)."""
    res = ttest_ind(x, y, equal_var=False)
    return float(res.statistic), float(res.df), float(res.pvalue)

def bh_correct(pvals: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.
    When testing many features at once, some will be 'significant' by chance.
    BH correction adjusts p-values upward to control the false discovery rate.
    Returns (adjusted_pvals, is_significant boolean array).
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    # BH formula: adjusted_p = raw_p * n_tests / rank
    adj = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest rank down
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    # Unsort back to original order
    adj_out = np.zeros(n)
    adj_out[order] = adj
    return adj_out, adj_out < q

def drift_table(
    boot_a_path: str, boot_b_path: str,
    feature_names: list[str], label: str, q: float = 0.05,
) -> pd.DataFrame:
    """Build drift test results for one year-pair.
    Loads the bootstrap IG importance arrays for each year and runs
    Welch's t-test per feature with BH correction.
    """
    a = np.load(boot_a_path)
    b = np.load(boot_b_path)
    display = pretty(feature_names)
    rows = []
    for i, (raw, disp) in enumerate(zip(feature_names, display)):
        t, df, p = welch_t(a[:, i], b[:, i])
        rows.append({
            "feature_raw": raw,
            "feature_display": disp,
            "t_stat": round(t, 4),
            "df": round(df, 2),
            "p_raw": p,
            "mean_a": float(a[:, i].mean()),
            "mean_b": float(b[:, i].mean()),
            "delta": float(b[:, i].mean() - a[:, i].mean()),
            "pct_change": float(
                (b[:, i].mean() - a[:, i].mean()) / max(abs(a[:, i].mean()), 1e-9) * 100
            ),
        })
    df_out = pd.DataFrame(rows)
    df_out["p_bh"], df_out["drifted"] = bh_correct(df_out["p_raw"].values, q=q)
    df_out["comparison"] = label
    return df_out.sort_values("p_raw")

def main():
    print("Phase 7-LITE: Welch's t-test on bootstrap attribution distributions")
    # Load feature names from the first available IG summary
    summary_path = None
    for year in ["2008", "2013", "2018", "2024"]:
        cand = f"outputs/explanations/{year}/summary.json"
        if os.path.exists(cand):
            summary_path = cand
            break
    if summary_path is None:
        print("ERROR: No IG summary found. Run ig_lite.py first.")
        sys.exit(1)
    with open(summary_path) as f:
        feats = json.load(f)["feature_names_raw"]
    print(f"  Features: {len(feats)}")
    all_tables = []
    for year_a, year_b, label in COMPARISONS:
        path_a = f"outputs/explanations/{year_a}/global_bootstrap.npy"
        path_b = f"outputs/explanations/{year_b}/global_bootstrap.npy"
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print(f"  SKIP {label}: missing bootstrap files")
            continue
        df = drift_table(path_a, path_b, feats, label)
        n_drifted = df["drifted"].sum()
        print(f"  {label}: {n_drifted}/{len(feats)} features drifted (BH q<0.05)")
        all_tables.append(df)
    if not all_tables:
        print("No comparisons could be computed. Exiting.")
        sys.exit(1)
    # Save results as both parquet (fast) and csv (human-readable)
    os.makedirs("outputs/drift", exist_ok=True)
    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_parquet("outputs/drift/welch_anchored.parquet", index=False)
    combined.to_csv("outputs/drift/welch_anchored.csv", index=False)
    print("\n  Results saved to outputs/drift/welch_anchored.{parquet,csv}")
    for label in combined["comparison"].unique():
        sub = combined[combined["comparison"] == label]
        drifted = sub[sub["drifted"]]
        print(f"\n  {label}:")
        if len(drifted) > 0:
            for _, r in drifted.iterrows():
                direction = "UP" if r["delta"] > 0 else "DOWN"
                print(f"    {direction} {r['feature_display']}: "
                      f"delta={r['delta']:+.4f} ({r['pct_change']:+.1f}%), "
                      f"p_bh={r['p_bh']:.4f}")
        else:
            print("    No significant drift detected.")
    print("\nDone.")

if __name__ == "__main__":
    main()
