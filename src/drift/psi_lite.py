"""Phase 7-LITE: Population Stability Index on calibrated score distributions.
Compares the distribution of predicted probabilities between anchor year pairs
to detect population shifts that might degrade model performance.
PSI interpretation thresholds (industry standard):
  < 0.10   -- No meaningful shift
  0.10-0.25 -- Moderate shift, investigate further
  > 0.25   -- Significant shift, retrain recommended
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Year pairs to compare for population drift
COMPARISONS = [
    ("2008", "2024", "2008_vs_2024"),
    ("2013", "2018", "2013_vs_2018"),
]
N_BINS = 10  # decile bins (splits into 10 equal-sized buckets)

def compute_psi(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = N_BINS,
) -> tuple[float, pd.DataFrame]:
    """Compute PSI between two probability distributions.
    expected: predicted probabilities from the reference (older) year.
    actual: predicted probabilities from the comparison (newer) year.
    Returns (total_psi, per-bin breakdown DataFrame).
    """
    # Define bin boundaries using the reference year's quantiles so
    # each bin starts with ~10% of the reference population.
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    rows = []
    psi_total = 0.0
    for i in range(n_bins):
        lo, hi = breakpoints[i], breakpoints[i + 1]
        pct_exp = np.mean((expected >= lo) & (expected < hi))
        pct_act = np.mean((actual >= lo) & (actual < hi))
        # Floor at 1e-6 to avoid log(0)
        pct_exp = max(pct_exp, 1e-6)
        pct_act = max(pct_act, 1e-6)
        psi_bin = (pct_act - pct_exp) * np.log(pct_act / pct_exp)
        psi_total += psi_bin
        rows.append({
            "bin": i + 1,
            "range_lo": round(lo, 4) if lo > -np.inf else 0.0,
            "range_hi": round(hi, 4) if hi < np.inf else 1.0,
            "pct_expected": round(pct_exp, 4),
            "pct_actual": round(pct_act, 4),
            "psi_bin": round(psi_bin, 6),
        })
    return round(psi_total, 6), pd.DataFrame(rows)

def _load_preds(year: str) -> np.ndarray:
    """Load test predictions for a year from the FT model outputs."""
    path = f"outputs/models/ft_{year}/test_preds.npy"
    if os.path.exists(path):
        return np.load(path)
    raise FileNotFoundError(f"No test predictions found at {path}")

def main():
    print("Phase 7-LITE: Population Stability Index (PSI)")
    os.makedirs("outputs/drift", exist_ok=True)
    results = []
    for year_a, year_b, label in COMPARISONS:
        try:
            preds_a = _load_preds(year_a)
            preds_b = _load_preds(year_b)
        except FileNotFoundError as e:
            print(f"  SKIP {label}: {e}")
            continue
        psi_total, psi_bins = compute_psi(preds_a, preds_b)
        if psi_total < 0.10:
            interpretation = "No meaningful shift"
        elif psi_total < 0.25:
            interpretation = "Moderate shift -- investigate"
        else:
            interpretation = "Significant shift -- retrain recommended"
        print(f"  {label}: PSI = {psi_total:.4f} ({interpretation})")
        psi_bins["comparison"] = label
        psi_bins["psi_total"] = psi_total
        results.append(psi_bins)
        psi_bins.to_csv(f"outputs/drift/psi_{label}.csv", index=False)
    if results:
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv("outputs/drift/psi_all.csv", index=False)
        summary = {}
        for year_a, year_b, label in COMPARISONS:
            sub = combined[combined["comparison"] == label]
            if len(sub) > 0:
                summary[label] = {
                    "psi_total": float(sub["psi_total"].iloc[0]),
                    "n_bins": N_BINS,
                }
        with open("outputs/drift/psi_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\n  Saved to outputs/drift/psi_*.csv and psi_summary.json")
    else:
        print("  No PSI comparisons could be computed.")
    print("Done.")

if __name__ == "__main__":
    main()
