"""Phase 8-LITE: Adverse Impact Ratio (AIR) by race and sex.
Compares baseline (WOE logistic regression) vs advanced (FT-Transformer)
approval rates for protected groups against the majority group.
AIR = approval_rate(minority) / approval_rate(majority)
  >= 0.80 : Four-Fifths Rule compliance (EEOC guideline)
  >= 0.90 : Industry best-practice target
Computes AIR with B=500 bootstrap CIs on test-set predictions.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.features.ft_preprocessor import FTPreprocessor  # noqa: E402
from src.models.ft_ensemble import FTEnsemble  # noqa: E402

ANCHOR_YEARS = [2008, 2024]
B = 500
THRESHOLD = 0.5  # binary approval threshold
# Protected attribute configurations.
# Majority group is the reference; each minority group's approval rate
# is divided by the majority rate to compute AIR.
RACE_GROUPS = {
    "majority": ["White"],
    "minorities": {
        "Black": ["Black or African American"],
        "Hispanic": ["Hispanic or Latino"],
        "Asian": ["Asian"],
    },
}
SEX_GROUPS = {
    "majority": ["Male"],
    "minorities": {
        "Female": ["Female"],
    },
}

def compute_air(
    y_pred: np.ndarray, group_labels: np.ndarray,
    majority_values: list[str], minority_values: list[str],
    threshold: float = THRESHOLD,
) -> dict:
    """Compute AIR for a single minority group vs majority.
    AIR = minority_approval_rate / majority_approval_rate.
    """
    majority_mask = np.isin(group_labels, majority_values)
    minority_mask = np.isin(group_labels, minority_values)
    majority_preds = y_pred[majority_mask]
    minority_preds = y_pred[minority_mask]
    if len(majority_preds) == 0 or len(minority_preds) == 0:
        return {"air": np.nan, "n_majority": 0, "n_minority": 0}
    majority_rate = (majority_preds >= threshold).mean()
    minority_rate = (minority_preds >= threshold).mean()
    air = minority_rate / max(majority_rate, 1e-9)
    return {
        "air": float(air),
        "majority_approval_rate": float(majority_rate),
        "minority_approval_rate": float(minority_rate),
        "n_majority": int(majority_mask.sum()),
        "n_minority": int(minority_mask.sum()),
    }

def bootstrap_air(
    y_pred: np.ndarray, group_labels: np.ndarray,
    majority_values: list[str], minority_values: list[str],
    b: int = B, threshold: float = THRESHOLD,
) -> dict:
    """Bootstrap confidence intervals for AIR.
    Resamples the test set B times and computes AIR each time to get
    2.5th and 97.5th percentile confidence bounds.
    """
    rng = np.random.RandomState(42)
    air_boot = []
    for _ in range(b):
        idx = rng.choice(len(y_pred), len(y_pred), replace=True)
        result = compute_air(y_pred[idx], group_labels[idx],
                             majority_values, minority_values, threshold)
        if not np.isnan(result["air"]):
            air_boot.append(result["air"])
    if len(air_boot) < 10:
        return {"air": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "B": 0}
    air_arr = np.array(air_boot)
    base = compute_air(y_pred, group_labels, majority_values, minority_values, threshold)
    base["ci_lo"] = float(np.percentile(air_arr, 2.5))
    base["ci_hi"] = float(np.percentile(air_arr, 97.5))
    base["B"] = len(air_boot)
    return base

def _load_baseline_preds(year: int) -> np.ndarray | None:
    """Load baseline model (WOE logistic regression) predictions on test data.
    Reconstructs the same train/test split used during training so the
    test predictions align with the protected attribute rows.
    """
    model_dir = f"outputs/models/baseline_{year}"
    manifest_path = os.path.join(model_dir, "_run_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    import yaml
    with open("config/config.yml") as f:
        config = yaml.safe_load(f)
    df = pd.read_parquet(
        os.path.join(config["data"]["processed"], f"processed_{year}.parquet")
    )
    y = df["loan_status"].values
    df_feat = df.drop(columns=["loan_status"])
    # Load saved WOE pipeline, model weights, and Platt calibrator
    with open(os.path.join(model_dir, "woe_pipeline.pkl"), "rb") as f:
        woe = pickle.load(f)
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "platt.pkl"), "rb") as f:
        platt = pickle.load(f)
    # Reconstruct the exact same split used during training
    from sklearn.model_selection import train_test_split
    from src.utils.reproducibility import seed_everything
    seed_everything(42)
    idx = np.arange(len(df))
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
    test_idx, cal_idx = train_test_split(
        rest_idx, test_size=1/3, random_state=42, stratify=y[rest_idx]
    )
    X_test = woe.transform(df_feat.iloc[test_idx])
    raw_logits = X_test @ model.weights + model.bias
    probs = platt.transform(raw_logits)
    return probs, test_idx

def _load_ft_preds(year: int):
    """Load FT-Transformer test predictions from saved numpy files."""
    preds_path = f"outputs/models/ft_{year}/test_preds.npy"
    labels_path = f"outputs/models/ft_{year}/test_labels.npy"
    if not os.path.exists(preds_path):
        return None, None
    preds = np.load(preds_path)
    labels = np.load(labels_path)
    return preds, labels

def run_year(year: int) -> dict:
    """Compute AIR for baseline and advanced model for one year."""
    t0 = time.time()
    print(f"\n  Year {year}")
    import yaml
    with open("config/config.yml") as f:
        config = yaml.safe_load(f)
    # Load protected attributes (race, sex, ethnicity) for this year
    protected_path = os.path.join(
        config["data"]["processed"], f"protected_{year}.parquet"
    )
    if not os.path.exists(protected_path):
        print(f"    SKIP: protected parquet not found at {protected_path}")
        return {}
    df_prot = pd.read_parquet(protected_path)
    # The preprocessing pipeline harmonizes column names so all years
    # use applicant_race_name_1 and applicant_sex_name regardless of schema
    race_col = "applicant_race_name_1" if "applicant_race_name_1" in df_prot.columns else None
    sex_col = "applicant_sex_name" if "applicant_sex_name" in df_prot.columns else None
    if race_col is None and sex_col is None:
        print(f"    SKIP: no protected attribute columns found")
        return {}
    results = {"year": year, "comparisons": []}
    # Load predictions from both models
    baseline_result = _load_baseline_preds(year)
    if baseline_result is not None:
        base_preds, base_test_idx = baseline_result
        print(f"    Baseline: {len(base_preds):,} test predictions")
    else:
        base_preds, base_test_idx = None, None
        print(f"    Baseline: NOT AVAILABLE")
    ft_preds, ft_labels = _load_ft_preds(year)
    if ft_preds is not None:
        print(f"    FT-Transformer: {len(ft_preds):,} test predictions")
    else:
        print(f"    FT-Transformer: NOT AVAILABLE")
    # Compute AIR for each combination of attribute and model
    for attr_name, col, groups in [
        ("race", race_col, RACE_GROUPS),
        ("sex", sex_col, SEX_GROUPS),
    ]:
        if col is None:
            continue
        for minority_name, minority_values in groups["minorities"].items():
            for model_name, preds, test_idx in [
                ("baseline", base_preds, base_test_idx),
                ("ft_transformer", ft_preds, None),
            ]:
                if preds is None:
                    continue
                # Get protected attributes aligned with test predictions
                if test_idx is not None:
                    group_labels = df_prot.iloc[test_idx][col].values
                else:
                    # FT test set uses same split logic; reconstruct test_idx
                    df = pd.read_parquet(
                        os.path.join(config["data"]["processed"], f"processed_{year}.parquet")
                    )
                    y = df["loan_status"].values
                    from sklearn.model_selection import train_test_split
                    idx = np.arange(len(df))
                    train_idx, rest_idx = train_test_split(
                        idx, test_size=0.30, random_state=42, stratify=y
                    )
                    test_idx_ft, cal_idx = train_test_split(
                        rest_idx, test_size=1/3, random_state=42, stratify=y[rest_idx]
                    )
                    group_labels = df_prot.iloc[test_idx_ft][col].values
                air_result = bootstrap_air(
                    preds, group_labels.astype(str),
                    groups["majority"], minority_values, b=B,
                )
                air_result["attribute"] = attr_name
                air_result["minority_group"] = minority_name
                air_result["model"] = model_name
                air_result["year"] = year
                # Check Four-Fifths Rule compliance
                air_val = air_result.get("air", np.nan)
                air_result["four_fifths_compliant"] = bool(air_val >= 0.80) if not np.isnan(air_val) else None
                air_result["best_practice_compliant"] = bool(air_val >= 0.90) if not np.isnan(air_val) else None
                results["comparisons"].append(air_result)
                status = "[OK]" if air_result.get("four_fifths_compliant") else "[WARN]"
                print(f"    {status} {model_name} {attr_name}/{minority_name}: "
                      f"AIR={air_val:.3f} [{air_result.get('ci_lo', '?'):.3f}, "
                      f"{air_result.get('ci_hi', '?'):.3f}]")
    results["wall_clock_s"] = round(time.time() - t0, 1)
    return results

def main():
    parser = argparse.ArgumentParser(description="Phase 8-LITE: AIR fairness analysis")
    parser.add_argument("--years", type=int, nargs="*", default=ANCHOR_YEARS,
                        help=f"Years to analyze (default: {ANCHOR_YEARS})")
    args = parser.parse_args()
    print("Phase 8-LITE: Adverse Impact Ratio (AIR) Analysis")
    print(f"  Years: {args.years}  |  B={B}  |  threshold={THRESHOLD}")
    os.makedirs("outputs/fairness", exist_ok=True)
    all_results = {}
    for year in args.years:
        all_results[year] = run_year(year)
    # Save as both JSON and CSV
    with open("outputs/fairness/air_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    rows = []
    for year, data in all_results.items():
        for comp in data.get("comparisons", []):
            rows.append(comp)
    if rows:
        pd.DataFrame(rows).to_csv("outputs/fairness/air_results.csv", index=False)
    print(f"\n  Saved to outputs/fairness/air_results.{{json,csv}}")
    print("Done.")

if __name__ == "__main__":
    main()
