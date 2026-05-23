"""Train FT-Transformer ensembles for all 17 HMDA years.
For each year, this script:
  1. Loads the preprocessed parquet
  2. Splits 70/20/10 train/test/calibration (same split as baseline)
  3. Fits the FTPreprocessor (StandardScaler + LabelEncoder) on train only
  4. Trains a 5-member bagged ensemble with early stopping on val AUC
  5. Calibrates with Platt scaling on the calibration set
  6. Evaluates on the held-out test set and saves all artifacts
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
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.features.ft_preprocessor import FTPreprocessor
from src.models.ft_ensemble import FTEnsemble
from src.models.calibration import PlattScaler
from src.utils.reproducibility import (
    seed_everything, select_device, assert_clean_git,
    write_manifest, file_sha256, get_library_versions,
)

# Hyperparameters for the FT-Transformer ensemble.
# d_block and n_blocks control model capacity.
# lr=3e-3 uses the linear scaling rule: base lr 1e-3 * (batch_size/1024).
FT_CONFIG = {
    "d_block": 192, "n_blocks": 3,
    "attention_dropout": 0.2, "ffn_dropout": 0.1, "residual_dropout": 0.0,
    "lr": 3e-3, "weight_decay": 1e-5,
    "n_epochs": 100, "batch_size": 4096, "patience": 10,
    "K": 5, "subsample_fraction": 0.15,
    "test_size": 0.30, "cal_fraction": 1 / 3, "seed": 42,
}

def checkpoint_exists(year: int, out_root: str) -> bool:
    """Check if a year's training is already complete."""
    out_dir = os.path.join(out_root, f"ft_{year}")
    return os.path.isfile(os.path.join(out_dir, "_run_manifest.json"))

def ks_stat(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for credit scoring."""
    order = np.argsort(y_prob)
    yt = y_true[order]
    npos = yt.sum()
    nneg = len(yt) - npos
    if npos == 0 or nneg == 0:
        return 0.0
    cdf_pos = np.cumsum(yt) / npos
    cdf_neg = np.cumsum(1 - yt) / nneg
    return float(np.max(np.abs(cdf_pos - cdf_neg)))

def train_year(year: int, config: dict, out_root: str, device: torch.device) -> dict:
    """Full FT-Transformer pipeline for one year. Returns metrics dict."""
    out_dir = os.path.join(out_root, f"ft_{year}")
    os.makedirs(out_dir, exist_ok=True)
    seed = FT_CONFIG["seed"]
    # Load preprocessed data
    parquet_path = os.path.join(config["data"]["processed"], f"processed_{year}.parquet")
    df = pd.read_parquet(parquet_path)
    # Leakage guardrail: same as baseline -- crash if post-decision columns exist
    _LEAKED_COLS = [
        "debt_to_income_ratio", "loan_to_value_ratio", "loan_term",
        "interest_rate", "rate_spread", "total_loan_costs",
        "origination_charges", "discount_points", "hoepa_status_name",
    ]
    for col in _LEAKED_COLS:
        assert col not in df.columns, (
            f"CRITICAL LEAKAGE: '{col}' found in {year} data! "
            f"Rebuild parquets: python scripts/analysis/02_preprocess.py --years {year} --force"
        )
    y = df["loan_status"].values.astype(np.int64)
    df_feat = df.drop(columns=["loan_status"])
    print(f"  Loaded {len(df):,} rows, {len(df_feat.columns)} features")
    # Split: 70 train / 20 test / 10 cal (matches baseline exactly)
    idx = np.arange(len(df))
    train_idx, rest_idx = train_test_split(
        idx, test_size=FT_CONFIG["test_size"], random_state=seed, stratify=y,
    )
    rest_y = y[rest_idx]
    test_rel, cal_rel = train_test_split(
        np.arange(len(rest_idx)),
        test_size=FT_CONFIG["cal_fraction"], random_state=seed, stratify=rest_y,
    )
    test_idx = rest_idx[test_rel]
    cal_idx = rest_idx[cal_rel]
    # Further split train into train/val for early stopping (90/10)
    train_y = y[train_idx]
    tr_rel, val_rel = train_test_split(
        np.arange(len(train_idx)), test_size=0.1, random_state=seed, stratify=train_y,
    )
    actual_train_idx = train_idx[tr_rel]
    val_idx = train_idx[val_rel]
    print(
        f"  Split: train={len(actual_train_idx):,}  val={len(val_idx):,}  "
        f"test={len(test_idx):,}  cal={len(cal_idx):,}"
    )
    print(f"  Class balance: {y[actual_train_idx].mean():.1%} approve")
    # Preprocess: fit scaler/encoder on training data only
    prep = FTPreprocessor(year=year)
    x_num_tr, x_cat_tr = prep.fit_transform(df_feat.iloc[actual_train_idx])
    x_num_val, x_cat_val = prep.transform(df_feat.iloc[val_idx])
    x_num_cal, x_cat_cal = prep.transform(df_feat.iloc[cal_idx])
    x_num_te, x_cat_te = prep.transform(df_feat.iloc[test_idx])
    y_tr = y[actual_train_idx]
    y_val = y[val_idx]
    y_cal = y[cal_idx]
    y_te = y[test_idx]
    feat_summary = prep.feature_summary()
    print(
        f"  Features: {feat_summary['n_num_features']} numeric, "
        f"{feat_summary['n_cat_features']} categorical"
    )
    # Train ensemble: 5 members, each on a unique 15% stratified subsample
    model_kwargs = {
        "n_num": feat_summary["n_num_features"],
        "cat_cardinalities": feat_summary["cat_cardinalities"],
        "d_block": FT_CONFIG["d_block"],
        "n_blocks": FT_CONFIG["n_blocks"],
        "attention_dropout": FT_CONFIG["attention_dropout"],
        "ffn_dropout": FT_CONFIG["ffn_dropout"],
        "residual_dropout": FT_CONFIG["residual_dropout"],
    }
    train_kwargs = {
        k: FT_CONFIG[k]
        for k in ["lr", "weight_decay", "n_epochs", "batch_size", "patience"]
    }
    # Use year-specific seed so each year's ensemble is independently reproducible
    seed_everything(seed + year)
    ensemble = FTEnsemble(
        K=FT_CONFIG["K"], base_seed=seed + year,
        subsample_fraction=FT_CONFIG["subsample_fraction"],
        model_kwargs=model_kwargs, train_kwargs=train_kwargs,
    )
    ensemble.fit(x_num_tr, x_cat_tr, y_tr, x_num_val, x_cat_val, y_val, device=device)
    # Calibrate on the held-out calibration set
    p_cal = ensemble.predict_proba(x_num_cal, x_cat_cal, device)
    platt = PlattScaler().fit(p_cal, y_cal)
    # Evaluate on the test set
    p_te_raw = ensemble.predict_proba(x_num_te, x_cat_te, device)
    p_te_cal = platt.transform(p_te_raw)
    auc = roc_auc_score(y_te, p_te_cal)
    ks = ks_stat(y_te, p_te_cal)
    gini = 2 * auc - 1
    metrics = {
        "auc": round(auc, 4), "ks": round(ks, 4), "gini": round(gini, 4),
        "n_test": int(len(y_te)), "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)), "n_cal": int(len(y_cal)),
    }
    print(f"\n  TEST  AUC={auc:.4f}  KS={ks:.4f}  Gini={gini:.4f}")
    # Save all artifacts
    ensemble.save(out_dir)
    prep.save(os.path.join(out_dir, "ft_preprocessor.pkl"))
    with open(os.path.join(out_dir, "platt.pkl"), "wb") as f:
        pickle.dump(platt, f)
    np.save(os.path.join(out_dir, "test_preds.npy"), p_te_cal)
    np.save(os.path.join(out_dir, "test_labels.npy"), y_te)
    return metrics, feat_summary

def main():
    parser = argparse.ArgumentParser(description="Train FT-Transformer ensembles for HMDA years")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Specific years to process (default: all from config)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing checkpoints")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (audit mode -- slower but bit-reproducible)")
    args = parser.parse_args()
    try:
        git_sha = assert_clean_git()
    except RuntimeError:
        print("WARNING: dirty git tree -- proceeding anyway for dev mode")
        git_sha = "dirty"
    with open("config/config.yml") as f:
        config = yaml.safe_load(f)
    years = args.years or config["pipeline"]["years"]
    out_root = config["outputs"]["models"]
    device = select_device(force_cpu=args.cpu)
    print("FT-Transformer ensemble training")
    print(f"Device: {device}  |  Years: {years}  |  K={FT_CONFIG['K']}")
    print(f"Git SHA: {git_sha[:8]}")
    results_all = {}
    t_total = time.time()
    for year in years:
        if not args.force and checkpoint_exists(year, out_root):
            print(f"\nYear {year}: checkpoint found, skipping (use --force to overwrite)")
            continue
        print(f"Year {year}")
        t0 = time.time()
        try:
            metrics, feat_summary = train_year(year, config, out_root, device)
            elapsed = time.time() - t0
            metrics["wall_clock_s"] = round(elapsed, 1)
            results_all[year] = metrics
            write_manifest(
                os.path.join(out_root, f"ft_{year}"),
                git_sha=git_sha, year=year, device=str(device),
                ft_config=FT_CONFIG, feature_summary=feat_summary,
                metrics=metrics, wall_clock_s=round(elapsed, 1),
                platt_a=PlattScaler.__dict__.get("a", "see platt.pkl"),
                library_versions=get_library_versions(),
            )
            print(f"  Saved to outputs/models/ft_{year}/  ({elapsed:.0f}s)")
        except Exception as e:
            print(f"  ERROR year {year}: {e}")
            import traceback
            traceback.print_exc()
            raise
    total_time = time.time() - t_total
    print(f"\nAll years complete. Total time: {total_time/60:.1f} min")
    # Print summary table
    if results_all:
        print(f"\n{'Year':<6} {'AUC':>7} {'KS':>7} {'Gini':>7} {'Time(s)':>9}")
        for yr, m in sorted(results_all.items()):
            print(
                f"  {yr}  {m['auc']:>6.4f}  {m['ks']:>6.4f}  "
                f"{m['gini']:>6.4f}  {m.get('wall_clock_s', 0):>8.0f}"
            )

if __name__ == "__main__":
    main()
