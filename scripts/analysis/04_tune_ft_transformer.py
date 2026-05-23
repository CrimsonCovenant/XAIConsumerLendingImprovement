"""Optuna hyperparameter tuning for FT-Transformer on a single HMDA year.
Tunes the Transformer architecture (d_block, n_blocks, dropout rates) and
training parameters (lr, weight_decay, batch_size) using Optuna's Bayesian
optimization. Runs each trial on a 10% subsample for speed.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.features.ft_preprocessor import FTPreprocessor
from src.models.ft_transformer import FTTransformerLending
from src.models.ft_train import train_one_member
from src.utils.reproducibility import seed_everything, select_device, write_manifest

def main():
    parser = argparse.ArgumentParser(description="Tune FT-Transformer hyperparameters with Optuna")
    parser.add_argument("--year", type=int, default=2008, help="Year to tune on (default: 2008)")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials (default: 20)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--subsample", type=float, default=0.10,
                        help="Fraction of training data for tuning speed (default: 0.10)")
    args = parser.parse_args()
    with open("config/config.yml") as f:
        config = yaml.safe_load(f)
    device = select_device(force_cpu=args.cpu)
    seed = 42
    year = args.year
    print(f"FT-Transformer Optuna tuning -- year {year}")
    print(f"Device: {device}  |  Trials: {args.n_trials}")
    # Load and split data
    parquet_path = os.path.join(config["data"]["processed"], f"processed_{year}.parquet")
    df = pd.read_parquet(parquet_path)
    y = df["loan_status"].values.astype(np.int64)
    df_feat = df.drop(columns=["loan_status"])
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    train_y = y[train_idx]
    # Split train into actual train + validation for early stopping
    tr_rel, val_rel = train_test_split(
        np.arange(len(train_idx)), test_size=0.1, random_state=seed, stratify=train_y,
    )
    actual_train_idx = train_idx[tr_rel]
    val_idx = train_idx[val_rel]
    # Preprocess features
    prep = FTPreprocessor(year=year)
    x_num_tr, x_cat_tr = prep.fit_transform(df_feat.iloc[actual_train_idx])
    x_num_val, x_cat_val = prep.transform(df_feat.iloc[val_idx])
    y_tr = y[actual_train_idx]
    y_val = y[val_idx]
    feat_summary = prep.feature_summary()
    n_num = feat_summary["n_num_features"]
    cat_cards = feat_summary["cat_cardinalities"]
    print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}")
    print(f"  Features: {n_num} numeric, {feat_summary['n_cat_features']} categorical")
    # Subsample training data for faster tuning iterations
    n_sample = max(int(len(y_tr) * args.subsample), 10000)
    _, sub_idx = train_test_split(
        np.arange(len(y_tr)), test_size=n_sample / len(y_tr),
        random_state=seed, stratify=y_tr,
    )
    x_num_sub = x_num_tr[sub_idx]
    x_cat_sub = x_cat_tr[sub_idx]
    y_sub = y_tr[sub_idx]
    print(f"  Tuning subsample: {len(y_sub):,} rows ({args.subsample:.0%})")
    # Each Optuna trial trains a model with a different set of hyperparameters
    # and returns the best validation AUC achieved during training.
    def objective(trial: optuna.Trial) -> float:
        d_block = trial.suggest_categorical("d_block", [96, 128, 192, 256])
        n_blocks = trial.suggest_int("n_blocks", 2, 4)
        attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.4, step=0.05)
        ffn_dropout = trial.suggest_float("ffn_dropout", 0.05, 0.3, step=0.05)
        residual_dropout = trial.suggest_float("residual_dropout", 0.0, 0.2, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192])
        seed_everything(seed + trial.number)
        model = FTTransformerLending(
            n_num=n_num, cat_cardinalities=cat_cards,
            d_block=d_block, n_blocks=n_blocks,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        ).to(device)
        result = train_one_member(
            model,
            x_num_sub, x_cat_sub, y_sub,
            x_num_val, x_cat_val, y_val,
            lr=lr, weight_decay=weight_decay,
            n_epochs=60, batch_size=batch_size,
            patience=8, device=device,
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        return result["best_val_auc"]
    # Run the Optuna study
    out_dir = os.path.join("outputs", "tuning", f"ft_optuna_{year}")
    os.makedirs(out_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(out_dir, 'study.db')}"
    study = optuna.create_study(
        study_name=f"ft_transformer_{year}",
        direction="maximize", storage=storage, load_if_exists=True,
    )
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - t0
    # Print and save results
    best = study.best_trial
    print(f"Best trial #{best.number}  |  val_auc={best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:25s} = {v}")
    print(f"\nTotal tuning time: {elapsed/60:.1f} min")
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({
            "best_val_auc": best.value,
            "best_trial": best.number,
            "params": best.params,
            "n_trials": len(study.trials),
        }, f, indent=2)
    write_manifest(
        out_dir,
        year=year, n_trials=len(study.trials),
        best_val_auc=best.value, best_params=best.params,
        wall_clock_s=round(elapsed, 1),
        subsample_fraction=args.subsample, device=str(device),
    )
    print(f"\nSaved to {out_dir}/")
    print(f"\nTo apply these params, update FT_CONFIG in 06_train_ft_ensembles.py")

if __name__ == "__main__":
    main()
