"""FT-Transformer preprocessing -- separate from the WOE baseline pipeline.
Produces numpy arrays ready for the FT-Transformer model:
  x_num: float32 (n, n_num_features) -- standardized numeric features
  x_cat: int64   (n, n_cat_features) -- integer-coded categorical features
  y:     int64   (n,)                -- binary target
Fit on training data only. Call transform() on val/test without re-fitting.
Saved alongside each year's ensemble for inference reproducibility.
"""
from __future__ import annotations
import os
import pickle
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.features.display_names import pretty

# Numeric features present in all 17 years.
# Order must be stable across train/val/test splits because the
# FT-Transformer expects (x_num, x_cat) in a fixed column order.
NUMERIC_ALL_YEARS: list[str] = [
    "loan_amount_000s", "applicant_income_000s",
    "loan_to_income", "log_loan_amount", "log_income",
    "tract_to_msamd_income", "hud_median_family_income",
    "log_area_median_income",
    "has_co_applicant",  # binary 0/1, treated as numeric
]
# Tier 2 numeric features (DTI, LTV, loan term) are intentionally excluded.
# Their NaN missingness is correlated with the target (2-3x higher for
# denied loans), causing near-perfect leakage (AUC 0.997 when included).
NUMERIC_POST2018: list[str] = []
# Categorical features (all years)
CATEGORICAL_ALL_YEARS: list[str] = [
    "loan_purpose_name", "lien_status_name", "property_type_name",
    "preapproval_name", "loan_type_name", "owner_occupancy_name",
    # hoepa_status_name excluded: post-decision leakage
]

class FTPreprocessor:
    """Stateful preprocessor for FT-Transformer inputs.
    Fit once on training data, persist to disk, reuse for val/test/bootstrap.
    Uses only Tier 1 features (available all 17 years) to avoid leakage.
    """
    def __init__(self, year: int):
        self.year = year
        self.is_post2018 = year >= 2018
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.cat_cardinalities_: list[int] = []
        self._scaler = StandardScaler()
        self._encoders: dict[str, LabelEncoder] = {}
        self._cat_unknown_code_: dict[str, int] = {}
        self._num_fill_values_: dict[str, float] = {}
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "FTPreprocessor":
        """Fit scalers and encoders on the training split only."""
        # Only use Tier 1 numeric features (Tier 2 excluded for leakage)
        candidate_num = list(NUMERIC_ALL_YEARS)
        self.numeric_cols_ = [c for c in candidate_num if c in df_train.columns]
        self.categorical_cols_ = [
            c for c in CATEGORICAL_ALL_YEARS if c in df_train.columns
        ]
        # Store training median for each numeric column (used to fill NaNs)
        for col in self.numeric_cols_:
            series = pd.to_numeric(df_train[col], errors="coerce")
            med = series.median()
            self._num_fill_values_[col] = float(med) if pd.notna(med) else 0.0
        # Fit StandardScaler so each numeric feature has mean=0, std=1.
        # This is important for Transformer attention to work well.
        X_num = self._extract_numeric(df_train, fit=True)
        self._scaler.fit(X_num)
        # Fit a LabelEncoder per categorical feature.
        # We add an explicit __UNKNOWN__ class so unseen categories at
        # test time get a valid code instead of crashing.
        self.cat_cardinalities_ = []
        for col in self.categorical_cols_:
            le = LabelEncoder()
            values = df_train[col].astype(str).fillna("__MISSING__").values
            le.fit(np.append(values, "__UNKNOWN__"))
            self._encoders[col] = le
            self._cat_unknown_code_[col] = int(
                np.where(le.classes_ == "__UNKNOWN__")[0][0]
            )
            self.cat_cardinalities_.append(len(le.classes_))
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_num float32, x_cat int64) ready for the model."""
        assert self.is_fitted, "Call fit() first."
        # Scale numeric features using the training set's mean and std
        X_num = self._extract_numeric(df, fit=False)
        X_num = self._scaler.transform(X_num).astype(np.float32)
        # Encode categorical features as integers
        if self.categorical_cols_:
            cats = []
            for col in self.categorical_cols_:
                le = self._encoders[col]
                raw = df[col].astype(str).fillna("__MISSING__").values
                known = set(le.classes_)
                # Map unseen values to the __UNKNOWN__ code
                coded = np.array([
                    le.transform([v])[0] if v in known
                    else self._cat_unknown_code_[col]
                    for v in raw
                ], dtype=np.int64)
                cats.append(coded.reshape(-1, 1))
            X_cat = np.hstack(cats).astype(np.int64)
        else:
            X_cat = np.empty((len(df), 0), dtype=np.int64)
        return X_num, X_cat

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one call."""
        return self.fit(df).transform(df)

    def feature_summary(self) -> dict:
        """Return a dict describing the fitted feature set (for manifests)."""
        return {
            "year": self.year,
            "n_num_features": len(self.numeric_cols_),
            "n_cat_features": len(self.categorical_cols_),
            "cat_cardinalities": self.cat_cardinalities_,
            "numeric_cols": self.numeric_cols_,
            "categorical_cols": self.categorical_cols_,
            "numeric_display": pretty(self.numeric_cols_),
            "categorical_display": pretty(self.categorical_cols_),
        }

    def save(self, path: str) -> None:
        """Persist fitted preprocessor to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FTPreprocessor":
        """Load a fitted preprocessor from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _extract_numeric(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract numeric columns, filling missing with training median.
        During fit: computes median on the fly.
        During transform: uses stored median values from training.
        """
        cols = []
        for col in self.numeric_cols_:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                fill_val = self._num_fill_values_.get(col, 0.0)
                cols.append(s.fillna(fill_val).values.astype(np.float32).reshape(-1, 1))
            else:
                cols.append(np.zeros((len(df), 1), dtype=np.float32))
        return np.hstack(cols) if cols else np.empty((len(df), 0), dtype=np.float32)
