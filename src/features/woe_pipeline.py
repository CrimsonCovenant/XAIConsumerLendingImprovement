"""WOE encoding pipeline -- industry-standard credit-scoring feature transform.
Weight of Evidence (WOE) converts every feature into a single number that
represents its relationship with the target (approve/deny). This makes
logistic regression coefficients directly interpretable as log-odds changes.
Numeric features are first binned into optimal monotonic bins (using the
optbinning library), then WOE-encoded. Categorical features are directly
WOE-encoded with Laplace smoothing (using category_encoders).
After fitting, features with Information Value (IV) below a threshold are
dropped. IV < 0.02 means the feature has almost no predictive power
(Siddiqi's standard credit-scoring threshold).
"""
from __future__ import annotations
import logging
from typing import Any
import numpy as np
import pandas as pd
from category_encoders import WOEEncoder
from optbinning import OptimalBinning
from src.features.display_names import pretty

logger = logging.getLogger(__name__)

class WOEPipeline:
    """Weight-of-Evidence encoding for numeric and categorical features.
    iv_threshold: minimum Information Value to keep a feature (default 0.02).
    """
    def __init__(self, iv_threshold: float = 0.02):
        self.iv_threshold = iv_threshold
        self._numeric_binners: dict[str, OptimalBinning] = {}
        self._cat_encoder: WOEEncoder | None = None
        self._numeric_features: list[str] = []
        self._categorical_features: list[str] = []
        self._retained_features: list[str] = []
        self._dropped_features: list[str] = []
        self._iv_scores: dict[str, float] = {}
        self._is_fitted: bool = False

    def fit(
        self, df_train: pd.DataFrame, y_train: np.ndarray | pd.Series,
        numeric_features: list[str], categorical_features: list[str],
    ) -> "WOEPipeline":
        """Fit WOE encoding on the training set only.
        df_train: raw feature values.
        y_train: binary target (0=denied, 1=approved).
        """
        y = np.asarray(y_train).ravel()
        self._numeric_features = list(numeric_features)
        self._categorical_features = list(categorical_features)
        # Numeric features: use OptimalBinning to find monotonic bins,
        # then compute WOE for each bin. monotonic_trend='auto' lets the
        # library decide whether the relationship is increasing or decreasing.
        for feat in self._numeric_features:
            x = df_train[feat].values.astype(float)
            binner = OptimalBinning(
                name=feat, dtype="numerical", monotonic_trend="auto",
                solver="cp", min_bin_size=0.05,
            )
            binner.fit(x, y)
            self._numeric_binners[feat] = binner
            # Extract total Information Value from the binning table
            table = binner.binning_table.build()
            iv = table["IV"].sum() if "IV" in table.columns else 0.0
            self._iv_scores[feat] = float(iv)
        # Categorical features: WOEEncoder with Laplace regularization
        # (regularization=1.0 adds a small count to each category to prevent
        # infinite WOE values when a category has zero events or non-events)
        if self._categorical_features:
            self._cat_encoder = WOEEncoder(
                cols=self._categorical_features, regularization=1.0,
            )
            self._cat_encoder.fit(df_train[self._categorical_features], y)
            for feat in self._categorical_features:
                iv = self._compute_cat_iv(df_train[feat], y)
                self._iv_scores[feat] = float(iv)
        # Drop features below the IV threshold. Features with IV < 0.02
        # have almost no predictive power and just add noise.
        all_features = self._numeric_features + self._categorical_features
        self._retained_features = [
            f for f in all_features if self._iv_scores.get(f, 0) >= self.iv_threshold
        ]
        self._dropped_features = [
            f for f in all_features if f not in self._retained_features
        ]
        if self._dropped_features:
            logger.info(
                "Dropped %d features with IV < %.3f: %s",
                len(self._dropped_features), self.iv_threshold,
                [pretty(f) for f in self._dropped_features],
            )
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features to WOE values. Returns a 2-D NumPy array.
        Only retained features (IV >= threshold) are included.
        """
        assert self._is_fitted, "Call fit() first."
        parts = []
        # Numeric: apply each binner's WOE transform
        retained_numeric = [
            f for f in self._retained_features if f in self._numeric_binners
        ]
        for feat in retained_numeric:
            x = df[feat].values.astype(float)
            woe_vals = self._numeric_binners[feat].transform(x, metric="woe")
            parts.append(woe_vals.reshape(-1, 1))
        # Categorical: transform all at once through the encoder, then pick retained
        retained_cat = [
            f for f in self._retained_features if f in self._categorical_features
        ]
        if retained_cat and self._cat_encoder is not None:
            all_cat_encoded = self._cat_encoder.transform(
                df[self._categorical_features]
            )
            for feat in retained_cat:
                parts.append(all_cat_encoded[feat].values.reshape(-1, 1))
        if not parts:
            raise ValueError("No features survived IV filtering!")
        return np.hstack(parts)

    def information_values(self) -> dict[str, float]:
        """Return {display_name: IV} for all features (before filtering).
        Used in visualizations and audit artifacts.
        """
        return {pretty(k): v for k, v in self._iv_scores.items()}

    def retained_feature_names(self) -> list[str]:
        """Raw column names of features that survived IV filtering."""
        return list(self._retained_features)

    def retained_display_names(self) -> list[str]:
        """Display names of retained features (for figure axes).
        Uses the display name mapping from src/features/display_names.py.
        """
        return [pretty(f) for f in self._retained_features]

    def bin_table(self, feature: str) -> pd.DataFrame:
        """Return the monotonic bin table for a numeric feature."""
        if feature not in self._numeric_binners:
            raise KeyError(f"{feature} is not a binned numeric feature.")
        return self._numeric_binners[feature].binning_table.build()

    @staticmethod
    def _compute_cat_iv(series: pd.Series, y: np.ndarray) -> float:
        """Compute Information Value for a single categorical feature.
        IV measures how well a feature separates approved from denied.
        Laplace smoothing (+0.5) prevents log(0) when a category has
        zero events or non-events.
        """
        df_tmp = pd.DataFrame({"feat": series.values, "target": y})
        total_events = y.sum()
        total_non_events = len(y) - total_events
        if total_events == 0 or total_non_events == 0:
            return 0.0
        iv = 0.0
        for _, grp in df_tmp.groupby("feat"):
            events = grp["target"].sum()
            non_events = len(grp) - events
            dist_events = (events + 0.5) / (total_events + 1)
            dist_non_events = (non_events + 0.5) / (total_non_events + 1)
            woe = np.log(dist_non_events / dist_events)
            iv += (dist_non_events - dist_events) * woe
        return iv
