"""Deterministic feature pipeline consumed by both the baseline and advanced model.
Handles the 2018 HMDA schema break by mapping post-2018 data back to
canonical pre-2018 column names (which align with the display name mappings
in src/features/display_names.py).
Protected attributes (race, sex, ethnicity) are separated from modeling
features and returned via feature_meta for the fairness audit in Phase 8.
All 17 years use the same 15 pre-decision features. Leakage traps that
were permanently excluded:
  - hoepa_status: APR/fees only exist for approved loans (denied = 'Not Applicable')
  - rate_spread: NaN encodes denial
  - interest_rate, total_loan_costs, origination_charges: pricing variables
  - debt_to_income_ratio, loan_to_value_ratio, loan_term: NaN rate 2-3x
    higher for denied loans, giving models a trivial shortcut (AUC 0.997)
"""
from __future__ import annotations
import logging
import os
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Which raw columns to read from pre-2018 HMDA files
PRE2018_USECOLS: list[str] = [
    "action_taken",
    "loan_amount_000s", "applicant_income_000s",
    "loan_purpose_name", "lien_status_name", "property_type_name",
    "owner_occupancy_name",
    # hoepa_status_name excluded: post-decision leakage
    "loan_type_name", "preapproval_name",
    "co_applicant_sex_name",  # used to derive has_co_applicant
    "tract_to_msamd_income", "hud_median_family_income",
    # Protected attributes (kept separately for fairness audit)
    "applicant_sex_name", "applicant_race_name_1", "applicant_ethnicity_name",
]

# Which raw columns to read from post-2018 HMDA files
POST2018_USECOLS: list[str] = [
    "action_taken",
    "loan_amount", "income",
    "loan_purpose", "lien_status",
    # hoepa_status excluded: post-decision leakage
    "occupancy_type", "derived_dwelling_category",
    "loan_type", "preapproval",
    "co_applicant_sex",
    "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income",
    # Tier 2 excluded: post-decision leakage traps
    "derived_sex", "derived_race", "derived_ethnicity",
]

# Post-2018 numeric codes to human-readable strings.
# These mappings come from the HMDA LAR data dictionary.
_LOAN_PURPOSE_MAP: dict[Any, str] = {
    1: "Home purchase", 2: "Home improvement",
    31: "Refinancing", 32: "Cash-out refinancing",
    4: "Other purpose", 5: "Not applicable",
}
_LIEN_STATUS_MAP: dict[Any, str] = {
    1: "Secured by a first lien", 2: "Secured by a subordinate lien",
}
_HOEPA_STATUS_MAP: dict[Any, str] = {
    1: "HOEPA loan", 2: "Not a HOEPA loan", 3: "Not applicable",
}
_OCCUPANCY_MAP: dict[Any, str] = {
    1: "Owner-occupied as a principal dwelling",
    2: "Not owner-occupied as a principal dwelling",
    3: "Not owner-occupied as a principal dwelling",
}
# derived_dwelling_category string to pre-2018 property_type_name
_DWELLING_TO_PROPERTY: dict[str, str] = {
    "Single Family (1-4 Units):Site-Built":  "One-to-four family dwelling",
    "Single Family (1-4 Units):Manufactured": "Manufactured housing",
    "Multifamily:Site-Built":                 "Multifamily dwelling",
    "Multifamily:Manufactured":               "Multifamily dwelling",
}
_LOAN_TYPE_MAP: dict[Any, str] = {
    1: "Conventional", 2: "FHA-insured",
    3: "VA-guaranteed", 4: "FSA/RHS-guaranteed",
}
_PREAPPROVAL_MAP: dict[Any, str] = {
    1: "Preapproval was requested", 2: "Preapproval was not requested",
}

# Canonical column names that come out of the pipeline (pre-2018 names)
_TIER1_MODELING_FEATURES: list[str] = [
    "loan_amount_000s", "applicant_income_000s",
    "loan_purpose_name", "lien_status_name", "property_type_name",
    "owner_occupancy_name",
    # hoepa_status_name excluded: post-decision leakage
    "loan_type_name", "preapproval_name",
    "has_co_applicant",
    "tract_to_msamd_income", "hud_median_family_income",
]
# Tier 2 is intentionally empty. These features (DTI, LTV, loan term)
# are post-decision variables whose NaN rate is 2-3x higher for denied
# loans, creating a trivial shortcut that inflated AUC to 0.997.
_TIER2_MODELING_FEATURES: list[str] = []
_PROTECTED_ATTRS: list[str] = [
    "applicant_sex_name", "applicant_race_name_1", "applicant_ethnicity_name",
]
_TARGET_COL = "loan_status"
# Derived features computed from Tier 1 base features
_DERIVED_FEATURES: list[str] = [
    "loan_to_income", "log_loan_amount", "log_income", "log_area_median_income",
]

# Post-2018 DTI is reported as binned strings like '<20%' or '30%-<36%'.
# We map each bin to its midpoint for numeric use.
_DTI_BIN_MIDPOINTS: dict[str, float] = {
    "<20%": 10.0, "20%-<30%": 25.0, "30%-<36%": 33.0,
    "36": 36.0, "37": 37.0, "38": 38.0, "39": 39.0,
    "40": 40.0, "41": 41.0, "42": 42.0, "43": 43.0,
    "44": 44.0, "45": 45.0, "46": 46.0, "47": 47.0,
    "48": 48.0, "49": 49.0,
    "50%-60%": 55.0, ">60%": 70.0,
}

def _parse_dti(series: pd.Series) -> pd.Series:
    """Parse post-2018 DTI binned strings to numeric midpoint values.
    Unrecognized bins (like 'Exempt' or 'NA') become NaN.
    """
    result = series.astype(str).str.strip().map(_DTI_BIN_MIDPOINTS)
    mask_nan = result.isna()
    numeric_attempt = pd.to_numeric(series[mask_nan], errors="coerce")
    result.loc[mask_nan] = numeric_attempt
    return result

def _load_pre2018(year: int, raw_dir: str) -> pd.DataFrame:
    """Load only the needed columns from a pre-2018 raw CSV."""
    path = os.path.join(raw_dir, f"year_{year}.csv")
    logger.info("Loading pre-2018 data for %d from %s", year, path)
    available_cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in PRE2018_USECOLS if c in available_cols]
    df = pd.read_csv(path, usecols=usecols, engine="pyarrow")
    return df

def _load_post2018(year: int, raw_dir: str) -> pd.DataFrame:
    """Load only the needed columns from a post-2018 raw CSV.
    Uses chunked reading because post-2018 files can be several GB.
    """
    path = os.path.join(raw_dir, f"year_{year}.csv")
    logger.info("Loading post-2018 data for %d from %s (chunked)", year, path)
    available_cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in POST2018_USECOLS if c in available_cols]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols,
                             chunksize=200_000, low_memory=False):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def _derive_co_applicant_pre2018(df: pd.DataFrame) -> pd.Series:
    """Derive has_co_applicant from co_applicant_sex_name (pre-2018).
    If the column says 'No co-applicant', that means 0; otherwise 1.
    """
    if "co_applicant_sex_name" in df.columns:
        return (~df["co_applicant_sex_name"]
                .astype(str).str.lower()
                .str.contains("no co-applicant", na=True)).astype(int)
    return pd.Series(np.nan, index=df.index)

def _derive_co_applicant_post2018(df: pd.DataFrame) -> pd.Series:
    """Derive has_co_applicant from co_applicant_sex (post-2018).
    Post-2018 codes: 5 means no co-applicant; anything else means has one.
    """
    col = None
    for candidate in ["co_applicant_sex", "co-applicant_sex"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is not None:
        val = pd.to_numeric(df[col], errors="coerce")
        return (val.notna() & (val != 5)).astype(int)
    return pd.Series(np.nan, index=df.index)

def _harmonize_post2018(df: pd.DataFrame) -> pd.DataFrame:
    """Map post-2018 columns to canonical pre-2018 column names and values.
    This is the heart of the schema harmonization: after this function,
    all years look identical regardless of which HMDA schema they came from.
    """
    out = pd.DataFrame()
    # Post-2018 loan_amount is in dollars; convert to thousands to match pre-2018
    out["loan_amount_000s"] = pd.to_numeric(df["loan_amount"], errors="coerce") / 1000.0
    # Post-2018 income is already in thousands
    out["applicant_income_000s"] = pd.to_numeric(df["income"], errors="coerce")
    # Map numeric codes to human-readable strings using HMDA data dictionary
    out["loan_purpose_name"] = (
        pd.to_numeric(df["loan_purpose"], errors="coerce").map(_LOAN_PURPOSE_MAP)
    )
    out["lien_status_name"] = (
        pd.to_numeric(df["lien_status"], errors="coerce").map(_LIEN_STATUS_MAP)
    )
    # hoepa_status excluded: post-decision leakage
    out["owner_occupancy_name"] = (
        pd.to_numeric(df["occupancy_type"], errors="coerce").map(_OCCUPANCY_MAP)
    )
    # Property type: map the derived_dwelling_category string
    out["property_type_name"] = df["derived_dwelling_category"].map(_DWELLING_TO_PROPERTY)
    out["loan_type_name"] = (
        pd.to_numeric(df["loan_type"], errors="coerce").map(_LOAN_TYPE_MAP)
    )
    if "preapproval" in df.columns:
        out["preapproval_name"] = (
            pd.to_numeric(df["preapproval"], errors="coerce").map(_PREAPPROVAL_MAP)
        )
    out["has_co_applicant"] = _derive_co_applicant_post2018(df)
    # Geographic features: rename to canonical names
    if "tract_to_msa_income_percentage" in df.columns:
        out["tract_to_msamd_income"] = pd.to_numeric(
            df["tract_to_msa_income_percentage"], errors="coerce"
        )
    if "ffiec_msa_md_median_family_income" in df.columns:
        out["hud_median_family_income"] = pd.to_numeric(
            df["ffiec_msa_md_median_family_income"], errors="coerce"
        )
    # Tier 2 features NOT created. Their NaN missingness is correlated with
    # the target (2-3x higher for denied loans), creating a leakage shortcut.
    out["action_taken"] = pd.to_numeric(df["action_taken"], errors="coerce")
    # Protected attributes: rename to canonical pre-2018 names
    out["applicant_sex_name"] = df["derived_sex"].astype(str)
    out["applicant_race_name_1"] = df["derived_race"].astype(str)
    out["applicant_ethnicity_name"] = df["derived_ethnicity"].astype(str)
    return out

def _add_pre2018_tier1(df: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Add has_co_applicant for pre-2018 data (derived from co_applicant_sex_name)."""
    df = df.copy()
    df["has_co_applicant"] = _derive_co_applicant_pre2018(df_raw)
    return df

def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from the base Tier 1 features.
    loan_to_income: winsorized at 1st/99th percentile to limit outliers.
    log transforms: stabilize skewed distributions so the model does not
    overweight extreme values (e.g. a $5M loan vs a $200K loan).
    """
    df = df.copy()
    # Loan-to-income ratio, winsorized to reduce outlier impact
    raw_ratio = df["loan_amount_000s"] / df["applicant_income_000s"].replace(0, np.nan)
    lo, hi = raw_ratio.quantile(0.01), raw_ratio.quantile(0.99)
    df["loan_to_income"] = raw_ratio.clip(lo, hi)
    # Log transforms: clip to positive first because some post-2018
    # records have zero or negative values after coercion from 'Exempt'
    df["log_loan_amount"] = np.log1p(df["loan_amount_000s"].clip(lower=0))
    df["log_income"] = np.log1p(df["applicant_income_000s"].clip(lower=0))
    if "hud_median_family_income" in df.columns:
        df["log_area_median_income"] = np.log1p(
            df["hud_median_family_income"].clip(lower=0)
        )
    return df

def preprocess_year(
    year: int, config: dict, schema_break: int = 2018,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full preprocessing pipeline for a single year.
    Returns (df_model, df_protected, meta):
      df_model: modeling features + target + derived features
      df_protected: protected attributes only, row-aligned with df_model
      meta: manifest metadata (row counts, drop counts, feature lists)
    """
    raw_dir = config["data"]["raw"]
    regime = "pre2018" if year < schema_break else "post2018"
    is_post2018 = regime == "post2018"
    # Load raw data using the appropriate schema loader
    if regime == "pre2018":
        df = _load_pre2018(year, raw_dir)
        df = _add_pre2018_tier1(df, df)
        df = df.drop(columns=["co_applicant_sex_name"], errors="ignore")
    else:
        df_raw = _load_post2018(year, raw_dir)
        df = _harmonize_post2018(df_raw)
        del df_raw
    rows_loaded = len(df)
    # Keep only originated (1=approved) and denied (3) applications.
    # Other action codes (withdrawn, incomplete, etc.) are excluded
    # because they do not represent a clear approve/deny decision.
    df["action_taken"] = pd.to_numeric(df["action_taken"], errors="coerce")
    df = df[df["action_taken"].isin([1, 3])].copy()
    rows_after_filter = len(df)
    # Create binary target: 1 = approved, 0 = denied
    df[_TARGET_COL] = (df["action_taken"] == 1).astype(int)
    df = df.drop(columns=["action_taken"])
    # Determine which modeling features exist for this year
    tier1_available = [c for c in _TIER1_MODELING_FEATURES if c in df.columns]
    tier2_available = (
        [c for c in _TIER2_MODELING_FEATURES if c in df.columns]
        if is_post2018 else []
    )
    base_modeling = tier1_available + tier2_available
    # Drop rows with missing values in required columns.
    # Geographic features are optional since they may be NaN for some tracts.
    optional_features = {"tract_to_msamd_income", "hud_median_family_income"}
    required_cols = [c for c in base_modeling if c not in optional_features]
    required_cols += [c for c in _PROTECTED_ATTRS if c in df.columns]
    required_cols += [_TARGET_COL]
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required)
    # Compute derived features (log transforms, loan-to-income ratio)
    df = _compute_derived(df)
    # Drop any NaNs introduced by derived features (e.g. zero income -> NaN ratio)
    core_derived = ["loan_to_income", "log_loan_amount", "log_income"]
    df = df.dropna(subset=core_derived)
    rows_after_dropna = len(df)
    # Separate protected attributes from modeling features
    existing_protected = [c for c in _PROTECTED_ATTRS if c in df.columns]
    df_protected = df[existing_protected].copy().reset_index(drop=True)
    # Build the final modeling DataFrame
    derived_available = [c for c in _DERIVED_FEATURES if c in df.columns]
    model_cols = (
        [c for c in base_modeling if c in df.columns]
        + derived_available
        + [_TARGET_COL]
    )
    # Deduplicate while preserving order
    seen = set()
    model_cols_deduped = []
    for c in model_cols:
        if c not in seen:
            seen.add(c)
            model_cols_deduped.append(c)
    model_cols = model_cols_deduped
    df_model = df[model_cols].copy().reset_index(drop=True)
    # Classify features by type for the WOE and FT-Transformer pipelines
    numeric_features = [
        c for c in [
            "loan_amount_000s", "applicant_income_000s",
            "tract_to_msamd_income", "hud_median_family_income",
            "loan_to_income", "log_loan_amount", "log_income",
            "log_area_median_income",
        ]
        if c in df_model.columns
    ]
    categorical_features = [
        c for c in [
            "loan_purpose_name", "lien_status_name", "property_type_name",
            "owner_occupancy_name",
            # hoepa_status_name excluded: post-decision leakage
            "loan_type_name", "preapproval_name",
        ]
        if c in df_model.columns
    ]
    # has_co_applicant is binary (0/1) but treated as numeric for WOE encoding
    binary_features = [c for c in ["has_co_applicant"] if c in df_model.columns]
    numeric_features += binary_features
    meta = {
        "year": year,
        "schema_regime": regime,
        "rows_loaded": rows_loaded,
        "rows_after_action_filter": rows_after_filter,
        "rows_after_dropna": rows_after_dropna,
        "rows_dropped_na": rows_after_filter - rows_after_dropna,
        "rows_dropped_action_filter": rows_loaded - rows_after_filter,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "protected_attributes": existing_protected,
        "target_col": _TARGET_COL,
        "derived_features": derived_available,
        "tier1_features": tier1_available,
        "tier2_features": tier2_available,
        "feature_tier": "tier1+tier2" if tier2_available else "tier1_only",
    }
    logger.info(
        "Year %d (%s): %d -> %d rows (dropped %d action, %d NA) -- %d features (%s)",
        year, regime, rows_loaded, rows_after_dropna,
        rows_loaded - rows_after_filter,
        rows_after_filter - rows_after_dropna,
        len(numeric_features) + len(categorical_features),
        meta["feature_tier"],
    )
    return df_model, df_protected, meta

def temporal_split(
    df: pd.DataFrame, year: int,
    test_size: float = 0.30, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split on the target within a single year.
    We split within each year (not across years) so that the drift
    analysis compares models trained on the same era, not contaminated
    by data from other time periods.
    Returns (train_idx, test_idx) as integer indices into df.
    """
    y = df[_TARGET_COL].values
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    return train_idx, test_idx
