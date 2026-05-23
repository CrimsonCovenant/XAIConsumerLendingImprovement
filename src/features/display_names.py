"""Single source of truth for feature name to human-readable display name mapping.
Every visualization in this project must route feature names through the
pretty() function below. No raw column names (like 'loan_amount_000s')
should ever appear in a chart title, axis label, or legend.
"""
from __future__ import annotations
from typing import Iterable, Union

# Maps raw column names from HMDA data to clean display names.
# Covers both the original pre-2018 column names AND the unified names
# produced by the wrangling pipeline. When a column could appear under
# two names (e.g. 'loan_amount_000s' vs 'loan_amount'), both are listed.
FEATURE_DISPLAY: dict[str, str] = {
    # Original pre-2018 raw names
    "loan_amount_000s":         "Loan Amount",
    "applicant_income_000s":    "Applicant Income",
    "loan_purpose_name":        "Loan Purpose",
    "lien_status_name":         "Lien Position",
    "property_type_name":       "Property Type",
    "owner_occupancy_name":     "Owner-Occupied Status",
    "applicant_sex_name":       "Applicant Sex",
    "applicant_race_name_1":    "Applicant Race",
    "applicant_ethnicity_name": "Applicant Ethnicity",
    "hoepa_status_name":        "High-Cost Loan Flag",
    # Unified column names (from the wrangling harmonization layer)
    "loan_amount":              "Loan Amount",
    "income":                   "Applicant Income",
    "loan_purpose":             "Loan Purpose",
    "lien_status":              "Lien Position",
    "property_type":            "Property Type",
    "occupancy_type":           "Owner-Occupied Status",
    "applicant_sex":            "Applicant Sex",
    "applicant_race":           "Applicant Race",
    "applicant_ethnicity":      "Applicant Ethnicity",
    "hoepa_status":             "High-Cost Loan Flag",
    # Available across all 17 years
    "loan_type_name":           "Loan Type",
    "loan_type":                "Loan Type",
    "preapproval_name":         "Preapproval Status",
    "preapproval":              "Preapproval Status",
    "has_co_applicant":         "Co-Applicant Present",
    "tract_to_msamd_income":    "Neighborhood Income Ratio",
    "hud_median_family_income": "Area Median Income",
    "log_area_median_income":   "Area Median Income (log scale)",
    # Post-2018 only (excluded from training due to leakage risk)
    "debt_to_income_ratio":     "Debt-to-Income Ratio",
    "loan_to_value_ratio":      "Loan-to-Value Ratio",
    "loan_term":                "Loan Term",
    # Derived features created during preprocessing
    "loan_to_income":           "Loan-to-Income Ratio",
    "log_loan_amount":          "Loan Amount (log scale)",
    "log_income":               "Income (log scale)",
}

# One-line plain-English context for each feature, used in figure
# annotations and captions so non-technical readers understand
# what each feature actually measures in the real world.
FEATURE_CONTEXT: dict[str, str] = {
    "Loan Amount":             "Size of mortgage requested, in thousands of dollars.",
    "Applicant Income":        "Yearly income reported on the application, in thousands.",
    "Loan Purpose":            "Home purchase, refinance, or home improvement.",
    "Lien Position":           "First lien (primary mortgage) vs. subordinate (second mortgage).",
    "Property Type":           "One-to-four family, manufactured, or multifamily home.",
    "Owner-Occupied Status":   "Borrower lives in the home vs. investment property.",
    "Applicant Sex":           "Reported demographic -- federally protected class.",
    "Applicant Race":          "Reported demographic -- federally protected class.",
    "Applicant Ethnicity":     "Reported demographic -- federally protected class.",
    "High-Cost Loan Flag":     "Loan exceeds HOEPA thresholds -- federal predatory-lending guardrail.",
    "Loan Type":               "Conventional, FHA, VA, or USDA -- FHA/VA applicants have very different approval rates.",
    "Preapproval Status":      "Whether the applicant sought and received preapproval before applying.",
    "Co-Applicant Present":    "Whether a co-applicant (usually a spouse) is on the loan, combining two incomes.",
    "Neighborhood Income Ratio": "Census tract income as a percentage of the metro-area median -- local market strength.",
    "Area Median Income":      "Metro-area median family income in dollars -- captures local cost of living.",
    "Area Median Income (log scale)": "Area Median Income, log-transformed to stabilize a skewed distribution.",
    "Debt-to-Income Ratio":    "Monthly debt payments divided by monthly income -- the key underwriting ratio (43% QM threshold).",
    "Loan-to-Value Ratio":     "Loan amount divided by property value -- 80% triggers PMI, 97% is the conventional limit.",
    "Loan Term":               "Loan duration in months -- 360 (30-year) vs. 180 (15-year) signals borrower risk profile.",
    "Loan-to-Income Ratio":    "Loan amount divided by income -- how much is being asked vs. earned.",
    "Loan Amount (log scale)": "Loan Amount, log-transformed so very large loans don't distort the analysis.",
    "Income (log scale)":      "Applicant Income, log-transformed to stabilize a skewed distribution.",
}

def pretty(name_or_names: Union[str, Iterable[str]]) -> Union[str, list[str]]:
    """Convert raw feature name(s) to display name(s).
    Pass a single raw name or a list. Falls back to a title-cased version
    of the raw name if not mapped, so unmapped features degrade gracefully.
    """
    if isinstance(name_or_names, str):
        return FEATURE_DISPLAY.get(
            name_or_names, name_or_names.replace("_", " ").title()
        )
    return [
        FEATURE_DISPLAY.get(n, n.replace("_", " ").title()) for n in name_or_names
    ]

def context_for(display_name: str) -> str:
    """Return a plain-English one-liner explaining what a feature means.
    Returns an empty string if the display name has no entry.
    """
    return FEATURE_CONTEXT.get(display_name, "")

def assert_no_raw_names(strings: Iterable[str]) -> None:
    """Safety check: call before rendering any figure to make sure no
    raw column names (like 'loan_amount_000s') leaked into the visual.
    Raises ValueError with the list of leaked names if any are found.
    """
    raw = set(FEATURE_DISPLAY.keys())
    leaked = [s for s in strings if any(r in s for r in raw)]
    if leaked:
        raise ValueError(
            f"Raw feature names leaked into visual strings: {leaked}"
        )
