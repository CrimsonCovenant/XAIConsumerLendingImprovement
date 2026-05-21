# Explanation Drift in Consumer Lending Models

> **Statistically rigorous XAI applied to 17 years of HMDA mortgage data (2008–2024)**

## Overview

This project builds a full pipeline for detecting **explanation drift** — statistically significant changes in *why* a model makes decisions, not just *what* it predicts. We train two parallel model architectures on Home Mortgage Disclosure Act (HMDA) data across all 17 years from 2008 to 2024, then compare how each model's internal logic shifts over time using bootstrapped confidence intervals, Welch's *t*-tests, and Integrated Gradients.

The result is a quantitative audit trail showing exactly how lending decision factors evolved through the 2008 financial crisis, the post-crisis recovery, the 2018 regulatory schema overhaul, the COVID-era rate environment, and the 2022–2024 tightening cycle.

---

## Architecture

```
data/raw/            ← 186 GB of HMDA LAR CSVs (2008–2024)
data/processed/      ← Cleaned parquets (one per year)

src/
├── features/        ← Preprocessing, WOE pipeline, FT-Transformer preprocessor
├── models/          ← Logistic regression (from scratch), FT-Transformer ensemble
├── drift/           ← Welch's t-test, drift detection
├── explainability/  ← Integrated Gradients, bootstrap attribution
├── fairness/        ← Adverse Impact Ratio (AIR) metrics
├── viz/             ← Plotting style system, caption generation
├── data_loaders/    ← Schema-aware HMDA loaders (pre/post-2018)
└── utils/           ← Reproducibility, seeding, manifests

scripts/
├── wrangling/       ← Raw CSV → processed parquets
└── analysis/        ← Model training, tuning, drift analysis

outputs/
├── models/          ← Trained weights, bootstrap coefficients
├── figures/         ← Publication-quality plots
├── explanations/    ← IG attribution matrices
└── audit_artifacts/ ← IV tables, drift test results
```

---

## Models

### Baseline — WOE Logistic Regression (from scratch)

A credit-scoring industry-standard pipeline:

1. **Weight-of-Evidence encoding** via OptimalBinning (monotonic binning + Laplace-smoothed categorical WOE)
2. **L2-regularized logistic regression** trained with Adam optimizer (custom NumPy implementation — no sklearn)
3. **Platt calibration** on a held-out calibration slice
4. **100× bootstrap** resampling for coefficient confidence intervals (warm-started from final weights for 30× speedup)

### Advanced — FT-Transformer Deep Ensemble

A modern tabular deep learning architecture:

1. **FT-Transformer** ([Gorishniy et al., 2021](https://arxiv.org/abs/2106.11959)) with feature tokenization for both numeric and categorical inputs
2. **5-member deep ensemble** with bagged subsampling (15% stratified per member) for calibrated uncertainty
3. **Platt calibration** on held-out data
4. **Integrated Gradients** for per-feature attribution with bootstrap resampling

Both models are trained on identical features and splits for direct comparability.

---

## Feature Set

All 17 years use the **same 15 pre-decision features** — strictly variables available at the moment of application, before any underwriting decision:

| Type | Features |
|---|---|
| **Numeric (8)** | Loan amount, applicant income, neighborhood income ratio, area median income, loan-to-income ratio, log(loan amount), log(income), log(area median income) |
| **Categorical (6)** | Loan purpose, lien position, property type, owner-occupied status, loan type, preapproval status |
| **Binary (1)** | Co-applicant present |

### Leakage Exclusions

The following features were identified and permanently excluded as **post-decision variables** (their values or missingness are determined by the approval/denial outcome, not available at application time):

- **HOEPA status** — APR/fee thresholds are only calculated for approved loans; denied loans are mass-coded "Not Applicable"
- **Interest rate, rate spread, total loan costs, origination charges** — pricing variables that only exist after approval
- **Debt-to-income ratio, loan-to-value ratio, loan term** — reported differently for approved vs. denied loans (NaN rate 2–3× higher for denials)

Runtime guardrails in both training scripts assert that no leaked column survives in the processed data.

---

## The 2018 Schema Break

HMDA underwent a major regulatory overhaul in 2018, completely changing column names, encoding schemes, and data definitions. This project handles the break with:

- **Dual-schema loaders** (`data_loaders/`) that read pre-2018 and post-2018 CSVs independently
- **A harmonization layer** (`features/preprocessing.py`) that maps both schemas to a unified set of canonical column names
- **Uniform feature space** — by excluding post-2018-only columns, the model sees identical features across all 17 years

This design means any detected drift reflects **real macroeconomic and population shifts**, not schema artifacts.

---

## Pipeline Phases

| Phase | Script | Description |
|---|---|---|
| **1 — Wrangling** | `scripts/wrangling/05_build_yearly_processed.py` | Raw CSV → cleaned parquet (handles schema break, derives features, separates protected attributes) |
| **2 — Baseline** | `scripts/analysis/03_bootstrap_baseline.py` | WOE + Adam LR + Platt + bootstrap coefficients per year |
| **3 — Tuning** | `scripts/analysis/04_tune_ft_transformer.py` | Optuna hyperparameter search for FT-Transformer |
| **4 — Advanced** | `scripts/analysis/06_train_ft_ensembles.py` | FT-Transformer 5-member ensemble per year |
| **5 — Explainability** | *(upcoming)* | Integrated Gradients + bootstrap attribution matrices |
| **6 — Drift** | *(upcoming)* | Welch's *t*-test on bootstrap coefficient distributions across years |
| **7 — Fairness** | *(upcoming)* | Adverse Impact Ratio analysis by race, sex, ethnicity |
| **8 — Visualization** | *(upcoming)* | Publication-quality figures with automated captions |

---

## Reproducibility

Every pipeline run writes a `_run_manifest.json` containing:

- Git SHA, library versions, random seed
- Feature set (raw + display names), hyperparameters
- Train/test/cal split sizes, class balance
- AUC, KS, Gini, F1, accuracy metrics
- Wall-clock time, Platt calibration parameters

All random operations use `seed_everything(42)` which seeds Python, NumPy, and PyTorch (including MPS).

---

## Setup

```bash
# Clone and create environment
git clone <repo-url>
cd xai-consumer-lending
conda env create -f environment.yml
conda activate xai-lending

# Download HMDA data (not included — 186 GB)
# Place yearly CSVs in data/raw/year_{2008..2024}.csv

# Build processed parquets
conda run -n xai-lending python scripts/wrangling/05_build_yearly_processed.py

# Train baselines (all 17 years, ~5 min/year)
conda run -n xai-lending python scripts/analysis/03_bootstrap_baseline.py

# Train FT-Transformer ensembles (all 17 years)
conda run -n xai-lending python scripts/analysis/06_train_ft_ensembles.py
```

---

## Key Design Decisions

1. **No raw feature names in outputs** — all visualizations, file names, and captions use human-readable display names via `src/features/display_names.py`
2. **CPU-only for audit artifacts** — MPS/CUDA used for dev speed, but final reproducible outputs are forced to CPU (`force_cpu=True`)
3. **Logistic regression from scratch** — custom NumPy implementation with Adam, cosine LR decay, and L2 regularization (no sklearn dependency for the core model)
4. **WOE encoding for interpretability** — industry-standard credit scoring transform that gives coefficients a direct monotonic relationship with log-odds

---

## Tech Stack

- **Python 3.11** (conda-managed)
- **PyTorch** — FT-Transformer + MPS acceleration
- **rtdl-revisiting-models** — FT-Transformer architecture (v0.0.2)
- **NumPy / Pandas** — core numerical computing
- **OptBinning** — monotonic WOE binning
- **category_encoders** — categorical WOE encoding
- **Optuna** — hyperparameter tuning
- **scikit-learn** — StandardScaler, LabelEncoder, train_test_split
