# BT4012 Group Project — Healthcare Provider Fraud Detection

## Team

| Name | Responsibility |
|---|---|
| Jada Loau | Repo and pipeline setup; data loading, cleaning, EDA and provider-level feature engineering (`src/`, notebooks 01–04) |
| Tan Yumin | EDA notebooks, feature engineering and train/test split (notebooks 03–05) |
| Loo Pei En | Boosting models, final model selection and held-out evaluation, high-recall review policy, Streamlit dashboard (06b, 06d, `dashboard/`) |
| Tan Wei Jian | Logistic regression and random forest models, fold-level VIF, SHAP evaluation (06c, 07, `src/fold_vif.py`) |

## Problem

Healthcare fraud is typically committed by **providers** (hospitals, physicians, clinics) rather than individual patients — through practices like billing for services not rendered, upcoding, or unbundling procedures. Because fraud signals emerge from *patterns across many claims*, not any single claim in isolation, this is naturally a **provider-level classification problem**: given all claims and beneficiary interactions tied to a provider, predict whether that provider is potentially fraudulent.

## Dataset

[Healthcare Provider Fraud Detection](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) (Kaggle, rohitrox) — a relational dataset with four linked tables:

| Table | Grain | Key contents |
|---|---|---|
| `Train` | Provider | `PotentialFraud` label (Yes/No) |
| `Train_Beneficiarydata` | Beneficiary (patient) | Demographics, chronic conditions, coverage dates |
| `Train_Inpatientdata` | Claim (inpatient) | Admission/discharge dates, diagnosis & procedure codes, reimbursement amounts |
| `Train_Outpatientdata` | Claim (outpatient) | Same structure as inpatient, without admission |

We use the **Train** files only (not the Kaggle Test files, which lack the fraud label) and construct our own train/test split at the provider level.

Class imbalance is significant: only ~9% of providers are labelled fraudulent.

We chose this dataset over alternatives (e.g. SGCC) for its relational, multi-table structure, which supports a wider range of course methods — supervised classifiers, unsupervised clustering, Benford's Law, and graph-based/network features — rather than a single flat feature table.

## Methodology

The pipeline runs in seven stages, detailed in `notebooks/`:

1. **Data loading & joining** — merge Beneficiary, Inpatient, Outpatient, and label tables
2. **Cleaning** — handle missing values, data types, inconsistent categories
3. **EDA & statistical tests** — chi-square, t-test, ANOVA, Benford's Law
4. **Provider-level feature engineering** — aggregate claim-level data up to the provider (claim counts, reimbursement statistics, diagnosis diversity, etc.), taking care to avoid **claim-level leakage** — features must be computed from claims, not from information only knowable after the fraud determination
5. **Train/test split** — stratified 80/20 split by **provider** (4,328 development / 1,082 test), so no provider appears in both sets. The test set is untouched until the single final evaluation. `05` also explores SMOTE and undersampling, but the final models do **not** use those files: they resample inside each CV fold (step 6)
6. **Core modeling** (`06b` boosting, `06c` logistic regression and random forest) — 18 candidates (6 models × original / class weighting / SMOTE), all on the same 5 provider folds (`artifacts/shared/cv_folds.csv`), selected by mean fold average precision. Imputation, SMOTE and logistic-regression VIF selection (`src/fold_vif.py`) are all fitted inside each training fold
7. **Final evaluation & review policy** — LightGBM on original data, threshold chosen on validation for ≥80% recall, evaluated once on the test set (`reports/final_model_evaluation.md`); `06d` compares 80/90/95% recall targets and records the 90% review policy
8. **SHAP evaluation** (`07`) — explains the frozen model's held-out scores (`reports/shap_evaluation.md`)
9. **Dashboard** (`dashboard/`) — Streamlit review queue using the frozen model and threshold

## Repo Structure & Setup

New to the repo? Start with **`bt4012_handover (internal)/00_setup_guide.ipynb`** — it covers cloning, environment setup, dataset download, folder conventions, and our Git workflow.

```
├── data/{raw, processed}/     # data (see 00_setup_guide for what goes where)
├── notebooks/                 # one notebook per pipeline stage above
├── src/                       # shared functions imported into notebooks
├── artifacts/                 # frozen model run, shared CV folds, final selection and test records
├── reports/                   # final evaluation, model comparison and SHAP write-ups + figures
├── dashboard/                 # Streamlit provider review app (see dashboard/README.md)
├── scripts/                   # final-test runner and provenance records
├── tests/                     # automated tests for the dashboard and the final-evaluation guard
├── run_pipeline.sh            # runs notebooks 01 -> 05 in order
├── requirements.txt
└── README.md
```

## Status

- [x] Data loading & joining
- [x] Cleaning
- [x] EDA & statistical tests
- [x] Feature engineering
- [x] Train/test split & imbalance handling
- [x] Core modeling and final held-out evaluation
- [x] SHAP evaluation
- [x] Dashboard
- [ ] Final report

## How to reproduce

1. **Set up** (full detail in the setup guide above):
   ```bash
   git clone https://github.com/citronwhitetea/bt4012-group14-healthcare-fraud-detection.git
   cd bt4012-group14-healthcare-fraud-detection
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   **macOS:** LightGBM needs OpenMP. If importing it fails with `Library not loaded: libomp.dylib`, run `brew install libomp` once.
   **Data:** download the four **Train** files from [Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) into `data/raw/` and rename them:
   `Train-1542865627584.csv` → `provider_labels.csv`, `Train_Beneficiarydata-1542865627584.csv` → `beneficiary_raw.csv`,
   `Train_Inpatientdata-1542865627584.csv` → `inpatient_raw.csv`, `Train_Outpatientdata-1542865627584.csv` → `outpatient_raw.csv`.
2. Rebuild the processed data by running notebooks 01 → 05 in dependency order:
   ```bash
   bash run_pipeline.sh          # or: bash run_pipeline.sh 03b  to start from a later notebook
   ```
   Note that `04a` runs before `03b`, because `03b` reads the claim-level table `04a` writes. The script finishes by checking that the regenerated train/test tables match the fingerprints in `artifacts/shared/final_model_selection.json`.
3. Modelling notebooks (`06b`, `06c`, `06d`, `07`) are **not** re-run by the script. Their final outputs are committed under `artifacts/` and `reports/`, and the held-out test cannot be evaluated twice (`src/final_evaluation.py` blocks it). To view the saved test results: `python scripts/run_final_test.py`.
4. Dashboard: see `dashboard/README.md`.
5. Check everything works: `python -m pytest tests/ -q` (all tests should pass).

## Grading Rubric Reference

| Criterion | Points |
|---|---|
| Data understanding / preprocessing | 3 |
| Method selection / justification | 4 |
| Experimental design / evaluation | 4 |
| Results / insights / applicability | 3 |
| Limitations / reproducibility | 2 |
