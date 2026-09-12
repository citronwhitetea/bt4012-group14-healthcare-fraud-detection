# AI Context Briefing — BT4012 Healthcare Provider Fraud Detection

**Instructions for use:** Paste this entire document as your first message in a new chat with any AI assistant (Claude, ChatGPT, etc.) before asking for help. It gives the assistant full context on the project so it can pick up work immediately without re-explaining the dataset, pipeline, or decisions already made. Update this file whenever a pipeline step is completed, and re-paste the latest version when starting a new chat.

---

## Project

BT4012 (Fraud Analytics) group project. Building a fraud detection pipeline on the [Healthcare Provider Fraud Detection dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) (Kaggle, rohitrox). Goal: predict whether a **healthcare provider** (not an individual claim or patient) is potentially fraudulent, using claims data aggregated to the provider level.

Repo: `bt4012-group14-healthcare-fraud-detection` (GitHub). Standard structure:
```
data/{raw,processed}/   notebooks/   src/   reports/figures/   docs/
```
Full column-by-column dataset documentation: `docs/DATA_DICTIONARY.md`.
Full step-by-step decision log: `docs/PIPELINE_LOG.md`.

## Dataset

Four raw CSVs in `data/raw/`, joined via `Provider`/`BeneID` keys:
- `provider_labels.csv` — Provider-level fraud label (`PotentialFraud`: Yes/No). 5,410 providers.
- `beneficiary_raw.csv` — Patient demographics, chronic conditions. 138,556 beneficiaries.
- `inpatient_raw.csv` / `outpatient_raw.csv` — Claim-level records (dates, amounts, diagnosis/procedure codes, physicians). 558,211 claims combined.

We use **Train files only** (Kaggle Test files lack the fraud label) and build our own train/test split.

**Key fact:** claim-level fraud rate (~38%) is much higher than provider-level fraud rate (~9%) — fraudulent providers generate disproportionately more claims. Don't confuse the two when discussing "the fraud rate."

## Pipeline status (7 steps)

| Step | Code | Notebook |
|---|---|---|
| 1. Data loading & joining | `src/data_loader.py` | `01_data_loading_joining.ipynb` |
| 2. Cleaning | `src/cleaning.py` | `02_cleaning.ipynb` |
| 3. EDA & statistical tests | `src/eda.py` | `03_eda_stats.ipynb` |
| 4. Feature engineering | `src/features.py`, `src/feature_selection.py` | `04_feature_engineering.ipynb` |
| 5. Train/test split & imbalance handling | `05_train_test_split.ipynb` | |
| 6. Core modeling | `06_modeling.ipynb` | |
| 7. SHAP evaluation & report | `07_shap_evaluation.ipynb` | |

## Code architecture (`src/`)

- **`data_loader.py`** → `build_claim_level_dataset()` — loads 4 CSVs, stacks Inpatient+Outpatient, merges beneficiary + label info. Returns claim-level DataFrame (558,211 rows).
- **`cleaning.py`** → `clean_claim_level_dataset(df)` — parses dates, recodes binary columns (`ChronicCond_*`, `RenalDiseaseIndicator`, `PotentialFraud`), derives `AgeAtClaim`/`IsDeceased`/`NumPhysiciansInvolved`, casts geo codes to `category`. No rows dropped.
- **`eda.py`** → `build_minimal_provider_summary(df)`, `chi_square_test()`, `t_test()`, `anova_test()`, `cramers_v()`, `benford_leading_digit_test()`, `benford_test_by_fraud_group()`, `benford_mean_absolute_deviation()`, plus `plot_*()` visualization functions.
- **`features.py`** → `build_provider_features(df)` — aggregates claim-level table to one row per provider across 6 feature blocks (claim volume, claim amounts, diagnosis/procedure complexity, physician involvement, beneficiary demographics, duration). Returns 5,410 rows × 19 columns (`Provider`, `PotentialFraud`, 17 features).
- **`feature_selection.py`** → `evaluate_feature_significance()` (t-test + Cohen's d per feature), `flag_weak_features()`, `compute_vif()`, `iterative_vif_selection()` (greedy multicollinearity removal).

## Key decisions & rationale (don't re-litigate these without reason)

1. **Provider is the unit of analysis.** Claims aren't independent within a provider — statistical tests and train/test splits must operate at Provider grain, never Claim grain, or results/leakage checks are invalid.
2. **No feature or split touches `PotentialFraud`** except as the final label — avoids target leakage.
3. **Beneficiary demographics are aggregated from a table deduplicated to one row per (Provider, BeneID)** before averaging — otherwise frequently-visiting patients skew a provider's demographic profile.
4. **Multicollinearity was resolved in two passes:** (a) removed exact arithmetic identities baked into the feature functions themselves (e.g. `TotalClaimAmt ≈ NumClaims × AvgClaimAmt`, `NumInpatientClaims`/`NumOutpatientClaims` vs `NumClaims`/`PropInpatient`), (b) used iterative VIF elimination for genuine domain correlations (e.g. `AvgClaimAmt` vs `PropInpatient`, r=0.84).
5. **Two feature sets exist**, saved in `data/processed/feature_sets.json`:
   - `full_features` (17 features) — for Random Forest / boosting (not sensitive to multicollinearity)
   - `logistic_features` (15 features, VIF < 10) — for logistic regression
6. **Benford's Law is not a strong fraud discriminator here** — raw chi-square looked significant but that's a sample-size artifact; Mean Absolute Deviation (scale-invariant) shows Fraud/Non-Fraud groups deviate almost identically from Benford's expected distribution. Report this as a tested-but-inconclusive hypothesis, don't oversell it.
7. **Strongest features found so far (by effect size, not p-value):** `NumClaims`, claim amount statistics (`AvgClaimAmt`/`MedianClaimAmt`), `PropInpatient`. Weak features: `AvgNumPhysiciansInvolved`, `AvgAgeOfBeneficiaries` — kept for tree models anyway, since a t-test only catches linear effects.

## Grading rubric (keep in mind for any deliverable)

| Criterion | Points |
|---|---|
| Data understanding / preprocessing | 3 |
| Method selection / justification | 4 |
| Experimental design / evaluation | 4 |
| Results / insights / applicability | 3 |
| Limitations / reproducibility | 2 |

## What's needed next

**Step 5: Train/test split & class imbalance handling.** Must split by **Provider** (not claim) to avoid the same provider's data leaking across train/test. Apply imbalance handling (SMOTE / class weights — not yet decided) **after** splitting, and **only** to the training set, never the test set. See `docs/PIPELINE_LOG.md` for the planned approach.