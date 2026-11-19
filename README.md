# BT4012 Group 14 — Healthcare Provider Fraud Detection

We rank healthcare providers by fraud risk from their Medicare claims, so an investigation team knows whom to audit first. The final model is **LightGBM trained on the original class balance**, with a review threshold of **0.0861**, chosen on validation data for at least 80% recall.

## Results at a glance

**Model comparison (development data, mean average precision over 5 shared folds; random guessing scores 0.094):**

| Model (original class balance) | Mean fold AP |
|---|---:|
| **LightGBM (selected)** | **0.674** |
| XGBoost | 0.670 |
| Logistic regression (fold-local VIF < 10) | 0.665 |
| Random forest | 0.663 |
| *Business rule: audit the largest claims first (`MaxClaimAmt`)* | *0.486* |

The business-rule row is a benchmark with nothing to train: it ranks providers by one feature, scored on the same folds (06c, section 13). Every model beats it by 0.18–0.19 AP.

**Held-out test (1,082 providers, 101 labelled fraudulent); the final model was evaluated once, on 24 Oct 2026:**

| Metric | Value | 95% bootstrap interval |
|---|---:|---|
| Average precision | 0.754 | 0.680–0.826 |
| ROC AUC | 0.960 | 0.944–0.973 |
| Recall (fraud providers caught) | 0.851 (86 of 101) | 0.772–0.921 |
| Precision (flags that are labelled fraud) | 0.486 | 0.435–0.539 |
| Providers flagged for review | 16% (177 of 1,082) | |

The test AP is above the development value (0.674). That is not overfitting, which would push the test score down: the test set holds only 101 fraud cases, so its estimate is noisy, and its interval starts just above the development value. Full write-up: [`reports/final_model_evaluation.md`](reports/final_model_evaluation.md).

---

## For markers: start here

Every notebook is committed **with its outputs**, so you can read the whole project without running anything. To check that it runs, follow steps 1–4 below; to rerun everything, see [Running everything](#running-everything).

### Where to look

| What you want to see | Open |
|---|---|
| Data, cleaning, EDA, feature engineering | `notebooks/00a` → `05` (table below) |
| Model training and comparison | `notebooks/06b` (boosting), `06c` (logistic regression, random forest) |
| Final model and held-out result | `reports/final_model_evaluation.md` |
| Model explanations (SHAP) | `notebooks/07`, `reports/shap_evaluation.md` |
| Review-threshold policy and missed cases | `notebooks/08` |
| Robustness checks after the test (leakage fix, wider tuning, ensembles) | `notebooks/09a`, `09b`, `09c`; `reports/leakfree_retraining.md` |
| Interactive review queue | `dashboard/` (step 4) |

### 1. Set up the environment

Use **Python 3.10 or 3.11**. The pinned packages do not install on Python 3.9 (too old) or 3.13 (too new). macOS ships Python 3.9 as `python3`, so check `python3 --version` first; if needed, install 3.11 with `brew install python@3.11` and use `python3.11 -m venv .venv`. If you use conda instead: `conda create -n bt4012 python=3.11 -y && conda activate bt4012`, then run the `pip install` line below.

```bash
git clone https://github.com/citronwhitetea/bt4012-group14-healthcare-fraud-detection.git
cd bt4012-group14-healthcare-fraud-detection
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**macOS only:** if importing LightGBM fails with `Library not loaded: libomp.dylib`, run `brew install libomp` once.

### 2. Add the data

Download the four **Train** files from [Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) into `data/raw/` and rename them:

| Kaggle file | Rename to |
|---|---|
| `Train-1542865627584.csv` | `provider_labels.csv` |
| `Train_Beneficiarydata-1542865627584.csv` | `beneficiary_raw.csv` |
| `Train_Inpatientdata-1542865627584.csv` | `inpatient_raw.csv` |
| `Train_Outpatientdata-1542865627584.csv` | `outpatient_raw.csv` |

The Kaggle **Test** files are not used, because they have no fraud label. We make our own train/test split.

### 3. Run the pipeline and checks (one command)

```bash
bash run_pipeline.sh
```

This takes a few minutes and does everything needed to check our results:
1. Runs the raw-table EDA notebooks and notebooks 01 → 05, saving the processed tables in `data/processed/`.
2. Checks that the rebuilt train and test tables are **byte-identical** to the ones the final model was trained and tested on (two `[ok]` lines).
3. Runs 07 and 08 (SHAP and review policy) on the frozen model.
4. Runs the automated tests, then prints the saved held-out result.

It ends with `All done.` and a list of the steps it completed. Retraining the models is a separate, optional command because it takes hours (see [Running everything](#running-everything)).

On Windows without Git Bash or WSL, open the notebooks in Jupyter and run them in number order instead: `00a, 00b, 00c, 01, 02, 03a, 03b, 03c, 04, 05, 07, 08`. Then run `python -m pytest tests/ -q` and `python scripts/run_final_test.py`.

### 4. Open the dashboard

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

It scores the held-out providers with the frozen model and shows each provider's score and the factors behind it. Details: [`dashboard/README.md`](dashboard/README.md).

### Running everything

`run_pipeline.sh` runs the notebooks in dependency order, in stages:

| Command | Runs | Time | Changes |
|---|---|---|---|
| `bash run_pipeline.sh` | **Standard check:** raw-table EDA, 01 → 05, data check, 07, 08, tests, saved result | minutes | those notebooks, `data/processed/` and `reports/` outputs |
| `bash run_pipeline.sh models` | Retraining: 06b, 06c, 09a, leak-free 06b/06c, 09b, 09c | hours | new run folders under `artifacts/`; executed copies in `artifacts/reruns/` |
| `bash run_pipeline.sh all` | **Every notebook:** the standard check, then retraining | hours | as above |
| `bash run_pipeline.sh data` / `explain` / `tests` | One part of the standard check on its own | minutes | that part only |

Add a notebook name to start partway through a stage, e.g. `bash run_pipeline.sh data 03c`. `QUICK_RUN=1 bash run_pipeline.sh models` makes 09b and 09c a quick smoke test.

**Nothing here can change the final result.**
- The `models` stage writes new timestamped run folders and saves the executed notebooks separately, so the committed notebooks, the frozen model and the saved test results stay as they are.
- The held-out test cannot be re-run: the test cells in `06b`/`06c` are disabled, and `src/final_evaluation.py` refuses a second evaluation. `07` and `08` read the saved test records rather than re-evaluating.

---

## Notebook map

| # | Notebook | What it does | Main output |
|---|---|---|---|
| **Explore the raw tables** | | | |
| 00a | `00a_inpatient_eda` | First look at the inpatient claims table | figures |
| 00b | `00b_outpatient_eda` | First look at the outpatient claims table | figures |
| 00c | `00c_beneficiary_eda` | First look at the beneficiary table | figures |
| **Prepare the data** | | | |
| 01 | `01_data_loading_joining` | Joins the four raw tables into one claim-level table (558,211 claims) | in-memory check |
| 02 | `02_cleaning` | Dates, recodes, structural missing values, negative amounts | `claims_cleaned.csv` |
| 03a | `03a_claim_level_eda` | Claim-level EDA, chi-square / Cramér's V, aggregate Benford's Law | figures |
| 03b | `03b_claim_level_feature_engineering` | 25 claim-level features | `claims_with_basic_features.csv` |
| 03c | `03c_provider_level_eda` | Provider-level tests (Welch's t, Cohen's d), candidate features | `provider_level_eda_table.csv` |
| 04 | `04_provider_feature_engineering` | 26 provider-level features; VIF to build the 22-feature logistic set | `provider_features.csv`, `feature_sets.json` |
| 05 | `05_train_test_split` | Stratified 80/20 split by provider; SMOTE and undersampling explored | `provider_features_{train,test}.csv` |
| **Model and test** | | | |
| 06b | `06b_boosting_models` | XGBoost, LightGBM × original / class weight / SMOTE-NC; holds the guarded final test | `artifacts/boosting/<run>/` |
| 06c | `06c_logistic_random_forest_models` | Logistic regression (all features, fold-local VIF<10, VIF<5) and random forest × the same strategies; single-feature business-rule benchmark | `artifacts/logistic_random_forest/<run>/` |
| **After the test** | | | |
| 07 | `07_shap_evaluation` | SHAP explanations of the frozen model's test scores | `reports/shap_evaluation/` |
| 08 | `08_high_recall_policy_and_misses` | Compares 80/90/95% recall review policies; analyses the missed cases | `reports/error_analysis/` |
| 09a | `09a_leakage_fixes_cv_check` | Fixes two features that used all providers; checks the effect on CV | `provider_features_{train,test}_leakfree.csv` |
| 09b | `09b_optuna_early_stopping` | Wider LightGBM search with Optuna and early stopping | `artifacts/tuning/<run>/` |
| 09c | `09c_ensemble` | Model averaging, stacking, CatBoost | `artifacts/ensemble/<run>/` |

Shared code lives in `src/` (`data_loader`, `cleaning`, `eda`, `features`, `feature_selection`, `fold_vif`, `final_evaluation`).

---

## Problem

Healthcare fraud is usually committed by **providers**, through billing for services never delivered, upcoding or unbundling. A single fraudulent claim is made to look normal; the signal appears in the pattern across a provider's claims. We therefore classify **providers**, not claims: given all of a provider's claims and patients, predict whether it is potentially fraudulent.

## Dataset

[Healthcare Provider Fraud Detection](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) (Kaggle, rohitrox). Four linked tables:

| Table | One row per | Key contents |
|---|---|---|
| `Train` | Provider (5,410) | `PotentialFraud` label (Yes/No) |
| `Train_Beneficiarydata` | Patient (138,556) | Demographics, chronic conditions, coverage |
| `Train_Inpatientdata` | Inpatient claim (40,474) | Admission/discharge dates, diagnosis and procedure codes, amounts |
| `Train_Outpatientdata` | Outpatient claim (517,737) | Same, without admission fields |

Only 9.4% of providers are labelled fraudulent, but they file 38% of all claims. Column definitions: [`data/raw/raw_data.md`](data/raw/raw_data.md).

## Method

1. **Join and clean** (01, 02). No rows dropped. Missing values are filled according to what they mean: a missing admission date marks an outpatient claim, and missing inpatient deductibles get the claim type's median, with a flag.
2. **EDA and statistical tests** (03a, 03c). Claim-level signals are weak (Cramér's V 0.11; Benford's Law does not separate the groups). Provider-level gaps are large (e.g. maximum claim amount, Cohen's d 1.61).
3. **Feature engineering** (03b, 04). 25 claim-level features are aggregated into 26 provider-level features, none using the label. Tree models use all 26; logistic regression drops overlapping features by VIF.
4. **Split** (05). Stratified 80/20 by provider: 4,328 development and 1,082 test providers. The test set was not used until the final evaluation.
5. **Models** (06b, 06c). 18 candidates (LightGBM, XGBoost, random forest and three logistic-regression variants, each with original data, class weighting or SMOTE-NC), compared on the same 5 provider folds (`artifacts/shared/cv_folds.csv`) by mean fold average precision. Imputation, SMOTE-NC and VIF selection are fitted inside each training fold.
6. **Selection and threshold.** Candidates within one standard error of the best were treated as tied; the simplest imbalance strategy was preferred (original, then class weighting, then SMOTE). The threshold is the most precise one reaching 80% recall on out-of-fold predictions. Both rules were recorded before testing in `artifacts/shared/final_model_selection.json`.
7. **One held-out evaluation** (24 Oct 2026), with bootstrap confidence intervals.
8. **Explanations, review policy and robustness checks** (07, 08, 09a–09c).

## Test-set use (disclosure)

The test set was used three times. Only the first is the project's final result.

| Date | What | Result | Status |
|---|---|---|---|
| 24 Oct | Frozen LightGBM, 80%-recall threshold | AP 0.754, 86/101 caught | **Final result** |
| 1 Nov | Same model, 90%-recall threshold (08) | 96/101 caught, 24% flagged | Supplementary; proposed after seeing the first result |
| 15 Nov | LightGBM + logistic regression average (09c) | AP 0.759, 84/101 caught | Locked in before testing, then withdrawn after testing (it missed 2 more cases); reported as supplementary |

The records are in `artifacts/shared/`. Anything done after the first evaluation is development data only, or is labelled supplementary.

## Limitations

- **Labels are "potential fraud", not proven fraud,** so the model learns to rank providers like past investigation outcomes, and its flags are for review, not for decisions.
- **One year of data (Nov 2008 – Dec 2009) from one source.** Fraud patterns change once detection rules are known, so the model would need regular retraining and monitoring.
- **One feature used test providers.** `AvgPhysicianProviderSpread` counted each physician's providers across all 5,410 providers, including the test set. Recomputed from training providers only, the selection is unchanged and LightGBM's mean fold AP stays within 0.007 of 0.674 (09a, [`reports/leakfree_retraining.md`](reports/leakfree_retraining.md)). Its effect on the held-out score is not estimated, because that would reuse the test labels.
- **Cross-validation estimates are slightly optimistic,** because the same folds were used to tune and to score; the single held-out test is the reliable figure.
- **The test set was reused after the final evaluation** (see the disclosure above); those later results are supplementary only.

## Repository layout

```
├── data/raw/            # the four Kaggle files go here (not committed)
├── data/processed/      # rebuilt by run_pipeline.sh (not committed, except feature_sets.json)
├── notebooks/           # one notebook per stage (see the map above)
├── src/                 # shared functions used by the notebooks
├── artifacts/           # saved model runs, shared CV folds, selection and test records
├── reports/             # write-ups, figures, SHAP tables, final evaluation snapshot
├── dashboard/           # Streamlit review queue
├── scripts/             # saved-result viewer and test runners; provenance/ holds archived copies
├── tests/               # automated tests (dashboard, final-evaluation guard)
├── run_pipeline.sh      # runs the notebooks in order, by stage (see Running everything)
└── requirements.txt     # pinned package versions
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Library not loaded: libomp.dylib` (macOS) | `brew install libomp` |
| `pip install` fails on a pinned version | Check the Python version is 3.10 or 3.11 |
| `FileNotFoundError` for `data/raw/...` | Download and rename the four Kaggle files (step 2) |
| `FileNotFoundError` for `..._leakfree.csv` in 09b/09c | Run `09a` first (`bash run_pipeline.sh models` does this in order) |
| `run_pipeline.sh` reports `MISMATCH` | The rebuilt data differs from what the final model used; check the raw files are the Kaggle **Train** files |
| `run_final_test.py --evaluate` refuses | Intended: the held-out evaluation is complete and cannot be repeated |

## Notes on saved runs

- `artifacts/shared/final_model_selection.json` names the 06c run `20261024_144141_full`. That folder was never committed. `artifacts/logistic_random_forest/20261115_010242_full` is a rerun of the same notebook, data, folds and seeds. All 12 of its candidate scores equal those recorded in `reports/06b_06c_final_candidates.csv`. The frozen model came from 06b, so neither 06c run affects the final result.
- `scripts/provenance/` keeps exact copies of the code run on 24 October, including one teammate's local file paths. They are records, not entry points; use `scripts/run_final_test.py`.
- Some `reports/06b_06c_model_comparison.md` figures come from earlier 13 October runs that are not committed. The report says so at the top, and its conclusion matches the final comparison.


## Acknowledgement of AI Use
Generative AI tools, including Claude and Claude Code (Anthropic), were used to help write and debug code, check analyses and edit documentation. All modelling decisions, results and interpretations were reviewed by the team, who take full responsibility for the submitted work. See the final report for the full acknowledgement.
