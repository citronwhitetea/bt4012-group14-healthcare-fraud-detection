# Pei's work: what I did and how to check it

**Owner:** Loo Pei En · **Branch:** `pei_branch` · **Part of the project:** boosting models, final model and threshold, higher-recall experiment, Streamlit dashboard

This note explains my part in plain language so you can check it. The companion file `pei_work_summary_ai.md` has the same content in a form an AI assistant can use to verify the repository automatically.

---

## The short version

I trained the boosting models (XGBoost and LightGBM). With Swei's baselines, this produced 18 candidate models, and **LightGBM trained on the original data** won. The held-out test set was then used to evaluate it at a threshold aiming for 80% recall. To reduce missed cases, I compared 80%, 90% and 95% recall targets on validation data, and we chose **90%**. I also built the Streamlit dashboard that reviewers would use.

| | 80% target (evaluated first) | 90% target (chosen) |
|---|---|---|
| Threshold | 0.0861 | 0.0241 |
| Fraud-labelled providers caught (of 101) | 86 | 96 |
| Missed | 15 | 5 |
| Providers flagged for review (of 1,082) | 177 (16.4%) | 260 (24.0%) |
| Precision | 48.6% | 36.9% |

The 90% target was evaluated after the 80% target, on the same test set, so its result is not an independent test. The report says this in one sentence.

---

## What I did, step by step

### 1. Boosting models (`notebooks/06b_boosting_models.ipynb`)

- **Training data:** the 4,328 training providers only. The 1,082 test providers stayed untouched during development.
- **Strategies:** each model was trained three ways: on the original data, with class weighting, and with SMOTE-NC. The weighting and SMOTE were done **inside each cross-validation fold**, so no information leaked between folds.
- **Shared folds:** I used the same five folds as Swei (`artifacts/shared/cv_folds.csv`), so all 18 candidates were compared on identical data.
- **Leakage fix:** three features had been filled with dataset-wide averages (`TopDiagnosisCodeShare`, `TopProcedureCodeShare`, `BenfordMAD`). I reset them to missing and re-filled them inside each fold.
- **Search:** 20 random settings per model, under all three strategies (600 model fits).
- **Metric:** average precision, because only 9.4% of providers are fraud-labelled.

**Results.** LightGBM on original data scored 0.674 average precision (seven times the 0.094 you'd get by chance). SMOTE was worse in all five folds, and class weighting made no consistent difference. Quadrupling the search barely changed the scores, so the features, not the tuning, are what limits performance. Rerunning everything gave identical numbers.

### 2. Final model and test

- **Selection:** any candidate within one standard error of the best counted as tied, and the simplest approach won. That chose LightGBM on original data. Logistic regression and random forest were close behind, so we don't claim boosting is clearly better.
- **Threshold:** set on validation data as the highest-precision threshold that still catches at least 80% of fraud-labelled providers (0.0861).
- **Test:** evaluated once. 86 of 101 caught, average precision 0.754, ROC-AUC 0.960.

### 3. Higher-recall experiment (`notebooks/06d_high_recall_policy_and_misses.ipynb`)

| Target (validation) | Caught of 405 | Providers reviewed | Extra reviews per extra case |
|---|---|---|---|
| 80% | 324 | 701 | — |
| 90% | 365 | 1,036 | about 8 (from 80%) |
| 95% | 385 | 1,422 | about 19 (from 90%) |

- **Choosing 90%:** it catches more without the workload jump that 95% brings. The decision was committed to git, with the rationale, before it was tested.
- **On test:** it recovered 10 of the 15 misses. The 83 extra reviews were those 10 fraud-labelled providers plus 73 that were labelled "no".
- **Who gets missed:** mostly small, outpatient-heavy providers that look like legitimate ones on our features. That's the main blind spot to work on.

### 4. Streamlit dashboard (`dashboard/`)

- **Review queue** with three bands: review first, additional review, not flagged.
- **Case panel** for each provider: key figures, a chart of what pushed its score up or down, a shortlist tick-box and notes.
- **Upload checks:** impossible values are rejected with a downloadable list of problems.
- **Model page:** both confusion matrices and the 80/90/95 comparison.
- **Integrity check:** the dashboard checks it is using exactly the frozen model and threshold.
- **Tests:** 41 automated tests and a self-check.

---

## How to check my work

```bash
git fetch origin && git checkout pei_branch && git pull
pip install -r requirements.txt -r requirements-dashboard.txt
python -m dashboard.core          # should end with "Self-check passed"
python -m pytest tests/ -q        # all tests should pass
streamlit run dashboard/app.py    # queue should show 260 of 1,082 flagged
```

You need `data/processed/` locally (run notebooks 01–05) and the run folder `artifacts/boosting/20261011_232303_full/`. Without the run folder, the dashboard still opens but can't score providers.

**Things worth checking:**
- **Notebooks:** 06b's printed results match the table in step 1, and 06d's Section 1 table matches the 80/90/95 table.
- **The dashboard's Model and evidence page:** it shows 86/15 and 96/5 in the two confusion matrices.
- **The policy file:** `artifacts/shared/exploratory_review_policy.json` names the 90% target and its rationale.

---

## Things to be aware of

- **The report must match the pipeline.**
  - Section 3.2 should say resampling for the final models happened inside each fold, not on pre-resampled datasets.
  - Section 3.1 should mention that the three averaged features were reset and re-filled within folds.
  - The boosting models used all 26 features; logistic regression was run with all 26 (`lr_all`) or with VIF selection refitted inside each fold (`lr_vif10`, `lr_vif5`). The full-data VIF<10 list in `feature_sets.json` has 22 features.
- **AI assistance.** I used Claude to help write the 06b and 06d notebooks, the dashboard and parts of the report text. A coding agent also ran the final model selection and test through a helper script, with my confirmation. The report needs to acknowledge this, as the brief requires.
- **Unrelated local changes.** These are on my machine and not committed: re-run outputs in 06c (Swei's notebook), and old run folders (`..._quick`, `..._230823_full`). Swei, tell me if you want the 06c outputs committed.
- **Ask me** if any number doesn't match. I can walk anyone through the notebooks or the dashboard before the presentation.
