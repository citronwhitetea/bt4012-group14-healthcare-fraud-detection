# Pei's work: handover for AI-assisted verification

Audience: an AI coding assistant asked to verify or extend Pei's part of the BT4012 Group 14 project (healthcare provider fraud detection). The human-readable version is `pei_work_summary_human.md`.

Repository: `github.com/citronwhitetea/bt4012-group14-healthcare-fraud-detection`, branch `pei_branch`. Python 3.11 (project `.venv`). Do not use syntax that needs Python 3.12. In particular, f-string expressions must not contain backslashes or reuse the enclosing quote character.

## 1. Scope and ownership

| Area | Owner | Location |
|---|---|---|
| Cleaning, EDA, features, 80/20 provider-level split | Jada, Yumin | notebooks 01–05, `src/features.py` |
| Logistic regression (all / VIF<10 / VIF<5), random forest, SHAP, overall evaluation | Swei | `notebooks/06c_logistic_random_forest_models.ipynb`, 07 |
| XGBoost + LightGBM, final model artefacts, thresholds, high-recall experiment, dashboard | **Pei** | 06b, 06d, `dashboard/`, `tests/test_dashboard_*.py` |

## 2. Data contract

- **Inputs:**
  - `data/processed/provider_features_train.csv`: 4,328 providers.
  - `data/processed/provider_features_test.csv`: 1,082 providers.
  - `data/processed/feature_sets.json`: key `full_features` = 26 features.
  - Both CSVs: ID column `Provider`, target `PotentialFraud` (Yes/No → 1/0). Positive rate 9.4%.
- **Shared folds:** `artifacts/shared/cv_folds.csv`, 5 stratified folds on the training providers, with columns `Provider`, `fold` and `PotentialFraud`. They are shared with 06c.
- **Restored missing values:** `restore_na_cols` = `TopDiagnosisCodeShare`, `TopProcedureCodeShare`, `BenfordMAD`. These were mean-filled upstream. They are set to NaN where `<col>Missing == 1`, then median-imputed inside each fold by the pipeline.

## 3. Frozen final run

Folder: `artifacts/boosting/20261011_232303_full/`.

| File | Content |
|---|---|
| `selected_pipeline.joblib` | imblearn Pipeline: `SimpleImputer(median, keep_empty_features=True, pandas output)` → `LGBMClassifier` |
| `selected_metadata.json` | `model=lgbm`, `strategy=original`, `threshold=0.08610464772643815`, `threshold_rule={"rule":"min_recall","value":0.8}`, `feature_order` (26), `restore_na_cols`, `fingerprints_sha256` (train, test, feature_sets, folds, pipeline), versions, seeds, `threshold_rationale` (added before testing) |
| `oof_predictions.csv` | Out-of-fold scores for every model × strategy; columns `Provider, fold, y_true, oof_prob, model, strategy` |
| `test_predictions.csv` | `Provider, y_true, score, flagged` at 0.0861 |
| `test_metrics.json` | `point` (threshold, tp, fp, fn, tn, recall, precision, f1, ap, roc_auc, flagged_share), `bootstrap_95ci` |
| Other | `cv_search_results.csv`, `cv_summary.csv`, `fold_metrics.csv`, `paired_strategy_comparison.csv`, `selected_fold_metrics.csv`, `threshold_tradeoff_oof.csv`, `cv_comparison.png`, `calibration_oof.png`, `smote_sanity_checks.csv`, copies of `cv_folds.csv` and `feature_sets.json` |

Other runs in `artifacts/boosting/` (`20261011_225931_quick`, `20261011_230823_full`) are superseded and untracked. Ignore them.

## 4. Method (06b)

- **Models:** XGBoost and LightGBM, each under three strategies:
  - **`original`:** no adjustment.
  - **`class_weight`:** `scale_pos_weight` = neg/pos, computed per fold.
  - **`smote`:** impute → StandardScaler → SMOTENC with binary flags as categorical → model.
  - All strategies share fold-local median imputation.
- **Search:** `ParameterSampler`, 20 configs per model. The same configs are used across strategies. 600 fits. Seed 4012.
- **Primary metric:** AP, as fold mean and SD.
- **Selection rule:** candidates within 1 SE (`ap_std/sqrt(5)`) of the best mean AP are tied. Among tied candidates, prefer strategy `original` > `class_weight` > `smote`, then higher AP. The same rule was applied across all 18 candidates from 06b and 06c (cut-off 0.6627).
- **Threshold rule:** the highest out-of-fold precision among thresholds with OOF recall ≥ target (via `precision_recall_curve`).

## 5. Expected numbers (verify these)

**CV (06b, full run; must be reproducible exactly):**

| model/strategy | AP mean ± SD |
|---|---|
| lgbm/original (selected) | 0.67396 ± 0.0258 |
| lgbm/class_weight | 0.67405 ± 0.0255 |
| xgb/original | 0.6705 ± 0.0267 |
| xgb/class_weight | 0.6656 ± 0.0350 |
| lgbm/smote | 0.6530 ± 0.0372 |
| xgb/smote | 0.6502 ± 0.0360 |
| dummy (prior) | 0.094 |

- **Paired fold differences vs original:**
  - SMOTE: lower in 5/5 folds (−0.020 for XGBoost, −0.021 for LightGBM).
  - Class weighting: lower in 3/5 folds (−0.005 for XGBoost, 0.000 for LightGBM).
- **Selected model OOF:** ROC-AUC 0.935 ± 0.012. Brier 0.0498, against a prior-only reference of 0.0848. Mean score 0.080, against a base rate of 0.094.
- **Baselines (06c, Swei):**
  - LR VIF<10: 0.66473
  - LR all: 0.66447
  - RF: 0.66329
  - LR VIF<5: 0.66223

**Validation thresholds (OOF, 405 positives / 4,328):**

| target | threshold | caught | reviewed | precision |
|---|---|---|---|---|
| 0.80 | 0.0861046 | 324 | 701 | 46.2% |
| 0.90 | 0.0240939 | 365 | 1,036 | 35.2% |
| 0.95 | 0.0068779 | 385 | 1,422 | 27.1% |

- **Reviews per extra catch:** 8.2 for 80→90; 19.3 for 90→95; 11.8 for 80→95.
- **Leave-one-fold-out recall at the 90% target:** 0.901, 0.877, 0.889, 0.864, 0.951.

**Held-out test (101 positives / 1,082):**

| | 80% target (evaluated first) | 90% target (evaluated second, same test set) |
|---|---|---|
| TP / FN / FP / TN | 86 / 15 / 91 / 890 | 96 / 5 / 164 / 817 |
| flagged | 177 (16.4%) | 260 (24.0%) |
| recall | 0.8515 (CI 0.772–0.921) | 0.950 (CI 0.901–0.990) |
| precision | 0.4859 (CI 0.435–0.539) | 0.369 |
| AP / ROC-AUC | 0.7536 / 0.9600 | (threshold-independent; same model) |

- **Original misses flagged by the 90% target:** PRV51482, PRV57667, PRV51005, PRV54743, PRV57096, PRV54361, PRV53787, PRV53671, PRV55057, PRV54876.
- **Still missed:** PRV56361 (0.0231), PRV54505, PRV55010, PRV57569, PRV52826 (0.0029).

## 6. Evaluation-order invariants (do not break)

1. The 80% threshold and model were fixed and recorded in `artifacts/shared/final_model_selection.json` before the test set was opened. The test was run once (`test_metrics.json`, `heldout_evaluation.json`; cleanup commit `aed2777`).
2. The 90% target was proposed **after** the 80% test result was known. It was chosen on OOF data only, and recorded in `artifacts/shared/exploratory_review_policy.json` (target 0.9, threshold 0.0240938744582718, pipeline and OOF fingerprints, `decided_by`, `rationale`, `committed_at` 2026-11-01T21:27:25, `disclosure`). The file was committed to git before `artifacts/shared/exploratory_policy_supplementary_evaluation.json` was produced. 06d enforces this through `git ls-files` and `git diff` checks.
3. The 95% target is **validation only**. Never evaluate it, or any other new threshold, on the test set, and never apply it to a queue of test providers.
4. Do not edit `selected_metadata.json`, `final_model_selection.json`, `test_metrics.json`, `test_predictions.csv` or the policy and evaluation JSONs. The dashboard and 06d verify fingerprints and thresholds, and will refuse mismatches.
5. Reporting convention agreed by the team:
   - Present the work as "experimented with 80/90/95% validation targets, chose 90%".
   - Always state that the 80% target was evaluated on the held-out set first, and the 90% target afterwards on the same set.
   - Do not describe the 90% result as an independent test.

## 7. Notebook 06d (`notebooks/06d_high_recall_policy_and_misses.ipynb`)

Sections, in order:
1. Validation comparison of 80/90/95, including a cross-check cell against the numbers above, which raises on mismatch. Also fold tables (pooled and leave-one-fold-out) and fragility checks.
2. Validation-miss profiles. Writes `reports/error_analysis/validation_missed_cases.csv`.
3. Commit the policy. Raises if `CHOSEN_TARGET`, `DECIDED_BY` or `RATIONALE` is blank. Never overwrites an existing policy file.
4. One evaluation of the committed policy on the test set. Requires the policy to be git-tracked and unchanged. Re-runs only read the saved JSON.
5. Descriptive analysis of the 15 original misses. Writes `heldout_original_misses.csv`, `heldout_group_profiles.csv` and `heldout_misses_vs_caught_shap.csv`.

**Miss profiles (validation, 80% target):** the 81 missed against the 324 caught had median claims 61 vs 197, inpatient share 13% vs 26%, and share with no primary procedure codes 26% vs 6%. The 5 test misses remaining at 90% are outpatient-heavy (≈6% inpatient), resembling the additionally flagged negatives.

## 8. Dashboard (`dashboard/`)

| File | Role |
|---|---|
| `core.py` | Streamlit-free logic: `find_run_dir` (reads `selected_run` from the selection record, falling back to inference), `load_bundle` (requires the pipeline fingerprint; checks model, strategy, threshold and fingerprint against the record and `test_metrics.json`; evaluation-only mode via `require_pipeline=False`), `read_csv_bytes` (IDs as strings, rejects duplicate headers, ≤50,000 rows), `validate_input` (impossible values are errors; out-of-range, blank or fractional counts are warnings; returns an issue report), `score_batch`, TreeSHAP via LightGBM `pred_contrib`, `waterfall`, `check_saved_predictions`, `load_policy` (verifies run, fingerprints, thresholds), `recall_threshold` / `what_if_options` (validation-only), `self_check` |
| `app.py` | UI. The queue uses the 90% target only, with no toggle; bands are Review first (≥0.0861), Additional review, Not flagged. Model and evidence has sections Performance (both confusion matrices, labelled "80% target: evaluated first", "90% target (chosen): same held-out test, evaluated after the 80% target", plus an order-of-evaluation row), Review thresholds, Review policies (80/90/95 validation table), This file, Features, Limitations, Run details |
| `glossary.py` | Feature labels, units (drive the impossible-value rules and display), blank-value rules |
| `make_demo_files.py` | Writes clean, warning and error demo uploads to `demo_uploads/` (gitignored) |
| `README.md` | Run, verification and design notes |

Tests:
- `tests/test_dashboard_edge_cases.py`: 32 tests. Self-contained; builds a tiny LightGBM run in a temporary directory.
- `tests/test_dashboard_core.py`: 9 tests against the real artifacts; skips if they are absent.

## 9. Verification commands

```bash
python -m dashboard.core
# expect: located via selected_run…; checks all "match" or "not recorded";
#         "Saved held-out predictions: match"; contribution gap ~1e-15; "Self-check passed."
python -m pytest tests/ -q                     # expect 41 passed
python - <<'EOF'
import json, pandas as pd
run = "artifacts/boosting/20261011_232303_full"
pol = json.load(open("artifacts/shared/exploratory_review_policy.json"))
p = pd.read_csv(f"{run}/test_predictions.csv", dtype={"Provider": str})
for t in (pol["original_threshold"], pol["threshold"]):
    f = p.score >= t
    print(round(t, 4), int(f.sum()), int((f & (p.y_true == 1)).sum()), int((~f & (p.y_true == 1)).sum()))
EOF
# expect: 0.0861 177 86 15  /  0.0241 260 96 5
git log --oneline -- artifacts/shared/exploratory_review_policy.json   # the policy commit must predate the evaluation file's commit
```

Also verify that `data/processed` matches the frozen run: the sha256 of the train and test CSVs should equal `selected_metadata.json → fingerprints_sha256.train/test`. A local re-run of 04b changed the `AvgAgeOfBeneficiaries` output values, so confirm the processed files were not regenerated.

## 10. Known issues and open items

- **Report consistency:**
  - §3.2 must state that the final experiments resampled inside CV folds; random undersampling was not used in the final comparison.
  - §3.1 must mention the fold-local re-imputation of the three restored columns.
  - Boosting used 26 features; VIF sets were for logistic regression only.
- **Uncommitted local changes on Pei's machine:** re-run outputs in `notebooks/06c_…` (Swei's call) and the superseded runs. Not part of Pei's work.
- **Limitations to keep in the report:** proxy labels; test-set reuse for the 90% target; full-data EDA before the split; selection optimism (non-nested CV); physician-network feature availability; missingness flags as proxies; 2008–2009 data; threshold not cost-optimised.
- **AI assistance:**
  - Claude helped write 06b, 06d, the dashboard, the tests and report text.
  - A coding agent executed the final selection and test through a helper script, with Pei's confirmation; that helper was committed in `aed2777`.
  - Acknowledge both in the report.
