# Provider risk review dashboard

A Streamlit app that scores healthcare providers with the team's frozen final model, ranks them for review and explains each score. It uses the exact pipeline, feature order and threshold that were evaluated on the held-out set. Nothing is retrained.

## Run it

From the repository root:

```bash
pip install -r requirements.txt -r requirements-dashboard.txt
python -m dashboard.core                 # self-check: must end with "Self-check passed"
streamlit run dashboard/app.py
```

**Pin the versions you tested.** Run this once on the machine used for the demo, and commit the file it writes:

```bash
pip freeze | grep -iE "^(streamlit|altair|lightgbm|scikit-learn|imbalanced-learn|pandas|numpy|joblib)==" > requirements-dashboard.lock
```

Teammates then install with `pip install -r requirements-dashboard.lock`.

## How the frozen model is found and verified

The app reads the `selected_run` entry of `artifacts/shared/final_model_selection.json`. If the record has no such entry, it infers the run and says so under **Model and evidence → Run details**. To point it elsewhere, set `FRAUD_DASHBOARD_RUN_DIR`.

Before scoring anything, it checks the following:
- **The pipeline fingerprint is present**, and the saved model matches it.
- **Model, strategy, threshold and pipeline fingerprint match the final-selection record**, wherever the record states them. Display names such as "LightGBM" are accepted for "lgbm".
- **The saved held-out evaluation (`test_metrics.json`) was produced at the same threshold.**

Any mismatch stops the app with a clear message. Values the record doesn't state appear as "not recorded" in Run details.

If the model can't be loaded (missing file, missing package), the app still opens. The **Model and evidence** tab shows the saved evaluation, and the **Review queue** explains why scoring is unavailable.

For the held-out test set, freshly computed scores are compared with the saved `test_predictions.csv`. The queue says when they match exactly, so it is the reported result.

## What it shows

**Header.**
- The title and a one-line model summary.
- **Score another file**, which opens a panel with an upload box, a sample to try, and a blank template.
- **Back to the held-out test set**, which appears while an upload is in use.

When a file is scored for the first time, a progress panel lists each step, then collapses.

**Review queue.**
- **Summary panel:** how many providers are flagged, and how to read a score. Scores are for prioritising review; they are not validated probabilities of fraud.
- **Filters:** Flagged, Below threshold, Shortlist and All, with counts beneath, plus a provider search.
- **Download:** exports exactly the providers listed, with scores, top factors, your shortlist and notes, and the model, run ID, threshold, source file and scoring time.
- **Data checks:** warnings, plus a downloadable list of the affected providers.
- **Case panel:** click a row to open its case beside the list. It has:
  - status chips (flagged or not, shortlisted, outside the current filter);
  - the score, with the disclaimer;
  - the provider's position against the review threshold;
  - **Provider at a glance:** claims, inpatient share, average claim amount and claims per patient, each beside the training median;
  - **Your review:** a shortlist tick-box and a note;
  - **Why it scored here:** a waterfall from the model's average prediction to this provider's score, plus a table of each factor's value, the training median (all providers) and its push.

  **Close**, or clicking the row again, closes the case. **Previous in list** and **Next in list** step through the current list.

The shortlist and notes last for the browser session only and are separate from the model's flags. Download the queue to keep them.

**Model and evidence** shows one topic at a time:
- **Performance:** the held-out outcome grid, using dataset-label wording, and the validation and test measures with 95% ranges.
- **Review threshold:** how the threshold was chosen on out-of-fold validation data (the highest precision with recall at or above 80%, then frozen), its validation figures, and the workload curve.
- **This file:** built only from the loaded file. It shows:
  - the flagged share against the training reference;
  - percentile bands split by each provider's actual flag;
  - how many providers fall outside the training range, per feature;
  - the factors the model leans on for this file.
- **Features:** the feature dictionary, with units, definitions and how blank values are handled.
- **Limitations:** six short cards.
- **Run details:** the verification checks, package versions, model settings and selection rule.

## Review policies

We experimented with 80%, 90% and 95% validation recall targets and chose 90%. All targets use the same frozen LightGBM model; only the threshold that decides who is flagged differs, so scores and explanations are identical.

| | 80% target | 90% target (chosen) | 95% target |
|---|---|---|---|
| Threshold | from `selected_metadata.json` | from `artifacts/shared/exploratory_review_policy.json` | computed from validation scores |
| Held-out test | evaluated first | evaluated second, on the same test set | not evaluated (validation only) |

**Review queue.** The queue uses the chosen 90% target, with no toggle. A short note names it and points to Model and evidence. Providers fall into three bands:
- **Review first:** the highest scores, at or above the 80% target's threshold.
- **Additional review:** flagged by the 90% target below that. These need real review, or the extra recall is not delivered.
- **Not flagged:** not automatically cleared.

If the policy file is missing or doesn't match the frozen run, the queue falls back to the 80% target's threshold, with a warning.

**Model and evidence:**
- **Performance** opens with the experiment ("we experimented with 80%, 90% and 95% … and chose 90%") and states the order: the 80% target was evaluated on the held-out set first, then the 90% target on the same set. It shows both confusion matrices (the 90% one with changes relative to the 80% target), a comparison table with an "Order of evaluation" row, and a one-line note that the 90% result is not an independent test.
- **Review thresholds** explains how both thresholds were set, with both marked on the workload curve.
- **Review policies** compares the 80%, 90% and 95% targets on validation data, with increments against the 80% target and against the row above. The 95% target appears only there and is never applied to a queue.

The 90% target is used only if its policy file matches the frozen run: the same run, model fingerprint, 80% threshold and validation predictions, and a threshold below the 80% one. Its evaluation file must also have used the same threshold. The 95% row is offered only if the validation rule reproduces the frozen threshold exactly.

## Upload checks

**Rejected**, with a downloadable list of each affected provider, column and value:
- missing columns;
- duplicate column names;
- blank provider IDs, or IDs that repeat once spaces are removed;
- text in number columns, or infinite values;
- negative counts, amounts, days or averages;
- shares outside 0–100%;
- ages outside 0–120;
- flags that are blank or anything other than 0 or 1;
- files over 50,000 rows or 20 MB.

**Scored with a warning:**
- blank values in non-flag columns (filled with the training median);
- values outside the range seen in training;
- counts that aren't whole numbers.

Provider IDs are read as text, so IDs like `00123` keep their leading zeros.

Units and impossible-value rules come from `dashboard/glossary.py`. Check its definitions against notebook 05 when the feature engineering changes.

## Trying the upload

```bash
python -m dashboard.make_demo_files
```

This writes three files to `demo_uploads/`:
- a clean 60-provider file;
- a file that scores with warnings;
- a file that is rejected, listing three specific problems.

The **Download a sample to try** button in the app gives you the clean version directly. Add `demo_uploads/` to `.gitignore` if you don't want the files committed.

## Tests

```bash
python -m pytest tests/ -q
```

- `tests/test_dashboard_edge_cases.py` builds a small LightGBM model in a temporary folder. It runs on any machine, without the project's data. It covers:
  - the frozen-model checks (missing fingerprint, changed threshold, changed model file, an evaluation from another threshold, evaluation-only mode);
  - every upload rule above;
  - scoring, explanations and the waterfall.
- `tests/test_dashboard_core.py` checks the real frozen run. It confirms the app reproduces `test_predictions.csv` exactly, and skips if the project's artifacts aren't present.

## Explanations and SHAP

Explanations are TreeSHAP values computed by LightGBM itself (`pred_contrib=True`). They are on the log-odds scale, computed on the features after the pipeline's median imputation, and they add up exactly to each score. They should match `shap.TreeExplainer(model)` with no background dataset, applied to the imputed features.

Each push is measured against the model's average prediction, not against the training median shown for context.

## Design

- **Typeface:** Public Sans, the typeface of the US federal design system.
- **Colour:** blue is the app colour. Amber is reserved for "flagged" and "raises the score". Red and green are avoided.
- **Headings:** plain HTML, so Streamlit adds no link icons.
- **Narrow screens:** padding and summary text are reduced.
- **Streamlit versions:** each Streamlit function is asked which full-width option it supports, so the app works from 1.35 to current releases.
