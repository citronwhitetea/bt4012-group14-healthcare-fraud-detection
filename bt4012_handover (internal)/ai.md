# AI Context Briefing — BT4012 Healthcare Provider Fraud Detection

**Instructions for use:** Paste this entire document as your first message in a new chat with any AI assistant (Claude, ChatGPT, etc.) before asking for help. It gives the assistant full context on the project so it can pick up work immediately without re-explaining the dataset, pipeline, or decisions already made. Update this file whenever a pipeline step is completed, and re-paste the latest version when starting a new chat.

---

## Project

BT4012 (Fraud Analytics) group project. Building a fraud detection pipeline on the [Healthcare Provider Fraud Detection dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) (Kaggle, rohitrox). Goal: predict whether a **healthcare provider** (not an individual claim or patient) is potentially fraudulent, using claims data aggregated to the provider level.

Repo: `bt4012-group14-healthcare-fraud-detection` (GitHub). Standard structure:
```
data/{raw,processed}/   notebooks/   src/   reports/figures/
```
There is no `docs/` folder — column-by-column dataset documentation lives in `data/raw/raw_data.md`, and the full step-by-step decision log is `bt4012_handover (internal)/human.md` (this file's companion).

## Dataset

Four raw CSVs in `data/raw/`, joined via `Provider`/`BeneID` keys:
- `provider_labels.csv` — Provider-level fraud label (`PotentialFraud`: Yes/No). 5,410 providers.
- `beneficiary_raw.csv` — Patient demographics, chronic conditions. 138,556 beneficiaries.
- `inpatient_raw.csv` / `outpatient_raw.csv` — Claim-level records (dates, amounts, diagnosis/procedure codes, physicians). 558,211 claims combined.

We use **Train files only** (Kaggle Test files lack the fraud label) and build our own train/test split.

**Key fact:** claim-level fraud rate (~38%) is much higher than provider-level fraud rate (~9%) — fraudulent providers generate disproportionately more claims. Don't confuse the two when discussing "the fraud rate."

Standalone raw-table EDA notebooks exist for each source file, run *before* joining/cleaning: `notebooks/inpatient_eda.ipynb`, `outpatient_eda.ipynb`, `beneficiary_eda.ipynb`.

## Pipeline status (7 steps, some split into `a`/`b` sub-notebooks by grain)

| Step | Code | Notebook(s) |
|---|---|---|
| 1. Data loading & joining | `src/data_loader.py` | `01_data_loading_joining.ipynb` |
| 2. Cleaning | `src/cleaning.py` | `02_cleaning.ipynb` |
| 3. EDA & statistical tests | `src/eda.py` | `03a_claim_level_eda.ipynb` (claim-grain descriptive EDA + fraud-rate/chi-square/Benford tests), `03b_provider_level_eda.ipynb` (provider-grain EDA, boxplots, t-tests, Extended Candidate Features significance/VIF testing) |
| 4. Feature engineering | `src/features.py`, `src/feature_selection.py` | `04a_basic_claim_level_feature_engineering.ipynb` (claim-grain feature exploration), `04b_feature_engineering.ipynb` (provider-grain, the real modeling feature table) |
| 5. Train/test split | — | `05_train_test_split.ipynb` (done: 80/20 provider-level stratified split, 4,328 train / 1,082 test; imbalance handling happens inside CV folds in 06b/06c) |
| 6. Core modeling | `src/fold_vif.py`, `src/final_evaluation.py` | `06b_boosting_models.ipynb` (XGBoost/LightGBM, Pei), `06c_logistic_random_forest_models.ipynb` (LR/RF, Swei), `06d_high_recall_policy_and_misses.ipynb` (90% recall policy, Pei). Done; final model frozen and tested once on 24 Oct. Details: `pei_work_summary_ai.md`, `reports/final_model_evaluation.md` |
| 6+. Post-evaluation checks (15 Nov) | `src/features.py` | `06e_leakage_fixes_cv_check.ipynb`, `06f_optuna_early_stopping.ipynb`, `06g_ensemble.ipynb`. Done; see "Post-evaluation work" below |
| 7. SHAP evaluation & report | — | `07_shap_evaluation.ipynb` (done; `reports/shap_evaluation.md`) |

**Why the `a`/`b` split:** `a` notebooks operate at claim grain (one row per claim), `b`/unsuffixed notebooks operate at provider grain (one row per provider) or are the "real" pipeline step. This mirrors a cleanup done mid-project — an earlier `03_eda_stats.ipynb` mixed both grains in one file and got split cleanly; `04` was similarly renamed to `04b` once `04a` existed, to avoid the same ambiguity. Both notebooks in each pair share code — see Code architecture below — so there's no duplicated logic, just different points at which each stops (claim-grain exploration vs. provider-grain aggregation/modeling).

## Code architecture (`src/`)

- **`data_loader.py`** → `build_claim_level_dataset()` — loads 4 CSVs, stacks Inpatient+Outpatient, merges beneficiary + label info. Returns claim-level DataFrame (558,211 rows × 56 columns). Uses `validate="many_to_one"` on both merges; verified zero unmatched keys and zero duplicate keys on either lookup table.
- **`cleaning.py`** → `clean_claim_level_dataset(df)` — parses dates, recodes binary columns (`ChronicCond_*`, `RenalDiseaseIndicator`, `PotentialFraud`), derives `AgeAtClaim`/`IsDeceased`/`NumPhysiciansInvolved`, casts geo codes to `category` and `ClmProcedureCode_*` to string (was accidentally `float64`), fills structural missingness (claim-type-aware for `DeductibleAmtPaid`, with a `DeductibleAmtPaidMissing` traceability flag), and clips invalid negative reimbursement/deductible values (`HadNegativeAmount` flag). No rows dropped. Returns 558,211 rows × 61 columns.
- **`eda.py`** → `build_minimal_provider_summary(df)`, `chi_square_test()`, `t_test()`, `anova_test()`, `cramers_v()`, `benford_leading_digit_test()`, `benford_test_by_fraud_group()`, `benford_mean_absolute_deviation()`, plus `plot_*()` visualization functions.
- **`features.py`** — the single source of truth for both claim-level and provider-level feature engineering:
  - `build_claim_level_features(df)` — orchestrates 11 `add_*` functions (claim type, financial, duration, diagnosis, procedure, complexity, physicians, beneficiary health, temporal, repeat-claiming, same-day-duplication), adding 25 claim-grain columns. Called by both `build_provider_features()` below *and* `04a_basic_claim_level_feature_engineering.ipynb` directly — never implemented twice.
  - `build_provider_features(df, reference_providers=None)` — calls `build_claim_level_features()` first, then aggregates across 10 feature blocks to one row per provider. Returns 5,410 rows × 28 columns (`Provider`, `PotentialFraud`, 26 features). `reference_providers` limits which providers the physician-network feature counts (see decision 10). Since 15 Nov it leaves `TopDiagnosisCodeShare`, `TopProcedureCodeShare` and `BenfordMAD` as NaN for the model pipeline to impute (see decision 8).
- **`feature_selection.py`** → `evaluate_feature_significance()` (t-test + Cohen's d per feature), `flag_weak_features()`, `compute_vif()`, `iterative_vif_selection()` (greedy multicollinearity removal).
- **`fold_vif.py`** → `FoldVIFSelector` (VIF selection fitted inside each CV fold, used by 06c's logistic regression) and `NamedSMOTE`.
- **`final_evaluation.py`** → the guard that allowed the single held-out evaluation and now only reads saved results.

## Key decisions & rationale (don't re-litigate these without reason)

1. **Provider is the unit of analysis.** Claims aren't independent within a provider — statistical tests and train/test splits must operate at Provider grain, never Claim grain, or results/leakage checks are invalid.
2. **No feature or split touches `PotentialFraud`** except as the final label — avoids target leakage.
3. **Beneficiary demographics are aggregated from a table deduplicated to one row per (Provider, BeneID)** before averaging — otherwise frequently-visiting patients skew a provider's demographic profile.
4. **Multicollinearity was resolved in two passes:** (a) removed exact arithmetic identities baked into the feature functions themselves (e.g. `TotalClaimAmt ≈ NumClaims × AvgClaimAmt`, `NumInpatientClaims`/`NumOutpatientClaims` vs `NumClaims`/`PropInpatient`), (b) used iterative VIF elimination for genuine domain correlations. After the latest feature additions, VIF elimination now drops 4 features (`AvgClaimAmt`, `TopBeneficiaryClaimShare`, `PropInpatient`, `NumUniqueOperatingPhysicians`), up from 3 previously.
5. **Two feature sets exist**, saved in `data/processed/feature_sets.json`:
   - `full_features` (26 features) — for Random Forest / boosting (not sensitive to multicollinearity)
   - `logistic_features` (22 features, VIF < 10) — for logistic regression
6. **Benford's Law is not a strong fraud discriminator at the aggregate level** — raw chi-square looked significant but that's a sample-size artifact; MAD (scale-invariant) shows Fraud/Non-Fraud groups deviate almost identically (0.0186 vs 0.0181). Per-provider `BenfordMAD` differs between groups (Cohen's d = -0.54), **but `07_shap_evaluation` section 9 shows this mostly reflects claim volume**: MAD shrinks with claim count and fraud providers are larger. Against the dataset's own digit pattern at the same claim count, d = +0.07 (CI includes 0). Team decision (27 Oct): present as tested with a qualified result, not as a fraud detector. Only computable for providers with ≥20 claims (61.4%); `BenfordMADMissing` flags the rest.
7. **A batch of 5 candidate features were added to `src/features.py`** after being validated in `03b`'s Extended Candidate Features section (Cohen's d ranking + redundancy check against existing features): `TopDiagnosisCodeShare`, `TopProcedureCodeShare`, `AvgPhysicianProviderSpread`, `TopBeneficiaryClaimShare`, `BenfordMAD`. **Counter-intuitive finding:** fraud providers show *lower* diagnosis/procedure code concentration and *lower* dependence on any single "super-patient" — more diverse, broader-based billing, not more repetitive/templated billing as hypothesized. One candidate (`RoundNumberClaimRate`, initially d=0.68) was a **false lead** — it correlated 0.99 with the existing `PropInpatient` and was dropped.
8. **A real bug was caught and fixed**: `TopProcedureCodeShare` was initially 0-filled for providers with zero procedure-coded claims (62.6% of providers), which silently flipped its effect size from d=-0.89 to d=+0.11 (0 means "many codes, none dominant" — the opposite of "no codes at all"). Fixed by filling with the mean among providers who *do* have a value, plus `TopProcedureCodeShareMissing`/`TopDiagnosisCodeShareMissing` traceability flags. **Update (15 Nov):** that mean was computed over all 5,410 providers, including test providers. `src/features.py` now leaves these values (and `BenfordMAD`) as NaN, so imputation happens inside the model pipeline using training rows only. `04b` re-applies the old mean fill explicitly, so the canonical `provider_features_{train,test}.csv` stay byte-identical to the frozen model's fingerprints. No score changes: 06b/06c already reset these columns to NaN and re-imputed them per fold.
10. **Physician-network leak (found 15 Nov).** `AvgPhysicianProviderSpread` counted each physician's providers across all providers' claims, so test providers shaped training providers' values. This is covariate leakage, not target leakage: no labels were involved. `build_physician_network_features(df, reference_providers=...)` now counts only a reference set. A reference provider counts itself once; any other provider is scored as if it alone had joined the set. Leak-free tables are `data/processed/provider_features_{train,test}_leakfree.csv`, built by `06e`. Measured effect: none beyond fold noise (see below).
9. **Strongest features (by effect size, not p-value):** `MaxClaimAmt` (d=1.61), `TopProcedureCodeShareMissing` (d=-1.47), `AvgLengthOfStayDays` (d=1.21), `StdClaimAmt` (d=1.11), `TopProcedureCodeShare` (d=-0.97), `BenfordMADMissing` (d=-0.89), `TopBeneficiaryClaimShare` (d=-0.75). Two of these (`TopProcedureCodeShareMissing`, `BenfordMADMissing`) are likely partly restating claim volume/`PropInpatient` rather than adding fully independent signal — worth a closer look before over-weighting them in the report. Weak features: `AvgAgeOfBeneficiaries` (d≈0.003, essentially flat), `DeductibleToReimbursementRatio`, `PropDeceasedBeneficiaries` — kept for tree models anyway, since a t-test only catches linear effects.

## Grading rubric (keep in mind for any deliverable)

| Criterion | Points |
|---|---|
| Data understanding / preprocessing | 3 |
| Method selection / justification | 4 |
| Experimental design / evaluation | 4 |
| Results / insights / applicability | 3 |
| Limitations / reproducibility | 2 |

## Final model (do not change)

**The frozen LightGBM is the final model:**
- **Run:** `artifacts/boosting/20261011_232303_full`.
- **Model:** original class balance, threshold 0.0861 for ≥80% validation recall.
- **Held-out result (evaluated once, 24 Oct):** AP 0.754, recall 0.851 (86 of 101 caught), precision 0.486, 177 of 1,082 flagged.
- **90% policy:** threshold 0.0241, evaluated afterwards on the same test set (1 Nov).

The selection record, test outputs and dashboard are fingerprint-checked. Never re-evaluate on the test set or edit those files. See `pei_work_summary_ai.md` §6.

## Post-evaluation work (15 Nov 2026, Yumin)

All development-only (training providers, cross-validation) unless stated otherwise. None of it changed the final model.

| Notebook / record | Question | Result |
|---|---|---|
| `06e` + `reports/leakfree_retraining.md` | Does fixing the physician-network leak change anything? | No. LightGBM mean fold AP 0.674 → 0.667–0.677 depending on how strictly the fix is applied (fold SD ≈ 0.025). 06b/06c rerun on leak-free tables (`FEATURE_VARIANT=leakfree`) still selects LightGBM / original. |
| `06f` | Does a wider Optuna search (150 trials, early stopping) beat 06b's 20-config random search? | No. +0.008 on the tuning folds was selection optimism: nested CV −0.005, 15 new splits −0.002 (wins 5 of 15). Tuning is at its ceiling (~0.67–0.68 AP). |
| `06g` | Do ensembles or CatBoost help? | Equal average of LightGBM + logistic regression: +0.012 mean fold AP on 15 new splits (wins 11 of 15), because the two models make different mistakes (rank correlation 0.87). CatBoost matches LightGBM alone but adds ~0 when averaged with it. Stacking and the 5-model average add nothing beyond the simple pair. |
| `artifacts/shared/ensemble_final_selection.json`, `scripts/run_ensemble_test.py` | Held-out check of the ensemble | The ensemble was committed as final (commits `492403e`, `317e260`) and then tested once. Test AP 0.759 vs 0.754; recall 0.832 (84 caught) vs 0.851 (86); precision 0.494 vs 0.486. No difference is statistically distinguishable. **Withdrawn after the test**: the team kept LightGBM because the ensemble missed 2 more fraud providers. This reversal is test-informed and is disclosed in `reports/final_model_evaluation.md`. The record's status is `withdrawn_after_test`. |

**Rules from this work:**
- **Test-set uses so far (three):** the 80% evaluation (24 Oct), the 90% policy (1 Nov) and the ensemble (15 Nov). Do not add a fourth. Any new threshold or model is validation-only.
- **Reporting:** present the ensemble as a supplementary experiment. It was better in CV and indistinguishable on test, and the reversal must be disclosed.
- **Dependencies:** `requirements.txt` now includes `optuna==4.1.0` and `catboost==1.2.8`. Do not install `catboost==1.2.7`: it downgrades numpy to 1.x.
- **Recall target as the lever:** if the team wants fewer missed fraud cases, change the recall target (validation-only), not the model. The ensemble's validation trade-off, computed from 06g's OOF predictions: 85% target = 803 reviews, 90% = 1,026, 95% = 1,435, out of 4,328 providers.
