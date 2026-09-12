# Pipeline Log

Running record of what's been done at each pipeline stage, why, and what to watch out for. Update this as you complete or modify a step — it's meant to save the next person (including future-you) from re-deriving decisions that were already made.

---

## Step 1: Data Loading & Joining
**Code:** `src/data_loader.py`
**Notebook:** `notebooks/01_data_loading_joining.ipynb`

**What it does:**
- Loads the 4 raw CSVs from `data/raw/`
- Stacks Inpatient + Outpatient claims into one table, tagging each row with `ClaimType`
- Merges in beneficiary info on `BeneID`, then provider fraud label on `Provider`
- Output: one claim-level table, 558,211 rows × 56 columns

**Key decisions:**
- Used `validate="many_to_one"` on both merges as a safety check against duplicate-key joins silently inflating row counts
- Kept the join at claim-level (not pre-aggregated) — aggregation to provider-level happens explicitly in Step 4, so intermediate steps can inspect claim-level detail

**Verified:**
- 5,410 unique providers, 138,556 unique beneficiaries — matches expected dataset scale
- Claim-level fraud rate (38.1%) is higher than provider-level fraud rate (~9%) — fraudulent providers generate disproportionately more claims. This is a real finding worth keeping for the report, not a bug.

---

## Step 2: Cleaning
**Code:** `src/cleaning.py`
**Notebook:** `notebooks/02_cleaning.ipynb`

**What it does, in order:**
1. `parse_dates()` — converts date columns from string to datetime
2. `recode_chronic_conditions()` — `ChronicCond_*` from `{1=Yes, 2=No}` to `{1, 0}`
3. `recode_renal_disease_indicator()` — `RenalDiseaseIndicator` from `{"Y","0"}` to `{1, 0}`
4. `recode_fraud_label()` — `PotentialFraud` from `{"Yes","No"}` to `{1, 0}`
5. `add_age_columns()` — derives `AgeAtClaim` and `IsDeceased`
6. `cast_categorical_codes()` — casts `State`, `County`, `Gender`, `Race` to `category` dtype (they're codes, not quantities)
7. `add_physician_involvement_count()` — derives `NumPhysiciansInvolved` (0-3) from the physician ID columns
8. `fill_structural_missing()` — fills `NaN` with `0` only for Deductible/Reimbursement amount columns

**Key decisions:**
- **No rows dropped.** Row-level filtering decisions are deferred to Step 4, where the aggregation context makes the right call clearer.
- **Missingness in `AdmissionDt`, `DischargeDt`, `DOD` is left as-is (NaT), not imputed.** These are structurally meaningful (outpatient claim / beneficiary alive), and filling them would destroy that signal. Verified: outpatient claim count (517,737) exactly matches missing `AdmissionDt` count — confirms the missingness is purely structural, not a join/data issue.
- Physician columns treated the same way — missing `OperatingPhysician`/`OtherPhysician` means "not needed for this claim," converted into a `NumPhysiciansInvolved` count rather than left as raw missingness.

**Verified:**
- Shape after cleaning: 558,211 rows × 58 columns (2 more than raw, from derived columns)
- Age range: 25.1 to 101.0 — no negative or implausible ages
- Deceased beneficiaries: 4,131 claim-rows have non-null `DOD` (note: this is claim-rows, not unique deceased people — get the beneficiary-level count separately if needed for reporting)

**Not yet handled (deferred to Step 4 — feature engineering):**
- Diagnosis/procedure code columns (`ClmDiagnosisCode_*`, `ClmProcedureCode_*`) are high-cardinality sparse categoricals — not usable as-is in most models. Plan: derive counts (`NumDiagnosisCodes`, `NumProcedureCodes`) and/or flags for specific high-risk codes.
- Physician ID columns are also high-cardinality — likely to be aggregated into provider-level features (e.g. number of distinct physicians a provider works with) rather than used directly.

---

## Step 3: EDA & Statistical Tests
**Code:** `src/eda.py`
**Notebook:** `notebooks/03_eda_stats.ipynb`

**What it does:**
- `build_minimal_provider_summary()` — lightweight provider-level aggregation (claim count, avg/total claim amount, avg age, % inpatient, fraud label) — just enough structure to run valid statistical tests. **This is not the full feature set** — that's built properly in Step 4.
- `chi_square_test()` — tests independence between a categorical variable and the fraud label
- `t_test()` — Welch's t-test comparing a numeric variable's mean between fraud/non-fraud groups
- `anova_test()` — one-way ANOVA for comparing a numeric variable across 3+ groups
- `benford_leading_digit_test()` — chi-square goodness-of-fit test of claim amounts' leading digits against Benford's Law's expected distribution

**Key decision:**
- **Formal hypothesis tests run on provider-level aggregates, not claim-level rows.** The fraud label is defined at the provider grain, and claims from the same provider aren't independent observations — testing at claim-level would inflate the effective sample size and risk misleadingly significant p-values. Purely descriptive/visual EDA (distributions, missingness patterns) is fine at claim-level.

**Results:**
- **Strongest signals:** `TotalClaimAmt` (r=0.58 with fraud), `NumClaims` (r=0.37) — both confirmed by boxplots showing clear separation between Fraud/Non-Fraud medians and spread
- **Moderate signal:** `ClaimType` — Inpatient claims show a notably higher fraud rate (~0.57) than Outpatient (~0.37)
- **Weak signal:** `AvgNumPhysicians`, `AvgAgeAtClaim` — little to no separation between Fraud/Non-Fraud groups
- **Multicollinearity flagged early:** `AvgClaimAmt` and `PropInpatient` correlate at 0.84 with each other (not with fraud) — carried forward into Step 4's VIF analysis
- **Benford's Law — corrected interpretation:** raw chi-square initially suggested Non-Fraud claims deviate more from Benford's Law than Fraud claims (chi2: ~12,365 vs ~7,588). This is misleading — chi-square scales with sample size, and Non-Fraud has ~1.6x more claims. Using Mean Absolute Deviation (MAD, scale-invariant), both groups deviate by nearly the same amount (Fraud: 0.0186, Non-Fraud: 0.0181). **Conclusion: Benford's Law is not a strong standalone fraud discriminator in this dataset**, likely because claim amounts are fee-schedule/policy-constrained rather than organically occurring — a known limitation of applying Benford's Law to regulated financial data. Reported as a tested-but-inconclusive hypothesis, not omitted.

---

## Step 4: Provider-Level Feature Engineering
**Code:** `src/features.py`, `src/feature_selection.py`
**Notebook:** `notebooks/04_feature_engineering.ipynb`

**What it does:**
Builds 6 feature blocks from the cleaned claim-level table, aggregated to one row per provider:
1. **Claim volume** — `NumClaims`, `PropInpatient`
2. **Claim amounts** — `MedianClaimAmt`, `StdClaimAmt`, `MaxClaimAmt`, `DeductibleToReimbursementRatio`
3. **Diagnosis/procedure complexity** — `AvgNumDiagnosisCodes`, `AvgNumProcedureCodes`
4. **Physician involvement** — `AvgNumPhysiciansInvolved`, `NumUniqueAttendingPhysicians`, `NumUniqueOperatingPhysicians`
5. **Beneficiary demographics** — `AvgAgeOfBeneficiaries`, `PropDeceasedBeneficiaries`, `AvgChronicConditions`, `AvgClaimsPerBeneficiary`
6. **Duration** — `AvgClaimDurationDays`, `AvgLengthOfStayDays`

Then: `evaluate_feature_significance()` (t-test + Cohen's d per feature) → `compute_vif()` / `iterative_vif_selection()` (multicollinearity check) → two feature sets saved to `data/processed/feature_sets.json`.

**Key decisions:**
- **Two aggregation units kept separate:** claim-level events aggregated directly; beneficiary demographics aggregated from a table deduplicated to one row per (Provider, BeneID) first, to avoid a frequently-visiting beneficiary skewing the provider's demographic profile.
- **No target leakage:** no feature computation touches `PotentialFraud`.
- **Multicollinearity fixed in two passes:**
  1. Removed exact arithmetic identities directly inside `features.py` (not just filtered in the notebook) so they can't resurface on re-runs: `TotalClaimAmt`/`TotalDeductiblePaid` (near-exact products of `NumClaims × Avg*`), `NumInpatientClaims`/`NumOutpatientClaims` (exact components of `NumClaims`/`PropInpatient`), `AvgDeductiblePaid` (redundant with `AvgClaimAmt` + the ratio), diversity-ratio features that turned out to duplicate `AvgNum*Codes`, `NumUniqueBeneficiaries` (redundant with `NumClaims`/`AvgClaimsPerBeneficiary`).
  2. Used `iterative_vif_selection()` for genuine (non-identity) domain correlations — e.g. `AvgClaimAmt` correlates with `MedianClaimAmt`/`StdClaimAmt`/`MaxClaimAmt` (all summary stats of the same distribution) and with `PropInpatient` (inpatient claims cost more).
- **Two feature sets maintained deliberately:** full 17-feature set for tree-based/boosting models (not sensitive to multicollinearity), and a 15-feature VIF-clean subset for logistic regression (drops `AvgClaimAmt`, `PropInpatient`, `NumUniqueOperatingPhysicians`).

**Verified:**
- Provider feature table: 5,410 rows × 19 columns (`Provider`, `PotentialFraud`, 17 features)
- No unexpected missing values after aggregation
- Final VIF for both feature sets: all values below 10

**Feature significance (by effect size, not p-value alone):** claim volume (`NumClaims`) and claim amount statistics remain the strongest features, consistent with Step 3's EDA. Some features (e.g. `AvgNumPhysiciansInvolved`) showed weak linear effect size — flagged but retained for tree-based models, which can exploit nonlinear/interaction signal a t-test can't detect.

---

## Step 5: Train/Test Split & Class Imbalance Handling
**Status:** ⬜ Not started

**Planned approach:**
- Split by **Provider**, not claim, to avoid the same provider's claims leaking across train and test sets
- Apply class imbalance handling (SMOTE / class weights — TBD) **after** splitting, and **only** to the training set, to avoid leaking synthetic/resampled signal into evaluation

## Step 6: Core Modeling
**Status:** ⬜ Not started

## Step 7: SHAP Evaluation & Report Assembly
**Status:** ⬜ Not started