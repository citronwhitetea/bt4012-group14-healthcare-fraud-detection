# Pipeline Log

Running record of what's been done at each pipeline stage, why, and what to watch out for. Update this as you complete or modify a step — it's meant to save the next person (including future-you) from re-deriving decisions that were already made.

---

## Step 0: Raw-Table EDA (before joining/cleaning)

**Notebooks:** `notebooks/inpatient_eda.ipynb`, `notebooks/outpatient_eda.ipynb`, `notebooks/beneficiary_eda.ipynb`

Standalone EDA on each raw CSV in isolation, before any join or cleaning — missingness, dtypes, reimbursement/duration distributions, top diagnosis codes, etc. Findings from these fed directly into Step 2's cleaning decisions (see below) and Step 4's feature ideas. Key findings:
- **Inpatient:** `DeductibleAmtPaid` is a near-constant ~$1,068 whenever present (2.2% missing). `ClaimDuration`/`LengthOfStay` are near-identical since `ClaimStartDt`≈`AdmissionDt` in 99.9% of rows.
- **Outpatient:** `DeductibleAmtPaid` behaves oppositely — mostly $0, real variance, 0% missing. Procedure codes are essentially unused (99.97% missing). Reimbursement is ~35x smaller than inpatient on average.
- **Beneficiary:** found 15 rows with negative `IPAnnualReimbursementAmt` (down to -$8,000) and 12 with negative `OPAnnualReimbursementAmt` — a real data-quality issue, fixed in Step 2.

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
- Zero unmatched join keys in either direction, zero duplicate keys in either lookup table, zero `ClaimID` overlap between Inpatient/Outpatient — the join is clean
- Claim-level fraud rate (38.1%) is higher than provider-level fraud rate (~9%) — fraudulent providers generate disproportionately more claims. This is a real finding worth keeping for the report, not a bug.

---

## Step 2: Cleaning
**Code:** `src/cleaning.py`
**Notebook:** `notebooks/02_cleaning.ipynb`

**What it does, in order:**
1. `parse_dates()` — converts date columns from string to datetime
2. `recode_chronic_conditions()` — `ChronicCond_*` from `{1=Yes, 2=No}` to `{1, 0}`
3. `recode_renal_disease_indicator()` — `RenalDiseaseIndicator` from `{"Y","0"}` to `{1, 0}`. **Hardened**: now raises a `ValueError` if an unmapped, non-null value shows up (e.g. a new code in a future data refresh), instead of silently collapsing it into "No" alongside genuine `"0"` values. Genuine missing (`NaN`) still defaults to 0.
4. `recode_fraud_label()` — `PotentialFraud` from `{"Yes","No"}` to `{1, 0}`
5. `add_age_columns()` — derives `AgeAtClaim` and `IsDeceased`. **Hardened**: now guards `DOD`/`DOB`/`ClaimStartDt` with `if col in df.columns` like every other function in the file, instead of hard-referencing them — harmless on the current pipeline but no longer fragile if reused on a differently-shaped frame.
6. `cast_categorical_codes()` — casts `State`, `County`, `Gender`, `Race` to `category` dtype (they're codes, not quantities)
7. `cast_procedure_codes()` — **new**: casts `ClmProcedureCode_1..6` from `float64` (an artifact of `NaN` forcing pandas to upcast during CSV read) to string, matching `ClmDiagnosisCode_*`/`DiagnosisGroupCode`. Left as float, a code like `9904` could get treated as a summable/orderable number by anything downstream that doesn't know better.
8. `add_physician_involvement_count()` — derives `NumPhysiciansInvolved` (0-3) from the physician ID columns
9. `fill_structural_missing()` — **rewritten**: `DeductibleAmtPaid` is now filled per-`ClaimType` median (resolves to ~$1,068 for Inpatient, ~$0 for Outpatient — matching each claim type's real distribution) instead of a blanket 0, with a `DeductibleAmtPaidMissing` flag recording which 899 inpatient rows were filled. *Why this mattered:* the old blanket-0 fill was quietly fabricating variance in inpatient `DeductibleAmtPaid` that shouldn't exist (post-fill mean dropped to $1,044 with std $157, purely from artificial zeros, when the true value is a constant $1,068). All other Deductible/Reimbursement columns still fill with 0 as before (they're genuinely ~0% missing already).
10. `clip_negative_amounts()` — **new**: clips negative values in any Deductible/Reimbursement column to 0 (73 claim rows / 27 beneficiaries affected — the negative `IPAnnualReimbursementAmt`/`OPAnnualReimbursementAmt` rows found in Step 0's beneficiary EDA), with a `HadNegativeAmount` flag for traceability. A negative reimbursement isn't a valid value; these were rare data-entry errors, not a legitimate credit/refund signal.

**Key decisions:**
- **No rows dropped.** Row-level filtering decisions are deferred to Step 4, where the aggregation context makes the right call clearer.
- **Missingness in `AdmissionDt`, `DischargeDt`, `DOD` is left as-is (NaT), not imputed.** These are structurally meaningful (outpatient claim / beneficiary alive), and filling them would destroy that signal. Verified: outpatient claim count (517,737) exactly matches missing `AdmissionDt` count — confirms the missingness is purely structural, not a join/data issue.
- Physician columns treated the same way — missing `OperatingPhysician`/`OtherPhysician` means "not needed for this claim," converted into a `NumPhysiciansInvolved` count rather than left as raw missingness.
- **Missing ≠ zero, in general.** The `DeductibleAmtPaid` and negative-amount fixes above are both instances of the same lesson: don't default-fill a numeric gap with 0 without checking what the *true* distribution around that gap actually looks like.

**Verified:**
- Shape after cleaning: 558,211 rows × 61 columns (5 more than raw: `IsDeceased`, `AgeAtClaim`, `NumPhysiciansInvolved`, `DeductibleAmtPaidMissing`, `HadNegativeAmount`)
- Age range: 25.1 to 101.0 — no negative or implausible ages
- Deceased beneficiaries: non-null `DOD` claim-rows exist as expected (note: this is claim-rows, not unique deceased people — get the beneficiary-level count separately if needed for reporting)
- Inpatient `DeductibleAmtPaid` is now a clean constant $1,068 (std = 0) post-fill, matching its true distribution
- Zero negative values remain in any Deductible/Reimbursement column after `clip_negative_amounts()`

**Not yet handled (deferred to Step 4 — feature engineering):**
- Diagnosis/procedure code columns (`ClmDiagnosisCode_*`, `ClmProcedureCode_*`) are high-cardinality sparse categoricals — not usable as-is in most models. Handled in Step 4 via counts (`NumDiagnosisCodes`, `NumProcedureCodes`) and concentration ratios (`TopDiagnosisCodeShare`, `TopProcedureCodeShare`).
- Physician ID columns are also high-cardinality — aggregated into provider-level features (`NumUniqueAttendingPhysicians`, `AvgPhysicianProviderSpread`, etc.) rather than used directly.

---

## Step 3a: Claim-Level EDA
**Notebook:** `notebooks/03a_claim_level_eda.ipynb`

Pure descriptive EDA at **claim grain** (one row per claim), no `src/eda.py`/`src/features.py` dependency for the descriptive parts — self-contained. Covers 12 topics: claim type, amounts, duration, diagnosis/procedure intensity, physician involvement, beneficiary characteristics, patient-claim relationships, temporal patterns, duplicate/suspicious patterns, missingness, and cross-variable relationships. Plus a 13th section — **Fraud-Rate Comparisons** — which *does* use the fraud label (chi-square + Cramér's V on `ClaimType`, fraud rate by `ClaimType`/`State`, aggregate Benford's Law), reusing `src/eda.py`'s already-validated functions rather than reimplementing the stats. This section originally lived in `03_eda_stats.ipynb`/`03b` but was moved here since it's computed directly on claim-level rows, not a provider aggregate.

**Key results:**
- Outpatient dominates claim volume ~13:1 over Inpatient
- `ClaimDuration` averages just 1.7 days (mostly same-day Outpatient visits) — do **not** confuse this with `LengthOfStay` (Inpatient-only, averages 5.7 days); they measure different things and were nearly conflated once already
- No weekend effect on claim volume (28.5% observed vs. 28.6% uniform-week baseline)
- 11.3% of claims fall within 7 days of the same beneficiary's prior claim, 35.4% within 30 days
- 2,798 claims share the same BeneID + Provider + ClaimStartDt
- `ClaimType` has a real but small-to-moderate association with fraud (Cramér's V = 0.11)
- Aggregate Benford's Law: not discriminative (MAD 0.0186 fraud vs. 0.0181 non-fraud) — the per-provider version in Step 3b differs between groups (d = -0.54), but `07` section 9 shows that mostly reflects claim volume, so it is not presented as a fraud signal

---

## Step 3b: Provider-Level EDA & Statistical Tests
**Code:** `src/eda.py`, `src/feature_selection.py` (for the significance ranking)
**Notebook:** `notebooks/03b_provider_level_eda.ipynb`

**History:** this notebook is the result of merging two things that used to be separate: the original `03_eda_stats.ipynb` (minus its claim-level-only sections, which moved to `03a`) and a standalone `05_provider_level_eda.ipynb` that was renamed into this `03b` slot to avoid colliding with `05_train_test_split.ipynb`. Both told a genuinely different, complementary part of the provider-level story, so they were combined rather than picking one.

**What it does:**
1. **Statistical testing on a minimal 6-column `provider_df`** (`build_minimal_provider_summary()`): fraud-group boxplots, correlation heatmap, Welch's t-tests + Cohen's d (`AvgClaimAmt`, `NumClaims`).
2. **Extended Candidate Features**: tested 12 new candidate provider-level signals (physician self-dealing/network sharing, diagnosis/procedure code concentration, round-number billing, claim-amount CV, per-provider Benford's Law, beneficiary concentration/geographic dispersion, zero-reimbursement rate, 30-day readmission rate) via `evaluate_feature_significance()`/`flag_weak_features()`, plus a redundancy check against the existing `src/features.py` feature set. This is where the 5 features now in `src/features.py` were validated (see Step 4 log).
3. **Broader provider-level aggregation & EDA**: a separate, richer 38-column `provider_features` table built directly from `claims_with_basic_features.csv` (Step 4a's output), explored across 12 topics (target distribution, claim volume, beneficiary activity, reimbursement, duration, diagnosis/procedure intensity, physician behavior, patient complexity, temporal behavior, repeated/suspicious claims, consolidated distributions, feature relationships). **This table is exploratory only — not wired into the actual modeling pipeline** (that's `src/features.py`/`04b`'s `provider_features.csv`). Saved to `data/processed/provider_level_eda_table.csv`.

**Key decision:** formal hypothesis tests run on provider-level aggregates, not claim-level rows — the fraud label is defined at the provider grain, and claims from the same provider aren't independent observations.

**Results:**
- **Strongest signals (Step 3 core):** `TotalClaimAmt` (r=0.58 with fraud), `NumClaims` (r=0.37)
- **Moderate signal:** `ClaimType` — Inpatient claims show a notably higher fraud rate (~0.57) than Outpatient (~0.37)
- **Weak signal:** `AvgNumPhysicians`, `AvgAgeAtClaim`
- **Multicollinearity flagged early:** `AvgClaimAmt` and `PropInpatient` correlate at 0.84 — carried into Step 4's VIF analysis
- **Extended candidates — 5 strong, 1 false lead, rest weak:** see Step 4 log for the full breakdown; the headline is a **reversed hypothesis** — fraud providers show *more* diverse diagnosis/procedure coding and *less* dependence on any single "super-patient," not more repetitive/templated billing.
- **Broader 38-feature table — top correlates with fraud:** `TotalReimbursement` (0.58), `NumUniqueProcedureCodes` (0.56, but correlates 0.97 with `NumClaims` — mostly restating volume, not real diversity), `NumInpatientClaims` (0.53), `MaxReimbursement` (0.51). Confirms the "breadth not concentration" pattern independently: `RepeatBeneficiaryRate` is *higher* for fraud while `TopBeneficiaryClaimShare` (from the Extended Candidates section) is *lower* — fraud providers see more beneficiaries more than once, but no single one dominates. `MeanAge` is essentially identical between groups (73.20 vs. 73.21) — billing differences aren't explained by patient age. Despite the "detect repeated billing" framing, `SameBeneProviderDayRate`/`ClaimWithin30DaysRate` are both essentially flat between fraud and non-fraud — tested, not discriminative.

---

## Step 4a: Basic Claim-Level Feature Engineering
**Code:** `src/features.py` (`build_claim_level_features()`)
**Notebook:** `notebooks/04a_basic_claim_level_feature_engineering.ipynb`

**What it does:** explores and documents the 25 claim-grain features built by `build_claim_level_features()` — `IsInpatient`, `LogReimbursement`/`HasDeductible`, `ClaimDuration`/`SameDayClaim`/`LengthOfStay`, `NumDiagnosisCodes`/`HasDiagnosis`/`PrimaryDiagnosis`, `NumProcedureCodes`/`HasProcedure`, `ClaimComplexity`, `HasOperatingPhysician`/`HasOtherPhysician`/`SameAttendingOperating`, `NumChronicConditions`/`HasRenalDisease`, `ClaimMonth`/`ClaimDayOfWeek`/`IsWeekend`, `DaysSincePreviousClaim`/`ClaimWithin7Days`/`ClaimWithin30Days`, `SameBeneProviderDay`/`NumClaimsSameBeneProviderDay`. Saves the enriched claim-level table (558,211 rows × 86 columns) to `data/processed/claims_with_basic_features.csv`, consumed by `03b`'s broader provider-level EDA section.

**Key decision — single source of truth:** this notebook originally computed these 25 features inline (self-contained, no `src/` dependency, per an earlier instruction to keep it that way). That created a duplication risk: `src/features.py`'s provider-level aggregators were independently recomputing very similar claim-level logic (e.g. `NumDiagnosisCodesOnClaim`, `ClaimDurationDays`) under different names, with no shared source. **Refactored** so `build_claim_level_features()` lives once in `src/features.py`, and both this notebook *and* `build_provider_features()` call it — the two can no longer drift out of sync. Verified via exact regression checks (6 decimal places) that the refactor didn't change any output value.

---

## Step 4b: Provider-Level Feature Engineering
**Code:** `src/features.py` (`build_provider_features()`), `src/feature_selection.py`
**Notebook:** `notebooks/04b_feature_engineering.ipynb` (renamed from `04_feature_engineering.ipynb` to sit alongside `04a`)

**What it does:**
Calls `build_claim_level_features()` first (see Step 4a), then builds 10 provider-level feature blocks:
1. **Claim volume** — `NumClaims`, `PropInpatient`
2. **Claim amounts** — `AvgClaimAmt`, `MedianClaimAmt`, `StdClaimAmt`, `MaxClaimAmt`, `DeductibleToReimbursementRatio`
3. **Diagnosis/procedure complexity** — `AvgNumDiagnosisCodes`, `AvgNumProcedureCodes`
4. **Physician involvement** — `AvgNumPhysiciansInvolved`, `NumUniqueAttendingPhysicians`, `NumUniqueOperatingPhysicians`
5. **Beneficiary demographics** — `AvgAgeOfBeneficiaries`, `PropDeceasedBeneficiaries`, `AvgChronicConditions`, `AvgClaimsPerBeneficiary`
6. **Duration** — `AvgClaimDurationDays`, `AvgLengthOfStayDays`
7. **Code concentration** (new) — `TopDiagnosisCodeShare`, `TopProcedureCodeShare` (+ `*Missing` flags)
8. **Physician network** (new) — `AvgPhysicianProviderSpread`
9. **Beneficiary concentration** (new) — `TopBeneficiaryClaimShare`
10. **Benford's Law** (new) — `BenfordMAD` (+ `BenfordMADMissing` flag)

Then: `evaluate_feature_significance()` → `compute_vif()` / `iterative_vif_selection()` → two feature sets saved to `data/processed/feature_sets.json`.

**Key decisions:**
- **Two aggregation units kept separate:** claim-level events aggregated directly; beneficiary demographics aggregated from a table deduplicated to one row per (Provider, BeneID) first.
- **No target leakage:** no feature computation touches `PotentialFraud`.
- **Multicollinearity fixed in two passes** (see `ai.md` point 4). Iterative VIF elimination now drops 4 features (`AvgClaimAmt`, `TopBeneficiaryClaimShare`, `PropInpatient`, `NumUniqueOperatingPhysicians`) instead of the previous 3 — the new `TopBeneficiaryClaimShare` correlates with the provider-size features once `AvgClaimAmt` is already gone.
- **Blocks 7-10 (the 5 new candidate features) were added after validation** in `03b`'s Extended Candidate Features section — each checked for effect size (Cohen's d) *and* redundancy against the existing 6 blocks before being added. `RoundNumberClaimRate` was tested and rejected (see `ai.md` point 7 — 0.99 correlation with `PropInpatient`, a false lead).
- **A real bug was caught and fixed** in `TopProcedureCodeShare`'s missing-value handling — see `ai.md` point 8. Do not 0-fill a concentration/share metric for "no data" cases; fill with the mean among valid rows and flag the fill instead.
- **Two feature sets maintained deliberately:** full 26-feature set for tree-based/boosting models, 22-feature VIF-clean subset for logistic regression.

**Verified:**
- Provider feature table: 5,410 rows × 28 columns (`Provider`, `PotentialFraud`, 26 features)
- No unexpected missing values after aggregation (all `*Missing` flags resolve any structural gap to a filled value)
- Final VIF for both feature sets: all values below 10
- Refactor regression-tested: 6 previously-recorded exact values (by fraud group) reproduced to 6 decimal places after the Step 4a single-source-of-truth refactor

**Feature significance (by effect size, not p-value alone):** `MaxClaimAmt` (d=1.61) is the single strongest feature. Several new candidates now rank above the previous top signal (`NumClaims`, d=0.67) — most notably `TopProcedureCodeShareMissing` (d=-1.47) and `TopProcedureCodeShare` (d=-0.97). **Caveat:** `TopProcedureCodeShareMissing` and `BenfordMADMissing` are both plausible proxies for claim volume/`PropInpatient` already captured elsewhere — worth checking their correlation with those features specifically before treating them as fully independent evidence in the report, even though VIF didn't eliminate either outright (VIF 5.1 and 2.1 respectively — real but not disqualifying collinearity).

---

## Step 5: Train/Test Split & Class Imbalance Handling
**Status:** ⬜ Not started

**Planned approach:**
- Split by **Provider**, not claim, to avoid the same provider's claims leaking across train and test sets
- Apply class imbalance handling (SMOTE / class weights — TBD) **after** splitting, and **only** to the training set, to avoid leaking synthetic/resampled signal into evaluation

## Step 6: Core Modeling
**Status:** ⬜ Not started

Candidates per the README: logistic regression (using `logistic_features`), random forest, gradient boosting (using `full_features`), plus a possible "novel method" for extra credit — Benford's Law deviation (`BenfordMAD`, already built) or a graph/network feature (shared physicians/beneficiaries across providers) would fit here, since the relational structure for it already exists in the raw data.

## Step 7: SHAP Evaluation & Report Assembly
**Status:** ⬜ Not started
