# Data Dictionary

Column-level reference for all four raw CSVs. Use this alongside `notebooks/00_setup_guide.ipynb` when writing cleaning/feature engineering code, so everyone has the same understanding of what each column means.

---

## `provider_label.csv` — Provider-level labels
**Grain:** one row per provider (5,410 rows)

| Column | Type | Description |
|---|---|---|
| `Provider` | string | Unique provider ID. Join key into Inpatient/Outpatient claims. |
| `PotentialFraud` | string (`Yes`/`No`) | Target label — whether this provider was flagged as potentially fraudulent. Recoded to `1`/`0` during cleaning. |

---

## `beneficiary_raw.csv` — Patient demographics & coverage
**Grain:** one row per beneficiary (138,556 rows)

| Column | Type | Description |
|---|---|---|
| `BeneID` | string | Unique beneficiary (patient) ID. Join key into Inpatient/Outpatient claims. |
| `DOB` | date | Date of birth. |
| `DOD` | date (mostly missing) | Date of death. **Missing means the beneficiary is alive** — not a data quality issue. |
| `Gender` | int code | Beneficiary gender code. |
| `Race` | int code | Beneficiary race code. |
| `RenalDiseaseIndicator` | string (`Y`/`0`) | Whether beneficiary has end-stage renal disease. Recoded to `1`/`0` during cleaning. |
| `State` | int code | State FIPS-style code. Categorical, not a quantity. |
| `County` | int code | County FIPS-style code. Categorical, not a quantity. |
| `NoOfMonths_PartACov` | int (0-12) | Months of Medicare Part A coverage in the year. |
| `NoOfMonths_PartBCov` | int (0-12) | Months of Medicare Part B coverage in the year. |
| `ChronicCond_Alzheimer` | int (`1`=Yes,`2`=No) | Chronic condition flag. Recoded to `1`/`0` during cleaning. |
| `ChronicCond_Heartfailure` | int (`1`/`2`) | Same encoding/recoding as above. |
| `ChronicCond_KidneyDisease` | int (`1`/`2`) | " |
| `ChronicCond_Cancer` | int (`1`/`2`) | " |
| `ChronicCond_ObstrPulmonary` | int (`1`/`2`) | " |
| `ChronicCond_Depression` | int (`1`/`2`) | " |
| `ChronicCond_Diabetes` | int (`1`/`2`) | " |
| `ChronicCond_IschemicHeart` | int (`1`/`2`) | " |
| `ChronicCond_Osteoporasis` | int (`1`/`2`) | " |
| `ChronicCond_rheumatoidarthritis` | int (`1`/`2`) | " |
| `ChronicCond_stroke` | int (`1`/`2`) | " |
| `IPAnnualReimbursementAmt` | float | Beneficiary's total annual inpatient reimbursement. |
| `IPAnnualDeductibleAmt` | float | Beneficiary's total annual inpatient deductible paid. |
| `OPAnnualReimbursementAmt` | float | Beneficiary's total annual outpatient reimbursement. |
| `OPAnnualDeductibleAmt` | float | Beneficiary's total annual outpatient deductible paid. |

---

## `inpatients_raw.csv` — Inpatient claims
**Grain:** one row per inpatient claim

| Column | Type | Description |
|---|---|---|
| `BeneID` | string | Join key to beneficiary. |
| `ClaimID` | string | Unique claim ID. |
| `ClaimStartDt`, `ClaimEndDt` | date | Claim period. |
| `Provider` | string | Join key to provider label. |
| `InscClaimAmtReimbursed` | float | Amount reimbursed for this claim. Likely the strongest single numeric fraud signal. |
| `AttendingPhysician` | string (ID, high cardinality) | Physician who attended the patient. |
| `OperatingPhysician` | string (ID, often missing) | Physician who performed the operation, if any. Missing = no operation, not unknown. |
| `OtherPhysician` | string (ID, often missing) | Additional physician involved, if any. |
| `AdmissionDt` | date | Hospital admission date. **Inpatient-only column.** |
| `ClmAdmitDiagnosisCode` | string (ICD-9 code) | Diagnosis code at admission. |
| `DeductibleAmtPaid` | float | Deductible portion paid on this claim. |
| `DischargeDt` | date | Hospital discharge date. **Inpatient-only column.** |
| `DiagnosisGroupCode` | string | DRG (Diagnosis-Related Group) code. **Inpatient-only column.** |
| `ClmDiagnosisCode_1` ... `_10` | string (ICD-9 codes) | Up to 10 diagnosis codes for this claim. Higher-numbered slots are sparsely filled — most claims only use 1-3. |
| `ClmProcedureCode_1` ... `_6` | string (procedure codes) | Up to 6 procedure codes for this claim. Sparsely filled for the same reason. |

---

## `outpatients_raw.csv` — Outpatient claims
**Grain:** one row per outpatient claim

Same schema as Inpatient, **except** it does not have `AdmissionDt`, `DischargeDt`, or `DiagnosisGroupCode` (these are inherently inpatient concepts). All other columns (`BeneID`, `ClaimID`, dates, `InscClaimAmtReimbursed`, physician columns, diagnosis/procedure codes, `DeductibleAmtPaid`) mean the same thing as in the Inpatient table.

---

## Derived columns (added during cleaning, not in raw data)

| Column | Source step | Description |
|---|---|---|
| `ClaimType` | `combine_claims()` | `"Inpatient"` or `"Outpatient"` — added when stacking the two claims tables together. |
| `IsDeceased` | `add_age_columns()` | `1` if `DOD` is present, else `0`. |
| `AgeAtClaim` | `add_age_columns()` | Beneficiary's age in years at `ClaimStartDt`, derived from `DOB`. |
| `NumPhysiciansInvolved` | `add_physician_involvement_count()` | Count of non-missing physician columns (0-3) for a claim. |