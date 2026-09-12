"""
Cleaning utilities for the healthcare provider fraud claim-level dataset.

Input: the claim-level DataFrame produced by src.data_loader.build_claim_level_dataset()
Output: the same shape, with corrected dtypes, recoded binary columns, and
derived helper columns (age, is_deceased). No rows are dropped here —
row-level filtering (if any) belongs in feature engineering, where you can
make that decision with full context of what you're aggregating.
"""

import pandas as pd

DATE_COLUMNS = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOB", "DOD"]

CHRONIC_CONDITION_COLUMNS = [
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known date columns from string to datetime. Missing dates stay as NaT."""
    df = df.copy()
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def recode_chronic_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Recode ChronicCond_* columns from {1=Yes, 2=No} to standard {1=Yes, 0=No}."""
    df = df.copy()
    for col in CHRONIC_CONDITION_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({1: 1, 2: 0})
    return df


def recode_renal_disease_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Recode RenalDiseaseIndicator from {'Y', '0'} strings to {1, 0} integers."""
    df = df.copy()
    if "RenalDiseaseIndicator" in df.columns:
        df["RenalDiseaseIndicator"] = df["RenalDiseaseIndicator"].map({"Y": 1, "0": 0}).fillna(0).astype(int)
    return df


def recode_fraud_label(df: pd.DataFrame) -> pd.DataFrame:
    """Recode PotentialFraud from {'Yes','No'} strings to {1, 0} integers, if present."""
    df = df.copy()
    if "PotentialFraud" in df.columns:
        df["PotentialFraud"] = df["PotentialFraud"].map({"Yes": 1, "No": 0})
    return df


def add_age_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive age-related columns from DOB/DOD.
    - IsDeceased: 1 if DOD is present, else 0
    - AgeAtClaim: age in years at the time of ClaimStartDt (uses DOD-based cutoff
      implicitly since dates after death shouldn't exist, but no explicit clipping here)
    """
    df = df.copy()
    df["IsDeceased"] = df["DOD"].notna().astype(int)

    if "ClaimStartDt" in df.columns:
        age_days = (df["ClaimStartDt"] - df["DOB"]).dt.days
        df["AgeAtClaim"] = (age_days / 365.25).astype("float")
    return df


def cast_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns that are numerically coded but semantically categorical
    (geographic FIPS-style codes, demographic codes) to 'category' dtype,
    so downstream stats/plots don't treat them as ordered quantities.
    """
    df = df.copy()
    categorical_code_cols = ["State", "County", "Gender", "Race"]
    for col in categorical_code_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def add_physician_involvement_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive NumPhysiciansInvolved: count of distinct physician role columns
    populated for a claim (Attending/Operating/Other). Missing here is
    structural (not every claim needs 3 physicians), so this converts that
    missingness into a usable numeric signal rather than leaving it as NaN.
    """
    df = df.copy()
    physician_cols = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
    present_cols = [c for c in physician_cols if c in df.columns]
    df["NumPhysiciansInvolved"] = df[present_cols].notna().sum(axis=1)
    return df


def fill_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values that are structurally meaningful rather than 'unknown':
    - Deductible/reimbursement amount columns: NaN -> 0 (no amount charged/paid)
    Admission/Discharge dates and DOD are intentionally left as NaT (missing) —
    they carry meaning (outpatient claim / beneficiary alive) that a fill value
    would destroy. Handle their meaning via IsDeceased and ClaimType instead.
    """
    df = df.copy()
    amount_cols = [c for c in df.columns if "Deductible" in c or "Reimbursement" in c]
    for col in amount_cols:
        df[col] = df[col].fillna(0)
    return df


def clean_claim_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in the correct order."""
    df = parse_dates(df)
    df = recode_chronic_conditions(df)
    df = recode_renal_disease_indicator(df)
    df = recode_fraud_label(df)
    df = add_age_columns(df)
    df = cast_categorical_codes(df)
    df = add_physician_involvement_count(df)
    df = fill_structural_missing(df)
    return df


if __name__ == "__main__":
    from data_loader import build_claim_level_dataset

    raw = build_claim_level_dataset()
    cleaned = clean_claim_level_dataset(raw)

    print(f"Shape: {cleaned.shape}")
    print(f"\nDtypes sample:\n{cleaned[['ClaimStartDt', 'DOB', 'AgeAtClaim', 'IsDeceased']].dtypes}")
    print(f"\nMissingness (top 10 columns):\n{cleaned.isna().sum().sort_values(ascending=False).head(10)}")
    print(f"\nAge range: {cleaned['AgeAtClaim'].min():.1f} to {cleaned['AgeAtClaim'].max():.1f}")
    print(f"\nPotentialFraud values: {cleaned['PotentialFraud'].unique()}")