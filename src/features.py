"""
Provider-level feature engineering.

Input: the cleaned claim-level DataFrame from src.cleaning.clean_claim_level_dataset()
Output: one row per Provider, with engineered features + the PotentialFraud label.

Two aggregation units are kept deliberately separate:
- Claim-level events (volume, amounts, diagnosis/procedure counts) are
  aggregated directly from the claim-level table.
- Beneficiary demographics (age, chronic conditions, deceased status) are
  aggregated from a deduplicated (Provider, BeneID) table first, so a
  beneficiary with many visits doesn't get counted multiple times and skew
  the provider's demographic profile.

No feature here uses PotentialFraud as an input — only as the final label,
to avoid leaking target information into the feature set.
"""

import pandas as pd

CHRONIC_CONDITION_COLUMNS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]

DIAGNOSIS_CODE_COLUMNS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROCEDURE_CODE_COLUMNS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
PHYSICIAN_COLUMNS = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]


# ---------------------------------------------------------------------------
# Claim volume & amount features (claim-level aggregation)
# ---------------------------------------------------------------------------

def build_claim_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Claim counts and inpatient/outpatient mix, per provider.

    NOTE: NumInpatientClaims and NumOutpatientClaims are computed internally
    but deliberately NOT returned as final features — NumClaims is their
    exact sum, and PropInpatient is their exact ratio, so keeping all four
    creates a perfect linear identity (infinite VIF). NumClaims + PropInpatient
    together already losslessly encode the same information.
    """
    grouped = df.groupby("Provider")
    raw = grouped.agg(
        NumClaims=("ClaimID", "count"),
        NumInpatientClaims=("ClaimType", lambda x: (x == "Inpatient").sum()),
    ).reset_index()
    raw["PropInpatient"] = raw["NumInpatientClaims"] / raw["NumClaims"]

    features = raw.drop(columns=["NumInpatientClaims"])
    return features


def build_claim_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reimbursement and deductible amount statistics, per provider.

    Note: TotalClaimAmt/TotalDeductiblePaid (near-exact products of
    NumClaims x Avg*) and AvgDeductiblePaid (highly correlated with
    AvgClaimAmt, since bigger claims carry bigger deductibles) are
    deliberately excluded — DeductibleToReimbursementRatio already captures
    the deductible signal as a scale-independent ratio.
    """
    features = df.groupby("Provider").agg(
        AvgClaimAmt=("InscClaimAmtReimbursed", "mean"),
        MedianClaimAmt=("InscClaimAmtReimbursed", "median"),
        StdClaimAmt=("InscClaimAmtReimbursed", "std"),
        MaxClaimAmt=("InscClaimAmtReimbursed", "max"),
    ).reset_index()

    # Ratio of deductible paid to amount reimbursed — a proxy for how much of
    # the claim burden falls on the patient vs. the insurer for this provider.
    total_reimbursed = df.groupby("Provider")["InscClaimAmtReimbursed"].sum()
    total_deductible = df.groupby("Provider")["DeductibleAmtPaid"].sum()
    features["DeductibleToReimbursementRatio"] = (
        features["Provider"].map(total_deductible) / (features["Provider"].map(total_reimbursed) + 1)
    )

    features["StdClaimAmt"] = features["StdClaimAmt"].fillna(0)  # providers with 1 claim have no std
    return features


# ---------------------------------------------------------------------------
# Diagnosis / procedure code features (claim-level, but code columns are wide)
# ---------------------------------------------------------------------------

def build_diagnosis_procedure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts derived from the sparse wide diagnosis/procedure columns.
    Raw codes (thousands of distinct ICD-9 values) aren't usable directly in
    most models, so we convert them into average codes logged per claim
    (a complexity signal).

    Note: unique-code diversity ratios (unique codes / NumClaims) were
    tried and dropped — they turned out to be near-duplicates of
    AvgNumDiagnosisCodes/AvgNumProcedureCodes (both measure per-claim code
    intensity from slightly different angles), causing severe
    multicollinearity between the two without adding new information.
    """
    df = df.copy()
    df["NumDiagnosisCodesOnClaim"] = df[DIAGNOSIS_CODE_COLUMNS].notna().sum(axis=1)
    df["NumProcedureCodesOnClaim"] = df[PROCEDURE_CODE_COLUMNS].notna().sum(axis=1)

    features = df.groupby("Provider").agg(
        AvgNumDiagnosisCodes=("NumDiagnosisCodesOnClaim", "mean"),
        AvgNumProcedureCodes=("NumProcedureCodesOnClaim", "mean"),
    ).reset_index()

    return features


# ---------------------------------------------------------------------------
# Physician features (claim-level)
# ---------------------------------------------------------------------------

def build_physician_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physician involvement and variety, per provider."""
    features = df.groupby("Provider").agg(
        AvgNumPhysiciansInvolved=("NumPhysiciansInvolved", "mean"),
        NumUniqueAttendingPhysicians=("AttendingPhysician", "nunique"),
        NumUniqueOperatingPhysicians=("OperatingPhysician", "nunique"),
    ).reset_index()
    return features


# ---------------------------------------------------------------------------
# Beneficiary demographic features (deduplicated to Provider-Beneficiary level
# to avoid overweighting frequently-visiting patients)
# ---------------------------------------------------------------------------

def build_beneficiary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demographic and chronic-condition features, per provider.

    Deduplicates to one row per (Provider, BeneID) BEFORE aggregating, so a
    beneficiary with 15 claims doesn't get their age/conditions counted 15
    times in the provider's average — that would conflate "this provider
    treats older/sicker patients" with "this provider has high claim volume,"
    which are different signals we already capture separately.
    """
    unique_bene = df.drop_duplicates(subset=["Provider", "BeneID"]).copy()

    unique_bene["NumChronicConditions"] = unique_bene[CHRONIC_CONDITION_COLUMNS].sum(axis=1)

    features = unique_bene.groupby("Provider").agg(
        AvgAgeOfBeneficiaries=("AgeAtClaim", "mean"),
        PropDeceasedBeneficiaries=("IsDeceased", "mean"),
        AvgChronicConditions=("NumChronicConditions", "mean"),
    ).reset_index()

    # Claims-per-beneficiary: total claims (from full df) / unique beneficiaries.
    # This ratio already captures the beneficiary-count relationship, so a
    # separate NumUniqueBeneficiaries feature (highly correlated with
    # NumClaims) is deliberately excluded to avoid multicollinearity.
    claims_per_bene = df.groupby("Provider")["ClaimID"].count() / df.groupby("Provider")["BeneID"].nunique()
    features["AvgClaimsPerBeneficiary"] = features["Provider"].map(claims_per_bene)

    return features


# ---------------------------------------------------------------------------
# Claim/admission duration features
# ---------------------------------------------------------------------------

def build_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average claim duration (ClaimEndDt - ClaimStartDt) and, for inpatient
    claims specifically, average length of hospital stay (DischargeDt - AdmissionDt).
    Unusually long/short durations relative to peers can be a fraud signal
    (e.g. billing for stays that didn't happen, or padding claim duration).
    """
    df = df.copy()
    df["ClaimDurationDays"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days

    inpatient = df[df["ClaimType"] == "Inpatient"].copy()
    inpatient["LengthOfStayDays"] = (inpatient["DischargeDt"] - inpatient["AdmissionDt"]).dt.days

    duration_features = df.groupby("Provider").agg(
        AvgClaimDurationDays=("ClaimDurationDays", "mean"),
    ).reset_index()

    los_features = inpatient.groupby("Provider").agg(
        AvgLengthOfStayDays=("LengthOfStayDays", "mean"),
    ).reset_index()

    features = duration_features.merge(los_features, on="Provider", how="left")
    # Providers with no inpatient claims have no length of stay — fill with 0,
    # since "no inpatient stays" is meaningfully different from "unknown."
    features["AvgLengthOfStayDays"] = features["AvgLengthOfStayDays"].fillna(0)
    return features


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_provider_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature blocks and merge into a single provider-level feature
    table, with the PotentialFraud label attached last (not used as an
    input to any feature above).
    """
    claim_volume = build_claim_volume_features(df)
    claim_amount = build_claim_amount_features(df)
    diag_proc = build_diagnosis_procedure_features(df)
    physician = build_physician_features(df)
    beneficiary = build_beneficiary_features(df)
    duration = build_duration_features(df)

    label = df.groupby("Provider")["PotentialFraud"].first().reset_index()

    feature_blocks = [claim_volume, claim_amount, diag_proc, physician, beneficiary, duration]
    features = feature_blocks[0]
    for block in feature_blocks[1:]:
        features = features.merge(block, on="Provider", how="left")

    features = features.merge(label, on="Provider", how="left")
    return features


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data_loader import build_claim_level_dataset
    from cleaning import clean_claim_level_dataset

    raw = build_claim_level_dataset()
    df = clean_claim_level_dataset(raw)
    provider_features = build_provider_features(df)

    print(f"Provider feature table shape: {provider_features.shape}")
    print(f"Columns: {list(provider_features.columns)}")
    print(f"\nMissing values per column:\n{provider_features.isna().sum()[provider_features.isna().sum() > 0]}")
    print(f"\nSample:\n{provider_features.head()}")