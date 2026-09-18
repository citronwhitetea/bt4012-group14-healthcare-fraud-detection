"""
Claim-level and provider-level feature engineering.

Input: the cleaned claim-level DataFrame from src.cleaning.clean_claim_level_dataset()
Output: either the same claim-level shape with engineered columns added
(build_claim_level_features), or one row per Provider (build_provider_features).

This module is the single source of truth for claim-level derived columns —
both build_provider_features() (which aggregates them up) and
notebooks/04a_basic_claim_level_feature_engineering.ipynb (which explores
them at claim grain) call build_claim_level_features() rather than each
computing their own version.

Two aggregation units are kept deliberately separate at the provider level:
- Claim-level events (volume, amounts, diagnosis/procedure counts) are
  aggregated directly from the claim-level table.
- Beneficiary demographics (age, chronic conditions, deceased status) are
  aggregated from a table deduplicated to one row per (Provider, BeneID)
  first, so a beneficiary with many visits doesn't get counted multiple
  times and skew the provider's demographic profile.

No feature here uses PotentialFraud as an input — only as the final label,
to avoid leaking target information into the feature set.
"""

import numpy as np
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
# Claim-level feature columns — single source of truth. Each add_* function
# takes the claim-level df and returns it with a few more columns; the
# orchestrator (build_claim_level_features) runs them in dependency order.
# Every function copies its input (matching src/cleaning.py's convention),
# so each is safe to call standalone, not just through the orchestrator.
# ---------------------------------------------------------------------------

def add_claim_type_flags(df: pd.DataFrame) -> pd.DataFrame:
    """IsInpatient: 1 if the claim is Inpatient, else 0 — numeric version of ClaimType."""
    df = df.copy()
    df["IsInpatient"] = (df["ClaimType"] == "Inpatient").astype(int)
    return df


def add_financial_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    LogReimbursement: log(1 + reimbursed amount) — tames the heavy right skew.
    HasDeductible: 1 if any deductible was paid, else 0 — collapses the
    bimodal (mostly-0-or-~1068) deductible pattern into a clean binary flag.
    """
    df = df.copy()
    df["LogReimbursement"] = np.log1p(df["InscClaimAmtReimbursed"])
    df["HasDeductible"] = (df["DeductibleAmtPaid"] > 0).astype(int)
    return df


def add_claim_duration_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ClaimDuration: days from claim start to end (all claims).
    SameDayClaim: 1 if ClaimDuration is 0.
    LengthOfStay: days admitted to discharged — Inpatient-only, NaN for
    Outpatient (structural, not missing data).
    """
    df = df.copy()
    df["ClaimDuration"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days
    df["SameDayClaim"] = (df["ClaimDuration"] == 0).astype(int)
    df["LengthOfStay"] = (df["DischargeDt"] - df["AdmissionDt"]).dt.days
    return df


def add_diagnosis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    NumDiagnosisCodes: count of non-null diagnosis code slots used (0-10).
    HasDiagnosis: 1 if any diagnosis code is present.
    PrimaryDiagnosis: friendly alias for ClmDiagnosisCode_1.
    """
    df = df.copy()
    df["NumDiagnosisCodes"] = df[DIAGNOSIS_CODE_COLUMNS].notna().sum(axis=1)
    df["HasDiagnosis"] = (df["NumDiagnosisCodes"] > 0).astype(int)
    df["PrimaryDiagnosis"] = df["ClmDiagnosisCode_1"]
    return df


def add_procedure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    NumProcedureCodes: count of non-null procedure code slots used (0-6).
    HasProcedure: 1 if any procedure code is present — only ~4% of claims
    overall, but the majority of Inpatient claims; carries most of the
    information NumProcedureCodes does, more robustly.
    """
    df = df.copy()
    df["NumProcedureCodes"] = df[PROCEDURE_CODE_COLUMNS].notna().sum(axis=1)
    df["HasProcedure"] = (df["NumProcedureCodes"] > 0).astype(int)
    return df


def add_complexity_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    ClaimComplexity: NumDiagnosisCodes + NumProcedureCodes — a single, simple
    measure of coding activity. Requires add_diagnosis_columns/
    add_procedure_columns to have already run.
    """
    df = df.copy()
    df["ClaimComplexity"] = df["NumDiagnosisCodes"] + df["NumProcedureCodes"]
    return df


def add_physician_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    HasOperatingPhysician / HasOtherPhysician: binary presence flags for the
    two structurally-sparse physician roles.
    SameAttendingOperating: 1 if the same physician is listed as both
    attending and operating — a possible self-referral signal.
    (NumPhysiciansInvolved is already built in src/cleaning.py.)
    """
    df = df.copy()
    df["HasOperatingPhysician"] = df["OperatingPhysician"].notna().astype(int)
    df["HasOtherPhysician"] = df["OtherPhysician"].notna().astype(int)
    df["SameAttendingOperating"] = (
        (df["AttendingPhysician"] == df["OperatingPhysician"]) & df["OperatingPhysician"].notna()
    ).astype(int)
    return df


def add_beneficiary_health_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    NumChronicConditions: sum of the 11 ChronicCond_* flags (0-11) —
    overall comorbidity burden.
    HasRenalDisease: alias of the already-cleaned RenalDiseaseIndicator,
    named to match the other Has* flags here.
    (AgeAtClaim is already built in src/cleaning.py.)
    """
    df = df.copy()
    df["NumChronicConditions"] = df[CHRONIC_CONDITION_COLUMNS].sum(axis=1)
    df["HasRenalDisease"] = df["RenalDiseaseIndicator"]
    return df


def add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ClaimMonth: calendar month (1-12) of the claim start date.
    ClaimDayOfWeek: numeric day of week, 0=Monday to 6=Sunday.
    IsWeekend: 1 if Saturday/Sunday.
    """
    df = df.copy()
    df["ClaimMonth"] = df["ClaimStartDt"].dt.month
    df["ClaimDayOfWeek"] = df["ClaimStartDt"].dt.dayofweek
    df["IsWeekend"] = df["ClaimDayOfWeek"].isin([5, 6]).astype(int)
    return df


def add_repeat_claiming_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    DaysSincePreviousClaim: gap in days since this beneficiary's prior claim
    — NaN for a beneficiary's first claim (structural, not missing data).
    ClaimWithin7Days / ClaimWithin30Days: 1 if that gap is <=7 / <=30 days
    (NaN treated as 0 — no prior claim means no rapid repeat).

    Sorts by (BeneID, ClaimStartDt) to compute the gap, which reorders rows
    — harmless for every downstream use (groupby-based aggregation, or a
    final re-sort before saving), since no feature here depends on row order.
    """
    df = df.copy()
    df = df.sort_values(["BeneID", "ClaimStartDt"])
    df["DaysSincePreviousClaim"] = df.groupby("BeneID")["ClaimStartDt"].diff().dt.days
    df["ClaimWithin7Days"] = (df["DaysSincePreviousClaim"] <= 7).fillna(False).astype(int)
    df["ClaimWithin30Days"] = (df["DaysSincePreviousClaim"] <= 30).fillna(False).astype(int)
    return df


def add_duplication_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    SameBeneProviderDay: 1 if this claim shares BeneID + Provider +
    ClaimStartDt with at least one other claim.
    NumClaimsSameBeneProviderDay: exact count of claims sharing that same
    combination — magnitude behind the flag above.
    """
    df = df.copy()
    df["SameBeneProviderDay"] = df.duplicated(
        subset=["BeneID", "Provider", "ClaimStartDt"], keep=False
    ).astype(int)
    df["NumClaimsSameBeneProviderDay"] = df.groupby(
        ["BeneID", "Provider", "ClaimStartDt"]
    )["ClaimID"].transform("count")
    return df


def build_claim_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all claim-level feature blocks in dependency order (complexity needs
    diagnosis/procedure counts first; repeat-claiming needs a stable sort).
    Returns the claim-level df with all engineered columns added — the
    single source of truth consumed by both build_provider_features() below
    and notebooks/04a_basic_claim_level_feature_engineering.ipynb.
    """
    df = add_claim_type_flags(df)
    df = add_financial_flags(df)
    df = add_claim_duration_columns(df)
    df = add_diagnosis_columns(df)
    df = add_procedure_columns(df)
    df = add_complexity_column(df)
    df = add_physician_flags(df)
    df = add_beneficiary_health_columns(df)
    df = add_temporal_columns(df)
    df = add_repeat_claiming_columns(df)
    df = add_duplication_columns(df)
    return df


# ---------------------------------------------------------------------------
# Claim volume & amount features (provider-level aggregation)
# ---------------------------------------------------------------------------

def build_claim_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Claim counts and inpatient/outpatient mix, per provider. Requires
    IsInpatient (build_claim_level_features) to already be present.

    NOTE: NumInpatientClaims and NumOutpatientClaims are computed internally
    but deliberately NOT returned as final features — NumClaims is their
    exact sum, and PropInpatient is their exact ratio, so keeping all four
    creates a perfect linear identity (infinite VIF). NumClaims + PropInpatient
    together already losslessly encode the same information.
    """
    grouped = df.groupby("Provider")
    raw = grouped.agg(
        NumClaims=("ClaimID", "count"),
        NumInpatientClaims=("IsInpatient", "sum"),
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
# Diagnosis / procedure code features (provider-level aggregation)
# ---------------------------------------------------------------------------

def build_diagnosis_procedure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average diagnosis/procedure code counts per claim, per provider.
    Requires NumDiagnosisCodes/NumProcedureCodes (build_claim_level_features)
    to already be present. Raw codes (thousands of distinct ICD-9 values)
    aren't usable directly in most models, so this is a complexity signal
    instead.

    Note: unique-code diversity ratios (unique codes / NumClaims) were
    tried and dropped — they turned out to be near-duplicates of
    AvgNumDiagnosisCodes/AvgNumProcedureCodes (both measure per-claim code
    intensity from slightly different angles), causing severe
    multicollinearity between the two without adding new information.
    """
    features = df.groupby("Provider").agg(
        AvgNumDiagnosisCodes=("NumDiagnosisCodes", "mean"),
        AvgNumProcedureCodes=("NumProcedureCodes", "mean"),
    ).reset_index()
    return features


# ---------------------------------------------------------------------------
# Physician features (provider-level aggregation)
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
    Demographic and chronic-condition features, per provider. Requires
    NumChronicConditions (build_claim_level_features) to already be present.

    Deduplicates to one row per (Provider, BeneID) BEFORE aggregating, so a
    beneficiary with 15 claims doesn't get their age/conditions counted 15
    times in the provider's average — that would conflate "this provider
    treats older/sicker patients" with "this provider has high claim volume,"
    which are different signals we already capture separately.
    """
    unique_bene = df.drop_duplicates(subset=["Provider", "BeneID"])

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
# Claim/admission duration features (provider-level aggregation)
# ---------------------------------------------------------------------------

def build_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average claim duration and, for inpatient claims specifically, average
    length of hospital stay, per provider. Requires ClaimDuration/
    LengthOfStay/IsInpatient (build_claim_level_features) to already be
    present. Unusually long/short durations relative to peers can be a
    fraud signal (e.g. billing for stays that didn't happen, or padding
    claim duration).
    """
    duration_features = df.groupby("Provider").agg(
        AvgClaimDurationDays=("ClaimDuration", "mean"),
    ).reset_index()

    inpatient = df[df["IsInpatient"] == 1]
    los_features = inpatient.groupby("Provider").agg(
        AvgLengthOfStayDays=("LengthOfStay", "mean"),
    ).reset_index()

    features = duration_features.merge(los_features, on="Provider", how="left")
    # Providers with no inpatient claims have no length of stay — fill with 0,
    # since "no inpatient stays" is meaningfully different from "unknown."
    features["AvgLengthOfStayDays"] = features["AvgLengthOfStayDays"].fillna(0)
    return features


# ---------------------------------------------------------------------------
# Candidate fraud-signal features (billing pattern, physician/beneficiary
# network) — surfaced in the "Extended Candidate Features" section of
# notebooks/03b_provider_level_eda.ipynb, where each one's effect size and
# redundancy against the blocks above was checked before being added here.
# ---------------------------------------------------------------------------

def build_code_concentration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Share of a provider's claims that use its single most common diagnosis /
    procedure code — a "billing templating" signal. Computed only among
    claims that have a code at all; a provider with zero coded claims of a
    given type gets NaN from value_counts.

    NaN is filled with the MEAN among providers who do have coded claims,
    not 0 — 0 would mean "many different codes, none dominant," which is
    the opposite of "no codes recorded at all," and would silently distort
    the signal (confirmed: a 0-fill flipped TopProcedureCodeShare's effect
    size from Cohen's d = -0.89 to +0.11, since 62.6% of providers have zero
    procedure-coded claims). The *Missing flags record which providers were
    filled, mirroring the BenfordMAD pattern below.

    Counter-intuitive EDA finding (computed by excluding, not 0-filling,
    the no-code providers — see notebooks/03b_provider_level_eda.ipynb):
    fraud providers have LOWER concentration than non-fraud (more diverse
    coding, not more repetitive/templated) — TopProcedureCodeShare has the
    largest effect size of any feature tested so far (Cohen's d = -0.89;
    fraud mean 0.19 vs non-fraud 0.44, among providers with at least one
    procedure-coded claim).
    """
    def top_code_share(s: pd.Series) -> float:
        s = s.dropna()
        return s.value_counts(normalize=True).iloc[0] if len(s) else np.nan

    features = df.groupby("Provider").agg(
        TopDiagnosisCodeShare=("ClmDiagnosisCode_1", top_code_share),
        TopProcedureCodeShare=("ClmProcedureCode_1", top_code_share),
    ).reset_index()

    features["TopDiagnosisCodeShareMissing"] = features["TopDiagnosisCodeShare"].isna().astype(int)
    features["TopProcedureCodeShareMissing"] = features["TopProcedureCodeShare"].isna().astype(int)

    features["TopDiagnosisCodeShare"] = features["TopDiagnosisCodeShare"].fillna(
        features["TopDiagnosisCodeShare"].mean()
    )
    features["TopProcedureCodeShare"] = features["TopProcedureCodeShare"].fillna(
        features["TopProcedureCodeShare"].mean()
    )
    return features


def build_physician_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    AvgPhysicianProviderSpread: for each physician a provider uses (in an
    attending or operating role), how many DIFFERENT providers does that
    physician also bill under, averaged across the provider's physicians.
    Requires looking across the whole claims table (not just one provider's
    rows), since a physician's provider count depends on their claims
    everywhere, not just at this provider.

    Counter-intuitive EDA finding: fraud providers' physicians are actually
    LESS shared with other providers (1.14 vs 1.38, Cohen's d = -0.45) —
    the opposite of a "shared ring physician" hypothesis.
    """
    phys_long = pd.concat([
        df[["Provider", "AttendingPhysician"]].rename(columns={"AttendingPhysician": "Physician"}),
        df[["Provider", "OperatingPhysician"]].rename(columns={"OperatingPhysician": "Physician"}),
    ]).dropna()

    physician_provider_counts = phys_long.drop_duplicates().groupby("Physician")["Provider"].nunique()
    phys_long["NumProvidersForPhysician"] = phys_long["Physician"].map(physician_provider_counts)

    features = phys_long.groupby("Provider").agg(
        AvgPhysicianProviderSpread=("NumProvidersForPhysician", "mean"),
    ).reset_index()
    return features


def build_beneficiary_concentration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TopBeneficiaryClaimShare: share of a provider's claims coming from its
    single most-billed beneficiary — does one "super-patient" account for
    an outsized share of this provider's business?

    Counter-intuitive EDA finding: fraud providers have LOWER concentration
    (0.05 vs 0.17, Cohen's d = -0.75) — they spread claims across a broader
    patient base rather than depending on a handful of repeat patients.
    """
    claims_per_bene = df.groupby(["Provider", "BeneID"]).size().reset_index(name="NumClaimsFromBene")
    features = (
        claims_per_bene.groupby("Provider")["NumClaimsFromBene"]
        .apply(lambda x: x.max() / x.sum())
        .reset_index(name="TopBeneficiaryClaimShare")
    )
    return features


def _benford_mad(amounts: pd.Series) -> float:
    """
    Mean absolute deviation of observed leading-digit proportions from
    Benford's Law — mirrors benford_mean_absolute_deviation() in src/eda.py
    (kept local here rather than imported, since no src module in this
    project imports another at module level; every cross-module usage goes
    through a notebook's `from src.x import y` instead).
    """
    positive_amounts = amounts[amounts > 0].dropna()
    leading_digits = positive_amounts.astype(str).str.lstrip("0.").str[0].astype(int)
    leading_digits = leading_digits[leading_digits.between(1, 9)]
    counts = leading_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    observed_props = counts / counts.sum()

    benford_expected_probs = pd.Series(
        [np.log10(1 + 1 / d) for d in range(1, 10)], index=range(1, 10)
    )
    return (observed_props - benford_expected_probs).abs().mean()


def build_benford_features(df: pd.DataFrame, min_claims: int = 20) -> pd.DataFrame:
    """
    Per-provider Benford's Law deviation (BenfordMAD) on
    InscClaimAmtReimbursed. Only computed for providers with >= min_claims
    claims — Benford's Law needs a reasonable sample to be meaningful, and
    38.6% of providers fall below that. Those providers get
    BenfordMADMissing=1 and BenfordMAD filled with the dataset-wide mean
    (not 0 — a 0-fill would misleadingly imply perfect Benford conformity
    for a provider we simply couldn't test), so the flag carries the real
    signal for "insufficient data" rather than the value itself.

    Nuance vs. the aggregate-level Benford check in Step 3
    (notebooks/03a_claim_level_eda.ipynb): pooling Fraud vs Non-Fraud
    together made the two groups look nearly identical (MAD 0.0186 vs
    0.0181). Per provider, a modest but real difference emerges (0.030 vs
    0.037, Cohen's d = -0.54) — aggregating had masked it.
    """
    claims_per_provider = df.groupby("Provider").size()
    eligible_providers = claims_per_provider[claims_per_provider >= min_claims].index

    benford_mad_by_provider = (
        df[df["Provider"].isin(eligible_providers)]
        .groupby("Provider")["InscClaimAmtReimbursed"]
        .apply(_benford_mad)
    )

    features = df[["Provider"]].drop_duplicates().reset_index(drop=True)
    features["BenfordMAD"] = features["Provider"].map(benford_mad_by_provider)
    features["BenfordMADMissing"] = features["BenfordMAD"].isna().astype(int)
    features["BenfordMAD"] = features["BenfordMAD"].fillna(features["BenfordMAD"].mean())
    return features


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_provider_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the claim-level feature pipeline, then all provider-level
    aggregation blocks, merging into a single provider-level feature table
    with the PotentialFraud label attached last (not used as an input to
    any feature above).
    """
    df = build_claim_level_features(df)

    claim_volume = build_claim_volume_features(df)
    claim_amount = build_claim_amount_features(df)
    diag_proc = build_diagnosis_procedure_features(df)
    physician = build_physician_features(df)
    beneficiary = build_beneficiary_features(df)
    duration = build_duration_features(df)
    code_concentration = build_code_concentration_features(df)
    physician_network = build_physician_network_features(df)
    beneficiary_concentration = build_beneficiary_concentration_features(df)
    benford = build_benford_features(df)

    label = df.groupby("Provider")["PotentialFraud"].first().reset_index()

    feature_blocks = [
        claim_volume, claim_amount, diag_proc, physician, beneficiary, duration,
        code_concentration, physician_network, beneficiary_concentration, benford,
    ]
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
