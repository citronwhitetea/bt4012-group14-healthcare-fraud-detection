"""Plain-language names, units and missing-value rules for the model's provider-level features.

Each entry is (label, description, unit). The unit decides how values are displayed and which
values the upload check treats as impossible:

    count       whole number of things, 0 or more            (e.g. 12)
    average     average of a count, 0 or more                 (e.g. 2.4)
    amount      money in US dollars, 0 or more                (e.g. $1,200)
    proportion  share between 0 and 1, shown as a percentage  (e.g. 37%)
    flag        1 = yes, 0 = no; blank not allowed
    days        number of days, 0 or more
    years       age in years, 0 to 120
    ratio       ratio of two amounts, 0 or more
    index       a deviation score, 0 or more

Descriptions follow the feature engineering in notebook 05. Edit freely: this file only affects
display and the upload check's impossible-value rules.
"""
from __future__ import annotations

import re

FEATURES: dict[str, tuple[str, str, str]] = {
    "NumClaims": ("Number of claims", "Total inpatient and outpatient claims billed by the provider.", "count"),
    "AvgClaimAmt": ("Average claim amount", "Mean reimbursed amount per claim.", "amount"),
    "MedianClaimAmt": ("Median claim amount", "Middle reimbursed amount, less affected by a few very large claims.", "amount"),
    "MaxClaimAmt": ("Largest claim amount", "Highest single reimbursed amount.", "amount"),
    "StdClaimAmt": ("Spread of claim amounts", "Standard deviation of reimbursed amounts across claims.", "amount"),
    "DeductibleToReimbursementRatio": ("Deductible-to-reimbursement ratio",
                                       "Deductible paid relative to the amount reimbursed.", "ratio"),
    "PropInpatient": ("Share of inpatient claims", "Proportion of the provider's claims that are inpatient.", "proportion"),
    "AvgNumDiagnosisCodes": ("Diagnosis codes per claim", "Average number of diagnosis codes recorded on a claim.", "average"),
    "AvgNumProcedureCodes": ("Procedure codes per claim", "Average number of procedure codes recorded on a claim.", "average"),
    "TopDiagnosisCodeShare": ("Most common primary diagnosis share",
                              "Among claims with a primary diagnosis code, the share carrying the provider's most "
                              "frequent primary diagnosis code.", "proportion"),
    "TopProcedureCodeShare": ("Most common primary procedure share",
                              "Among claims with a primary procedure code, the share carrying the provider's most "
                              "frequent primary procedure code.", "proportion"),
    "TopDiagnosisCodeShareMissing": ("No primary diagnosis codes",
                                     "Yes when none of the provider's claims has a primary diagnosis code, so the "
                                     "diagnosis share cannot be computed.", "flag"),
    "TopProcedureCodeShareMissing": ("No primary procedure codes",
                                     "Yes when none of the provider's claims has a primary procedure code, so the "
                                     "procedure share cannot be computed. Common for outpatient-only providers.", "flag"),
    "AvgNumPhysiciansInvolved": ("Physician fields filled per claim",
                                 "Average number of physician fields (attending, operating, other) filled in per "
                                 "claim. The same physician in two roles counts twice.", "average"),
    "NumUniqueAttendingPhysicians": ("Attending physicians",
                                     "Number of different attending physicians across the provider's claims.", "count"),
    "NumUniqueOperatingPhysicians": ("Operating physicians",
                                     "Number of different operating physicians across the provider's claims.", "count"),
    "AvgPhysicianProviderSpread": ("Physician network spread",
                                   "On average, how many providers the provider's physicians also bill through.", "average"),
    "AvgAgeOfBeneficiaries": ("Average patient age", "Mean age of the beneficiaries the provider billed for.", "years"),
    "AvgChronicConditions": ("Chronic conditions per patient", "Average number of chronic conditions per beneficiary.", "average"),
    "PropDeceasedBeneficiaries": ("Share of deceased patients", "Proportion of beneficiaries recorded as deceased.", "proportion"),
    "AvgClaimsPerBeneficiary": ("Claims per patient", "Average number of claims per beneficiary.", "average"),
    "TopBeneficiaryClaimShare": ("Top patient's claim share",
                                 "Share of claims belonging to the provider's single most frequent beneficiary.", "proportion"),
    "AvgClaimDurationDays": ("Claim duration", "Average days between a claim's start and end dates.", "days"),
    "AvgLengthOfStayDays": ("Length of stay", "Average inpatient stay between admission and discharge.", "days"),
    "BenfordMAD": ("Benford deviation",
                   "How far the leading digits of claim amounts depart from Benford's law (mean absolute deviation).",
                   "index"),
    "BenfordMADMissing": ("Benford deviation unavailable",
                          "Yes when too few claims were available to compute the Benford deviation.", "flag"),
}

UNIT_NAMES = {
    "count": "Count", "average": "Average count", "amount": "US dollars", "proportion": "Percentage",
    "flag": "Yes / No", "days": "Days", "years": "Years", "ratio": "Ratio", "index": "Index", "number": "Number",
}

# Smallest and largest possible values for each unit (None = no limit)
LIMITS = {
    "count": (0, None), "average": (0, None), "amount": (0, None), "proportion": (0, 1),
    "flag": (0, 1), "days": (0, None), "years": (0, 120), "ratio": (0, None), "index": (0, None),
    "number": (None, None),
}


def _split_camel(name: str) -> str:
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).split()
    return " ".join(words).capitalize()


def label(feature: str) -> str:
    return FEATURES.get(feature, (_split_camel(feature), "", "number"))[0]


def description(feature: str) -> str:
    return FEATURES.get(feature, ("", "No description recorded.", "number"))[1]


def unit(feature: str) -> str:
    return FEATURES.get(feature, ("", "", "number"))[2]


def is_missing_flag(feature: str) -> bool:
    return unit(feature) == "flag" and feature.endswith("Missing")


def missing_rule(feature: str, restore_cols: list[str]) -> str:
    """How a blank value in this column is handled."""
    if unit(feature) == "flag":
        return "Not allowed: must be 0 or 1."
    if feature in restore_cols:
        return f"Set to missing whenever {feature}Missing is 1; the model then uses the training median."
    return "Allowed with a warning; the model uses the training median."


def fmt(feature: str, value) -> str:
    """Format a value in the feature's unit."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:  # NaN
        return "missing"
    u = unit(feature)
    if u == "flag":
        return "Yes" if v >= 0.5 else "No"
    if u == "proportion":
        return f"{v:.1%}"
    if u == "amount":
        return f"${v:,.0f}" if abs(v) >= 100 else f"${v:,.2f}"
    if u == "days":
        return f"{v:,.1f} days"
    if u == "years":
        return f"{v:,.1f} years"
    if u == "count" and v.is_integer():
        return f"{int(v):,}"
    return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"
