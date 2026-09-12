"""
Data loading and joining utilities for the healthcare provider fraud dataset.
"""

from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Update these if your downloaded filenames have different numeric suffixes
FILENAMES = {
    "labels": "provider_labels.csv",
    "beneficiary": "beneficiary_raw.csv",
    "inpatient": "inpatient_raw.csv",
    "outpatient": "outpatient_raw.csv",
}


def load_raw_tables(raw_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """
    Load the four raw CSVs into a dict of DataFrames.
    Keys: 'labels', 'beneficiary', 'inpatient', 'outpatient'
    """
    tables = {}
    for key, filename in FILENAMES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected file not found: {path}\n"
                f"Check that '{filename}' is in data/raw/, or update "
                f"FILENAMES in src/data_loader.py to match your actual filename."
            )
        tables[key] = pd.read_csv(path)
    return tables


def combine_claims(inpatient: pd.DataFrame, outpatient: pd.DataFrame) -> pd.DataFrame:
    """
    Stack inpatient and outpatient claims into a single claim-level table,
    tagged with a claim_type column. Inpatient-only columns (e.g. admission
    dates) will be NaN for outpatient rows, and vice versa.
    """
    inpatient = inpatient.copy()
    outpatient = outpatient.copy()

    inpatient["ClaimType"] = "Inpatient"
    outpatient["ClaimType"] = "Outpatient"

    combined = pd.concat([inpatient, outpatient], axis=0, ignore_index=True, sort=False)
    return combined


def build_claim_level_dataset(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Full join pipeline: load raw tables, combine inpatient+outpatient claims,
    attach beneficiary info, attach provider-level fraud label.

    Returns a single claim-level DataFrame — one row per claim, with
    beneficiary and provider label columns attached. This is the input
    to Step 4 (provider-level feature engineering), which will aggregate
    it up to one row per provider.
    """
    tables = load_raw_tables(raw_dir)

    claims = combine_claims(tables["inpatient"], tables["outpatient"])

    claims_with_bene = claims.merge(
        tables["beneficiary"], on="BeneID", how="left", validate="many_to_one"
    )

    full = claims_with_bene.merge(
        tables["labels"], on="Provider", how="left", validate="many_to_one"
    )

    return full


if __name__ == "__main__":
    df = build_claim_level_dataset()
    print(f"Claim-level dataset shape: {df.shape}")
    print(f"Unique providers: {df['Provider'].nunique()}")
    print(f"Unique beneficiaries: {df['BeneID'].nunique()}")
    print(f"Fraud label distribution:\n{df['PotentialFraud'].value_counts(normalize=True)}")
