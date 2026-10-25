"""Create upload files for trying the dashboard, taken from your own held-out providers.

Run from the repository root:
    python -m dashboard.make_demo_files

Writes three CSVs to demo_uploads/ (labels removed):
  1_clean_60_providers.csv     scores straight away
  2_with_warnings.csv          scores, but the data-checks panel lists warnings
  3_with_errors.csv            is rejected, with a message explaining each problem
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard import core

N_PROVIDERS = 60
SEED = 4012


def main() -> None:
    bundle = core.load_bundle()
    test_path = core.resolve_path(bundle.meta["test_path"])
    sample = (pd.read_csv(test_path).drop(columns=[bundle.target], errors="ignore")
              .sample(N_PROVIDERS, random_state=SEED).reset_index(drop=True))
    out = core.REPO_ROOT / "demo_uploads"
    out.mkdir(exist_ok=True)

    continuous = [f for f in bundle.features if f not in bundle.binary_features and f not in bundle.restore_cols]
    big = "MaxClaimAmt" if "MaxClaimAmt" in continuous else continuous[0]
    blank = "AvgClaimAmt" if "AvgClaimAmt" in continuous else continuous[1]

    clean = out / "1_clean_60_providers.csv"
    sample.to_csv(clean, index=False)

    warn = sample.copy()
    warn[[big, blank]] = warn[[big, blank]].astype(float)
    if bundle.train_stats is not None:
        warn.loc[0, big] = float(bundle.train_stats.at[big, "max"]) * 10   # far outside the training range
    warn.loc[[1, 2], blank] = np.nan                                       # blank values
    warn["Notes"] = "example extra column"                                 # ignored column
    warned = out / "2_with_warnings.csv"
    warn.to_csv(warned, index=False)

    bad = sample.copy().astype({big: object})
    bad.loc[3, bundle.id_col] = bad.loc[4, bundle.id_col]                  # duplicate provider ID
    bad.loc[5, big] = "not a number"                                       # text in a numeric column
    if bundle.binary_features:
        flag = bundle.binary_features[0]
        bad[flag] = bad[flag].astype(int)
        bad.loc[6, flag] = 2                                               # a flag that isn't 0 or 1
    errors = out / "3_with_errors.csv"
    bad.to_csv(errors, index=False)

    print(f"Wrote {N_PROVIDERS}-provider demo files to {out.relative_to(core.REPO_ROOT)}/:")
    for path in (clean, warned, errors):
        print(f"  {path.name}")
    print("Upload them with 'Score another file' at the top right of the dashboard.")


if __name__ == "__main__":
    try:
        main()
    except core.DashboardError as exc:
        raise SystemExit(f"ERROR: {exc}")
