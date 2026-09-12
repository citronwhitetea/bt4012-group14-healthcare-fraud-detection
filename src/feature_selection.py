"""
Feature significance and multicollinearity evaluation.

Run this AFTER build_provider_features() and BEFORE train/test split +
modeling, to decide which engineered features are actually worth keeping,
and to catch redundant (highly correlated) feature pairs before they cause
instability in linear models like logistic regression.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


def evaluate_feature_significance(df: pd.DataFrame, feature_cols: list[str],
                                   target_col: str = "PotentialFraud") -> pd.DataFrame:
    """
    Run a Welch's t-test + Cohen's d for every feature against the fraud
    label, and return a single sorted summary table.

    Works for both continuous and binary (0/1) features — for binary
    features, the "mean difference" is just a difference in proportions,
    which is still a meaningful comparison.

    Sorted by absolute Cohen's d (effect size), not p-value — with n=5410
    providers, even small effects can hit p < 0.05, so effect size is the
    more honest way to rank feature importance here.
    """
    results = []
    for col in feature_cols:
        group_fraud = df.loc[df[target_col] == 1, col].dropna()
        group_non_fraud = df.loc[df[target_col] == 0, col].dropna()

        if len(group_fraud) < 2 or len(group_non_fraud) < 2:
            continue

        t_stat, p_value = stats.ttest_ind(group_fraud, group_non_fraud, equal_var=False)
        pooled_std = np.sqrt((group_fraud.var() + group_non_fraud.var()) / 2)
        cohens_d = (group_fraud.mean() - group_non_fraud.mean()) / pooled_std if pooled_std > 0 else np.nan

        results.append({
            "feature": col,
            "mean_fraud": group_fraud.mean(),
            "mean_non_fraud": group_non_fraud.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "abs_cohens_d": abs(cohens_d) if not np.isnan(cohens_d) else -1,
        })

    result_df = pd.DataFrame(results).sort_values("abs_cohens_d", ascending=False)
    result_df = result_df.drop(columns="abs_cohens_d").reset_index(drop=True)
    return result_df


def flag_weak_features(significance_df: pd.DataFrame, cohens_d_threshold: float = 0.2,
                        p_threshold: float = 0.05) -> pd.DataFrame:
    """
    Flag features as 'weak' if they fail EITHER the significance threshold
    (p >= 0.05) OR the effect size threshold (|Cohen's d| < 0.2, conventionally
    a 'small' effect). Weak features aren't necessarily useless in a model —
    tree-based models can still exploit weak/nonlinear signals — but they're
    worth flagging explicitly rather than silently including everything.
    """
    df = significance_df.copy()
    df["is_weak"] = (df["p_value"] >= p_threshold) | (df["cohens_d"].abs() < cohens_d_threshold)
    return df


def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Variance Inflation Factor for each feature — quantifies how much a
    feature's variance is inflated by its correlation with other features
    in the list. Rule of thumb: VIF > 5 suggests problematic multicollinearity,
    VIF > 10 is a strong warning sign (mainly matters for linear/logistic
    regression; tree-based models are largely unaffected by multicollinearity).

    Drops rows with any NaN in the given columns, since VIF requires a
    complete numeric matrix.
    """
    X = df[feature_cols].dropna()
    X = X.assign(intercept=1)  # VIF calculation requires a constant term

    vif_data = pd.DataFrame()
    vif_data["feature"] = feature_cols
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(len(feature_cols))
    ]
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def iterative_vif_selection(df: pd.DataFrame, feature_cols: list[str],
                             threshold: float = 10.0) -> tuple[list[str], pd.DataFrame]:
    """
    Greedily drop the single highest-VIF feature, recompute, and repeat until
    every remaining feature's VIF is below `threshold`. Useful for building a
    decorrelated feature set specifically for logistic regression, while the
    full feature set (unmodified) can still be used for tree-based models,
    which aren't sensitive to multicollinearity.

    Returns (final_feature_list, elimination_log) where elimination_log
    records which feature was dropped at each step and its VIF at drop time.
    """
    remaining = list(feature_cols)
    elimination_log = []

    while True:
        vif_df = compute_vif(df, remaining)
        max_vif_row = vif_df.iloc[0]

        if max_vif_row["VIF"] < threshold or len(remaining) <= 1:
            break

        elimination_log.append({"dropped_feature": max_vif_row["feature"], "vif_at_drop": max_vif_row["VIF"]})
        remaining.remove(max_vif_row["feature"])

    return remaining, pd.DataFrame(elimination_log)


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data_loader import build_claim_level_dataset
    from cleaning import clean_claim_level_dataset
    from features import build_provider_features

    raw = build_claim_level_dataset()
    df = clean_claim_level_dataset(raw)
    provider_features = build_provider_features(df)

    feature_cols = [c for c in provider_features.columns if c not in ("Provider", "PotentialFraud")]

    print("=== Feature Significance (sorted by |Cohen's d|) ===")
    sig_df = evaluate_feature_significance(provider_features, feature_cols)
    print(sig_df.to_string())

    print("\n=== Weak Features Flagged ===")
    flagged = flag_weak_features(sig_df)
    print(flagged[flagged["is_weak"]]["feature"].tolist())

    print("\n=== Variance Inflation Factor (multicollinearity check) ===")
    vif_df = compute_vif(provider_features, feature_cols)
    print(vif_df.to_string())