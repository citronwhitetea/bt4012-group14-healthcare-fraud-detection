"""
EDA and statistical testing utilities.

Statistical tests are run on PROVIDER-LEVEL aggregates, not claim-level rows,
since claims from the same provider are not independent observations and
the fraud label itself is defined at the provider grain. Running tests on
raw claim rows would inflate sample size and risk false-positive significance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Provider-level aggregation (minimal, just enough for EDA — full version in
# Step 4)
# ---------------------------------------------------------------------------

def build_minimal_provider_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("Provider").agg(
        NumClaims=("ClaimID", "count"),
        AvgClaimAmt=("InscClaimAmtReimbursed", "mean"),
        TotalClaimAmt=("InscClaimAmtReimbursed", "sum"),
        AvgAgeAtClaim=("AgeAtClaim", "mean"),
        PropInpatient=("ClaimType", lambda x: (x == "Inpatient").mean()),
        AvgNumPhysicians=("NumPhysiciansInvolved", "mean"),
        PotentialFraud=("PotentialFraud", "first"),
    ).reset_index()
    return summary


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------

def plot_missingness(df: pd.DataFrame, top_n: int = 20):
    """Bar chart of missingness % for the top_n columns with the most missing values."""
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    missing_pct.plot(kind="barh", ax=ax, color="indianred")
    ax.invert_yaxis()
    ax.set_xlabel("% Missing")
    ax.set_title(f"Top {top_n} Columns by Missingness")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame):
    """Grid of key distributions: claim amount (log), age, claims per provider."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(np.log1p(df["InscClaimAmtReimbursed"].dropna()), bins=50, color="steelblue")
    axes[0].set_title("log(1 + Claim Amount)")
    axes[0].set_xlabel("log(InscClaimAmtReimbursed + 1)")

    axes[1].hist(df["AgeAtClaim"].dropna(), bins=40, color="seagreen")
    axes[1].set_title("Age at Claim")
    axes[1].set_xlabel("Age")

    claims_per_provider = df.groupby("Provider")["ClaimID"].count()
    axes[2].hist(np.log1p(claims_per_provider), bins=40, color="darkorange")
    axes[2].set_title("log(1 + Claims per Provider)")
    axes[2].set_xlabel("log(NumClaims + 1)")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fraud group comparisons
# ---------------------------------------------------------------------------

def plot_fraud_boxplots(provider_df: pd.DataFrame, numeric_cols: list[str]):
    """Boxplots comparing each numeric column's distribution across fraud groups."""
    n = len(numeric_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(data=provider_df, x="PotentialFraud", y=col, ax=ax, showfliers=False)
        ax.set_xticklabels(["Non-Fraud", "Fraud"])
        ax.set_title(col)

    plt.tight_layout()
    return fig


def plot_fraud_rate_by_category(df: pd.DataFrame, categorical_col: str, top_n: int = 15):
    """
    Bar chart of fraud rate (% of claims/providers labeled Fraud) across
    categories of a column, e.g. ClaimType, State, or a chronic condition flag.
    """
    rate = df.groupby(categorical_col)["PotentialFraud"].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    rate.plot(kind="bar", ax=ax, color="tomato")
    ax.set_ylabel("Fraud Rate")
    ax.set_title(f"Fraud Rate by {categorical_col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(provider_df: pd.DataFrame, numeric_cols: list[str]):
    """Correlation heatmap among provider-level numeric features, including the fraud label."""
    corr = provider_df[numeric_cols + ["PotentialFraud"]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Provider-Level Features)")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def chi_square_test(df: pd.DataFrame, categorical_col: str, target_col: str = "PotentialFraud") -> dict:
    contingency = pd.crosstab(df[categorical_col], df[target_col])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    return {
        "test": "chi-square", "column": categorical_col,
        "chi2_statistic": chi2, "p_value": p_value, "degrees_of_freedom": dof,
        "contingency_table": contingency,
    }


def cramers_v(df: pd.DataFrame, categorical_col: str, target_col: str = "PotentialFraud") -> float:
    """
    Cramer's V: effect size for chi-square, scaled 0 (no association) to 1
    (perfect association). Use this alongside the chi-square p-value —
    at large n, p-value alone can be significant even for a trivial effect.
    """
    contingency = pd.crosstab(df[categorical_col], df[target_col])
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan


def t_test(df: pd.DataFrame, numeric_col: str, target_col: str = "PotentialFraud") -> dict:
    group_yes = df.loc[df[target_col] == 1, numeric_col].dropna()
    group_no = df.loc[df[target_col] == 0, numeric_col].dropna()
    t_stat, p_value = stats.ttest_ind(group_yes, group_no, equal_var=False)

    # Cohen's d: effect size for the t-test
    pooled_std = np.sqrt((group_yes.var() + group_no.var()) / 2)
    cohens_d = (group_yes.mean() - group_no.mean()) / pooled_std if pooled_std > 0 else np.nan

    return {
        "test": "welch t-test", "column": numeric_col,
        "t_statistic": t_stat, "p_value": p_value, "cohens_d": cohens_d,
        "mean_fraud": group_yes.mean(), "mean_non_fraud": group_no.mean(),
    }


def anova_test(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict:
    groups = [g[numeric_col].dropna().values for _, g in df.groupby(group_col)]
    f_stat, p_value = stats.f_oneway(*groups)
    return {
        "test": "one-way ANOVA", "column": numeric_col, "group_col": group_col,
        "f_statistic": f_stat, "p_value": p_value, "num_groups": len(groups),
    }


# ---------------------------------------------------------------------------
# Benford's Law
# ---------------------------------------------------------------------------

def _leading_digit_proportions(amounts: pd.Series) -> pd.Series:
    positive_amounts = amounts[amounts > 0].dropna()
    leading_digits = positive_amounts.astype(str).str.lstrip("0.").str[0].astype(int)
    leading_digits = leading_digits[leading_digits.between(1, 9)]
    counts = leading_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    return counts / counts.sum()


def benford_leading_digit_test(amounts: pd.Series) -> dict:
    """Chi-square goodness-of-fit test of leading digits against Benford's Law."""
    observed_props = _leading_digit_proportions(amounts)
    n = amounts[amounts > 0].dropna().shape[0]
    benford_expected_probs = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

    observed_counts = observed_props * n
    expected_counts = benford_expected_probs * n
    chi2, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

    return {
        "test": "Benford's Law (chi-square goodness of fit)",
        "chi2_statistic": chi2, "p_value": p_value,
        "observed_proportions": observed_props.round(4).to_dict(),
        "expected_proportions": dict(zip(range(1, 10), benford_expected_probs.round(4))),
    }


def benford_mean_absolute_deviation(amounts: pd.Series) -> float:
    """
    Nigrini's MAD conformity metric: mean absolute difference between
    observed and expected leading-digit proportions. Unlike the chi-square
    statistic, MAD does NOT scale with sample size, making it the fair way
    to compare Benford deviation between groups of different sizes (e.g.
    Fraud vs Non-Fraud, which have different claim counts).
    """
    observed_props = _leading_digit_proportions(amounts)
    benford_expected_probs = pd.Series(
        [np.log10(1 + 1 / d) for d in range(1, 10)], index=range(1, 10)
    )
    return (observed_props - benford_expected_probs).abs().mean()


def benford_test_by_fraud_group(df: pd.DataFrame, amount_col: str = "InscClaimAmtReimbursed",
                                 target_col: str = "PotentialFraud") -> dict:
    """
    Run Benford's Law separately for Fraud and Non-Fraud groups. Reports both
    the chi-square statistic (sensitive to sample size — do NOT compare raw
    chi2 across groups of different sizes) and MAD (scale-invariant — this
    is the metric to actually compare between groups).
    """
    fraud_amounts = df.loc[df[target_col] == 1, amount_col]
    non_fraud_amounts = df.loc[df[target_col] == 0, amount_col]

    fraud_result = benford_leading_digit_test(fraud_amounts)
    non_fraud_result = benford_leading_digit_test(non_fraud_amounts)

    fraud_result["mad"] = benford_mean_absolute_deviation(fraud_amounts)
    non_fraud_result["mad"] = benford_mean_absolute_deviation(non_fraud_amounts)

    return {"fraud_group": fraud_result, "non_fraud_group": non_fraud_result}


def plot_benford_comparison(df: pd.DataFrame, amount_col: str = "InscClaimAmtReimbursed",
                             target_col: str = "PotentialFraud"):
    """Grouped bar chart: observed leading-digit distribution for Fraud vs Non-Fraud vs Benford expected."""
    fraud_props = _leading_digit_proportions(df.loc[df[target_col] == 1, amount_col])
    non_fraud_props = _leading_digit_proportions(df.loc[df[target_col] == 0, amount_col])
    benford_expected = pd.Series(
        [np.log10(1 + 1 / d) for d in range(1, 10)], index=range(1, 10)
    )

    plot_df = pd.DataFrame({
        "Fraud": fraud_props, "Non-Fraud": non_fraud_props, "Benford Expected": benford_expected,
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_xlabel("Leading Digit")
    ax.set_ylabel("Proportion")
    ax.set_title("Leading Digit Distribution: Fraud vs Non-Fraud vs Benford's Law")
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data_loader import build_claim_level_dataset
    from cleaning import clean_claim_level_dataset

    raw = build_claim_level_dataset()
    df = clean_claim_level_dataset(raw)
    provider_summary = build_minimal_provider_summary(df)

    print("=== T-test: AvgClaimAmt vs Fraud ===")
    print(t_test(provider_summary, "AvgClaimAmt"))

    print("\n=== T-test: NumClaims vs Fraud ===")
    print(t_test(provider_summary, "NumClaims"))

    print("\n=== Chi-square + Cramer's V: ClaimType vs Fraud ===")
    print(chi_square_test(df, "ClaimType"))
    print(f"Cramer's V: {cramers_v(df, 'ClaimType'):.4f}")

    print("\n=== Benford's Law by Fraud Group ===")
    result = benford_test_by_fraud_group(df)
    print(f"Fraud group chi2: {result['fraud_group']['chi2_statistic']:.2f}")
    print(f"Non-fraud group chi2: {result['non_fraud_group']['chi2_statistic']:.2f}")