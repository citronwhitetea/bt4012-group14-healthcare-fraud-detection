# BT4012 Group Project — Healthcare Provider Fraud Detection

## Team

| Name | Responsibility |
|---|---|
| Jada Loau | _fill in_ |
| Tan Yumin | _fill in_ |
| Loo Pei En | _fill in_ |
| Tan Wei Jian | _fill in_ |

## Problem

Healthcare fraud is typically committed by **providers** (hospitals, physicians, clinics) rather than individual patients — through practices like billing for services not rendered, upcoding, or unbundling procedures. Because fraud signals emerge from *patterns across many claims*, not any single claim in isolation, this is naturally a **provider-level classification problem**: given all claims and beneficiary interactions tied to a provider, predict whether that provider is potentially fraudulent.

## Dataset

[Healthcare Provider Fraud Detection](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) (Kaggle, rohitrox) — a relational dataset with four linked tables:

| Table | Grain | Key contents |
|---|---|---|
| `Train` | Provider | `PotentialFraud` label (Yes/No) |
| `Train_Beneficiarydata` | Beneficiary (patient) | Demographics, chronic conditions, coverage dates |
| `Train_Inpatientdata` | Claim (inpatient) | Admission/discharge dates, diagnosis & procedure codes, reimbursement amounts |
| `Train_Outpatientdata` | Claim (outpatient) | Same structure as inpatient, without admission |

We use the **Train** files only (not the Kaggle Test files, which lack the fraud label) and construct our own train/test split at the provider level.

Class imbalance is significant: only ~9% of providers are labelled fraudulent.

We chose this dataset over alternatives (e.g. SGCC) for its relational, multi-table structure, which supports a wider range of course methods — supervised classifiers, unsupervised clustering, Benford's Law, and graph-based/network features — rather than a single flat feature table.

## Methodology

The pipeline runs in seven stages, detailed in `notebooks/`:

1. **Data loading & joining** — merge Beneficiary, Inpatient, Outpatient, and label tables
2. **Cleaning** — handle missing values, data types, inconsistent categories
3. **EDA & statistical tests** — chi-square, t-test, ANOVA, Benford's Law
4. **Provider-level feature engineering** — aggregate claim-level data up to the provider (claim counts, reimbursement statistics, diagnosis diversity, etc.), taking care to avoid **claim-level leakage** — features must be computed from claims, not from information only knowable after the fraud determination
5. **Train/test split & imbalance handling** — split by **provider** (not claim, to prevent leakage across the same provider appearing in both sets), then address class imbalance on the training set only
6. **Core modeling** — logistic regression, random forest, boosting; optionally a novel/less-standard method for extra credit
7. **SHAP evaluation & report assembly** — interpret model outputs, assemble final write-up

## Repo Structure & Setup

New to the repo? Start with **`notebooks/00_setup_guide.ipynb`** — it covers cloning, environment setup, dataset download, folder conventions, and our Git workflow.

```
├── data/{raw, processed}/     # data (see 00_setup_guide for what goes where)
├── notebooks/                 # one notebook per pipeline stage above
├── src/                       # shared functions imported into notebooks
├── reports/figures/           # exported plots for the final write-up
├── requirements.txt
└── README.md
```

## Status

- [ ] Data loading & joining
- [ ] Cleaning
- [ ] EDA & statistical tests
- [ ] Feature engineering
- [ ] Train/test split & imbalance handling
- [ ] Core modeling
- [ ] SHAP evaluation & report

## Grading Rubric Reference

| Criterion | Points |
|---|---|
| Data understanding / preprocessing | 3 |
| Method selection / justification | 4 |
| Experimental design / evaluation | 4 |
| Results / insights / applicability | 3 |
| Limitations / reproducibility | 2 |
