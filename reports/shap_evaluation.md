# SHAP evaluation of the frozen final model

Notebook: `notebooks/07_shap_evaluation.ipynb`. Outputs: `reports/shap_evaluation/` (tables, `figures/`, `summary.json`).

**Model explained:** 06b LightGBM, original class balance, run `20261011_232303_full`. The full pipeline is median `SimpleImputer` → `LGBMClassifier`, with the frozen review threshold **0.0861** (log-odds −2.36).
**Providers explained:** the 1,082 held-out test providers (101 fraud-labelled; 177 flagged: TP 86, FP 91, FN 15, TN 890).
**Method:** TreeSHAP, path-dependent, on the log-odds scale, using LightGBM `pred_contrib`. These are the same values the dashboard shows under "Why it scored here".

## Integrity checks

- The run, threshold, pipeline file and both feature tables match the fingerprints in `artifacts/shared/final_model_selection.json`.
- Recomputed held-out scores match the saved `test_predictions.csv` (max difference 1.1 × 10⁻¹⁶). Flags and confusion counts are identical.
- Additivity holds. The base value (−5.69 log-odds) plus each provider's contributions reproduces its log-odds score to 3.7 × 10⁻¹⁴, and its probability to 3.6 × 10⁻¹⁵.
- `shap.TreeExplainer` gives identical values (max difference 0).
- Nothing was refitted, retuned or re-thresholded. Test labels are used only after the fact, to group explanations by outcome.

## Key findings

### 1. Scores are driven mainly by claim size and claim volume
| Rank | Feature | Mean \|SHAP\| | Share | Effect direction |
|---:|---|---:|---:|---|
| 1 | MaxClaimAmt | 0.743 | 14.9% | higher → higher risk |
| 2 | NumClaims | 0.582 | 11.7% | higher → higher risk |
| 3 | AvgClaimsPerBeneficiary | 0.456 | 9.2% | higher → higher risk |
| 4 | AvgLengthOfStayDays | 0.355 | 7.1% | higher → higher risk |
| 5 | BenfordMAD | 0.254 | 5.1% | U-shaped (see 4) |
| 6 | MedianClaimAmt | 0.240 | 4.8% | step up at about $80–90 |
| 7 | PropDeceasedBeneficiaries | 0.207 | 4.2% | higher → higher (see 5) |
| 8 | TopProcedureCodeShare | 0.196 | 3.9% | **lower** → higher risk |
| 9 | DeductibleToReimbursementRatio | 0.170 | 3.4% | higher → higher risk |
| 10 | AvgClaimAmt | 0.159 | 3.2% | higher → higher risk |

- **Concentration.** The top three features carry 36% of total mean |SHAP|; the top ten carry 68%.
- **Stable leaders.** In 1,000 label-stratified bootstraps of the test providers, the top four ranks never change. BenfordMAD and MedianClaimAmt share ranks 5–6. Below rank 9, neighbouring ranks overlap, so their order should not be over-read.
- **Feature groups.** Exact sums within domain groups put **claim amounts** at 30% of attribution and **volume and patient concentration** at 25%. The other groups are care setting and stay length (12%), coding patterns (10%), physician network (8%), beneficiary profile (8%) and Benford's Law (7%).
- **Monotone effects.** Dependence plots show near-monotone, plausible effects for the top four. The push from MaxClaimAmt turns positive at about $14,000 and reaches +2 log-odds above $44,000. NumClaims turns positive above about 75 claims and rises steeply above about 260.
- **Agreement with gain importance.** The ranking agrees moderately with LightGBM's gain importance: Spearman 0.70, and 6 of the top 10 in common. The main disagreements are MedianClaimAmt (SHAP 6 vs gain 23), PropDeceasedBeneficiaries (7 vs 17), TopProcedureCodeShare (8 vs 3) and AvgAgeOfBeneficiaries (17 vs 6). Gain measures training-loss reduction; SHAP measures effect on held-out scores. They are not interchangeable.

### 2. Concentration features agree with the EDA
Lower **TopProcedureCodeShare**, **TopBeneficiaryClaimShare** and **TopDiagnosisCodeShare** all raise the score (Spearman of value vs SHAP: −0.85, −0.81 and −0.45). In other words, the model treats more diverse coding and a broader patient base as riskier. This matches the counter-intuitive effects found in EDA (04b: fraud providers have lower concentration; Cohen's d −0.89 and −0.75). The model therefore uses these features in the direction the data support, not the "templated billing" direction one might expect.

### 3. The three missing-value flags are never used
`TopDiagnosisCodeShareMissing`, `TopProcedureCodeShareMissing` and `BenfordMADMissing` have **zero splits** and zero SHAP. They duplicate information the model already has:
- `BenfordMADMissing` is exactly `NumClaims < 20` (missing providers have at most 19 claims; non-missing at least 20).
- `TopProcedureCodeShareMissing` has Spearman −0.96 with `AvgNumProcedureCodes`.

So "missing" reaches the model only through the **imputed training median** and through correlated features. What the imputed value means then depends on where the median falls:
- **BenfordMAD:** 416 test providers are missing. They are imputed to 0.0336, which sits at the bottom of BenfordMAD's U-shaped curve, so it pushes the score down (mean −0.27). Their fraud rate is 1.2%, against 14.4% when present.
- **TopProcedureCodeShare:** 674 are missing. They are imputed to 0.25, with a mean push of −0.07. Their fraud rate is 1.6%, against 22.1%.

These pushes happen to point the right way, but they come from where the median lies, not from a learned "missing" effect. A different fill value would change these explanations.

### 4. The Benford signal is confounded with claim volume
BenfordMAD ranks 5th, but its effect is **U-shaped**:
- **Very conforming providers** (MAD ≤ 0.023) get a mean push of +0.45. Their fraud rate is 31%, and their median is 280 claims.
- **Strongly deviating providers** (MAD ≥ 0.048) also get +0.48. Their fraud rate is 6.8%, and their median is 30 claims.

MAD falls as the number of claims grows (Spearman −0.77 with NumClaims on the test providers). The "low MAD → risk" arm therefore largely re-encodes high volume, and the "high MAD" arm mostly picks up small, noisy providers. **The Benford feature should not be presented as evidence of digit manipulation.** A sample-size-adjusted Benford statistic would be needed to test that claim (see Limitations).

### 5. PropDeceasedBeneficiaries looks weakly supported
It ranks 7th by SHAP, but its mean push is about zero in both classes (+0.004 fraud vs −0.004 non-fraud). Its largest pushes go to providers with 2–5% deceased beneficiaries (mean +0.75), whose fraud rate is only 7%. By contrast, the 0–2% band has a 29% fraud rate but a mean push of only +0.08. These are marginal comparisons and SHAP is conditional on the other features, but this is the feature most likely to be moving scores without discriminating. It is a candidate for review in any retraining.

### 6. Domain and network features play a small role
`AvgPhysicianProviderSpread`, the physician–provider network feature, ranks 20th (2.5%). The physician-network group as a whole has 8%. The relational features add some signal, but the model rests mainly on amounts and volume.

### 7. Interactions are substantial
SHAP interaction values put the summed mean |pairwise interaction| (5.15) close to the summed mean |main effect| (5.28). The strongest pair is NumClaims × MaxClaimAmt (0.11). Much of the vertical spread in the dependence plots is therefore interaction, not noise. Single-feature readings ("MaxClaimAmt adds +2") describe this provider in its context; they are not a fixed per-feature rule.

## Errors and decision fragility

**False positives look like smaller versions of true positives.** Both are mostly inpatient providers (89% and 88% have inpatient claims). FPs have a lower median NumClaims (117 vs 252) and lower median MaxClaimAmt ($36,000 vs $52,000). Two numbers show how much the FPs rest on claim size:
- In the mean SHAP profile, MaxClaimAmt pushes FPs almost as much as TPs (+1.35 vs +1.61), while NumClaims pushes them much less (+0.60 vs +1.76).
- **MaxClaimAmt is the largest positive driver for 61 of the 91 FPs (67%)**, against 51 of 86 TPs.

The maximum claim amount is capped at $57,000 for 58 test providers. Of the 76 providers at or above that value, 83% are flagged and 55% are fraud-labelled.

**Many flags rest on a single feature.** For 58 of 91 FPs (64%) and 28 of 86 TPs (33%), one contribution is larger than the provider's distance above the threshold. MaxClaimAmt is that feature in 47 cases and NumClaims in 30. This is an additive-attribution statement, not a counterfactual. It still means a reviewer should expect many flags whose explanation comes down to "one very large claim".

**False negatives are mostly not near misses.** Of the 15, 3 fall within ±0.5 log-odds of the threshold, and their mean log-odds score is −3.58 against a threshold of −2.36. They are smaller providers (median 74 claims; MaxClaimAmt $14,000) with little coding, physician-network or Benford signal (group pushes +0.05, −0.08 and 0.00). Their fraud labels are not expressed in the features this model relies on.

**Near the threshold** (46 providers with scores 0.054–0.134), the same top four features dominate. TopProcedureCodeShare (rank 5 vs 8 overall) and StdClaimAmt (8 vs 18) matter more there.

**Case studies.** The cases were selected by fixed rules (figures `07_case_*`).
- **Strongest TP, PRV52340 (score 0.987):** 1,743 claims (+4.24) plus a $57,000 maximum claim (+1.57).
- **TP closest above the threshold, PRV53951 (0.087):** flagged essentially by one $63,000 claim (+2.42).
- **Strongest FP, PRV57715 (0.942):** the same profile as the strongest TP: 860 claims (+2.29), a $57,000 maximum (+2.22) and 2.2 claims per beneficiary (+1.10).
- **FN closest below the threshold, PRV51482 (0.077):** a $99,000 claim (+2.23), offset by few claims per beneficiary (−0.50) and concentrated procedure coding (−0.49).
- **Lowest-scoring FN, PRV52826 (0.003):** many claims per patient (+1.66) but a maximum claim of only $3,300 (−0.58). The model sees no claim-size signal for this provider.

## Robustness

| Check | Result |
|---|---|
| Additivity (log-odds / probability) | max gap 3.7 × 10⁻¹⁴ / 3.6 × 10⁻¹⁵ |
| `pred_contrib` vs `shap.TreeExplainer` | identical |
| Bootstrap rank intervals | top 4 fixed; ranks 5–6 interchangeable |
| Path-dependent vs interventional (200 development providers as background) | Spearman 0.961 on global importance; top-10 overlap 9/10; same top driver for 78% of providers; median per-provider Spearman 0.97 |

- **Where the two methods differ.** The interventional variant swaps ranks 1 and 2 (NumClaims 0.61, MaxClaimAmt 0.60), and DeductibleToReimbursementRatio drops from 9th to 15th. MaxClaimAmt's credit shrinks by about 20%. Some of its path-dependent credit is shared with correlated amount and inpatient features (Spearman 0.80–0.92 with StdClaimAmt, AvgLengthOfStayDays and PropInpatient).
- **What holds under both.** The headline conclusions do not depend on the method: the same four leading features, the same directions, and the dominance of claim size and volume.

## Implications

- **For the dashboard.** "Why it scored here" uses these same values. Reviewers should expect MaxClaimAmt or NumClaims as the top factor for most flags (156 of 177). A single very large claim is often enough to cross the threshold, so the case review should check whether a large claim is clinically plausible rather than treat it as the finding.
- **For the report.** The model behaves as a "large, high-volume, inpatient-heavy provider" detector with sensible directions. That is consistent with the EDA, but it explains why its false positives are large legitimate providers and its misses are small ones.
- **For any future retraining** (out of scope for this frozen evaluation):
  - drop or replace the unused missing-value flags;
  - test a size-adjusted Benford statistic;
  - re-examine PropDeceasedBeneficiaries;
  - consider whether features normalised by provider size would reduce the volume dependence.

## Limitations

- **SHAP explains the model, not fraud.** `PotentialFraud` is a proxy label, and attributions describe how the frozen model uses its inputs, not what causes fraud.
- **Correlated features.** Credit can shift between correlated features (see `correlated_pairs.csv`). Group-level totals are more reliable than individual ranks within a block.
- **Path-dependent SHAP** credits features through the training distribution stored in the trees. The interventional comparison bounds this effect but does not remove it.
- **Small subgroups.** All statistics come from one held-out set of 1,082 providers. Outcome subgroups are small (FN n = 15), and the bootstrap covers test-sample variation only, not retraining.
- **Imputation.** For providers with missing values, explanations depend on the training-median fill. The model never uses the missing-value flags.
- **Upstream limitations carry over** from `reports/final_model_evaluation.md`. They include full-dataset feature development and the physician-network feature, which is computed across the whole claims table.
