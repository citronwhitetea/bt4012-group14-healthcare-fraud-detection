# Comparison of 06b and 06c model results

> **Earlier analysis.** This compares the 13 October runs of 06b and 06c, when logistic regression used a fixed
> 22-feature list. The final model choice uses the later 06c run with fold-level VIF (`lr_all`, `lr_vif10`, `lr_vif5`);
> see `reports/final_model_evaluation.md`. The conclusion here (boosting is not clearly better than logistic
> regression) is consistent with that final comparison.

Analysis date: 18 October 2026. All reported performance is development cross-validation or out-of-fold (OOF) performance, not held-out test performance.

## Recommendation

**Select LightGBM trained on the original class distribution as the provisional primary model under the project's declared average-precision selection rule.** It has the highest pooled OOF average precision (AP), nearly the highest mean fold AP, and lower fold AP variability than logistic regression. Its advantage is modest and has not been confirmed on an independent test set.

**Retain original logistic regression as the simpler benchmark and original XGBoost as the operational challenger.** Logistic regression nearly matches boosting, is competitive at larger review budgets, and has a compact linear score. XGBoost has the best observed precision under the current recall-at-least-80% threshold rule. There is no single model that dominates all relevant operating points.

## Scope and provenance

- 06b: `artifacts/boosting/20261013_231908_full`.
- 06c: `artifacts/logistic_random_forest/20261013_233654_full`.
- Both use 4,328 development providers, including 405 labelled potentially fraudulent providers (9.36%), with the same five provider folds. Each fold has 81 positive providers.
- The training data, test file, feature-list, shared-fold and saved-pipeline hashes all match their recorded metadata.
- Each saved notebook run searched 20 configurations for each of two model families and three imbalance strategies: 120 configurations and 600 fold fits per run.
- The earlier 06c run, `20261013_232621_full`, has identical summary metrics, fold metrics and saved pipeline. Its OOF probabilities differ only at floating-point precision (maximum absolute difference 2.22e-16). It is not an independent replication.
- No held-out test metrics are present in these run folders. Both selected metadata files have an empty threshold rationale. This analysis did not evaluate test labels, retrain models or change the notebooks.

**Version mismatch:** the saved 06c results are for `lr`, using a fixed 22-feature list, and random forest using 26 features. The current 06c notebook defines `lr_all`, `lr_vif10` and `lr_vif5`, with feature selection inside each training fold. No results for those new LR variants are saved here. Conclusions about them require a new run. Inspection of the saved LR pipeline confirms a fixed `ColumnTransformer` selection step and 22 coefficients, 18 of them nonzero.

## Overall predictive performance

Mean fold AP is the notebook's model-selection metric. Pooled OOF AP combines predictions across all folds; these two summaries need not be identical. AP summarizes the precision–recall trade-off and is not the precision at any one threshold.

| Original-distribution model | Mean fold AP | Fold AP SD | Pooled OOF AP | Pooled ROC AUC | OOF Brier score ↓ |
|---|---:|---:|---:|---:|---:|
| LightGBM | **0.67396** | 0.02579 | **0.67198** | 0.93527 | 0.04982 |
| XGBoost | 0.67046 | 0.02670 | 0.66729 | **0.93569** | **0.04915** |
| Logistic regression | 0.66473 | 0.04538 | 0.66238 | 0.93098 | 0.04941 |
| Random forest | 0.66329 | **0.01913** | 0.66126 | 0.93238 | 0.04994 |

All four models provide substantial ranking signal relative to the roughly 0.0936 AP baseline. Accuracy would be a poor primary selection criterion: predicting every provider as negative already gives 90.64% accuracy and zero recall.

LightGBM's mean AP advantage over logistic regression is 0.00923, or 0.92 percentage points. Its pooled AP advantage is 0.00960. This is useful evidence for choosing a provisional ranking model, but not a large separation. Random forest is the most stable across these five folds by AP SD, but has lower average AP than the other original-distribution models. Low observed fold SD alone does not establish better future robustness.

The comparable performance of logistic regression suggests that a substantial part of the useful signal can be captured by a regularized linear score over the engineered features. The current comparison does not establish which features or interactions cause the differences. That requires coefficient stability, attribution or ablation analysis.

## Does LightGBM reliably beat logistic regression?

| Saved fold | LightGBM AP | Logistic AP | Difference |
|---|---:|---:|---:|
| 0 | 0.68647 | 0.60693 | +0.07954 |
| 1 | 0.65056 | 0.68029 | -0.02973 |
| 2 | 0.67082 | 0.65585 | +0.01497 |
| 3 | 0.65057 | 0.64984 | +0.00073 |
| 4 | 0.71137 | 0.73072 | -0.01935 |

LightGBM wins three of five folds, with one almost tied. Its average advantage is heavily influenced by fold 0. Across the other four folds, logistic regression averages about 0.00834 higher AP. This is a diagnostic of sensitivity, not a reason to discard fold 0.

A paired bootstrap with 3,000 resamples, seed 4012, and provider resampling within each label class gives the following approximate 95% percentile intervals for **LightGBM minus logistic regression**, using their frozen OOF predictions:

- AP difference: **-0.01718 to +0.03673**.
- ROC AUC difference: **-0.00223 to +0.01126**.
- Brier score difference: **-0.00175 to +0.00250**; negative favors LightGBM.

Each interval includes zero. These are conditional descriptive intervals: they hold the selected configurations and predictions fixed, assume providers can be resampled within class, and omit uncertainty from retraining, hyperparameter search and feature selection. They do not correct selection optimism or dependence from overlapping CV training sets. The evidence does not establish clear superiority over logistic regression.

## Investigation performance at the current threshold rule

For each model, choose the OOF threshold with the highest precision among thresholds achieving recall of at least 80%. LightGBM and LR reproduce the saved choices. XGBoost and RF thresholds below are newly calculated comparisons, not saved deployment selections.

| Original-distribution model | Threshold | Precision | Recall | F1 | True positives | False positives | False negatives | Reviews |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.08610 | 46.22% | 80.00% | 0.58590 | 324 | 377 | 81 | 701 |
| XGBoost | 0.14023 | **46.97%** | **80.49%** | **0.59327** | **326** | **368** | **79** | **694** |
| Logistic regression | 0.17341 | 46.49% | 80.25% | 0.58877 | 325 | 374 | 80 | 699 |
| Random forest | 0.18943 | 46.42% | 80.00% | 0.58749 | 324 | 374 | 81 | 698 |

At these thresholds, XGBoost catches two more positive providers than LightGBM while producing nine fewer false alarms and seven fewer reviews. Logistic regression catches one more than LightGBM with three fewer false alarms. These differences are small and were evaluated on the same OOF data used for development.

All four require reviewing roughly 16% of providers to capture about 80% of labelled positives, and more than half the flags are false positives against the available labels. Review capacity therefore matters as much as the small differences in global AP. No investigation costs or recoverable fraud amounts are available, so the results cannot establish a financially optimal model.

## Performance under a fixed review budget

Rank providers by OOF score and review the top `ceil(budget × 4,328)` providers. There are no score ties at the reported boundaries. Cells show **positive providers detected / recall**.

| Review budget | Providers reviewed | LightGBM | XGBoost | Logistic regression | Random forest |
|---|---:|---:|---:|---:|---:|
| 5% | 217 | **168 / 41.48%** | 163 / 40.25% | 165 / 40.74% | 159 / 39.26% |
| 10% | 433 | 255 / 62.96% | **261 / 64.44%** | **261 / 64.44%** | 256 / 63.21% |
| 15% | 650 | 315 / 77.78% | 315 / 77.78% | 312 / 77.04% | **317 / 78.27%** |
| 20% | 866 | 340 / 83.95% | **351 / 86.67%** | 350 / 86.42% | 342 / 84.44% |

LightGBM leads at a 5% review budget. At 10%, XGBoost and LR tie for the lead. At 20%, XGBoost and LR capture approximately ten more positive providers than LightGBM. A 10% review budget is insufficient to achieve 80% recall for any of these four models in the saved results.

Higher recall targets also favor LR at some operating points. Under the same precision-maximizing threshold rule, LR reaches 90.12% recall with 1,029 reviews and 664 false positives; LightGBM reaches the same recall with 1,036 reviews and 671 false positives. At 95.06% recall, LR needs 1,395 reviews versus LightGBM's 1,422.

LightGBM and LR make different errors: at their saved thresholds, 303 positives are flagged by both, 21 only by LightGBM, 22 only by LR, and 59 by neither. Combining their flags would increase recall but also false positives. Complementarity motivates a separately validated ensemble experiment; it does not demonstrate an ensemble improvement by itself.

## Imbalance strategy comparison

The table reports mean fold AP for each strategy's own best sampled configuration.

| Model | Original | Class weighting | SMOTE-NC |
|---|---:|---:|---:|
| LightGBM | 0.67396 | **0.67405** | 0.65299 |
| XGBoost | **0.67046** | 0.66557 | 0.65019 |
| Logistic regression | 0.66473 | 0.65214 | **0.66625** |
| Random forest | **0.66329** | 0.63975 | 0.63408 |

- LightGBM class weighting gains just 0.000092 AP and loses on three of five folds. There is little evidence that it improves the model.
- SMOTE-NC reduces mean AP for LightGBM by 0.02097, XGBoost by 0.02027 and RF by 0.02921. It is worse on all five folds for each tree family.
- LR's SMOTE-NC improvement is only 0.00152 AP, with improvement in three folds and deterioration in two. Class weighting reduces LR AP by 0.01258.
- RF weighting reduces AP by 0.02354 and is worse in every fold.

The one-standard-error selection rule explains why the saved winners use the original distribution even though weighted LightGBM and SMOTE LR have slightly higher fold means. Within the tolerance, the code prefers `original`, then `class_weight`, then `smote`, and finally higher AP. It does **not** explicitly prefer simpler model families. Applying the rule jointly to all saved candidates still chooses original LightGBM; all four original models fall within the global cutoff of approximately 0.66267.

There is no evidence here that balancing the class distribution is necessary to obtain good ranking. Threshold adjustment already provides control over the precision–recall trade-off. The observed SMOTE degradation is consistent with synthetic samples adding little useful information or altering class geometry, but these runs do not identify its cause. Conclusions are specific to the configurations and budgets searched.

## Probability quality and tuning sensitivity

The prior-reference Brier score is 0.08482. All original models improve substantially on it. XGBoost has the lowest observed Brier score, followed closely by LR, LightGBM and RF. A lower Brier score alone does not prove better calibration because it also reflects discrimination; see the [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

LR's mean original OOF score is 9.48%, close to the 9.36% observed prevalence. LightGBM's is 7.96%, indicating aggregate underprediction in these OOF scores. Weighted LR and SMOTE LR have mean scores of 26.52% and 23.97%, with Brier scores of 0.10627 and 0.09848, respectively, both worse than the prior reference. Their uncalibrated outputs should not be treated as reliable fraud probabilities. Matching the overall prevalence is also insufficient to establish calibration across score ranges.

LightGBM is relatively consistent across its 20 original-distribution configurations: mean fold AP ranges from 0.6512 to 0.6740, with median 0.6677. XGBoost ranges from 0.6066 to 0.6705, median 0.6535. LR ranges from the dummy-level 0.0936 to 0.6647; excessive L1 regularization collapses at least one sampled configuration to a constant predictor. These summaries describe the chosen search spaces, not intrinsic robustness of the algorithms.

The selected LightGBM uses 384 trees, learning rate 0.02046, 60 leaves, minimum 45 child samples and row subsampling around 0.705. LR uses L1 regularization with C = 0.42981. The winning XGBoost original configuration uses 524 trees, learning rate 0.01680 and maximum depth 4. No model timing benchmark was saved, so computational-cost claims have not been quantified.

## Limits and next decision

1. **Development selection optimism.** The same folds choose hyperparameters and estimate performance. OOF predictions exclude each provider from that particular fit, but the provider's fold contributes to choosing the configuration. Thresholds are also optimized on those predictions. Nested CV or a final independent evaluation is needed to assess the complete selection procedure; see [scikit-learn's nested-CV explanation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
2. **Old LR feature selection.** The fixed 22-feature list was derived upstream using the complete provider table. This gives validation/test covariates a role in feature selection, even though VIF is unsupervised. The current fold-local variants address that particular issue but have not produced saved results.
3. **Upstream feature development.** Full-data EDA and physician-network feature construction precede the split. The network feature uses cross-provider claim relationships. Whether that is appropriate depends on which network information would exist when scoring future providers. Provider-disjoint folds do not by themselves establish temporal or network independence.
4. **Limited replication and labels.** There is one effective shared five-fold experiment, and the label is `PotentialFraud`. Performance describes agreement with these labels, not confirmed recoverable losses or prospective investigation outcomes.

Finish the new 06c comparison using the same development split. Choose the operational objective before final selection: retain mean AP for general ranking, or explicitly choose precision at the required recall or recall at a fixed review capacity. Freeze the winner and threshold rationale, then evaluate the held-out set once under the existing protocol. Do not choose a winner by repeatedly inspecting held-out results.

## Verification

All 13 distinct model/strategy combinations, including the shared dummy, were joined by provider and checked for matching labels/folds and complete predictions. AP, ROC AUC and precision-maximizing thresholds were independently checked against scikit-learn. Saved selected-model metrics and Brier scores were reproduced. The chart was visually inspected. Source artifacts and notebook files were left unchanged.

![Model comparison](/Users/tanwj/bt4012-group14-healthcare-fraud-detection/reports/figures/06b_06c_model_comparison.png)
