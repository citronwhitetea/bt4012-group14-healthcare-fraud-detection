# Leak-free features: retraining check

Analysis date: 15 November 2026, after the final held-out evaluation. Development (cross-validation) results only; no test labels were read.

**Result: removing the physician-network leak does not change model selection.** The team's joint validation rule still selects LightGBM on the original class distribution. Every estimate of LightGBM's mean fold AP stays within 0.007 of the frozen run's 0.674, against a fold standard deviation of about 0.025.

## What was fixed

- **Physician network.** `AvgPhysicianProviderSpread` counted each physician's providers across all 5,410 providers' claims, so test providers shaped training providers' values. `build_physician_network_features(df, reference_providers=...)` now counts only a reference set. A reference provider counts itself once; any other provider is scored as if it alone had joined the reference set.
- **All-provider fills.** `TopDiagnosisCodeShare`, `TopProcedureCodeShare` and `BenfordMAD` were filled with means over all providers. `build_provider_features()` now leaves them NaN for the model pipeline to impute from training rows. The modelling notebooks already re-imputed these per fold, so this changes no score.

The canonical tables (`provider_features_{train,test}.csv`) still reproduce the frozen fingerprints, so the frozen model, its threshold, the held-out evaluation and the dashboard are unchanged. Leak-free tables are `provider_features_{train,test}_leakfree.csv` (built by notebook 06e).

## Two measurements

| Measurement | Reference providers for the physician feature | LightGBM / original, mean fold AP |
|---|---|---:|
| Frozen run (`boosting/20261011_232303_full`) | All 5,410 providers | 0.6740 |
| 06e, frozen configuration | Each fold's training providers only (fully fold-clean) | 0.6672 |
| 06e, best of the same 20 configurations | Each fold's training providers only | 0.6723 |
| Leak-free retrain, frozen configuration | The 4,328 training providers | 0.6766 |
| Leak-free retrain, selected configuration | The 4,328 training providers | 0.6774 |

- **The leak-free retrain** removes test providers from the feature, which is what matters for a model evaluated on the held-out set. Inside its cross-validation, validation-fold providers still count towards training-fold values, so its CV figure is slightly optimistic for this one feature.
- **06e** removes the validation fold as well. It is the strictest estimate, but it uses fewer providers per count, which also makes the feature noisier.

The true effect lies between these and is indistinguishable from fold noise. In 06e the bootstrap interval for the pooled-AP difference was −0.0116 to +0.0060.

## Retrain: joint selection, canonical vs leak-free

Runs: `boosting/20261115_134434_full_leakfree` and `logistic_random_forest/20261115_135414_full_leakfree` (06b/06c with `FEATURE_VARIANT=leakfree`; same split, shared folds, search spaces and seeds). Canonical comparison: `boosting/20261011_232303_full` and `logistic_random_forest/20261115_010242_full`. The selection rule is the one in `final_model_evaluation.md`: candidates within one standard error of the best mean fold AP qualify; prefer original, then class weighting, then SMOTE, then higher AP.

| Candidate | Canonical mean fold AP | Leak-free mean fold AP | Difference |
|---|---:|---:|---:|
| lgbm / class_weight | 0.67405 | 0.67783 | +0.00378 |
| **lgbm / original (selected in both)** | **0.67396** | **0.67743** | **+0.00348** |
| xgb / original | 0.67046 | 0.66810 | −0.00236 |
| rf / original | 0.66329 | 0.66744 | +0.00414 |
| lr_vif10 / smote | 0.66832 | 0.66814 | −0.00018 |
| lr_vif10 / original | 0.66473 | 0.66469 | −0.00003 |

- **Tolerance cutoff:** 0.66267 canonical, 0.66682 leak-free. Both select LightGBM / original.
- **Leak-free selected configuration:** config 3 (298 trees, learning rate 0.033, 60 leaves, min. 27 samples per leaf). At the ≥80% recall rule its OOF threshold is 0.0455, with precision 0.454 and 16.5% of providers reviewed (frozen run: 0.462 and 16.2%).
- **Logistic regression is essentially unaffected.** Tree models move by up to ±0.005, consistent with the feature's small SHAP role (rank 20, about 2.5% of attribution).

All 18 candidates are in `leakfree_vs_canonical_candidates.csv`.

## Limits

- **The leak-free run is not a replacement final model.** The held-out set has been used, so testing this run would be test-informed. The frozen LightGBM remains the evaluated model.
- **The leak was covariate leakage, not target leakage.** Other providers' claims entered the feature; no labels did.
- **CV remains optimistic.** The same folds tune and score. The other upstream limitations in `final_model_evaluation.md` still apply.
