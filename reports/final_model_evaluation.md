# Final model selection and held-out evaluation

Decision confirmed and evaluation performed on 24 October 2026 (Asia/Kuala_Lumpur).

**Selected model: LightGBM, original class distribution.** The selection used only saved validation results from 06b and 06c. The model and its threshold were frozen before the held-out evaluation.

## Validation comparison

| Candidate | Strategy | Mean fold AP | Pooled OOF AP | Within tolerance |
|---|---|---:|---:|---|
| lgbm | class_weight | 0.67405 | 0.67088 | Yes |
| lgbm | original | 0.67396 | 0.67198 | Yes |
| xgb | original | 0.67046 | 0.66729 | Yes |
| lr_vif10 | smote | 0.66832 | 0.66542 | Yes |
| lr_all | smote | 0.66812 | 0.66576 | Yes |
| xgb | class_weight | 0.66557 | 0.65883 | Yes |
| lr_vif10 | original | 0.66473 | 0.66238 | Yes |
| lr_all | original | 0.66447 | 0.66248 | Yes |
| rf | original | 0.66329 | 0.66126 | Yes |
| lr_vif5 | original | 0.66223 | 0.65907 | No |
| lr_vif5 | smote | 0.66069 | 0.65777 | No |
| lr_all | class_weight | 0.65483 | 0.65204 | No |
| lgbm | smote | 0.65299 | 0.64348 | No |
| lr_vif10 | class_weight | 0.65214 | 0.65006 | No |
| xgb | smote | 0.65019 | 0.64272 | No |
| lr_vif5 | class_weight | 0.64386 | 0.64063 | No |
| rf | class_weight | 0.63975 | 0.63622 | No |
| rf | smote | 0.63408 | 0.62554 | No |

Tolerance cutoff: 0.662666. Eligible candidates are ranked by imbalance strategy (original, class weighting, SMOTE), then mean AP. This rule does not prefer logistic regression over tree models. It is not an equivalence test.

All 18 non-dummy candidate combinations were checked against their saved OOF predictions. Data, feature-list and shared-fold fingerprints match between the two runs. The current files and selected pipelines match their recorded fingerprints.

## Frozen threshold rationale

Pei confirmed team agreement in this chat on 24 October 2026, before final held-out evaluation. Use the existing joint mean-CV-AP selection policy: candidates within one standard error of the highest mean AP qualify; prefer original, then class_weight, then smote, then higher mean AP. The operational objective is at least 80% recall on development OOF predictions. Among thresholds meeting this objective, choose the highest precision. This prioritises identifying potentially fraudulent providers while accepting the observed validation review workload of 701/4328 providers. This is a project assumption, not an empirically established business cost optimum or a guarantee of 80% recall on unseen data.

Threshold: **0.086104647726**. No model fitting or threshold tuning was performed on the test set.

## Held-out results

Test providers: 1082; positive labels: 101.

| Metric | Development OOF | Held-out test | Test 95% bootstrap interval |
|---|---:|---:|---|
| ap | 0.6720 | 0.7536 | 0.6795–0.8265 |
| roc_auc | 0.9353 | 0.9600 | 0.9443–0.9729 |
| precision | 0.4622 | 0.4859 | 0.4348–0.5394 |
| recall | 0.8000 | 0.8515 | 0.7723–0.9208 |
| f1 | 0.5859 | 0.6187 | 0.5645–0.6715 |

True positives: **86**; false positives: **91**; false negatives: **15**; true negatives: **890**.

Providers flagged: **177/1082 (16.36%)**.

Intervals use the existing notebook protocol: 2,000 label-stratified provider bootstrap samples, seed 4012, with fixed predictions and threshold. They do not include retraining/model-selection uncertainty or uncertainty in prevalence, and assume providers can be resampled independently within class.

## Interpretation and limits

The 80% recall objective was selected on validation predictions; test recall is an observation, not a constraint to retune against. Do not change the selected model or threshold based on these results. Upstream full-dataset EDA/feature development and network-feature dependencies remain limitations, so this is not a completely isolated end-to-end evaluation. PotentialFraud is a proxy label, not proof of misconduct.

## Reproducibility disclosures

The threshold rationale was recorded in the run metadata after the training run was frozen but before the held-out evaluation; the threshold itself was selected during validation and remained unchanged. The pipeline fingerprint was verified unchanged. No claim is made that retraining would reproduce byte-identical serialized files.

Observed original modification times on 24 October 2026 (UTC+08:00): final selection record 15:03:19.390704; selected-run metadata 15:03:19.391110; test predictions 15:03:23.984827; test metrics 15:03:23.985541. The archived helper's control flow also writes the selection and rationale before calling the test function. File times alone are not immutable proof of label-access timing.

Pei confirmed the strategy-first AP rule and the 80% recall objective before testing on 24 October. This was a deliberate choice to prefer simpler imbalance handling and then higher AP, **not** a preference for simpler model families. The tolerance rule had been introduced during validation development, so it is not represented as a policy specified before all validation results were seen.

**Log-transformed logistic regression was discussed but was not run in the completed experiments.** The final choice compares the saved 06b and 06c candidates, without claiming to exhaust all plausible model variants.

The final model was fitted on 4,328 development providers, versus 3,462–3,463 providers in each CV training fold. More training data and differences in sampled providers are plausible contributors to the higher test AP, but this experiment does not quantify or separate their effects. The validation threshold required at least 80% recall and happened to select exactly 80%; test recall is above the target at 85.15%, with an approximate 77.23%–92.08% bootstrap interval. Neither this increase nor the AP gap proves improved future performance. Upstream full-data feature development remains a limitation; its differential effect on CV versus test is unknown.

The original execution helper and exact executed section-11 source are preserved under `scripts/provenance/`. The portable `scripts/run_final_test.py` now displays saved results by default. The shared guard and read-only notebook cells were added **after** evaluation and did not generate new predictions, refit models, or recompute the bootstrap. See [the evidence README](final_evaluation/README.md) for the archived code, snapshot hashes and handover procedure.

## Handover

- Frozen run: `artifacts/boosting/20261011_232303_full`.
- Use `selected_pipeline.joblib` with `selected_metadata.json` for the dashboard; preserve feature order and restore the three missing-value columns before prediction. Label the output a risk score.
- `test_predictions.csv` and `test_metrics.json` are saved alongside that pipeline.
- `artifacts/shared/final_model_selection.json` records the pre-test choice; `artifacts/shared/heldout_evaluation.json` records completion.
- Paired fold AP differences and joint selection candidates are saved in `reports/06b_06c_paired_ap.csv` and `reports/06b_06c_final_candidates.csv`.
- Training/search cells were not rerun or changed. After evaluation, section 11 in both notebooks was updated with a shared gate and a read-only results viewer. The saved table shown in either notebook is the final 06b LightGBM result. Swei can use the frozen model for SHAP; Pei can build the dashboard.
