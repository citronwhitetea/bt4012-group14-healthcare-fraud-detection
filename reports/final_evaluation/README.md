# Final evaluation evidence and execution provenance

The final evaluation completed on 24 October 2026. Do not repeat it to refresh notebook outputs.

## Read saved results

From the repository root, with the project environment activated:

```bash
python scripts/run_final_test.py
python -m unittest discover -s tests -p 'test_final_evaluation_guard.py'
```

The first command only reads JSON. Both modelling notebooks also have a read-only saved-results cell in section 11. It displays the same final **06b LightGBM** result, including when viewed from 06c; it is not a test of the LR/RF winner.

The viewer uses the original run when available and otherwise this committed snapshot. It verifies the recorded hashes. No pipeline, raw test labels, model fitting, prediction or bootstrapping is needed to view the saved results. The snapshot includes the exact test metrics, provider-level test predictions and post-amendment model metadata. Prediction labels and scores are final evaluation evidence: do not use them for further model/threshold selection.

## What executed originally

`scripts/provenance/finalize_healthcare_20261024.py` is an exact copy of the helper used on 24 October. It is preserved as historical evidence, including its original machine-specific paths; it is not the current entry point. It validated saved development outputs, wrote the approved selection and rationale, extracted the existing 06b section-11 function, and called it once. It did not retrain or tune a threshold on test data.

`scripts/provenance/06b_section11_executed.py` preserves the exact original test cell. Its SHA-256 matches `test_code_sha256` in the pre-test selection record. The helper executed only imports and function definitions from that cell, then called `run_final_test` with 2,000 bootstrap samples and seed 4012. The original training notebook has since been changed to use the shared guard, so the archived cell—not today's notebook—is the historical source.

`provenance.json` records the code hashes, snapshot hashes and observed original file modification times. The snapshot was created after evaluation; it does not claim to be a pre-test metadata fingerprint. The selection record and execution order establish the documented decision before the test function was called.

## Current shared gate

`src/final_evaluation.py` and `scripts/run_final_test.py` are portable cleanup code added after the result was produced. They are not claimed to be the original execution helper. Both notebooks use this same gate. A request must match the complete selected run path, model, strategy, threshold, rationale and fingerprints. Existing evaluation markers or test outputs prevent another evaluation. An exclusive marker creation protects simultaneous requests. A started or failed evaluation remains blocked pending explicit investigation.

The current records say **completed**, so `--evaluate --run ...` intentionally refuses. Do not delete the records to bypass the guard. The check is an accidental-reexecution safeguard, not protection against deliberately rewriting files.

## Local artifacts and dashboard handover

The trained model remains in the run referenced by `artifacts/shared/final_model_selection.json`; that file should be the dashboard's entry point. The binary pipeline and processed training/test data are not bundled in this small evidence snapshot. Teammates who need to score providers or compute SHAP must obtain the original run artifacts from Pei and verify the recorded fingerprints. Do not create a replacement model and present it as the already evaluated binary.

For scoring: restore the three missing-value columns using the recorded flags, preserve `feature_order`, and call the fitted pipeline. For SHAP: feed the same restored inputs through the fitted imputer, then explain the LightGBM step. Default LightGBM TreeExplainer contributions are log-odds, not percentage-point probability changes. Choose any explanation background from development data and label the output scale.

## Decisions and limitations

The strategy-first AP rule and 80% validation-recall objective were confirmed by Pei in the chat on 24 October before testing. Simpler model families were deliberately not the tie-breaker. The rationale was added to the frozen run's metadata before evaluation; the model and validation-selected threshold did not change. Log-transformed logistic regression was discussed but not run in these completed experiments. The original CV and upstream full-data feature-development limitations remain.
