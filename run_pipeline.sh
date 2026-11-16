#!/usr/bin/env bash
# Run the project's notebooks in dependency order.
#
# Usage (from the repo root, with the project environment set up):
#   bash run_pipeline.sh                 # the standard check (minutes): data + explain + tests + saved result
#   bash run_pipeline.sh data            # only: raw-table EDA, notebooks 01 -> 05, then check the rebuilt tables
#   bash run_pipeline.sh explain         # only: 07 and 08 (SHAP and review policy, using the frozen model)
#   bash run_pipeline.sh tests           # only: automated tests, then print the saved held-out result
#   bash run_pipeline.sh models          # retrain everything (hours): 06b, 06c, 09a, leak-free 06b/06c, 09b, 09c
#   bash run_pipeline.sh all             # every notebook and check: the standard check, then models
#   bash run_pipeline.sh data 03c        # start a stage from a later notebook (e.g. after editing 03c)
#   QUICK_RUN=1 bash run_pipeline.sh models   # smoke test: 09b and 09c use a few trials only
#
# What each stage touches:
#   data     Rewrites the raw-table EDA notebooks and 01-05 in place, and data/processed/.
#            Ends by checking that the rebuilt train/test tables are byte-identical to the
#            ones the frozen final model was selected and tested on.
#   explain  Rewrites 07 and 08 in place and their outputs under reports/. Nothing is
#            retrained and the held-out test is not re-evaluated (08 reads the saved results).
#   tests    Runs pytest (dashboard and final-evaluation guard), then prints the saved
#            held-out result with scripts/run_final_test.py (read only).
#   models   Slow: retrains hundreds of models (09b alone runs 150 Optuna trials plus nested CV).
#            Every run writes a NEW timestamped folder under artifacts/, and the executed
#            notebooks are saved to artifacts/reruns/<timestamp>/. The committed notebooks,
#            the frozen model and the saved test results are never changed. The held-out
#            test is not touched: the final-test cells are disabled and src/final_evaluation.py
#            refuses a second evaluation.
#
# Notebooks run in number order. 09a must run before the leak-free reruns, 09b and 09c:
# it writes the *_leakfree.csv tables. (In the models stage, 09a comes after the canonical 06b/06c.)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Use whichever environment is already active (venv or conda); otherwise fall back to ./.venv.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ] && [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
if ! command -v jupyter >/dev/null 2>&1; then
  echo "jupyter not found. Activate your environment and run: pip install -r requirements.txt" >&2
  exit 1
fi
PYTHON="$(command -v python || command -v python3)"

STAGE="${1:-standard}"
START="${2:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RERUN_DIR="artifacts/reruns/$STAMP"

# The three *_eda notebooks look at each raw table before joining; they read data/raw only.
DATA_NOTEBOOKS=(00a_inpatient_eda 00b_outpatient_eda 00c_beneficiary_eda
                01_data_loading_joining 02_cleaning 03a_claim_level_eda
                03b_claim_level_feature_engineering 03c_provider_level_eda
                04_provider_feature_engineering 05_train_test_split)
EXPLAIN_NOTEBOOKS=(07_shap_evaluation 08_high_recall_policy_and_misses)
# name[:FEATURE_VARIANT]  - the leak-free entries rerun 06b/06c on 09a's tables
MODEL_NOTEBOOKS=(06b_boosting_models 06c_logistic_random_forest_models
                 09a_leakage_fixes_cv_check
                 06b_boosting_models:leakfree 06c_logistic_random_forest_models:leakfree
                 09b_optuna_early_stopping 09c_ensemble)

RAN=()
started=true
[ -n "$START" ] && started=false

# run_notebook <entry> <inplace|copy> <timeout seconds, -1 = none>
run_notebook() {
  local entry="$1" mode="$2" timeout="$3"
  local nb="${entry%%:*}" variant="canonical"
  [[ "$entry" == *:* ]] && variant="${entry#*:}"
  if ! $started; then
    [[ "$nb" == "$START"* ]] && started=true || return 0
  fi
  local label="$nb"
  [ "$variant" != "canonical" ] && label="$nb [FEATURE_VARIANT=$variant]"
  echo "==> Running notebooks/$label"
  local t0; t0=$(date +%s)
  # Force the environment's python3 kernel, whatever kernel name a notebook was saved with.
  local args=(--to notebook --execute
              --ExecutePreprocessor.kernel_name=python3
              --ExecutePreprocessor.timeout="$timeout")
  if [ "$mode" = inplace ]; then
    args+=(--inplace)
  else
    mkdir -p "$RERUN_DIR"
    local out="$nb"; [ "$variant" != "canonical" ] && out="${nb}_$variant"
    args+=(--output-dir "$RERUN_DIR" --output "$out")
  fi
  FEATURE_VARIANT="$variant" jupyter nbconvert "${args[@]}" "notebooks/$nb.ipynb"
  echo "    $label done in $(( $(date +%s) - t0 ))s"
  RAN+=("$label")
}

check_frozen_data() {
  # The rebuilt train/test tables must be byte-identical to the ones the frozen final model
  # was selected and tested on. A mismatch means the data changed, so the saved results,
  # dashboard and SHAP outputs no longer correspond to these files.
  "$PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path
rec = json.loads(Path("artifacts/shared/final_model_selection.json").read_text())["fingerprints_sha256"]
ok = True
for key, path in [("train", "data/processed/provider_features_train.csv"),
                  ("test", "data/processed/provider_features_test.csv")]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{path} not found: run the data stage first (bash run_pipeline.sh data).")
    match = hashlib.sha256(p.read_bytes()).hexdigest() == rec[key]
    ok &= match
    print(f"[{'ok' if match else 'MISMATCH'}] {path} matches the frozen final selection")
if not ok:
    raise SystemExit("Rebuilt data differs from the frozen selection. Check the raw files before going further.")
PY
}

run_data()    { for nb in "${DATA_NOTEBOOKS[@]}";    do run_notebook "$nb" inplace 1800; done; check_frozen_data; }
run_explain() { check_frozen_data; for nb in "${EXPLAIN_NOTEBOOKS[@]}"; do run_notebook "$nb" inplace 1800; done; }
run_tests()   {
  echo "==> Running the automated tests"
  "$PYTHON" -m pytest tests/ -q
  echo "==> Saved held-out result of the final model (read only)"
  "$PYTHON" scripts/run_final_test.py
  RAN+=("tests + saved result")
}
run_models()  {
  check_frozen_data
  echo "Model stage: this retrains hundreds of models and can take hours (QUICK_RUN=1 shortens 09b/09c)."
  for nb in "${MODEL_NOTEBOOKS[@]}"; do run_notebook "$nb" copy -1; done
  echo "Executed notebooks saved to $RERUN_DIR; new run folders are under artifacts/."
}

case "$STAGE" in
  standard) run_data; run_explain; run_tests ;;
  data)     run_data ;;
  explain)  run_explain ;;
  tests)    run_tests ;;
  models)   run_models ;;
  all)      run_data; run_explain; run_tests; run_models ;;
  *)        echo "Unknown stage '$STAGE'. Use one of: standard (default), data, explain, tests, models, all." >&2; exit 1 ;;
esac

if ! $started; then
  echo "Start notebook '$START' is not in stage '$STAGE'." >&2
  exit 1
fi
echo "All done. Completed ${#RAN[@]} steps:"
printf '  - %s\n' "${RAN[@]}"
