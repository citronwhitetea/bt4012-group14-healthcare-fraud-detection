#!/usr/bin/env bash
# Run the data-preparation notebooks (01 -> 05) in dependency order.
#
# Usage (from the repo root, with the project venv set up):
#   bash run_pipeline.sh            # run everything from 01
#   bash run_pipeline.sh 03b        # start from a later notebook, e.g. after editing 03b
#
# Order matters: 04a must run before 03b, because 03b reads
# data/processed/claims_with_basic_features.csv, which 04a writes.
#
# This script deliberately STOPS before modelling (06b, 06c, 06d, 07):
#   - 06b/06c retrain 600+ models and write NEW run folders the frozen selection does not use;
#   - the held-out test set has already been used once and src/final_evaluation.py blocks a rerun;
#   - their final outputs are committed under artifacts/ and reports/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Use whichever environment is already active (venv or conda); otherwise fall back to ./.venv.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
if ! command -v jupyter >/dev/null 2>&1; then
  echo "jupyter not found. Activate your environment and run: pip install -r requirements.txt" >&2
  exit 1
fi

NOTEBOOKS=(
  01_data_loading_joining
  02_cleaning
  03a_claim_level_eda
  04a_basic_claim_level_feature_engineering
  03b_provider_level_eda
  04b_feature_engineering
  05_train_test_split
)

START="${1:-01}"
started=false
for nb in "${NOTEBOOKS[@]}"; do
  if [[ "$nb" == "$START"* ]]; then started=true; fi
  $started || continue
  echo "==> Running notebooks/$nb.ipynb"
  t0=$(date +%s)
  # Force the venv's python3 kernel (05 was saved with a custom kernel name).
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=1800 \
    "notebooks/$nb.ipynb"
  echo "    done in $(( $(date +%s) - t0 ))s"
done

if ! $started; then
  echo "Unknown start notebook '$START'. Choose one of: ${NOTEBOOKS[*]}" >&2
  exit 1
fi

# Check that the regenerated train/test tables are byte-identical to the ones the
# frozen final model was selected and tested on. A mismatch means the data changed
# and the saved results/dashboard no longer correspond to these files.
python - <<'PY'
import hashlib, json
from pathlib import Path
rec = json.loads(Path("artifacts/shared/final_model_selection.json").read_text())["fingerprints_sha256"]
ok = True
for key, path in [("train", "data/processed/provider_features_train.csv"),
                  ("test", "data/processed/provider_features_test.csv")]:
    p = Path(path)
    if not p.exists():
        print(f"[skip] {path} not found (05 not run yet)"); continue
    match = hashlib.sha256(p.read_bytes()).hexdigest() == rec[key]
    ok &= match
    print(f"[{'ok' if match else 'MISMATCH'}] {path} matches the frozen final selection: {match}")
if not ok:
    raise SystemExit("Regenerated data differs from the frozen selection. Tell the team before using it.")
PY
echo "All done."
