"""Checks that the dashboard scores exactly like the evaluated model and rejects bad input.

Run from the repository root:  python -m pytest tests/test_dashboard_core.py -q
Tests skip if the frozen run or the processed data are not present locally.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard import core


@pytest.fixture(scope="module")
def bundle():
    try:
        return core.load_bundle()
    except core.DashboardError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def test_frame(bundle):
    path = core.resolve_path(bundle.meta["test_path"])
    if not path.exists():
        pytest.skip(f"{path} not found")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def scored(bundle, test_frame):
    validation = core.validate_input(test_frame.astype({bundle.id_col: str}), bundle)
    assert validation.ok, validation.errors
    return core.score_batch(validation.frame, bundle)


def test_reproduces_frozen_test_predictions(bundle, scored):
    saved_path = bundle.run_dir / "test_predictions.csv"
    if not saved_path.exists():
        pytest.skip("no frozen test predictions")
    saved = pd.read_csv(saved_path, dtype={bundle.id_col: str})
    merged = scored.table.merge(saved, on=bundle.id_col, suffixes=("", "_saved"))
    assert len(merged) == len(scored.table)
    np.testing.assert_allclose(merged["score"], merged["score_saved"], rtol=0, atol=1e-9)
    assert (merged["flagged"] == (merged["score_saved"] >= bundle.threshold)).all()


def test_contributions_add_up_to_scores(scored):
    if scored.contributions is None:
        pytest.skip(scored.explain_note)
    logit = np.log(scored.table["score"] / (1 - scored.table["score"]))
    np.testing.assert_allclose(scored.contributions.sum(axis=1) + scored.base_value, logit, atol=1e-6)


def test_labels_do_not_change_scores(bundle, test_frame, scored):
    without = test_frame.drop(columns=[bundle.target], errors="ignore")
    again = core.score_batch(core.validate_input(without, bundle).frame, bundle)
    np.testing.assert_array_equal(again.table["score"], scored.table["score"])


def test_missing_feature_is_rejected(bundle, test_frame):
    v = core.validate_input(test_frame.drop(columns=[bundle.features[0]]), bundle)
    assert not v.ok and "missing" in v.errors[0]


def test_duplicate_provider_is_rejected(bundle, test_frame):
    v = core.validate_input(pd.concat([test_frame, test_frame.head(1)]), bundle)
    assert not v.ok and any("more than once" in e for e in v.errors)


def test_non_numeric_value_is_rejected(bundle, test_frame):
    bad = test_frame.copy()
    col = next(f for f in bundle.features if f not in bundle.binary_features)
    bad[col] = bad[col].astype(object)
    bad.loc[bad.index[0], col] = "abc"
    v = core.validate_input(bad, bundle)
    assert not v.ok and any("not a number" in e or "not numbers" in e for e in v.errors)


def test_non_binary_flag_is_rejected(bundle, test_frame):
    if not bundle.binary_features:
        pytest.skip("no binary features")
    bad = test_frame.copy()
    col = bundle.binary_features[0]
    bad[col] = bad[col].astype(float)
    bad.loc[bad.index[0], col] = 0.5
    v = core.validate_input(bad, bundle)
    assert not v.ok and any("0 or 1" in e for e in v.errors)


def test_flagged_values_are_reset_to_missing(bundle):
    if not bundle.restore_cols:
        pytest.skip("no restore columns")
    col = bundle.restore_cols[0]
    X = pd.DataFrame({col: [0.5, 0.5], f"{col}Missing": [1, 0]})
    out = core.restore_missing(X, [col])
    assert pd.isna(out.loc[0, col]) and out.loc[1, col] == 0.5


def test_long_free_text_in_selection_record_is_ignored(bundle):
    """A long sentence containing '/' must not be treated as a path (macOS: 'File name too long')."""
    record = {"decision": {"run": bundle.run_dir.name, "model": bundle.meta.get("model"),
                           "strategy": bundle.meta.get("strategy")},
              "rationale": "Missing a fraudulent provider costs more than an extra review. " * 8
                           + "We balanced recall/precision at 80% recall."}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "final_model_selection.json"
        path.write_text(json.dumps(record))
        run_dir, _, _ = core.find_run_dir(path)
    assert run_dir.resolve() == bundle.run_dir.resolve()
