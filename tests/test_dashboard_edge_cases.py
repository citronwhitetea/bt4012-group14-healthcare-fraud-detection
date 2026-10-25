"""Edge-case tests for the dashboard that run anywhere, without the project's data or artifacts.

A small LightGBM pipeline and a frozen-run folder are built in a temporary directory, then the
loader, the upload checks, scoring and explanations are tested against it.

Run from the repository root:  python -m pytest tests/test_dashboard_edge_cases.py -q
"""
from __future__ import annotations

import atexit
import io
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

from dashboard import core, glossary

FEATURES = list(glossary.FEATURES)
RESTORE = ["TopDiagnosisCodeShare", "TopProcedureCodeShare", "BenfordMAD"]
FLAGS = [f for f in FEATURES if glossary.unit(f) == "flag"]


def _fake_providers(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for f in FEATURES:
        u = glossary.unit(f)
        cols[f] = {"count": rng.poisson(40, n).astype(float), "amount": rng.lognormal(7, 1, n),
                   "proportion": rng.random(n), "flag": (rng.random(n) < 0.3).astype(int),
                   "days": rng.gamma(2, 2, n), "years": rng.uniform(60, 90, n),
                   }.get(u, rng.gamma(2, 1, n))
    df = pd.DataFrame(cols)
    for c in RESTORE:
        df.loc[df[f"{c}Missing"] == 1, c] = df[c].mean()
    z = -2 + 0.04 * (df["NumClaims"] - 40) + 1.5 * df["PropInpatient"]
    df.insert(0, "Provider", [f"PRV{seed}{i:05d}" for i in range(n)])
    df["PotentialFraud"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")
    return df


def _build_repo() -> dict:
    root = Path(tempfile.mkdtemp(prefix="dashboard_test_"))
    atexit.register(shutil.rmtree, root, True)
    data = root / "data"
    data.mkdir()
    train, test = _fake_providers(600, 1), _fake_providers(150, 2)
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)

    X = core.restore_missing(train[FEATURES], RESTORE)
    y = (train["PotentialFraud"] == "Yes").astype(int).to_numpy()
    pipe = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")),
                     ("model", LGBMClassifier(n_estimators=40, num_leaves=8, min_child_samples=10,
                                              random_state=0, verbose=-1))]).fit(X, y)
    run = root / "artifacts" / "boosting" / "20260101_120000_full"
    run.mkdir(parents=True)
    joblib.dump(pipe, run / "selected_pipeline.joblib")
    pipe_sha = core.sha256(run / "selected_pipeline.joblib")
    threshold = float(np.quantile(pipe.predict_proba(X)[:, 1], 0.8))

    Xt = core.restore_missing(test[FEATURES], RESTORE)
    pt = pipe.predict_proba(Xt)[:, 1]
    pd.DataFrame({"Provider": test["Provider"], "score": pt}).to_csv(run / "test_predictions.csv", index=False)
    json.dump({"point": {"threshold": threshold, "tp": 1, "fp": 1, "fn": 1, "tn": 1, "flagged_share": 0.5}},
              open(run / "test_metrics.json", "w"))
    oof = pipe.predict_proba(X)[:, 1]
    pd.DataFrame({"Provider": train["Provider"], "y_true": y, "oof_prob": oof, "model": "lgbm",
                  "strategy": "original"}).to_csv(run / "oof_predictions.csv", index=False)
    meta = {"run_id": run.name, "quick_run": False, "model": "lgbm", "strategy": "original", "threshold": threshold,
            "threshold_rule": {"rule": "min_recall", "value": 0.8}, "feature_order": FEATURES,
            "binary_features": FLAGS, "restore_na_cols": RESTORE, "id_col": "Provider", "target": "PotentialFraud",
            "train_path": str(data / "train.csv"), "test_path": str(data / "test.csv"), "n_train": len(train),
            "fingerprints_sha256": {"pipeline": pipe_sha}, "versions": {}}
    (run / "selected_metadata.json").write_text(json.dumps(meta))
    selection = root / "final_model_selection.json"
    selection.write_text(json.dumps({
        "selected_run": str(run),
        "decision": {"model": "LightGBM", "strategy": "original", "threshold": threshold,
                     "pipeline_sha256": pipe_sha},
        "candidates": [{"model": "lr_vif10"}, {"model": "rf"}],
        "rationale": "Missing a fraudulent provider costs more than an extra review. " * 6 + "recall/precision",
    }))
    return {"root": root, "run": run, "selection": selection, "test": test, "meta": meta, "threshold": threshold}


@pytest.fixture(scope="module")
def repo():
    return _build_repo()


@pytest.fixture(scope="module")
def bundle(repo):
    return core.load_bundle(selection_path=repo["selection"])


def _copy_run(repo) -> tuple[Path, Path]:
    """An isolated copy of the run folder and its selection record, safe to tamper with."""
    tmp = Path(tempfile.mkdtemp(prefix="tamper_"))
    atexit.register(shutil.rmtree, tmp, True)
    run = tmp / repo["run"].name
    shutil.copytree(repo["run"], run)
    record = json.loads(repo["selection"].read_text())
    record["selected_run"] = str(run)
    selection = tmp / "final_model_selection.json"
    selection.write_text(json.dumps(record))
    return run, selection


def _edit_meta(run: Path, **changes) -> None:
    meta = json.loads((run / "selected_metadata.json").read_text())
    meta.update(changes)
    (run / "selected_metadata.json").write_text(json.dumps(meta))


# ---------------------------------------------------------------- linking to the frozen model
def test_selected_run_is_read_exactly(repo, bundle):
    assert bundle.run_dir.resolve() == repo["run"].resolve()
    assert bundle.source.startswith("selected_run")
    assert {name: status for name, status, _ in bundle.checks}["Threshold"] == "match"


def test_display_names_in_the_record_are_accepted(bundle):
    checks = {name: status for name, status, _ in bundle.checks}
    assert checks["Model"] == "match" and checks["Strategy"] == "match" and checks["Pipeline fingerprint"] == "match"


def test_missing_fingerprint_is_rejected(repo):
    run, selection = _copy_run(repo)
    _edit_meta(run, fingerprints_sha256={})
    with pytest.raises(core.DashboardError, match="fingerprint"):
        core.load_bundle(selection_path=selection)


def test_changed_threshold_is_rejected(repo):
    run, selection = _copy_run(repo)
    _edit_meta(run, threshold=0.5)
    with pytest.raises(core.DashboardError, match="Threshold"):
        core.load_bundle(selection_path=selection)


def test_changed_model_file_is_rejected(repo):
    run, selection = _copy_run(repo)
    with open(run / "selected_pipeline.joblib", "ab") as f:
        f.write(b"tampered")
    with pytest.raises(core.DashboardError, match="fingerprint"):
        core.load_bundle(selection_path=selection)


def test_evaluation_from_another_threshold_is_rejected(repo):
    run, selection = _copy_run(repo)
    metrics = json.loads((run / "test_metrics.json").read_text())
    metrics["point"]["threshold"] = 0.123
    (run / "test_metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(core.DashboardError, match="Held-out evaluation"):
        core.load_bundle(selection_path=selection)


def test_evaluation_pages_work_without_the_model(repo):
    run, selection = _copy_run(repo)
    (run / "selected_pipeline.joblib").unlink()
    with pytest.raises(core.DashboardError):
        core.load_bundle(selection_path=selection)
    evaluation_only = core.load_bundle(selection_path=selection, require_pipeline=False)
    assert not evaluation_only.can_score and evaluation_only.test_metrics is not None


# ---------------------------------------------------------------- upload checks
def _validate(bundle, df):
    return core.validate_input(core.read_csv_bytes(df.to_csv(index=False).encode(), bundle.id_col), bundle)


def _rows(repo, n=10):
    return repo["test"].head(n).copy()


@pytest.mark.parametrize("column,value,expect", [
    ("NumClaims", -3, "negative"),
    ("PropInpatient", 1.7, "above 1"),
    ("AvgAgeOfBeneficiaries", 150, "above 120"),
    ("AvgClaimAmt", -10, "negative"),
    ("TopDiagnosisCodeShareMissing", 2, "0 or 1"),
])
def test_impossible_values_are_rejected(repo, bundle, column, value, expect):
    df = _rows(repo)
    df[column] = df[column].astype(float)
    df.loc[0, column] = value
    v = _validate(bundle, df)
    assert not v.ok and any(expect in e for e in v.errors)
    report = v.report()
    assert (report["column"] == column).any() and (report["provider"] == df.loc[0, "Provider"]).any()


def test_blank_flag_is_rejected(repo, bundle):
    df = _rows(repo)
    df[FLAGS[0]] = df[FLAGS[0]].astype(float)
    df.loc[0, FLAGS[0]] = np.nan
    v = _validate(bundle, df)
    assert not v.ok and any("blank" in e for e in v.errors)


def test_blank_and_trimmed_duplicate_ids_are_rejected(repo, bundle):
    df = _rows(repo)
    df["Provider"] = df["Provider"].astype(object)
    df.loc[0, "Provider"] = "   "
    df.loc[2, "Provider"] = df.loc[1, "Provider"] + "  "
    v = _validate(bundle, df)
    assert any("blank" in e for e in v.errors) and any("more than once" in e for e in v.errors)


def test_text_in_a_number_column_is_rejected(repo, bundle):
    df = _rows(repo).astype({"MaxClaimAmt": object})
    df.loc[0, "MaxClaimAmt"] = "twelve"   # ("n/a" would be read as a blank cell, which only warns)
    assert any("not a number" in e or "not numbers" in e for e in _validate(bundle, df).errors)


def test_duplicate_column_names_are_rejected(repo, bundle):
    text = _rows(repo).to_csv(index=False)
    header, rest = text.split("\n", 1)
    with pytest.raises(core.DashboardError, match="more than once"):
        core.read_csv_bytes((header + ",NumClaims\n" + rest.replace("\n", ",1\n")).encode(), bundle.id_col)


def test_ids_stay_text(bundle):
    df = core.read_csv_bytes(b"Provider,x\n00123,1\n", "Provider")
    assert df["Provider"].iloc[0] == "00123"


def test_row_limit(repo, bundle, monkeypatch):
    monkeypatch.setattr(core, "MAX_ROWS", 5)
    with pytest.raises(core.DashboardError, match="up to 5"):
        core.read_csv_bytes(_rows(repo, 10).to_csv(index=False).encode(), bundle.id_col)


def test_unusual_values_warn_but_score(repo, bundle):
    df = _rows(repo)
    df["NumClaims"] = df["NumClaims"].astype(float)
    df.loc[0, "NumClaims"] = 10 * bundle.train_stats.at["NumClaims", "max"]
    df.loc[1, "AvgClaimAmt"] = np.nan
    v = _validate(bundle, df)
    assert v.ok and any("outside the range" in w for w in v.warnings) and any("blank" in w for w in v.warnings)


# ---------------------------------------------------------------- scoring and explanations
@pytest.fixture(scope="module")
def scored(repo, bundle):
    v = _validate(bundle, repo["test"])
    assert v.ok, v.errors
    return core.score_batch(v.frame, bundle)


def test_scores_match_saved_predictions(bundle, scored):
    assert core.check_saved_predictions(scored.table, bundle)["status"] == "match"


def test_explanations_add_up_to_each_score(scored):
    logit = np.log(scored.table["score"] / (1 - scored.table["score"]))
    np.testing.assert_allclose(scored.contributions.sum(axis=1) + scored.base_value, logit, atol=1e-6)


def test_waterfall_runs_from_average_to_score(scored):
    steps = core.waterfall(scored, 0, n=5)
    final = steps.iloc[-1]["end"]
    assert steps.iloc[0]["kind"] == "base" and steps.iloc[-1]["kind"] == "final"
    assert abs(core.sigmoid(final) - scored.table["score"].iloc[0]) < 1e-6
