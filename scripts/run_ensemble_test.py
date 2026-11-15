"""Second, pre-registered held-out evaluation: the LightGBM + logistic regression ensemble (06g).

By default this displays the saved results. `--run` performs the one permitted evaluation. It
runs only if the selection record is approved, committed, and matches every fingerprint, and if
no evaluation has started. The ensemble's result is reported whatever it scores; the earlier
LightGBM test result is shown next to it as a paired comparison, not as a choice.

    python scripts/run_ensemble_test.py          # show saved results
    python scripts/run_ensemble_test.py --run    # the one permitted evaluation
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # the saved LR pipeline references src.fold_vif
SELECTION = ROOT / "artifacts/shared/ensemble_final_selection.json"
MARKER = ROOT / "artifacts/shared/ensemble_heldout_evaluation.json"
FIRST_SELECTION = ROOT / "artifacts/shared/final_model_selection.json"
FIRST_PREDICTIONS = ROOT / "artifacts/boosting/20261011_232303_full/test_predictions.csv"
N_BOOT, BOOT_SEED = 2000, 4012   # same protocol as the first evaluation
REPORTED = ["ap", "roc_auc", "precision", "recall", "f1", "flagged_share"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def check_allowed():
    """Validate the pre-registered decision before any test labels or model are loaded."""
    sel = json.loads(SELECTION.read_text())
    if sel.get("status") != "approved_before_test":
        raise ValueError("The ensemble selection has not been approved.")
    if MARKER.exists():
        raise RuntimeError("The ensemble evaluation has already started or completed. Read the saved results.")
    run = ROOT / sel["run_dir"]
    if any((run / name).exists() for name in ["test_metrics.json", "test_predictions.csv"]):
        raise RuntimeError("Existing held-out outputs found in the ensemble run. Do not evaluate again.")
    for key, rel in sel["files"].items():
        if sha(ROOT / rel) != sel["fingerprints_sha256"][key]:
            raise ValueError(f"{key} ({rel}) changed after selection.")
    if sha(__file__) != sel["test_code_sha256"]:
        raise ValueError("This evaluation script differs from the fingerprint in the selection record.")
    for path in [SELECTION, Path(__file__)]:
        rel = str(path.relative_to(ROOT))
        if not git("log", "-1", "--format=%H", "--", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
            raise ValueError(f"{rel} must be committed, unchanged, before the evaluation runs.")
    return sel, run


def metrics(y, p, t):
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"threshold": float(t), "ap": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "f1": float(f1_score(y, yhat, zero_division=0)),
            "flagged_share": float(yhat.mean()),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate(sel, run):
    import joblib
    import numpy as np
    import pandas as pd

    ensemble = joblib.load(run / "ensemble_pipeline.joblib")
    features, threshold = ensemble["feature_order"], float(ensemble["threshold"])
    assert features == sel["feature_order"] and threshold == sel["threshold"]

    test = pd.read_csv(ROOT / sel["files"]["test"])
    target = sel["target"]
    if not pd.api.types.is_numeric_dtype(test[target]):
        test[target] = test[target].map({"Yes": 1, "No": 0})
    assert test[target].isin([0, 1]).all() and test[sel["id_col"]].is_unique
    for col in sel["missing_value_cols"]:   # the leak-free table keeps these NaN; imputation is in the pipelines
        assert (test[col].isna() == (test[f"{col}Missing"] == 1)).all(), col

    y = test[target].to_numpy()
    member_scores = {n: m.predict_proba(test[features])[:, 1] for n, m in ensemble["members"].items()}
    p = sum(ensemble["weights"][n] * s for n, s in member_scores.items())
    point = metrics(y, p, threshold)

    # Paired comparison with the first evaluation's saved LightGBM scores (same providers, same order)
    first = pd.read_csv(FIRST_PREDICTIONS)
    assert (first["Provider"].to_numpy() == test[sel["id_col"]].to_numpy()).all()
    assert (first["y_true"].to_numpy() == y).all()
    first_threshold = json.loads(FIRST_SELECTION.read_text())["threshold"]
    p_first = first["score"].to_numpy()
    first_point = metrics(y, p_first, first_threshold)

    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boot, diffs = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m, m_first = metrics(y[idx], p[idx], threshold), metrics(y[idx], p_first[idx], first_threshold)
        boot.append({k: m[k] for k in REPORTED})
        diffs.append({k: m[k] - m_first[k] for k in REPORTED})
    ci = pd.DataFrame(boot).quantile([0.025, 0.975]).T.set_axis(["ci_low", "ci_high"], axis=1)
    diff_ci = pd.DataFrame(diffs).quantile([0.025, 0.975]).T.set_axis(["ci_low", "ci_high"], axis=1)

    pd.DataFrame({sel["id_col"]: test[sel["id_col"]], "y_true": y, "score": p,
                  "flagged": (p >= threshold).astype(int),
                  **{f"score_{n}": s for n, s in member_scores.items()}}).to_csv(run / "test_predictions.csv", index=False)
    result = {"point": point, "bootstrap_95ci": ci.to_dict(orient="index"),
              "comparison_with_first_evaluation": {
                  "first_evaluation_point": first_point,
                  "point_difference": {k: point[k] - first_point[k] for k in REPORTED},
                  "paired_bootstrap_95ci_of_difference": diff_ci.to_dict(orient="index"),
                  "note": "Paired over the same bootstrap samples. Reported for context; it did not choose the model."},
              "n_test": int(len(y)), "n_test_fraud": int(y.sum()), "n_boot": N_BOOT, "boot_seed": BOOT_SEED}
    (run / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def show(result, sel):
    import pandas as pd
    first = result["comparison_with_first_evaluation"]
    table = pd.DataFrame({
        "Ensemble OOF": {k: sel["oof_metrics"][k] for k in REPORTED},
        "Ensemble test": {k: result["point"][k] for k in REPORTED},
        "Ensemble 95% CI": {k: "{ci_low:.4f}–{ci_high:.4f}".format(**result["bootstrap_95ci"][k]) for k in REPORTED},
        "LightGBM test (first evaluation)": {k: first["first_evaluation_point"][k] for k in REPORTED},
        "Difference": first["point_difference"],
        "Difference 95% CI": {k: "{ci_low:+.4f} to {ci_high:+.4f}".format(**first["paired_bootstrap_95ci_of_difference"][k])
                              for k in REPORTED},
    })
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(table)
    p = result["point"]
    print(f"\nEnsemble at threshold {p['threshold']:.4f}: caught {p['tp']}, missed {p['fn']}, "
          f"false alerts {p['fp']}, flagged {p['tp'] + p['fp']}/{result['n_test']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true", help="perform the one permitted evaluation")
    args = parser.parse_args()
    sel = json.loads(SELECTION.read_text())
    run = ROOT / sel["run_dir"]
    if args.run:
        sel, run = check_allowed()
        now = lambda: datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat()
        record = {"status": "in_progress", "selected_run": sel["run_dir"], "started_at": now(),
                  "selection_record": str(SELECTION.relative_to(ROOT)),
                  "selection_commit": git("log", "-1", "--format=%H %cI", "--", str(SELECTION.relative_to(ROOT))).stdout.strip()}
        with MARKER.open("x") as stream:   # exclusive create: only one evaluation can start
            json.dump(record, stream, indent=2)
        try:
            result = evaluate(sel, run)
            record.update(status="completed", completed_at=now(),
                          test_metrics_sha256=sha(run / "test_metrics.json"),
                          test_predictions_sha256=sha(run / "test_predictions.csv"))
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            MARKER.write_text(json.dumps(record, indent=2) + "\n")
    else:
        completion = json.loads(MARKER.read_text()) if MARKER.exists() else {}
        if completion.get("status") != "completed":
            print("The ensemble evaluation has not been run. Use --run once, after the selection is committed.")
            return
        if sha(run / "test_metrics.json") != completion["test_metrics_sha256"]:
            raise ValueError("Saved ensemble test metrics changed after the evaluation.")
        result = json.loads((run / "test_metrics.json").read_text())
    show(result, sel)


if __name__ == "__main__":
    main()
