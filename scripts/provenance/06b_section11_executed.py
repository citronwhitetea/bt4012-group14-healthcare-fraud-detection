import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

TEST_RUN_ID = ""           # e.g. "20261112_143000_full"
I_CONFIRM_FINAL = False    # set True only when model + threshold are final
N_BOOT = 2000
BOOT_SEED = 4012
BOOSTING_DIR = Path("../artifacts/boosting")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _metrics(y_true, p, t):
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    return {"threshold": float(t), "ap": float(average_precision_score(y_true, p)),
            "roc_auc": float(roc_auc_score(y_true, p)),
            "precision": float(precision_score(y_true, yhat, zero_division=0)),
            "recall": float(recall_score(y_true, yhat, zero_division=0)),
            "f1": float(f1_score(y_true, yhat, zero_division=0)),
            "flagged_share": float(yhat.mean()),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def run_final_test(run_dir: Path) -> pd.DataFrame:
    assert run_dir.is_dir(), f"No run folder {run_dir}"
    assert not (run_dir / "test_metrics.json").exists(), "This run has already been tested once."
    with open(run_dir / "selected_metadata.json") as f:
        meta = json.load(f)

    if meta["quick_run"]:
        raise ValueError("This is a quick exploratory run. Complete and freeze a full search before final testing.")
    if not meta.get("threshold_rationale", "").strip():
        raise ValueError("No threshold rationale recorded. Agree the rule with Swei and document it before testing.")
    fp = meta["fingerprints_sha256"]
    frozen_files = {"test": Path(meta["test_path"]),
                    "feature_sets": run_dir / "feature_sets.json",
                    "folds": run_dir / "cv_folds.csv",
                    "pipeline": run_dir / "selected_pipeline.joblib"}
    for key, path in frozen_files.items():
        if _sha256(path) != fp[key]:
            raise ValueError(f"{key} ({path}) changed after the model run was frozen.")

    pipe = joblib.load(run_dir / "selected_pipeline.joblib")
    thr, feats = meta["threshold"], meta["feature_order"]
    id_col, target = meta["id_col"], meta["target"]

    test = pd.read_csv(meta["test_path"])
    if not pd.api.types.is_numeric_dtype(test[target]):
        test[target] = test[target].map({"Yes": 1, "No": 0})
    assert test[target].isin([0, 1]).all() and test[id_col].is_unique
    for col in meta["restore_na_cols"]:
        test.loc[test[f"{col}Missing"] == 1, col] = np.nan

    y_t = test[target].to_numpy()
    p_t = pipe.predict_proba(test[feats])[:, 1]
    point = _metrics(y_t, p_t, thr)

    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y_t == 1)[0], np.where(y_t == 0)[0]
    boot = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = _metrics(y_t[idx], p_t[idx], thr)
        boot.append({k: m[k] for k in ["ap", "roc_auc", "precision", "recall", "f1"]})
    ci = pd.DataFrame(boot).quantile([0.025, 0.975]).T
    ci.columns = ["ci_low", "ci_high"]

    pd.DataFrame({id_col: test[id_col], "y_true": y_t, "score": p_t,
                  "flagged": (p_t >= thr).astype(int)}).to_csv(run_dir / "test_predictions.csv", index=False)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump({"point": point, "bootstrap_95ci": ci.to_dict(orient="index"),
                   "n_test": int(len(y_t)), "n_test_fraud": int(y_t.sum()),
                   "n_boot": N_BOOT, "boot_seed": BOOT_SEED}, f, indent=2)

    out = pd.DataFrame({"OOF (train CV)": meta["oof_metrics"], "Held-out test": point})
    return out.join(ci)


if I_CONFIRM_FINAL and TEST_RUN_ID:
    final_results = run_final_test(BOOSTING_DIR / TEST_RUN_ID)
    display(final_results)
else:
    print("Held-out test NOT run. Set TEST_RUN_ID and I_CONFIRM_FINAL = True, then run this cell alone.")