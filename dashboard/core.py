"""Model loading, input checks, scoring and explanations for the provider risk dashboard.

This module has no Streamlit dependency, so it can be tested and run directly:

    python -m dashboard.core        # self-check against the frozen held-out predictions

The frozen run is read from artifacts/shared/final_model_selection.json (its selected_run entry).
Set the environment variable FRAUD_DASHBOARD_RUN_DIR to a run folder to override that.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SELECTION_PATH = ARTIFACTS_DIR / "shared" / "final_model_selection.json"
POLICY_PATH = ARTIFACTS_DIR / "shared" / "exploratory_review_policy.json"
POLICY_EVAL_PATH = ARTIFACTS_DIR / "shared" / "exploratory_policy_supplementary_evaluation.json"
RUN_DIR_ENV = "FRAUD_DASHBOARD_RUN_DIR"
RUN_ID_PATTERN = re.compile(r"\d{8}_\d{6}_(?:full|quick)")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_ROWS = 50_000

# Saved pipelines may reference project modules (e.g. src.fold_vif), so the repo root
# must be importable before unpickling.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard import glossary  # noqa: E402

MODEL_NAMES = {
    "lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random forest",
    "lr_all": "Logistic regression (all features)",
    "lr_vif10": "Logistic regression (VIF < 10)", "lr_vif5": "Logistic regression (VIF < 5)",
}
STRATEGY_NAMES = {
    "original": "original class balance", "class_weight": "class weighting",
    "smote": "SMOTE-NC oversampling",
}
_DISTRIBUTIONS = {
    "sklearn": "scikit-learn", "imblearn": "imbalanced-learn", "lightgbm": "lightgbm",
    "xgboost": "xgboost", "numpy": "numpy", "pandas": "pandas",
}
_SELECTED_RUN_KEYS = ("selected_run", "selected_run_dir", "selected_run_id", "final_run", "final_run_dir")
_SELECTED_WORDS = ("select", "final", "chosen", "winner", "decision")


class DashboardError(Exception):
    """A problem the person running the dashboard can act on. The message is shown as-is."""


# --------------------------------------------------------------------------------------
# Locating the frozen run and checking it against the final-selection record
# --------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(value: str | Path) -> Path:
    """Resolve a path saved by a notebook (relative to notebooks/) or relative to the repo root."""
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    for base in (NOTEBOOK_DIR, REPO_ROOT):
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return (NOTEBOOK_DIR / p).resolve()


def _leaves(obj, path=()):
    """Yield (key path, value) for every leaf in a JSON structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, path + (str(k).lower(),))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _leaves(v, path)
    else:
        yield path, obj


def _run_dirs_in(value: str) -> set[Path]:
    """Run folders a string refers to, as a path or as a run ID. Free text is ignored."""
    dirs: set[Path] = set()
    if "/" in value or "\\" in value:
        try:
            candidate = resolve_path(value)
            if candidate.is_dir() and (candidate / "selected_metadata.json").exists():
                dirs.add(candidate.resolve())
            elif candidate.name == "selected_metadata.json" and candidate.exists():
                dirs.add(candidate.parent.resolve())
        except (OSError, ValueError):
            pass  # free text such as "recall/precision" is not a path
    for run_id in RUN_ID_PATTERN.findall(value):
        for d in ARTIFACTS_DIR.glob(f"*/{run_id}"):
            if (d / "selected_metadata.json").exists():
                dirs.add(d.resolve())
    return dirs


def find_run_dir(selection_path: Path = SELECTION_PATH) -> tuple[Path, str, dict | None]:
    """Return (run folder, how it was located, the final-selection record or None)."""
    override = os.environ.get(RUN_DIR_ENV)
    if override:
        run_dir = resolve_path(override)
        if not (run_dir / "selected_metadata.json").exists():
            raise DashboardError(f"{RUN_DIR_ENV} points to {run_dir}, which has no selected_metadata.json.")
        record = json.loads(selection_path.read_text()) if selection_path.exists() else None
        return run_dir, f"{RUN_DIR_ENV} override", record

    if not selection_path.exists():
        raise DashboardError(
            f"No final model selection found at {selection_path}. "
            f"Pull the latest repo changes, or set {RUN_DIR_ENV} to a run folder."
        )
    record = json.loads(selection_path.read_text())
    leaves = list(_leaves(record))

    # 1. The record's own selected_run entry, read exactly
    exact: set[Path] = set()
    for path, value in leaves:
        if isinstance(value, str) and any(k in _SELECTED_RUN_KEYS for k in path):
            exact |= _run_dirs_in(value)
    if len(exact) == 1:
        return exact.pop(), f"selected_run in {selection_path.name}", record
    if len(exact) > 1:
        listed = ", ".join(sorted(d.name for d in exact))
        raise DashboardError(f"selected_run in {selection_path.name} refers to several runs ({listed}).")

    # 2. Fallback for records without a selected_run entry: narrow down every run mentioned
    found: dict[Path, set[tuple]] = {}
    for path, value in leaves:
        if isinstance(value, str):
            for d in _run_dirs_in(value):
                found.setdefault(d, set()).add(path)
    if not found:
        raise DashboardError(
            f"{selection_path.name} does not point to a run folder that exists locally. "
            f"Check that the artifacts folder is present, or set {RUN_DIR_ENV}."
        )
    strings = {v.lower() for _, v in leaves if isinstance(v, str)}

    def meta_of(d):
        return json.loads((d / "selected_metadata.json").read_text())

    filters = [
        lambda d: any(any(w in key for w in _SELECTED_WORDS) for p in found[d] for key in p),
        lambda d: {str(meta_of(d).get("model", "")).lower(), str(meta_of(d).get("strategy", "")).lower()} <= strings,
        lambda d: (d / "test_metrics.json").exists(),
    ]
    candidates = list(found)
    for keep in filters:
        if len(candidates) == 1:
            break
        narrowed = [d for d in candidates if keep(d)]
        if narrowed:
            candidates = narrowed
    if len(candidates) == 1:
        return candidates[0], f"inferred from {selection_path.name} (no selected_run entry)", record
    listed = ", ".join(sorted(d.name for d in candidates))
    raise DashboardError(f"{selection_path.name} refers to several run folders ({listed}). Set {RUN_DIR_ENV}.")


def verify_selection(record: dict | None, meta: dict, pipeline_sha: str | None) -> list[tuple[str, str, str]]:
    """Compare the run with the final-selection record. Returns (check, status, detail) rows.

    status is "match", "not recorded" or "mismatch". The record's 'selected'/'final'/'decision'
    entries are preferred over values listed elsewhere in it (e.g. among candidates).
    """
    if record is None:
        return [("Final-selection record", "not recorded", "No final_model_selection.json was found.")]
    leaves = list(_leaves(record))
    preferred = [(p, v) for p, v in leaves if any(any(w in k for w in _SELECTED_WORDS) for k in p)]
    pool = preferred or leaves

    def values(key_test, kind):
        return [v for p, v in pool if p and key_test(p[-1]) and isinstance(v, kind) and not isinstance(v, bool)]

    def norm(text) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

    def same(recorded: str, code: str, display: str) -> bool:
        # Accept the code ("lgbm", "class_weight") or a display name ("LightGBM", "class weighting").
        r = norm(recorded)
        names = {norm(code), norm(display)} - {""}
        return any(n == r or n in r or r in n for n in names)

    checks = []
    for name, key, names in (("Model", "model", MODEL_NAMES), ("Strategy", "strategy", STRATEGY_NAMES)):
        recorded = sorted({str(v) for v in values(lambda k, key=key: k == key, str)})
        code = str(meta.get(key, ""))
        if not recorded:
            checks.append((name, "not recorded", f"The record does not state the {key}."))
        elif any(same(r, code, names.get(code, code)) for r in recorded):
            checks.append((name, "match", code))
        else:
            checks.append((name, "mismatch", f"record says {', '.join(recorded)}; run says {code}"))

    thresholds = [float(v) for v in values(lambda k: "threshold" in k, (int, float))]
    if not thresholds:
        checks.append(("Threshold", "not recorded", "The record does not state the threshold."))
    elif any(abs(t - float(meta["threshold"])) < 1e-9 for t in thresholds):
        checks.append(("Threshold", "match", f"{float(meta['threshold']):.6f}"))
    else:
        checks.append(("Threshold", "mismatch",
                       f"record says {thresholds[0]:.6f}; run says {float(meta['threshold']):.6f}"))

    hashes = [v for p, v in leaves if isinstance(v, str) and SHA256_PATTERN.match(v)
              and any("pipeline" in k for k in p)]
    if not hashes:
        checks.append(("Pipeline fingerprint", "not recorded", "The record does not state the pipeline hash."))
    elif pipeline_sha in hashes:
        checks.append(("Pipeline fingerprint", "match", pipeline_sha[:12] + "…"))
    else:
        checks.append(("Pipeline fingerprint", "mismatch", "the saved pipeline differs from the one in the record"))
    return checks


# --------------------------------------------------------------------------------------
# Loading the frozen run
# --------------------------------------------------------------------------------------
@dataclass
class ModelBundle:
    run_dir: Path
    source: str
    meta: dict
    pipeline: object | None
    pipeline_sha: str | None
    features: list[str]
    id_col: str
    target: str
    threshold: float
    restore_cols: list[str]
    binary_features: list[str]
    reference_scores: np.ndarray | None          # sorted out-of-fold scores of training providers
    reference_labels: np.ndarray | None
    train_stats: pd.DataFrame | None              # per-feature median/min/max of training providers
    impute_medians: pd.Series | None              # medians the pipeline's imputer actually uses
    test_metrics: dict | None
    checks: list[tuple[str, str, str]] = field(default_factory=list)
    version_mismatches: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return MODEL_NAMES.get(self.meta.get("model", ""), self.meta.get("model", "Model"))

    @property
    def strategy_name(self) -> str:
        return STRATEGY_NAMES.get(self.meta.get("strategy", ""), self.meta.get("strategy", ""))

    @property
    def run_id(self) -> str:
        return self.meta.get("run_id", self.run_dir.name)

    @property
    def can_score(self) -> bool:
        return self.pipeline is not None


def restore_missing(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Reset upstream mean-filled values to NaN where their <col>Missing flag is 1 (as in training)."""
    X = X.copy()
    for col in cols:
        flag = f"{col}Missing"
        if col in X.columns and flag in X.columns:
            X.loc[X[flag] == 1, col] = np.nan
    return X


def _installed_version(module: str) -> str | None:
    try:
        return importlib_metadata.version(_DISTRIBUTIONS.get(module, module))
    except importlib_metadata.PackageNotFoundError:
        return None


def load_bundle(run_dir: Path | None = None, require_pipeline: bool = True,
                selection_path: Path = SELECTION_PATH) -> ModelBundle:
    """Load the frozen run and verify it.

    With require_pipeline=False the saved evaluation is still loaded and checked, but the model
    itself is not, so the evaluation pages work on a machine without the model or its packages.
    """
    record = None
    if run_dir is None:
        run_dir, source, record = find_run_dir(selection_path)
    else:
        source = "given folder"
        if selection_path.exists():
            record = json.loads(selection_path.read_text())
    meta_path = run_dir / "selected_metadata.json"
    if not meta_path.exists():
        raise DashboardError(f"{meta_path} is missing.")
    meta = json.loads(meta_path.read_text())
    if meta.get("quick_run"):
        raise DashboardError(f"{run_dir.name} is a quick exploratory run, not a frozen final model.")

    expected = meta.get("fingerprints_sha256", {}).get("pipeline")
    if not expected:
        raise DashboardError("The run has no recorded pipeline fingerprint, so the saved model cannot be verified.")
    pipeline_path = run_dir / "selected_pipeline.joblib"
    pipeline = pipeline_sha = None
    if pipeline_path.exists():
        pipeline_sha = sha256(pipeline_path)
    if require_pipeline:
        if pipeline_sha is None:
            raise DashboardError(f"The saved model {pipeline_path.name} is missing from {run_dir.name}.")
        if pipeline_sha != expected:
            raise DashboardError("The saved model does not match the fingerprint recorded when it was frozen.")

    checks = [("Pipeline fingerprint in run metadata",
               "match" if pipeline_sha == expected else ("not recorded" if pipeline_sha is None else "mismatch"),
               "saved model matches the frozen fingerprint" if pipeline_sha == expected else "model file unavailable")]
    checks += verify_selection(record, meta, expected)

    test_metrics = None
    if (run_dir / "test_metrics.json").exists():
        test_metrics = json.loads((run_dir / "test_metrics.json").read_text())
        tested_at = test_metrics.get("point", {}).get("threshold")
        if tested_at is not None and abs(float(tested_at) - float(meta["threshold"])) > 1e-9:
            checks.append(("Held-out evaluation", "mismatch",
                           f"evaluated at threshold {float(tested_at):.6f}, run uses {float(meta['threshold']):.6f}"))
        else:
            checks.append(("Held-out evaluation", "match", "evaluated with this model's threshold"))
    else:
        checks.append(("Held-out evaluation", "not recorded", "no test_metrics.json in the run folder"))

    mismatched = [c for c in checks if c[1] == "mismatch" and not (c[0].startswith("Pipeline fingerprint in")
                                                                    and not require_pipeline)]
    if mismatched:
        details = "; ".join(f"{c[0]}: {c[2]}" for c in mismatched)
        raise DashboardError(f"The saved run does not match the final-selection record ({details}).")

    if require_pipeline:
        try:
            pipeline = joblib.load(pipeline_path)
        except ModuleNotFoundError as exc:
            raise DashboardError(f"Loading the model needs the package or module '{exc.name}'. "
                                 "Install the project requirements.") from exc

    mismatches = []
    for module, trained in meta.get("versions", {}).items():
        if module == "python":
            continue
        installed = _installed_version(module)
        if installed is not None and installed != trained:
            mismatches.append((module, trained, installed))

    features = list(meta["feature_order"])
    restore_cols = list(meta.get("restore_na_cols", []))

    reference_scores = reference_labels = None
    oof_path = run_dir / "oof_predictions.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        mask = (oof["model"] == meta.get("model")) & (oof["strategy"] == meta.get("strategy"))
        if mask.any():
            order = np.argsort(oof.loc[mask, "oof_prob"].to_numpy())
            reference_scores = oof.loc[mask, "oof_prob"].to_numpy()[order]
            reference_labels = oof.loc[mask, "y_true"].to_numpy()[order]

    train_stats = None
    train_path = resolve_path(meta["train_path"]) if meta.get("train_path") else None
    if train_path is not None and train_path.exists():
        wanted = set(features)
        train_X = pd.read_csv(train_path, usecols=lambda c: c in wanted)[features]
        train_X = restore_missing(train_X, restore_cols)
        train_stats = pd.DataFrame({"median": train_X.median(), "min": train_X.min(), "max": train_X.max()})

    impute_medians = None
    steps = getattr(pipeline, "named_steps", {}) if pipeline is not None else {}
    imputer = steps.get("impute") if hasattr(steps, "get") else None
    if imputer is not None and hasattr(imputer, "statistics_"):
        impute_medians = pd.Series(imputer.statistics_, index=features)

    return ModelBundle(
        run_dir=run_dir, source=source, meta=meta, pipeline=pipeline, pipeline_sha=pipeline_sha,
        features=features, id_col=meta.get("id_col", "Provider"), target=meta.get("target", "PotentialFraud"),
        threshold=float(meta["threshold"]), restore_cols=restore_cols,
        binary_features=list(meta.get("binary_features", [])),
        reference_scores=reference_scores, reference_labels=reference_labels,
        train_stats=train_stats, impute_medians=impute_medians, test_metrics=test_metrics,
        checks=checks, version_mismatches=mismatches,
    )


# --------------------------------------------------------------------------------------
# Reading and checking an input file
# --------------------------------------------------------------------------------------
@dataclass
class Validation:
    frame: pd.DataFrame | None                    # ID + numeric features, as supplied (IDs trimmed)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)   # one row per affected provider and column

    @property
    def ok(self) -> bool:
        return not self.errors

    def report(self) -> pd.DataFrame:
        cols = ["file_row", "provider", "column", "value", "problem", "severity"]
        return pd.DataFrame(self.issues, columns=cols)


def _preview(items, n=5) -> str:
    items = [str(i) for i in items]
    shown = ", ".join(items[:n])
    return shown + (f" and {len(items) - n} more" if len(items) > n else "")


def _plural(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def read_csv_bytes(data: bytes, id_col: str | None = None) -> pd.DataFrame:
    """Read an uploaded CSV, keeping provider IDs as text and rejecting duplicate column names."""
    if not data.strip():
        raise DashboardError("The file is empty.")
    head = data[:1 << 16].decode("utf-8-sig", errors="replace")
    try:
        header = [h.strip() for h in next(csv.reader(io.StringIO(head)))]
    except (StopIteration, csv.Error) as exc:
        raise DashboardError(f"The file could not be read as CSV ({exc}).") from exc
    dupes = sorted({h for h in header if header.count(h) > 1})
    if dupes:
        raise DashboardError(f"These column names appear more than once: {_preview(dupes)}. "
                             "Each column needs a unique name.")
    try:
        df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", dtype={id_col: str} if id_col else None,
                         keep_default_na=True)
    except Exception as exc:  # noqa: BLE001 - surface any parse failure to the user
        raise DashboardError(f"The file could not be read as CSV ({exc}).") from exc
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) > MAX_ROWS:
        raise DashboardError(f"The file has {len(df):,} rows; the dashboard scores up to {MAX_ROWS:,} at a time. "
                             "Split it into smaller files.")
    return df


def validate_input(df: pd.DataFrame, bundle: ModelBundle) -> Validation:
    """Reject impossible values; warn about plausible but unusual ones."""
    id_col, features = bundle.id_col, bundle.features
    v = Validation(frame=None)
    df = df.reset_index(drop=True)                                 # unique row labels, in file order
    file_row = pd.Series(np.arange(len(df)) + 2, index=df.index)   # row 1 is the header

    if df.empty:
        v.errors.append("The file has no rows.")
        return v
    if id_col not in df.columns:
        v.errors.append(f"The file needs a '{id_col}' column identifying each provider.")
    missing = [c for c in features if c not in df.columns]
    if missing:
        v.errors.append(f"{_plural(len(missing), 'model feature is', 'model features are')} missing: "
                        f"{_preview(missing, 8)}.")
    if v.errors:
        return v

    ids = df[id_col].astype("string").str.strip()
    shown_id = ids.fillna("(blank)").replace("", "(blank)")

    def flag_rows(mask: pd.Series, column: str, problem: str, severity: str, values=None) -> list[str]:
        idx = mask[mask].index
        for i in idx[:500]:
            v.issues.append({"file_row": int(file_row[i]), "provider": shown_id[i], "column": column,
                             "value": "" if values is None else str(values[i]), "problem": problem,
                             "severity": severity})
        return list(shown_id[idx])

    blank_id = ids.isna() | (ids == "")
    if blank_id.any():
        flag_rows(blank_id, id_col, "blank provider ID", "error", df[id_col])
        v.errors.append(f"{_plural(int(blank_id.sum()), 'row has', 'rows have')} a blank {id_col} "
                        f"(file rows {_preview(file_row[blank_id])}).")
    dupe = ids.duplicated(keep=False) & ~blank_id
    if dupe.any():
        flag_rows(dupe, id_col, "provider ID appears more than once", "error", df[id_col])
        v.errors.append(f"{_plural(ids[dupe].nunique(), 'provider ID appears', 'provider IDs appear')} more than "
                        f"once, after removing spaces: {_preview(ids[dupe].unique())}.")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    for c in features:
        bad = X[c].isna() & df[c].notna() & (df[c].astype("string").str.strip() != "")
        if bad.any():
            who = flag_rows(bad, c, "not a number", "error", df[c])
            v.errors.append(f"{glossary.label(c)} ({c}): {_plural(int(bad.sum()), 'value is not a number', 'values are not numbers')} "
                            f"({_preview(who)}).")
    for c in features:
        inf = np.isinf(X[c])
        if inf.any():
            who = flag_rows(inf, c, "infinite value", "error", X[c])
            v.errors.append(f"{glossary.label(c)} ({c}): infinite values ({_preview(who)}).")

    for c in features:
        unit = glossary.unit(c)
        if c in bundle.binary_features and unit == "number":
            unit = "flag"
        lo, hi = glossary.LIMITS[unit]
        col = X[c]
        if unit == "flag":
            blank = col.isna() & ~(df[c].notna() & (df[c].astype("string").str.strip() != ""))
            if blank.any():
                who = flag_rows(blank, c, "blank flag (must be 0 or 1)", "error")
                v.errors.append(f"{glossary.label(c)} ({c}): {_plural(int(blank.sum()), 'flag is', 'flags are')} "
                                f"blank; flags must be 0 or 1 ({_preview(who)}).")
            not01 = col.notna() & ~col.isin([0, 1])
            if not01.any():
                who = flag_rows(not01, c, "flag is not 0 or 1", "error", col)
                v.errors.append(f"{glossary.label(c)} ({c}): flags must be 0 or 1 ({_preview(who)}).")
            continue
        if lo is not None:
            low = col < lo
            if low.any():
                what = "negative" if lo == 0 else f"below {lo}"
                who = flag_rows(low, c, f"{what} value (impossible for {glossary.UNIT_NAMES[unit].lower()})",
                                "error", col)
                v.errors.append(f"{glossary.label(c)} ({c}): {_plural(int(low.sum()), 'provider has', 'providers have')} "
                                f"{what} values ({_preview(who)}).")
        if hi is not None:
            high = col > hi + 1e-9
            if high.any():
                limit = "1 (100%)" if unit == "proportion" else str(hi)
                who = flag_rows(high, c, f"above {limit} (impossible for {glossary.UNIT_NAMES[unit].lower()})",
                                "error", col)
                v.errors.append(f"{glossary.label(c)} ({c}): {_plural(int(high.sum()), 'provider has', 'providers have')} "
                                f"values above {limit} ({_preview(who)}).")
    if v.errors:
        return v

    # Plausible but unusual: warnings only
    for c in features:
        if glossary.unit(c) == "count":
            frac = X[c].notna() & (X[c] % 1 != 0)
            if frac.any():
                who = flag_rows(frac, c, "count is not a whole number", "warning", X[c])
                v.warnings.append(f"{glossary.label(c)} ({c}) is a count, but {_plural(int(frac.sum()), 'value is', 'values are')} "
                                  f"not whole numbers ({_preview(who)}).")
    blanks = X.isna()
    if blanks.any().any():
        cols = blanks.any()[lambda s: s].index.tolist()
        for c in cols:
            flag_rows(blanks[c], c, "blank value (filled with the training median)", "warning")
        v.warnings.append(
            f"{_plural(int(blanks.any(axis=1).sum()), 'provider has', 'providers have')} blank values in "
            f"{_preview(cols)}. The model fills blanks with training medians, so these scores are less reliable.")
    if bundle.train_stats is not None:
        restored = restore_missing(X, bundle.restore_cols)
        outside = restored.lt(bundle.train_stats["min"]) | restored.gt(bundle.train_stats["max"])
        n_out = int(outside.any(axis=1).sum())
        if n_out:
            for c in outside.columns[outside.any()]:
                flag_rows(outside[c], c, "outside the range seen in training", "warning", X[c])
            top = outside.sum().sort_values(ascending=False)
            top = top[top > 0].index.tolist()
            v.warnings.append(
                f"{_plural(n_out, 'provider has', 'providers have')} values outside the range seen in training "
                f"(most often {_preview(top, 3)}). Their scores extrapolate beyond the training data.")

    if bundle.target in df.columns:
        v.notes.append(f"The '{bundle.target}' column is ignored; scores use the model features only.")
    extra = [c for c in df.columns if c not in features and c not in (id_col, bundle.target)]
    if extra:
        v.notes.append(f"{_plural(len(extra), 'extra column is', 'extra columns are')} ignored: {_preview(extra)}.")

    v.frame = pd.concat([ids.astype(str).rename(id_col), X], axis=1).reset_index(drop=True)
    return v


def outside_training_range(frame: pd.DataFrame, bundle: ModelBundle) -> pd.Series | None:
    """Per feature, how many providers in the file fall outside the training range."""
    if bundle.train_stats is None:
        return None
    X = restore_missing(frame[bundle.features], bundle.restore_cols)
    outside = X.lt(bundle.train_stats["min"]) | X.gt(bundle.train_stats["max"])
    return outside.sum().sort_values(ascending=False)


# --------------------------------------------------------------------------------------
# Scoring and explanations
# --------------------------------------------------------------------------------------
@dataclass
class ScoredBatch:
    table: pd.DataFrame                  # id, score, percentile, flagged, rank (row order = input order)
    model_input: pd.DataFrame            # features after NaN restoration (what the pipeline receives)
    model_view: pd.DataFrame             # features after the pipeline's preprocessing (what the model sees)
    contributions: pd.DataFrame | None   # per-feature contributions on the log-odds scale
    base_value: float | None
    explain_note: str


def _model_and_preprocessed(pipeline, X: pd.DataFrame):
    steps = getattr(pipeline, "steps", None)
    if not steps:
        return pipeline, X
    Xt = X
    for _, step in steps[:-1]:
        if step is None or step == "passthrough":
            continue
        if hasattr(step, "fit_resample") and not hasattr(step, "transform"):
            continue  # samplers only act during training
        Xt = step.transform(Xt)
        if not isinstance(Xt, pd.DataFrame):
            names = step.get_feature_names_out() if hasattr(step, "get_feature_names_out") else X.columns
            Xt = pd.DataFrame(Xt, columns=list(names), index=X.index)
    return steps[-1][1], Xt


def _contributions(model, Xt: pd.DataFrame):
    module = type(model).__module__
    if module.startswith("lightgbm") and hasattr(model, "booster_"):
        values = np.asarray(model.booster_.predict(Xt, pred_contrib=True))
        return values, "TreeSHAP values from LightGBM (pred_contrib), on the log-odds scale."
    if module.startswith("xgboost") and hasattr(model, "get_booster"):
        import xgboost
        values = np.asarray(model.get_booster().predict(xgboost.DMatrix(Xt), pred_contribs=True))
        return values, "TreeSHAP values from XGBoost (pred_contribs), on the log-odds scale."
    try:
        import shap
        exp = shap.Explainer(model, Xt)(Xt)
        values = np.column_stack([exp.values, np.broadcast_to(exp.base_values, (len(Xt),))])
        return values, "SHAP values from the shap package."
    except Exception:  # noqa: BLE001 - explanations are optional
        return None, "Feature explanations are not available for this model type."


def score_batch(frame: pd.DataFrame, bundle: ModelBundle) -> ScoredBatch:
    if bundle.pipeline is None:
        raise DashboardError("The model is not loaded on this machine, so files cannot be scored.")
    ids = frame[bundle.id_col].to_numpy()
    X = restore_missing(frame[bundle.features], bundle.restore_cols)
    scores = bundle.pipeline.predict_proba(X)[:, 1]

    if bundle.reference_scores is not None:
        percentile = np.searchsorted(bundle.reference_scores, scores, side="left") / len(bundle.reference_scores) * 100
    else:
        percentile = np.full(len(scores), np.nan)

    table = pd.DataFrame({bundle.id_col: ids, "score": scores, "percentile": percentile,
                          "flagged": scores >= bundle.threshold})
    table["rank"] = table["score"].rank(ascending=False, method="first").astype(int)

    model, Xt = _model_and_preprocessed(bundle.pipeline, X)
    values, note = _contributions(model, Xt)
    contributions = base = None
    if values is not None and values.ndim == 2 and values.shape[1] == Xt.shape[1] + 1:
        contributions = pd.DataFrame(values[:, :-1], columns=list(Xt.columns))
        base = float(values[0, -1])
    elif values is not None:
        note = "Feature explanations had an unexpected shape and are not shown."

    return ScoredBatch(table=table, model_input=X.reset_index(drop=True), model_view=Xt.reset_index(drop=True),
                       contributions=contributions, base_value=base, explain_note=note)


def check_saved_predictions(table: pd.DataFrame, bundle: ModelBundle) -> dict:
    """Compare freshly computed held-out scores with the saved test_predictions.csv."""
    path = bundle.run_dir / "test_predictions.csv"
    if not path.exists():
        return {"status": "absent", "max_diff": None}
    saved = pd.read_csv(path, dtype={bundle.id_col: str})
    score_col = "score" if "score" in saved else "prob" if "prob" in saved else None
    if score_col is None:
        return {"status": "absent", "max_diff": None}
    merged = table.merge(saved[[bundle.id_col, score_col]].rename(columns={score_col: "saved"}),
                         on=bundle.id_col, how="inner")
    if len(merged) != len(table):
        return {"status": "differ", "max_diff": None, "detail": "provider IDs differ from the saved predictions"}
    diff = float(np.max(np.abs(merged["score"] - merged["saved"])))
    flags_differ = int(((merged["saved"] >= bundle.threshold) != merged["flagged"]).sum())
    status = "match" if diff <= 1e-9 and not flags_differ else "differ"
    return {"status": status, "max_diff": diff, "flags_differ": flags_differ}


def sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


def provider_drivers(batch: ScoredBatch, row: int, bundle: ModelBundle, n: int = 10) -> pd.DataFrame | None:
    """Top-n features by absolute contribution for one provider, with its values and the training median."""
    if batch.contributions is None:
        return None
    contrib = batch.contributions.iloc[row]
    order = contrib.abs().sort_values(ascending=False).index[:n]
    rows = []
    for f in order:
        raw = batch.model_input.at[row, f]
        used = batch.model_view.at[row, f] if f in batch.model_view.columns else raw
        median = bundle.train_stats.at[f, "median"] if bundle.train_stats is not None else np.nan
        rows.append({"feature": f, "contribution": float(contrib[f]), "value": raw, "value_used": used,
                     "was_missing": bool(pd.isna(raw)), "train_median": median})
    return pd.DataFrame(rows)


def waterfall(batch: ScoredBatch, row: int, n: int = 8) -> pd.DataFrame | None:
    """Steps from the model's average prediction to this provider's score, on the log-odds scale."""
    if batch.contributions is None or batch.base_value is None:
        return None
    contrib = batch.contributions.iloc[row]
    top = list(contrib.abs().sort_values(ascending=False).index[:n])
    rest = contrib.drop(top)
    steps = [("Average prediction", None, 0.0, "base")]
    steps += [(f, f, float(contrib[f]), "raise" if contrib[f] > 0 else "lower") for f in top]
    if len(rest):
        steps.append((f"Other {len(rest)} factors", None, float(rest.sum()), "raise" if rest.sum() > 0 else "lower"))
    out, level = [], batch.base_value
    for name, feature, delta, kind in steps:
        start = level if kind != "base" else batch.base_value
        end = start + delta if kind != "base" else batch.base_value
        out.append({"step": name, "feature": feature, "delta": delta, "start": start, "end": end, "kind": kind})
        level = end
    out.append({"step": "This provider", "feature": None, "delta": 0.0, "start": level, "end": level, "kind": "final"})
    return pd.DataFrame(out)


def driver_summary(batch: ScoredBatch, label, k: int = 2) -> list[str]:
    """Short text listing the k features that raise each provider's score most."""
    if batch.contributions is None:
        return [""] * len(batch.table)
    out = []
    for _, contrib in batch.contributions.iterrows():
        top = contrib[contrib > 0].sort_values(ascending=False).index[:k]
        out.append(", ".join(label(f) for f in top))
    return out


def batch_importance(batch: ScoredBatch) -> pd.Series | None:
    if batch.contributions is None:
        return None
    return batch.contributions.abs().mean().sort_values(ascending=False)


def workload_curve(bundle: ModelBundle, points: int = 200) -> pd.DataFrame | None:
    """Share of training providers flagged vs share of labelled positives caught, from out-of-fold scores."""
    if bundle.reference_scores is None or bundle.reference_labels is None:
        return None
    s, y = bundle.reference_scores, bundle.reference_labels
    thresholds = np.unique(np.concatenate([np.quantile(s, np.linspace(0, 1, points)), [bundle.threshold]]))
    rows = []
    for t in thresholds:
        flagged = s >= t
        rows.append({"threshold": t, "share_flagged": flagged.mean(),
                     "recall": (flagged & (y == 1)).sum() / max((y == 1).sum(), 1),
                     "precision": (flagged & (y == 1)).sum() / max(flagged.sum(), 1)})
    return pd.DataFrame(rows)


def threshold_percentile(bundle: ModelBundle, threshold: float | None = None) -> float | None:
    """Where a threshold (default: the frozen one) sits among training providers' out-of-fold scores (0-100)."""
    if bundle.reference_scores is None:
        return None
    t = bundle.threshold if threshold is None else threshold
    return float(np.searchsorted(bundle.reference_scores, t, side="left") / len(bundle.reference_scores) * 100)


def validation_outcome(bundle: ModelBundle, threshold: float) -> dict | None:
    """Caught / reviewed / recall / precision on training providers' out-of-fold scores at a threshold."""
    if bundle.reference_scores is None or bundle.reference_labels is None:
        return None
    s, y = bundle.reference_scores, bundle.reference_labels
    flagged = s >= threshold
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    return {"caught": tp, "missed": int((y == 1).sum()) - tp, "reviewed": tp + fp, "share_reviewed": (tp + fp) / len(s),
            "recall": tp / max(int((y == 1).sum()), 1), "precision": tp / max(tp + fp, 1)}


def recall_threshold(bundle: ModelBundle, target: float) -> float | None:
    """Threshold for a recall target on validation (out-of-fold) scores, with the rule used in 06b and 06d:
    the highest precision among thresholds whose recall is at least the target."""
    if bundle.reference_scores is None or bundle.reference_labels is None:
        return None
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(bundle.reference_labels, bundle.reference_scores)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= target
    if not ok.any():
        return None
    return float(thr[int(np.argmax(np.where(ok, prec, -1)))])


@dataclass
class WhatIf:
    """A recall target shown for discussion only: computed from validation scores, never committed or tested."""
    target_recall: float
    threshold: float

    @property
    def name(self) -> str:
        return f"{self.target_recall:.0%} validation target (what-if, validation only)"


def what_if_options(bundle: ModelBundle, targets=(0.90, 0.95), exclude: float | None = None) -> tuple[list[WhatIf], str | None]:
    """Validation-only what-if thresholds. Offered only if the same rule reproduces the frozen threshold."""
    if bundle.reference_scores is None:
        return [], None
    rule = bundle.meta.get("threshold_rule", {})
    if rule.get("rule") != "min_recall":
        return [], None
    reproduced = recall_threshold(bundle, float(rule["value"]))
    if reproduced is None or abs(reproduced - bundle.threshold) > 1e-12:
        return [], ("What-if thresholds are unavailable: the validation predictions do not reproduce the frozen "
                    "threshold with its recorded rule.")
    out = []
    for target in targets:
        if exclude is not None and abs(target - exclude) < 1e-9:
            continue
        threshold = recall_threshold(bundle, target)
        if threshold is not None and threshold < bundle.threshold:
            out.append(WhatIf(target_recall=target, threshold=threshold))
    return out, None


# --------------------------------------------------------------------------------------
# The exploratory review policy (notebook 06d), kept separate from the frozen final model
# --------------------------------------------------------------------------------------
@dataclass
class ReviewPolicy:
    threshold: float
    target_recall: float
    record: dict
    evaluation: dict | None

    @property
    def name(self) -> str:
        return f"Higher-recall policy \u2014 {self.target_recall:.0%} validation target"


def load_policy(bundle: ModelBundle, path: Path = POLICY_PATH,
                eval_path: Path = POLICY_EVAL_PATH) -> tuple[ReviewPolicy | None, str | None]:
    """Return (policy, problem). The policy is used only if it matches the frozen run exactly."""
    if not path.exists():
        return None, None
    try:
        record = json.loads(path.read_text())
        threshold, target = float(record["threshold"]), float(record["target_recall"])
    except (OSError, ValueError, KeyError) as exc:
        return None, f"The exploratory review policy file could not be read ({exc})."
    problems = []
    if record.get("pipeline_sha256") != bundle.meta.get("fingerprints_sha256", {}).get("pipeline"):
        problems.append("it was committed for a different model file")
    if record.get("run_id") != bundle.run_id:
        problems.append("it names a different run")
    if abs(float(record.get("original_threshold", -1)) - bundle.threshold) > 1e-12:
        problems.append("its original threshold differs from the frozen one")
    if not 0 < threshold < bundle.threshold:
        problems.append("its threshold is not below the original one")
    oof_path = bundle.run_dir / "oof_predictions.csv"
    if oof_path.exists() and record.get("oof_predictions_sha256") not in (None, sha256(oof_path)):
        problems.append("the validation predictions have changed since it was committed")
    evaluation = None
    if eval_path.exists():
        try:
            evaluation = json.loads(eval_path.read_text())
        except (OSError, ValueError):
            problems.append("its supplementary evaluation file could not be read")
        else:
            if abs(float(evaluation.get("threshold", -1)) - threshold) > 1e-12:
                problems.append("its supplementary evaluation used a different threshold")
    if problems:
        return None, "The exploratory review policy was not loaded: " + "; ".join(problems) + "."
    return ReviewPolicy(threshold=threshold, target_recall=target, record=record, evaluation=evaluation), None


def common_raisers(batch: ScoredBatch, mask: np.ndarray, k: int = 3) -> pd.Series | None:
    """How often each feature is among the top-k score-raising factors, across the selected providers."""
    if batch.contributions is None or not mask.any():
        return None
    counts: dict[str, int] = {}
    for _, contrib in batch.contributions[mask].iterrows():
        for f in contrib[contrib > 0].sort_values(ascending=False).index[:k]:
            counts[f] = counts.get(f, 0) + 1
    return pd.Series(counts, dtype=int).sort_values(ascending=False)


# --------------------------------------------------------------------------------------
# Self-check: the dashboard must reproduce the frozen held-out predictions exactly
# --------------------------------------------------------------------------------------
def self_check() -> int:
    bundle = load_bundle()
    print(f"Run: {bundle.run_dir.name} (located via {bundle.source})")
    print(f"Model: {bundle.model_name}, {bundle.strategy_name}; threshold {bundle.threshold:.6f}")
    for name, status, detail in bundle.checks:
        print(f"  {name}: {status} ({detail})")
    for module, trained, installed in bundle.version_mismatches:
        print(f"WARNING version mismatch: {module} trained {trained}, installed {installed}")

    test_path = resolve_path(bundle.meta["test_path"])
    validation = validate_input(read_csv_bytes(test_path.read_bytes(), bundle.id_col), bundle)
    if not validation.ok:
        print("Input errors:", *validation.errors, sep="\n  ")
        return 1
    batch = score_batch(validation.frame, bundle)
    print(f"Scored {len(batch.table)} providers; flagged {int(batch.table['flagged'].sum())}")

    status = 0
    saved = check_saved_predictions(batch.table, bundle)
    if saved["status"] == "absent":
        print("No test_predictions.csv in the run folder; skipped reproduction check.")
    else:
        print(f"Saved held-out predictions: {saved['status']} (max score difference {saved.get('max_diff')})")
        status |= saved["status"] != "match"

    if batch.contributions is not None:
        logit = np.log(batch.table["score"] / (1 - batch.table["score"]))
        gap = float(np.max(np.abs(batch.contributions.sum(axis=1) + batch.base_value - logit)))
        print(f"Explanations: {batch.explain_note} Max gap between contributions and score: {gap:.2e}")
        status |= gap > 1e-6
    else:
        print(f"Explanations: {batch.explain_note}")

    print("Self-check passed." if status == 0 else "Self-check FAILED.")
    return int(status)


if __name__ == "__main__":
    try:
        sys.exit(self_check())
    except DashboardError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
