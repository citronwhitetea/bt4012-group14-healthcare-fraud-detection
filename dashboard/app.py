"""Provider risk review dashboard (BT4012 Group 14).

Run from the repository root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import hashlib
import html
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard import core, glossary  # noqa: E402

# --------------------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------------------
INK = "#14213D"        # headings, key numbers
BODY = "#2B3445"       # running text
MUTED = "#5C6676"      # secondary text
RULE = "#D9DEE5"       # hairlines and borders
GRID = "#EDF0F4"       # chart gridlines
FLAG = "#A65E0B"       # amber: flagged / raises the score
FLAG_TINT = "#FBEFD9"
FLAG_TEXT = "#7A4304"
BLUE = "#2E5E8E"       # app colour: logo, tabs, buttons, progress bars
BLUE_SOFT = "#8DAED0"
PANEL = "#F3F7FB"
PANEL_EDGE = "#E1E9F2"
CALM = "#3F72A6"       # below threshold / lowers the score
CALM_TINT = "#E3ECF6"
FONT = "Public Sans"
TOP_N_DRIVERS = 8
SCORE_DISCLAIMER = "Model score for prioritising review; not a validated probability of fraud."
CONTEXT_FEATURES = ["NumClaims", "PropInpatient", "AvgClaimAmt", "AvgClaimsPerBeneficiary"]
LOGO = (f'<svg width="38" height="38" viewBox="0 0 38 38" aria-hidden="true"><rect width="38" height="38" rx="10" '
        f'fill="{BLUE}"/><circle cx="17" cy="17" r="7" fill="none" stroke="#fff" stroke-width="2.6"/>'
        f'<path d="M22.3 22.3l6 6" stroke="#fff" stroke-width="2.8" stroke-linecap="round"/>'
        f'<circle cx="17" cy="17" r="2.2" fill="{FLAG_TINT}"/></svg>')

st.set_page_config(page_title="Provider risk review", page_icon="🩺", layout="wide")
alt.data_transformers.disable_max_rows()

# Streamlit is replacing use_container_width with width="stretch", one function at a time.
# Ask each function what it accepts instead of guessing from the version number.
_ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())


def _fill_kwargs(fn) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {"use_container_width": True}
    if "use_container_width" not in params:
        return {"width": "stretch"} if "width" in params else {}
    if "width" in params and _ST_VERSION >= (1, 50):
        return {"width": "stretch"}
    return {"use_container_width": True}


FILL = {name: _fill_kwargs(getattr(st, name))
        for name in ("dataframe", "altair_chart", "button", "download_button", "popover") if hasattr(st, name)}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"], .stApp [data-baseweb="tab"], .stApp [data-testid="stExpander"] summary {{
  font-family: '{FONT}', 'Helvetica Neue', Arial, sans-serif;
}}
.stApp {{ color: {BODY}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding: 2rem 3rem 3rem; max-width: none; }}
.stApp [data-baseweb="tab"] p {{ font-size: 0.98rem; font-weight: 600; }}
[data-testid="stHeaderActionElements"], .stApp h1 a, .stApp h2 a, .stApp h3 a {{ display: none !important; }}

.masthead {{ display: flex; align-items: center; gap: 0.85rem; }}
.masthead svg {{ flex: none; }}
.masthead h1 {{ font-size: 1.75rem; font-weight: 700; color: {INK}; margin: 0; padding: 0; letter-spacing: -0.01em; }}
.masthead .context {{ color: {MUTED}; font-size: 0.9rem; line-height: 1.45; margin-top: 0.15rem; }}
.rule {{ border-bottom: 1px solid {RULE}; margin: 0.6rem 0 0.3rem; }}
.pop-title {{ font-size: 1rem; font-weight: 600; color: {INK}; margin-bottom: 0.25rem; }}

.triage {{ margin: 1.1rem 0 1.2rem; background: {PANEL}; border: 1px solid {PANEL_EDGE}; border-radius: 12px;
  padding: 1.15rem 1.35rem 0.95rem; }}
.triage .sentence {{ font-size: 1.45rem; font-weight: 600; color: {INK}; line-height: 1.25; margin-bottom: 0.3rem; }}
.triage .detail {{ color: {BODY}; font-size: 0.95rem; line-height: 1.5; margin-bottom: 0.85rem; max-width: 58rem; }}
.triage .bar {{ display: flex; height: 16px; border-radius: 8px; overflow: hidden; background: #fff;
  box-shadow: inset 0 0 0 1px {PANEL_EDGE}; }}
.triage .bar .flagged {{ background: {FLAG}; }}
.triage .legend {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.3rem 1rem; margin-top: 0.4rem;
  font-size: 0.88rem; color: {MUTED}; }}
.triage .legend b {{ color: {INK}; font-weight: 600; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: 0; }}

.chip {{ display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
  white-space: nowrap; margin-right: 0.35rem; }}
.chip.flag {{ background: {FLAG_TINT}; color: {FLAG_TEXT}; }}
.chip.calm {{ background: {CALM_TINT}; color: {CALM}; }}
.chip.out {{ background: #fff; color: {MUTED}; box-shadow: inset 0 0 0 1px {RULE}; }}
.chip.short {{ background: {BLUE}; color: #fff; }}

.case-id {{ font-size: 1.45rem; font-weight: 700; color: {INK}; margin: 0.35rem 0 0; line-height: 1.2; word-break: break-all; }}
.case-line {{ color: {BODY}; margin: 0.45rem 0 0.2rem; line-height: 1.5; }}
.disclaimer {{ font-size: 0.82rem; color: {MUTED}; margin-bottom: 0.2rem; }}
.position {{ position: relative; height: 30px; margin: 0 0 0.1rem; }}
.position .track {{ position: absolute; top: 11px; left: 0; right: 0; height: 8px; border-radius: 4px; }}
.position .threshold {{ position: absolute; top: 3px; width: 2px; height: 24px; background: {INK}; margin-left: -1px; }}
.position .dot {{ position: absolute; top: 6px; width: 18px; height: 18px; margin-left: -9px; border-radius: 50%;
  border: 3px solid #fff; box-shadow: 0 0 0 1.5px {INK}; }}
.position-labels {{ position: relative; height: 1.2rem; font-size: 0.78rem; color: {MUTED}; }}
.position-labels span {{ position: absolute; white-space: nowrap; }}
.position-labels.top {{ margin-top: 0.7rem; color: {INK}; font-weight: 500; }}
.context-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.5rem; margin: 0.9rem 0 0.4rem; }}
.ctx {{ background: {PANEL}; border: 1px solid {PANEL_EDGE}; border-radius: 8px; padding: 0.5rem 0.7rem; }}
.ctx b {{ display: block; font-size: 1.05rem; color: {INK}; }}
.ctx span {{ font-size: 0.78rem; color: {MUTED}; line-height: 1.3; display: block; }}

.empty-title {{ font-size: 1.1rem; font-weight: 600; color: {INK}; margin: 0.2rem 0 0.3rem; }}
.reason-row {{ display: grid; grid-template-columns: 1fr 5.5rem; gap: 0.6rem; align-items: center; padding: 0.42rem 0;
  border-bottom: 1px solid {GRID}; font-size: 0.92rem; }}
.reason-row .meter {{ height: 6px; border-radius: 3px; background: {CALM_TINT}; overflow: hidden; }}
.reason-row .meter div {{ height: 100%; background: {FLAG}; }}
.reason-row small {{ color: {MUTED}; display: block; font-size: 0.78rem; }}

.section-title {{ font-size: 1.15rem; font-weight: 650; color: {INK}; margin: 1.6rem 0 0.3rem; }}
.section-sub {{ color: {BODY}; font-size: 0.95rem; line-height: 1.55; max-width: 58rem; margin-bottom: 1rem; }}
.sub-head {{ font-size: 0.95rem; font-weight: 600; color: {INK}; margin: 0.6rem 0 0.5rem; }}
.scope-note {{ font-size: 0.88rem; color: {BODY}; line-height: 1.5; background: {PANEL}; border-left: 3px solid {BLUE};
  border-radius: 0 8px 8px 0; padding: 0.6rem 0.9rem; margin: 0.9rem 0 0.3rem; max-width: 62rem; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.8rem; margin: 0.3rem 0 0.9rem; }}
.stat {{ background: {PANEL}; border: 1px solid {PANEL_EDGE}; border-radius: 10px; padding: 0.85rem 1rem; }}
.stat b {{ display: block; font-size: 1.6rem; font-weight: 700; color: {INK}; line-height: 1.15; }}
.stat span {{ font-size: 0.84rem; color: {MUTED}; }}
.stat.flag b {{ color: {FLAG_TEXT}; }}
.limits {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.9rem; margin-top: 0.4rem; }}
.limit {{ background: {PANEL}; border: 1px solid {PANEL_EDGE}; border-radius: 10px; padding: 0.95rem 1.05rem; }}
.limit b {{ display: block; color: {INK}; font-weight: 600; margin-bottom: 0.3rem; }}
.limit span {{ font-size: 0.9rem; color: {BODY}; line-height: 1.5; }}
.note {{ color: {MUTED}; font-size: 0.86rem; line-height: 1.5; max-width: 48rem; }}
.lead {{ font-size: 1.15rem; color: {INK}; font-weight: 500; line-height: 1.45; max-width: 52rem; margin: 1rem 0 0.4rem; }}

.outcome {{ border-collapse: separate; border-spacing: 6px; width: 100%; margin-left: -6px; border: none !important; }}
.outcome tr, .outcome th, .outcome td {{ border: none !important; }}
.outcome tr {{ background: transparent !important; }}
.outcome th {{ text-align: left; font-weight: 500; color: {MUTED}; font-size: 0.85rem; padding: 0 0.3rem; }}
.outcome td {{ border-radius: 8px; padding: 0.8rem 0.9rem; background: #F3F5F8; width: 38%; }}
.outcome td b {{ display: block; font-size: 1.5rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
.outcome td span {{ font-size: 0.82rem; color: {MUTED}; }}
.outcome td.caught {{ background: {FLAG_TINT}; }}
.outcome td.caught b {{ color: {FLAG_TEXT}; }}

@media (max-width: 900px) {{
  .block-container {{ padding: 1rem 1rem 2rem; }}
  .masthead h1 {{ font-size: 1.35rem; }}
  .masthead .context {{ display: none; }}
  .triage {{ padding: 0.8rem 0.9rem; margin: 0.6rem 0 0.8rem; }}
  .triage .sentence {{ font-size: 1.12rem; }}
  .triage .detail {{ display: none; }}
  .limits {{ grid-template-columns: 1fr; }}
  .context-grid {{ grid-template-columns: 1fr; }}
}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def esc(text) -> str:
    return html.escape(str(text))


def section(title: str, sub: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{esc(title)}</div>'
                + (f'<div class="section-sub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


def sub_head(title: str) -> None:
    st.markdown(f'<div class="sub-head">{esc(title)}</div>', unsafe_allow_html=True)


def note(text: str, top: float = 0) -> None:
    st.markdown(f'<div class="note" style="margin-top:{top}rem">{text}</div>', unsafe_allow_html=True)


def percentile_phrase(p: float) -> str:
    if np.isnan(p):
        return ""
    if p >= 99.5:
        return "in the top 1% of training providers"
    if p < 0.5:
        return "lower than almost every training provider"
    return f"higher than {p:.0f}% of training providers"


def run_date(run_id: str) -> str:
    try:
        d = datetime.strptime(run_id[:8], "%Y%m%d")
        return f"{d.day} {d:%b %Y}"
    except ValueError:
        return run_id


def style(chart):
    return (chart.configure(font=FONT)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=RULE,
                            tickColor=RULE, labelFontSize=12, titleFontSize=12, titleFontWeight=500,
                            labelLimit=260)
            .configure_legend(labelColor=BODY, labelFontSize=12, symbolType="square")
            .configure_view(strokeWidth=0))


def pick_one(label: str, options: list[str], key: str) -> str:
    """A segmented control that never ends up empty (falls back to a radio on old Streamlit)."""
    last_key = f"_{key}_last"

    def keep() -> None:
        if ss.get(key) is None:
            ss[key] = ss.get(last_key, options[0])

    if ss.get(key) not in options and key in ss:
        del ss[key]
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control(label, options, default=options[0], key=key, on_change=keep,
                                      label_visibility="collapsed")
    else:
        choice = st.radio(label, options, key=key, horizontal=True, label_visibility="collapsed")
    choice = choice if choice in options else ss.get(last_key, options[0])
    if choice not in options:
        choice = options[0]
    ss[last_key] = choice
    return choice


# --------------------------------------------------------------------------------------
# Loading the frozen model (or only its saved evaluation, if the model can't be loaded)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the final model")
def get_bundle(require_pipeline: bool) -> core.ModelBundle:
    return core.load_bundle(require_pipeline=require_pipeline)


ss = st.session_state
ss.setdefault("selected_provider", None)
ss.setdefault("table_version", 0)
ss.setdefault("upload_version", 0)
ss.setdefault("reviews", {})

model_error = None
try:
    bundle = get_bundle(True)
except core.DashboardError as exc:
    model_error = str(exc)
    try:
        bundle = get_bundle(False)
    except core.DashboardError as exc2:
        st.error(str(exc2))
        st.stop()

thr_pct = core.threshold_percentile(bundle)
n_train = bundle.meta.get("n_train") or 0
meta, tm = bundle.meta, bundle.test_metrics


def close_case() -> None:
    ss["selected_provider"] = None
    ss["table_version"] += 1          # a fresh table clears the row highlight


def open_case(provider_id: str) -> None:
    ss["selected_provider"] = provider_id
    ss["table_version"] += 1


def back_to_test_set() -> None:
    ss["upload_version"] += 1          # a fresh upload widget drops the uploaded file
    ss["selected_provider"] = None
    ss["table_version"] += 1


test_path = core.resolve_path(meta["test_path"]) if meta.get("test_path") else None
test_bytes = test_path.read_bytes() if test_path is not None and test_path.exists() else None
template = pd.DataFrame(columns=[bundle.id_col] + bundle.features).to_csv(index=False).encode()


@st.cache_data(show_spinner=False)
def sample_upload(source: bytes) -> bytes:
    """60 held-out providers without their labels: a realistic file to try the upload with."""
    df = core.read_csv_bytes(source, bundle.id_col).drop(columns=[bundle.target], errors="ignore")
    return df.sample(min(60, len(df)), random_state=4012).to_csv(index=False).encode()


# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
_valign = {"vertical_alignment": "center"} if "vertical_alignment" in inspect.signature(st.columns).parameters else {}
head_left, head_right = st.columns([4, 1.3], gap="large", **_valign)
with head_left:
    status_line = (f"{esc(bundle.model_name)} model, frozen {run_date(bundle.run_dir.name)} and tested once on "
                   f"held-out providers. Review threshold {bundle.threshold:.3f}.")
    if model_error:
        status_line += " The model is not loaded on this computer, so only the saved evaluation is shown."
    st.markdown(f'<div class="masthead">{LOGO}<div><h1>Provider risk review</h1><div class="context">'
                f'{status_line}</div></div></div>', unsafe_allow_html=True)
uploaded = None
with head_right:
    if bundle.can_score:
        panel = (st.popover("Score another file", **FILL.get("popover", {})) if hasattr(st, "popover")
                 else st.expander("Score another file"))
        with panel:
            st.markdown(
                '<div class="pop-title">Score another file</div><div class="note">A CSV with one row per provider: '
                f'a <b>{esc(bundle.id_col)}</b> column and the {len(bundle.features)} model features, laid out like '
                f'provider_features_test.csv. Up to {core.MAX_ROWS:,} rows. Label columns are ignored.</div>',
                unsafe_allow_html=True)
            uploaded = st.file_uploader("Provider file (CSV)", type="csv", key=f"upload-{ss['upload_version']}",
                                        label_visibility="collapsed")
            if test_bytes is not None:
                st.download_button("Download a sample to try", sample_upload(test_bytes),
                                   file_name="sample_providers.csv", mime="text/csv", **FILL["download_button"])
            st.download_button("Download a blank template", template, file_name="provider_template.csv",
                               mime="text/csv", **FILL["download_button"])
        if uploaded is not None:
            st.button("Back to the held-out test set", on_click=back_to_test_set, **FILL["button"])
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

if bundle.version_mismatches:
    details = "; ".join(f"{m} {t} (installed {i})" for m, t, i in bundle.version_mismatches)
    st.warning(f"Installed package versions differ from the ones the model was trained with: {details}. "
               "Scores may differ slightly from the evaluated model until the trained versions are installed.")


# --------------------------------------------------------------------------------------
# Scoring, with a visible progress panel the first time a file is scored
# --------------------------------------------------------------------------------------
def score_with_progress(data: bytes, name: str, is_test_set: bool) -> dict:
    store = ss.setdefault("_scored", {})
    key = hashlib.sha256(data).hexdigest()[:16] + "-" + (bundle.pipeline_sha or "")[:16]
    if key in store:
        return store[key]

    has_status = hasattr(st, "status")
    box = st.status(f"Scoring {name}", expanded=True) if has_status else st.spinner(f"Scoring {name}")
    result: dict = {"key": key, "errors": [], "warnings": [], "notes": [], "report": None}
    with box as status:
        def step(text: str) -> None:
            if has_status:
                st.write(text)

        def finish(label: str, state: str) -> None:
            if has_status:
                status.update(label=label, state=state, expanded=False)

        step(f"Reading {name}")
        try:
            frame = core.read_csv_bytes(data, bundle.id_col)
        except core.DashboardError as exc:
            result["errors"] = [str(exc)]
            finish(f"{name} could not be read", "error")
        else:
            step(f"Checking {len(frame):,} rows against the model's {len(bundle.features)} features")
            validation = core.validate_input(frame, bundle)
            result.update(errors=validation.errors, warnings=validation.warnings, notes=validation.notes,
                          report=validation.report() if validation.issues else None)
            if not validation.ok:
                finish(f"{name} needs fixes before it can be scored", "error")
            else:
                step(f"Scoring and explaining {len(validation.frame):,} providers")
                batch = core.score_batch(validation.frame, bundle)
                table = batch.table.copy()
                table["drivers"] = core.driver_summary(batch, glossary.label)
                result.update(batch=batch, table=table, frame=validation.frame,
                              scored_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                              outside=core.outside_training_range(validation.frame, bundle))
                if is_test_set:
                    step("Matching the saved held-out predictions")
                    result["saved_check"] = core.check_saved_predictions(table, bundle)
                finish(f"Scored {len(table):,} providers from {name}", "complete")

    store[key] = result
    while len(store) > 4:
        store.pop(next(iter(store)))
    return result


queue_state, result = "ok", {}
if not bundle.can_score:
    queue_state = "no_model"
elif uploaded is not None:
    data, data_name, data_label = uploaded.getvalue(), uploaded.name, f"providers in {uploaded.name}"
elif test_bytes is not None:
    data, data_name, data_label = test_bytes, test_path.name, "held-out test providers"
else:
    queue_state = "no_data"

if queue_state == "ok":
    result = score_with_progress(data, data_name, is_test_set=uploaded is None)
    if result["errors"]:
        queue_state = "errors"

batch: core.ScoredBatch | None = result.get("batch")
table: pd.DataFrame | None = result.get("table")
id_col = bundle.id_col
if table is not None:
    n_total, n_flag = len(table), int(table["flagged"].sum())
    ranked = table.sort_values("rank").reset_index(drop=True)
    reviews = ss["reviews"].setdefault(result["key"], {})
    if ss["selected_provider"] not in set(table[id_col]):
        ss["selected_provider"] = None
file_desc = ("the held-out test set" if uploaded is None else data_name) if queue_state == "ok" else "no file"

tab_queue, tab_model = st.tabs(["Review queue", "Model and evidence"])


# --------------------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------------------
def queue_unavailable() -> None:
    if queue_state == "no_model":
        st.markdown('<div class="lead">The model is not loaded on this computer, so providers cannot be scored.</div>',
                    unsafe_allow_html=True)
        st.error(model_error)
        note("The Model and evidence tab still shows the saved evaluation. To score providers, install the project "
             "requirements and make sure the run folder holds the original selected_pipeline.joblib.")
    elif queue_state == "no_data":
        st.markdown('<div class="lead">The held-out test file isn\'t on this computer.</div>', unsafe_allow_html=True)
        note(f"Expected at {esc(test_path)}. Run notebook 05 to create it, or use \"Score another file\" at the top right.")
    else:
        st.markdown(f'<div class="lead">{esc(data_name)} needs a few fixes before it can be scored.</div>',
                    unsafe_allow_html=True)
        for e in result["errors"]:
            st.error(e)
        if result.get("report") is not None:
            st.download_button("Download the list of problems (CSV)", result["report"].to_csv(index=False).encode(),
                               file_name=f"problems_{Path(data_name).stem}.csv", mime="text/csv")
        note("Correct the file and upload it again through \"Score another file\", or go back to the held-out test "
             "set. The report lists each affected provider, column and value.", top=0.6)


def triage_summary() -> None:
    share = n_flag / n_total
    top = f", about the top {100 - thr_pct:.0f}% of training providers," if thr_pct is not None else ""
    st.markdown(f"""
    <div class="triage">
      <div class="sentence">{n_flag:,} of {n_total:,} {esc(data_label)} are flagged for review</div>
      <div class="detail">Risk scores run from 0 to 1. Providers scoring {bundle.threshold:.3f} or more{top} are
        flagged. The score is for prioritising review, not a validated probability of fraud, and a flag is a reason
        to look, not a finding.</div>
      <div class="bar" role="img" aria-label="{share:.1%} of providers flagged">
        <div class="flagged" style="width:{share * 100:.2f}%"></div>
      </div>
      <div class="legend">
        <span><span class="swatch" style="background:{FLAG}"></span><b>{n_flag:,}</b> flagged ({share:.1%})</span>
        <span><span class="swatch" style="background:#fff; box-shadow: inset 0 0 0 1px {PANEL_EDGE}"></span>
          <b>{n_total - n_flag:,}</b> below the threshold</span>
      </div>
    </div>""", unsafe_allow_html=True)
    saved = result.get("saved_check")
    if saved and saved["status"] == "match":
        st.caption("These scores match the saved held-out predictions exactly, so this queue is the reported result.")
    elif saved and saved["status"] == "differ":
        st.warning("These scores differ from the saved held-out predictions. Check the installed package versions "
                   "before using this queue in a presentation.")


def data_checks() -> None:
    warnings, notes = result["warnings"], result["notes"]
    label = (f"Data checks: {len(warnings)} {'warning' if len(warnings) == 1 else 'warnings'} to review"
             if warnings else "Data checks passed")
    with st.expander(label, expanded=False):
        for w in warnings:
            st.warning(w)
        if not warnings:
            note("Every provider has the required columns, valid values, and values within the range seen in training.")
        for n in notes:
            st.caption(n)
        if result.get("report") is not None:
            st.download_button("Download the affected providers (CSV)", result["report"].to_csv(index=False).encode(),
                               file_name=f"data_checks_{Path(data_name).stem}.csv", mime="text/csv")


def position_bar(percentile: float, score: float) -> str:
    if np.isnan(percentile) or thr_pct is None:
        x, t, left, right = score * 100, bundle.threshold * 100, "Score 0", "Score 1"
    else:
        x, t, left, right = percentile, thr_pct, "Lowest scores", "Highest scores"
    color = FLAG if score >= bundle.threshold else CALM
    anchor = "transform:translateX(-100%)" if t > 75 else "transform:translateX(-50%)" if t > 25 else ""
    return f"""
    <div class="position-labels top"><span style="left:{t:.2f}%; {anchor}">Review threshold</span></div>
    <div class="position" role="img" aria-label="Position against training providers">
      <div class="track" style="background: linear-gradient(to right, {CALM_TINT} 0 {t:.2f}%, {FLAG_TINT} {t:.2f}% 100%)"></div>
      <div class="threshold" style="left:{t:.2f}%"></div>
      <div class="dot" style="left:{min(max(x, 1), 99):.2f}%; background:{color}"></div>
    </div>
    <div class="position-labels"><span style="left:0">{left}</span><span style="right:0">{right}</span></div>"""


def waterfall_chart(steps: pd.DataFrame):
    steps = steps.copy()
    steps["Step"] = [glossary.label(f) if isinstance(f, str) else s for s, f in zip(steps["step"], steps["feature"])]
    base_score, final_score = core.sigmoid(steps["start"].iloc[0]), core.sigmoid(steps["end"].iloc[-1])
    steps.loc[steps.index[0], "Step"] = f"Average prediction (score {base_score:.3f})"
    steps.loc[steps.index[-1], "Step"] = f"This provider (score {final_score:.3f})"
    steps["Effect"] = steps["kind"].map({"raise": "Raises the score", "lower": "Lowers the score",
                                         "base": "Start and end", "final": "Start and end"})
    steps["lo"], steps["hi"] = steps[["start", "end"]].min(axis=1), steps[["start", "end"]].max(axis=1)
    order = list(steps["Step"])
    y = alt.Y("Step:N", sort=order, title=None)
    bars = alt.Chart(steps[steps["kind"].isin(["raise", "lower"])]).mark_bar(size=16, cornerRadius=2).encode(
        x=alt.X("lo:Q", title="Log-odds of the score (each bar adds one factor's push)"), x2="hi:Q", y=y,
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises the score", "Lowers the score"], range=[FLAG, CALM]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Step:N", title="Factor"), alt.Tooltip("delta:Q", format="+.3f", title="Push")],
    )
    ends = alt.Chart(steps[steps["kind"].isin(["base", "final"])]).mark_tick(color=INK, thickness=3, size=22).encode(
        x="start:Q", y=y, tooltip=[alt.Tooltip("Step:N", title="")])
    return style((bars + ends).properties(height=30 * len(steps) + 40))


def set_review(provider_id: str, field: str, widget_key: str) -> None:
    entry = reviews.setdefault(provider_id, {"shortlisted": False, "note": ""})
    entry[field] = ss[widget_key]


def case_panel(view_ids: list[str]) -> None:
    pid = ss["selected_provider"]
    row = int(table.index[table[id_col] == pid][0])
    rec = table.loc[row]
    flagged = bool(rec["flagged"])
    review = reviews.get(pid, {"shortlisted": False, "note": ""})
    chips = ('<span class="chip flag">Flagged for review</span>' if flagged
             else '<span class="chip calm">Below the threshold</span>')
    if review["shortlisted"]:
        chips += '<span class="chip short">Shortlisted</span>'
    if pid not in view_ids:
        chips += '<span class="chip out">Outside current filter</span>'

    with st.container(border=True):
        head, close = st.columns([4, 1])
        with head:
            st.markdown(f'{chips}<div class="case-id">{esc(pid)}</div>', unsafe_allow_html=True)
        with close:
            st.button("Close", key="close_case", on_click=close_case, **FILL["button"])

        pct = rec["percentile"]
        pct_text = "" if np.isnan(pct) else f", {percentile_phrase(pct)}"
        st.markdown(f'<div class="case-line">Risk score <b>{rec["score"]:.3f}</b>{pct_text}. '
                    f'Ranked {int(rec["rank"]):,} of {n_total:,} in this file.</div>'
                    f'<div class="disclaimer">{SCORE_DISCLAIMER}</div>', unsafe_allow_html=True)
        st.markdown(position_bar(pct, rec["score"]), unsafe_allow_html=True)

        if pid in view_ids:
            i = view_ids.index(pid)
            prev_col, next_col = st.columns(2)
            prev_col.button("Previous in list", key="prev_case", disabled=i == 0, on_click=open_case,
                            args=(view_ids[max(i - 1, 0)],), **FILL["button"])
            next_col.button("Next in list", key="next_case", disabled=i == len(view_ids) - 1, on_click=open_case,
                            args=(view_ids[min(i + 1, len(view_ids) - 1)],), **FILL["button"])

        context = [f for f in CONTEXT_FEATURES if f in bundle.features]
        if context:
            sub_head("Provider at a glance")
            cells = []
            for f in context:
                median = (f"training median {glossary.fmt(f, bundle.train_stats.at[f, 'median'])}"
                          if bundle.train_stats is not None else "")
                cells.append(f'<div class="ctx"><b>{esc(glossary.fmt(f, batch.model_input.at[row, f]))}</b>'
                             f'<span>{esc(glossary.label(f))}<br>{esc(median)}</span></div>')
            st.markdown(f'<div class="context-grid">{"".join(cells)}</div>', unsafe_allow_html=True)

        sub_head("Your review")
        short_key, note_key = f"short-{result['key']}-{pid}", f"note-{result['key']}-{pid}"
        st.checkbox("Add to shortlist", value=review["shortlisted"], key=short_key,
                    on_change=set_review, args=(pid, "shortlisted", short_key))
        st.text_area("Review note", value=review["note"], key=note_key, height=80,
                     placeholder="What to check, who to ask, what you found…",
                     on_change=set_review, args=(pid, "note", note_key))
        st.caption("Your shortlist and notes are kept for this session and included in the download. "
                   "They are separate from the model's flags.")

        sub_head("Why it scored here")
        steps = core.waterfall(batch, row, n=TOP_N_DRIVERS)
        drivers = core.provider_drivers(batch, row, bundle, n=TOP_N_DRIVERS)
        if steps is None or drivers is None:
            st.info(batch.explain_note)
            return
        st.altair_chart(waterfall_chart(steps), **FILL["altair_chart"])
        note("The chart starts at the model's average prediction and adds each factor's push (TreeSHAP) until it "
             "reaches this provider's score. Pushes add up on the log-odds scale, so they are measured against the "
             "model's average prediction, not against the training median shown below.")
        drivers["Factor"] = drivers["feature"].map(glossary.label)
        drivers["This provider"] = [
            f"missing (model used {glossary.fmt(f, u)})" if m else glossary.fmt(f, v)
            for f, v, u, m in zip(drivers["feature"], drivers["value"], drivers["value_used"], drivers["was_missing"])
        ]
        drivers["Training median (all providers)"] = [glossary.fmt(f, m) for f, m in
                                                      zip(drivers["feature"], drivers["train_median"])]
        drivers["Push"] = drivers["contribution"].map(lambda x: f"{x:+.3f}")
        st.dataframe(drivers[["Factor", "This provider", "Training median (all providers)", "Push"]],
                     hide_index=True, **FILL["dataframe"])
        note("The training median is descriptive context across all providers, not a comparable peer group: a "
             "high-volume inpatient provider can legitimately differ from it.", top=0.3)
        if drivers["feature"].map(glossary.is_missing_flag).any():
            note("One of these factors is a missing-data flag. Such flags often reflect a provider's mix of claims, "
                 "for example no primary procedure codes, rather than suspicious behaviour on their own.", top=0.5)
        with st.expander(f"All {len(bundle.features)} values for {pid}"):
            st.dataframe(pd.DataFrame({
                "Factor": [glossary.label(f) for f in bundle.features],
                "This provider": [glossary.fmt(f, batch.model_input.at[row, f]) for f in bundle.features],
                "Training median (all providers)": [glossary.fmt(f, bundle.train_stats.at[f, "median"])
                                                    if bundle.train_stats is not None else "n/a"
                                                    for f in bundle.features],
                "Unit": [glossary.UNIT_NAMES[glossary.unit(f)] for f in bundle.features],
                "What it measures": [glossary.description(f) for f in bundle.features],
            }), hide_index=True, **FILL["dataframe"])


def empty_panel() -> None:
    with st.container(border=True):
        st.markdown('<div class="empty-title">Select a provider</div>'
                    '<div class="note">Click a row in the list to open its case: the score, provider context, the '
                    'factors behind the score, and space for your review notes.</div>', unsafe_allow_html=True)
        reasons = core.common_raisers(batch, table["flagged"].to_numpy(), k=3)
        if reasons is None or reasons.empty:
            return
        sub_head("Most common reasons for a flag")
        rows = []
        for f, count in reasons.head(6).items():
            rows.append(
                f'<div class="reason-row"><div>{esc(glossary.label(f))}'
                f'<small>A top-three factor for {count:,} of {n_flag:,} flags</small></div>'
                f'<div class="meter" aria-hidden="true"><div style="width:{count / n_flag * 100:.1f}%"></div></div></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)


def export_frame(view: pd.DataFrame) -> pd.DataFrame:
    out = view.rename(columns={"score": "risk_score", "percentile": "percentile_vs_training",
                               "flagged": "flagged_for_review", "drivers": "top_factors"})
    out["shortlisted"] = [reviews.get(p, {}).get("shortlisted", False) for p in out[id_col]]
    out["review_note"] = [reviews.get(p, {}).get("note", "") for p in out[id_col]]
    out["model"] = bundle.model_name
    out["run_id"] = bundle.run_id
    out["review_threshold"] = bundle.threshold
    out["source_file"] = data_name
    out["scored_at"] = result.get("scored_at", "")
    return out


with tab_queue:
    if queue_state != "ok":
        queue_unavailable()
    else:
        triage_summary()
        shortlisted = {p for p, r in reviews.items() if r.get("shortlisted")}
        # Fixed labels: a label that changes (e.g. with a count) would reset the filter.
        filters = {"Flagged": "flagged", "Below threshold": "below", "Shortlist": "short", "All": "all"}
        c_scope, c_search, c_download = st.columns([3, 2.2, 1.4], gap="medium")
        with c_scope:
            st.markdown('<div class="sub-head" style="margin-top:0">Show</div>', unsafe_allow_html=True)
            scope = filters[pick_one("Show", list(filters), key="scope")]
            st.caption(f"{n_flag:,} flagged, {n_total - n_flag:,} below the threshold, "
                       f"{len(shortlisted):,} shortlisted")
        with c_search:
            query = st.text_input("Find a provider", placeholder="Type part of a provider ID", key="query").strip()

        view = {"flagged": ranked[ranked["flagged"]], "below": ranked[~ranked["flagged"]],
                "short": ranked[ranked[id_col].isin(shortlisted)], "all": ranked}[scope]
        if query:
            view = view[view[id_col].str.contains(query, case=False, regex=False)]
        view_ids = view[id_col].tolist()

        with c_download:
            st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
            st.download_button(f"Download these {len(view):,} (CSV)", export_frame(view).to_csv(index=False).encode(),
                               file_name=f"review_queue_{Path(data_name).stem}_{scope}{'_search' if query else ''}.csv",
                               mime="text/csv", disabled=view.empty, **FILL["download_button"],
                               help="Exports the providers listed below with scores, top factors, your shortlist and "
                                    "notes, plus the model, run ID, threshold and scoring time.")
        data_checks()

        left, right = st.columns([1.35, 1], gap="large")
        with left:
            if view.empty:
                empty_msg = {"short": "Your shortlist is empty. Open a provider and tick \"Add to shortlist\"."}
                st.info(empty_msg.get(scope, "No providers match. Clear the search box or choose another filter."))
            else:
                display = pd.DataFrame({
                    "Rank": view["rank"].to_numpy(),
                    "Provider": view[id_col].to_numpy(),
                    "Risk score": view["score"].to_numpy(),
                    "Percentile": view["percentile"].to_numpy(),
                    "Status": np.where(view["flagged"], "Flagged", "Below"),
                    "Shortlist": ["Yes" if p in shortlisted else "" for p in view[id_col]],
                    "Top factors": view["drivers"].to_numpy(),
                })
                config = {
                    "Rank": st.column_config.NumberColumn(width="small"),
                    "Provider": st.column_config.TextColumn(width="small"),
                    "Risk score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f",
                                                                  width="medium", help=SCORE_DISCLAIMER),
                    "Percentile": st.column_config.NumberColumn(
                        format="%.0f", width="small",
                        help="Share of training providers (out-of-fold scores) with a lower score."),
                    "Status": st.column_config.TextColumn(width="small", help="Flagged = at or above the review threshold."),
                    "Shortlist": st.column_config.TextColumn(width="small", help="Your shortlist, separate from model flags."),
                    "Top factors": st.column_config.TextColumn(help="The two factors that raise the score most."),
                }
                table_key = f"queue-{scope}-{query}-{ss['table_version']}"
                try:
                    event = st.dataframe(display, column_config=config, hide_index=True, height=640,
                                         on_select="rerun", selection_mode="single-row", key=table_key,
                                         **FILL["dataframe"])
                    rows = list(event.selection.rows) if event is not None else []
                    last = ss.get("_last_pick")
                    if rows and last != (table_key, rows[0]):
                        ss["_last_pick"] = (table_key, rows[0])
                        ss["selected_provider"] = display["Provider"].iloc[rows[0]]
                    elif not rows and last is not None and last[0] == table_key:
                        ss["_last_pick"] = None
                        ss["selected_provider"] = None
                except TypeError:  # Streamlit < 1.35: no row selection
                    st.dataframe(display, column_config=config, hide_index=True, height=640, **FILL["dataframe"])
                    pick = st.selectbox("Open a provider", ["Choose a provider"] + view_ids, key="fallback_pick")
                    if pick != "Choose a provider":
                        ss["selected_provider"] = pick
        with right:
            if ss["selected_provider"] is None:
                empty_panel()
            else:
                case_panel(view_ids)


# --------------------------------------------------------------------------------------
# Model and evidence: one topic at a time
# --------------------------------------------------------------------------------------
MODEL_SECTIONS = ["Performance", "Review threshold", "This file", "Features", "Limitations", "Run details"]


def rule_sentence() -> str:
    rule = meta.get("threshold_rule", {})
    v = rule.get("value", 0)
    if rule.get("rule") == "min_recall":
        return (f"We chose the threshold with the highest out-of-fold validation precision while keeping recall at "
                f"or above {v:.0%}, then froze it before final testing. It does not guarantee {v:.0%} recall on new data.")
    if rule.get("rule") == "review_share":
        return f"We set the threshold to flag the top {v:.0%} of providers in validation, then froze it before final testing."
    if rule.get("rule") == "f1":
        return "We chose the threshold with the best out-of-fold validation F1, then froze it before final testing."
    return "The threshold was chosen on validation data and frozen before final testing."


def model_performance() -> None:
    n_test = (tm or {}).get("n_test")
    tested = f", then frozen and tested once on {n_test:,} held-out providers" if n_test else ""
    section("How the model performed on providers it had never seen",
            f"{esc(bundle.model_name)}, trained on {n_train:,} providers with {esc(bundle.strategy_name)}. It was "
            f"chosen from 18 candidate models using validation results only{tested}. These results come from that "
            "one-time evaluation, so they do not change with the file you score.")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        sub_head("Held-out test outcome")
        if not tm:
            st.info("The held-out test results are not in this run folder.")
        else:
            p = tm["point"]
            pos, neg = p["tp"] + p["fn"], p["fp"] + p["tn"]
            st.markdown(f"""
            <table class="outcome">
              <tr><th></th><th>Flagged</th><th>Not flagged</th></tr>
              <tr><th>Dataset label: potential fraud ({pos:,})</th>
                  <td class="caught"><b>{p['tp']:,}</b><span>caught</span></td>
                  <td><b>{p['fn']:,}</b><span>missed</span></td></tr>
              <tr><th>Dataset label: no potential fraud ({neg:,})</th>
                  <td><b>{p['fp']:,}</b><span>flagged, label says no</span></td>
                  <td><b>{p['tn']:,}</b><span>not flagged, label says no</span></td></tr>
            </table>""", unsafe_allow_html=True)
            note(f"Reviewing {p['tp'] + p['fp']:,} providers ({p['flagged_share']:.1%} of the test set) would reach "
                 f"{p['tp']:,} of the {pos:,} providers labelled potential fraud. About "
                 f"{p['fp'] / max(p['tp'] + p['fp'], 1):.0%} of flags carry a negative dataset label. Dataset labels "
                 "are not proof of misconduct either way, so every flag needs a person to review it.", top=0.7)
    with right:
        sub_head("Measures")
        oof, ci = meta.get("oof_metrics", {}), (tm or {}).get("bootstrap_95ci", {})
        rows = []
        for key, name, pct in [("recall", "Labelled providers caught (recall)", True),
                               ("precision", "Flags with a positive label (precision)", True),
                               ("f1", "F1", True), ("ap", "Average precision", False),
                               ("roc_auc", "ROC-AUC", False)]:
            fmt = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.3f}")
            test_val = tm["point"].get(key) if tm else None
            interval = ci.get(key)
            rows.append({"Measure": name,
                         "Validation (out-of-fold)": fmt(oof[key]) if key in oof else "n/a",
                         "Held-out test": fmt(test_val) if test_val is not None else "not run",
                         "Test range (95%)": f"{fmt(interval['ci_low'])} to {fmt(interval['ci_high'])}" if interval else "n/a"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, **FILL["dataframe"])
        note("Validation figures come from cross-validation on the training providers. Test ranges come from 2,000 "
             "bootstrap resamples. Average precision summarises ranking quality across every threshold; it is not accuracy.")


def model_threshold() -> None:
    section("How the review threshold was set",
            f"The threshold is <b>{bundle.threshold:.3f}</b>. {rule_sentence()} The held-out test then measured how "
            "it performs on new providers.")
    curve = core.workload_curve(bundle)
    if curve is None:
        st.info("The training providers' out-of-fold scores are not in this run folder.")
        return
    frozen = curve.iloc[(curve["threshold"] - bundle.threshold).abs().argmin()]
    st.markdown(f"""
    <div class="stats">
      <div class="stat flag"><b>{frozen['share_flagged']:.0%}</b><span>of training providers flagged (validation)</span></div>
      <div class="stat"><b>{frozen['recall']:.0%}</b><span>of labelled providers caught (validation)</span></div>
      <div class="stat"><b>{frozen['precision']:.0%}</b><span>of flags carry a positive label (validation)</span></div>
    </div>""", unsafe_allow_html=True)
    line = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=2.2).encode(
        x=alt.X("share_flagged:Q", title="Share of providers flagged for review", axis=alt.Axis(format="%")),
        y=alt.Y("recall:Q", title="Labelled providers caught", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("threshold:Q", format=".3f", title="Threshold"),
                 alt.Tooltip("share_flagged:Q", format=".1%", title="Flagged"),
                 alt.Tooltip("recall:Q", format=".1%", title="Caught"),
                 alt.Tooltip("precision:Q", format=".1%", title="Precision")])
    point = alt.Chart(pd.DataFrame([frozen])).mark_point(color=FLAG, size=150, filled=True, opacity=1).encode(
        x="share_flagged:Q", y="recall:Q")
    label = alt.Chart(pd.DataFrame([frozen])).mark_text(align="left", dx=12, dy=14, color=FLAG_TEXT, fontWeight=600,
                                                        font=FONT).encode(
        x="share_flagged:Q", y="recall:Q", text=alt.value("Chosen threshold"))
    st.altair_chart(style((line + point + label).properties(height=360)), **FILL["altair_chart"])
    st.caption("Each point on the line is a possible threshold, from the training providers' out-of-fold scores. "
               "Flagging more providers catches more labelled ones at the cost of more reviews. Hover to compare.")


def model_this_file() -> None:
    if queue_state != "ok":
        section("This file", "No file is scored on this computer at the moment, so there is nothing to describe. "
                             "See the Review queue tab for the reason.")
        return
    section(f"This file: {file_desc}",
            "Everything in this section is calculated from the providers currently loaded, so it changes when you "
            "score another file. Labels are never used here.")
    share_file = n_flag / n_total
    share_train = (100 - thr_pct) / 100 if thr_pct is not None else None
    train_block = (f'<div class="stat"><b>{share_train:.1%}</b><span>flagged among training providers '
                   '(reference)</span></div>' if share_train is not None else "")
    st.markdown(f"""
    <div class="stats">
      <div class="stat"><b>{n_total:,}</b><span>providers scored</span></div>
      <div class="stat flag"><b>{share_file:.1%}</b><span>flagged in this file</span></div>
      {train_block}
    </div>""", unsafe_allow_html=True)
    if share_train is not None:
        note("A similar flagged share alone does not show that the providers or their feature values are similar to "
             "the training data. The feature-range check below looks at the values themselves.")

    left, right = st.columns([1, 1], gap="large")
    with left:
        sub_head("Where this file's providers fall among training providers")
        pct = table["percentile"].to_numpy()
        if np.isnan(pct).all():
            st.info("Percentiles need the training providers' out-of-fold scores, which are not in this run folder.")
        else:
            band = np.clip((pct // 10).astype(int), 0, 9)
            dist = (pd.DataFrame({"band": band, "Status": np.where(table["flagged"], "Flagged", "Below the threshold")})
                    .groupby(["band", "Status"]).size().rename("n").reset_index())
            dist["share"] = dist["n"] / len(table)
            dist["Band"] = dist["band"].map(lambda b: f"{10 * b}–{10 * b + 10}")
            bands = [f"{10 * b}–{10 * b + 10}" for b in range(10)]
            bars = alt.Chart(dist).mark_bar().encode(
                x=alt.X("Band:N", sort=bands, scale=alt.Scale(domain=bands), axis=alt.Axis(labelAngle=0),
                        title="Percentile band among training providers"),
                y=alt.Y("share:Q", stack="zero", title="Share of this file", axis=alt.Axis(format="%")),
                color=alt.Color("Status:N", scale=alt.Scale(domain=["Below the threshold", "Flagged"],
                                                            range=[BLUE_SOFT, FLAG]),
                                legend=alt.Legend(orient="top", title=None)),
                order=alt.Order("Status:N", sort="ascending"),
                tooltip=[alt.Tooltip("Band:N", title="Percentile band"), alt.Tooltip("Status:N"),
                         alt.Tooltip("n:Q", title="Providers"), alt.Tooltip("share:Q", format=".1%", title="Share")])
            even = alt.Chart(pd.DataFrame({"y": [0.1]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
            st.altair_chart(style((bars + even).properties(height=300)), **FILL["altair_chart"])
            st.caption("Each bar is split by each provider's actual flag. The review threshold sits at about the "
                       f"{thr_pct:.1f}th percentile, inside its band. If the file's scores were spread like the training "
                       "providers', each band would hold about 10% (dashed line).")
        sub_head("Values outside the training range")
        outside = result.get("outside")
        if outside is None:
            st.info("The training data is not on this computer, so ranges cannot be compared.")
        elif outside.sum() == 0:
            note("No provider in this file has a value outside the range seen in training.")
        else:
            top = outside[outside > 0].head(8)
            st.dataframe(pd.DataFrame({"Factor": [glossary.label(f) for f in top.index],
                                       "Providers outside the training range": top.to_numpy()}),
                         hide_index=True, **FILL["dataframe"])
    with right:
        sub_head("Factors the model leans on in this file")
        imp = core.batch_importance(batch)
        if imp is None:
            st.info(batch.explain_note)
        else:
            top = imp.head(10).rename_axis("feature").reset_index(name="mean_abs")
            top["Factor"] = top["feature"].map(glossary.label)
            chart = alt.Chart(top).mark_bar(color=BLUE, cornerRadiusEnd=3, size=16).encode(
                x=alt.X("mean_abs:Q", title="Average push on the score (log-odds)"),
                y=alt.Y("Factor:N", sort=list(top["Factor"]), title=None),
                tooltip=["Factor", alt.Tooltip("mean_abs:Q", format=".3f", title="Average push")]).properties(height=300)
            st.altair_chart(style(chart), **FILL["altair_chart"])
            st.caption("Average size of each factor's push across the providers in this file. A large value means "
                       "scores move with that factor, not that it causes fraud.")


def model_features() -> None:
    section("Feature dictionary",
            f"The {len(bundle.features)} provider-level features the model uses, with their units and how blank values "
            "are handled. Hover over a column heading in the queue for a short description.")
    st.dataframe(pd.DataFrame({
        "Factor": [glossary.label(f) for f in bundle.features],
        "Column": bundle.features,
        "Unit": [glossary.UNIT_NAMES[glossary.unit(f)] for f in bundle.features],
        "What it measures": [glossary.description(f) for f in bundle.features],
        "If blank": [glossary.missing_rule(f, bundle.restore_cols) for f in bundle.features],
    }), hide_index=True, height=600, **FILL["dataframe"])
    note("Uploads with impossible values (negative counts or amounts, shares outside 0–100%, flags other than 0 or 1) "
         "are rejected. Unusual but possible values, such as a value outside the training range, are scored with a "
         "warning.", top=0.5)


LIMITS = [
    ("A proxy label", "The potential-fraud label comes from the dataset. It is not a finding of misconduct, so every "
                      "flag needs human review."),
    ("Recall can vary", "The threshold kept validation recall at or above 80%. Held-out recall was higher, but its "
                        "range is wide, so a new file can land below 80%."),
    ("Associations, not causes", "Factors show what the model responded to. Missing-data flags in particular can stand "
                                 "in for a provider's mix of claims."),
    ("Network information", "Physician-network factors describe links across the provider population. Scoring a new "
                            "provider assumes comparable network information is available."),
    ("Similar providers only", "Scores are most reliable for providers like the training data. A similar flagged share "
                               "does not by itself show that a new file is similar."),
    ("Earlier full-data exploration", "Exploratory analysis used the full provider dataset, so the held-out result is the "
                                      "least biased estimate available rather than a perfectly isolated one."),
]


def model_limits() -> None:
    section("Reading the scores responsibly", "What the model can and cannot tell you. These apply to every file.")
    st.markdown('<div class="limits">' + "".join(f'<div class="limit"><b>{esc(t)}</b><span>{esc(d)}</span></div>'
                                                 for t, d in LIMITS) + "</div>", unsafe_allow_html=True)


def model_run_details() -> None:
    section("Run details", f"Run {esc(bundle.run_id)}, located via {esc(bundle.source)}.")
    sub_head("Checks against the final-selection record")
    st.dataframe(pd.DataFrame(bundle.checks, columns=["Check", "Result", "Detail"]), hide_index=True,
                 **FILL["dataframe"])
    if any(status == "not recorded" for _, status, _ in bundle.checks):
        note("\"Not recorded\" means the record does not state that value, so it could not be compared.")
    if batch is not None:
        note(esc(batch.explain_note), top=0.4)
    left, right = st.columns([1, 1], gap="large")
    with left:
        sub_head("Package versions")
        versions = meta.get("versions", {})
        st.dataframe(pd.DataFrame({
            "Package": list(versions), "Trained with": list(versions.values()),
            "Installed": [(core._installed_version(m) or "n/a") if m != "python" else sys.version.split()[0]
                          for m in versions]}), hide_index=True, **FILL["dataframe"])
    with right:
        sub_head("Model settings and selection rule")
        st.code(json.dumps({"params": meta.get("params", {}), "selection_rule": meta.get("selection_rule", {})},
                           indent=2), language="json")


with tab_model:
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
    chosen = pick_one("Section", MODEL_SECTIONS, key="model_section")
    st.markdown(
        '<div class="scope-note"><b>Performance</b> comes from the one-time held-out evaluation. The <b>review '
        'threshold</b> was set on validation data before that evaluation, which then measured how it performs. Both, '
        'like the <b>Features</b> and <b>Limitations</b>, stay the same for any file. <b>This file</b> describes '
        f'{esc(file_desc)}.</div>', unsafe_allow_html=True)
    {"Performance": model_performance, "Review threshold": model_threshold, "This file": model_this_file,
     "Features": model_features, "Limitations": model_limits, "Run details": model_run_details}[chosen]()
