"""Turn qc_by_session into one tidy CSV per session, and expand a metric on demand.
 
Two steps, deliberately separate:
 
    get_qc_to_csv(qc)             ONE ROW PER METRIC. A metric whose value is
                                  a table keeps that table verbatim as JSON in
                                  `value_str`. Nothing is exploded, nothing is
                                  lost, and the row count stays small.
                                  Preprocessing evaluations (baseline fits,
                                  motion correction) are skipped by default.
 
    expand_qc_metric(df, "...")   Pull ONE metric back out as a wide table:
                                  one row per session, one column per ROI x
                                  channel (G_0, R_2, Iso_3). Every session gets
                                  a row; one without the metric reads
                                  status="missing" with NaN values.
 
Sessions do not have to carry the same QC, so nothing has to line up and no
schema is declared.
 
Key columns are parsed from the payload key
``behavior_<subject_id>_<session_date>_<session_time>``:
 
    session_id    behavior_859953_2026-07-28_09-37-40   (the key, verbatim)
    ses_idx       859953_2026-07-28                     (subject_id_session_date)
    nwb_suffix    093740                                (session_time, no dashes)
    subject_id / session_date / session_time
 
ses_idx and nwb_suffix match the columns of the same name in df_curation.
"""
 
from __future__ import annotations
 
import json
import re
 
import numpy as np
import pandas as pd
 
KEY_COLUMNS = ["session_id", "ses_idx", "nwb_suffix",
               "subject_id", "session_date", "session_time"]
 
LONG_COLUMNS = KEY_COLUMNS + [
    "stage", "modality", "evaluation", "metric",
    "status", "value_num", "value_str",
    "criterion", "reference", "allow_failed_metrics",
    "evaluation_status", "evaluated_at",
]
 
SPINE = ["stage", "modality", "evaluation", "metric"]
MISSING = "missing"
 
# Preprocessing evaluations hold the baseline-fit and motion-correction blocks.
# They are skipped by default; pass skip_evaluations=() to keep them.
DEFAULT_SKIP = ("Preprocessing using method",)
 
_KEY_RE = re.compile(
    r"^behavior_(?P<subject_id>[^_]+)_(?P<session_date>\d{4}-\d{2}-\d{2})"
    r"_(?P<session_time>\d{2}-\d{2}-\d{2})$"
)
 
 
# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------
 
def parse_session_key(key: str) -> dict:
    """behavior_<subject>_<date>_<time> -> the six key columns.
 
    A key that does not match still yields session_id; the derived fields come
    back as NaN rather than raising, so one odd key never costs you the row.
    """
    rec = {c: np.nan for c in KEY_COLUMNS}
    rec["session_id"] = key
    mo = _KEY_RE.match(key or "")
    if not mo:
        return rec
    g = mo.groupdict()
    rec.update(g)
    rec["ses_idx"] = f"{g['subject_id']}_{g['session_date']}"
    rec["nwb_suffix"] = g["session_time"].replace("-", "")
    return rec
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)
 
 
def _latest_status(metric: dict) -> str:
    hist = metric.get("status_history") or []
    return hist[-1].get("status", "NO_STATUS") if hist else "NO_STATUS"
 
 
def _dedupe_latest(evals: list) -> list:
    """Keep only the newest run of each evaluation, by `created`."""
    best: dict = {}
    for e in evals or []:
        key = (e.get("stage"), (e.get("modality") or {}).get("name"), e.get("name"))
        prev = best.get(key)
        if prev is None or (e.get("created") or "") > (prev.get("created") or ""):
            best[key] = e
    return list(best.values())
 
 
def _split_value(value) -> tuple[float | None, str | None]:
    """-> (value_num, value_str). Tables go to value_str as JSON, verbatim."""
    if value is None:
        return None, None
    if _is_number(value):
        return float(value), None
    if isinstance(value, bool):
        return None, str(value)
    if isinstance(value, str):
        # '1.016s' and '-0.943s' are numbers wearing a unit.
        stripped = value[:-1] if value.endswith("s") else value
        try:
            return float(stripped), None
        except ValueError:
            return None, value
    if isinstance(value, (dict, list, tuple)):
        return None, json.dumps(value, separators=(",", ":"))
    return None, str(value)
 
 
# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------
 
def get_qc_to_csv(
    qc_by_session: dict,
    evaluations: str | list[str] | None = None,
    metrics: str | list[str] | None = None,
    skip_evaluations: str | tuple[str, ...] | None = DEFAULT_SKIP,
) -> pd.DataFrame:
    """One row per (session, evaluation, metric).
 
    Scalars land in `value_num`. Anything structured (ROI p-value tables,
    {start, end}, fit coefficients, lists) is kept whole as JSON in
    `value_str`; use `expand_qc_metric` to open one back up.
 
    evaluations / metrics   keep only these, by exact name
    skip_evaluations        drop whole evaluations by name prefix, e.g.
                            "Preprocessing using method" to leave out the
                            baseline-fit blocks that dominate the row count
    """
    want_evals = {evaluations} if isinstance(evaluations, str) else (
        set(evaluations) if evaluations else None
    )
    want_metrics = {metrics} if isinstance(metrics, str) else (
        set(metrics) if metrics else None
    )
    skip = (skip_evaluations,) if isinstance(skip_evaluations, str) else (
        tuple(skip_evaluations) if skip_evaluations else ()
    )
 
    rows = []
    for key, evals in qc_by_session.items():
        keys = parse_session_key(key)
        for e in _dedupe_latest(evals):
            name = e.get("name") or ""
            if want_evals is not None and name not in want_evals:
                continue
            if skip and name.startswith(skip):
                continue
            eval_desc = e.get("description") or ""
            for m in e.get("metrics") or []:
                if want_metrics is not None and m.get("name") not in want_metrics:
                    continue
                vnum, vstr = _split_value(m.get("value"))
                hist = m.get("status_history") or [{}]
                rows.append({
                    **keys,
                    "stage": e.get("stage"),
                    "modality": (e.get("modality") or {}).get("name"),
                    "evaluation": name,
                    "metric": m.get("name"),
                    "status": _latest_status(m),
                    "value_num": vnum,
                    "value_str": vstr,
                    "criterion": m.get("description") or eval_desc,
                    "reference": m.get("reference") or "",
                    "allow_failed_metrics": e.get("allow_failed_metrics"),
                    "evaluation_status": e.get("latest_status"),
                    "evaluated_at": hist[-1].get("timestamp", ""),
                })
    if not rows:
        return pd.DataFrame(columns=LONG_COLUMNS)
    return pd.DataFrame(rows)[LONG_COLUMNS]
 
 
# ---------------------------------------------------------------------------
# Expand one metric back into a table
# ---------------------------------------------------------------------------
 
def expand_qc_metric(
    df: pd.DataFrame,
    metric: str = "Significant response Delay before Go Cue",
) -> pd.DataFrame:
    """One row per session, one column per ROI x channel.
 
        ses_idx            session_date  status  G_0   R_0   Iso_0  G_1
        859953_2026-07-28  2026-07-28    Fail    0.69  0.95  0.0034 8.7e-10
 
    Column names come out as ``<channel>_<roi>`` (G_2, R_2, Iso_2) to match the
    `fiber` convention in df_curation. Channel order is read from each session's
    own `channel` list, so a session storing G/Iso/R maps correctly rather than
    being silently transposed.
 
    For a non-ROI table such as {"start": .., "end": ..} each key becomes its
    own column. Sessions lacking the metric contribute no row, so comparing
    len(result) with the session count tells you how many are absent.
    """
    sel = df[(df["metric"] == metric) & df["value_str"].notna()]
    meta = [c for c in ("ses_idx", "nwb_suffix", "session_date", "status")
            if c in df.columns]
 
    rows = []
    for _, r in sel.iterrows():
        try:
            value = json.loads(r["value_str"])
        except (TypeError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
 
        rec = {c: r[c] for c in meta}
        channels = [str(c) for c in (value.get("channel") or [])]
        for key, item in value.items():
            if key == "channel":
                continue
            if channels and isinstance(item, (list, tuple)) and len(item) == len(channels):
                roi = str(key).replace("ROI ", "").strip()
                for chan, p in zip(channels, item):
                    rec[f"{chan}_{roi}"] = p if _is_number(p) else np.nan
            elif isinstance(item, (list, tuple)):
                for i, p in enumerate(item):
                    rec[f"{key}_{i}"] = p if _is_number(p) else np.nan
            else:
                rec[str(key)] = item
        rows.append(rec)
 
    # Every session in `df` gets a row, including the ones without this metric:
    # their value columns are NaN and status reads "missing", so absent never
    # looks like passing.
    sessions = df[meta[:-1]].drop_duplicates() if "status" in meta else df[meta].drop_duplicates()
    # one row per recording: two recordings on one day share ses_idx but not nwb_suffix
    sessions = sessions.drop_duplicates(subset=["ses_idx", "nwb_suffix"])
 
    if not rows:
        out = sessions.copy()
        out["status"] = MISSING
    else:
        out = sessions.merge(pd.DataFrame(rows), on=[c for c in meta if c != "status"],
                             how="left", suffixes=("", "_found"))
        if "status_found" in out.columns:
            out["status"] = out.pop("status_found")
        out["status"] = out["status"].fillna(MISSING)
 
    if "session_date" in out.columns:
        out = out.sort_values("session_date")
    return out.reset_index(drop=True)
 
 
def value_columns(expanded: pd.DataFrame) -> list[str]:
    """The ROI x channel columns of an expanded table, without the key columns.
 
    Saves re-deriving it every time you threshold:
        cols = value_columns(t);  t.set_index("ses_idx")[cols] < 1e-10
    """
    meta = {"ses_idx", "nwb_suffix", "session_date", "status"}
    return [c for c in expanded.columns if c not in meta]
 
 
# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------
 
def fill_missing_session_metrics(df: pd.DataFrame, spine: pd.DataFrame | None = None) -> pd.DataFrame:
    """Give every session a row for every metric seen anywhere in the dataset.
 
    Long format omits what a session does not have, which makes "absent" and
    "fine" look identical once you filter. This fills the holes explicitly, so
    a missing metric reads as missing rather than as a silent pass:
 
        status = "missing",  value_num = NaN,  present = False
    """
    if df.empty:
        return df.assign(present=pd.Series(dtype=bool))
    if spine is None:
        spine = df[SPINE].drop_duplicates()
    sessions = df[KEY_COLUMNS].drop_duplicates()
 
    out = sessions.merge(spine, how="cross").merge(
        df.assign(present=True), on=KEY_COLUMNS + SPINE, how="left"
    )
    out["present"] = out["present"].fillna(False).astype(bool)
    out["status"] = out["status"].where(out["present"], MISSING)
    return out.sort_values(["session_date", "ses_idx"] + SPINE).reset_index(drop=True)
 
 
def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Which metrics exist in how many sessions -- the QC mismatch, made visible.
 
    Sort by pct to find metrics only some pipeline versions emit. Everything at
    100% means your QC is uniform.
    """
    if df.empty:
        return pd.DataFrame(columns=["evaluation","metric","n_sessions","of_total","pct"])
    total = df["ses_idx"].nunique()
    return (
        df.groupby(["evaluation", "metric"])["ses_idx"].nunique()
        .rename("n_sessions").reset_index()
        .assign(of_total=total, pct=lambda d: (100 * d.n_sessions / total).round(1))
        .sort_values(["pct", "metric"])
        .reset_index(drop=True)
    )