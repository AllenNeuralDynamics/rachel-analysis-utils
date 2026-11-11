"""Load fiber curation CSVs and apply them to FIP data."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from aind_bwnm_fiber_data_curation_utils import data as _curation_data
from aind_bwnm_fiber_data_curation_utils import generate_curation_results

# the data/ subpackage ships an __init__.py, so this follows the install location
CURATION_DATA_DIR = Path(_curation_data.__file__).parent
# Iso is the isosbestic control channel and carries no intended measurement, so no
# curation CSV has rows for it. Leaving it out here passes it through untouched rather
# than tripping the missing-coverage check in apply_curation_df_fip.
CURATABLE_PATCH_CORD = r"(G|R)_\d+"
NO_FIBER = "no_fiber"
# a fiber whose signal steps at a fixed delay before the go cue is carrying a square
# pulse artifact rather than a neural signal
PVALUE_DELAY_THRESH = 1e-7
# sessions where the animal ignored more than this fraction of trials are too disengaged
IGNORE_RATE_THRESH = 0.40
# sessions whose fitted Q-values spread less than this give RPEs too flat to use;
# they are flagged (not dropped) so RPE plots can skip them
MIN_QVALUE_SPREAD = 0.45
POOR_FIT_COLUMN = "poor_model_fit"


def _to_ses_idx(session_id):
    """'behavior_808054_2025-09-02_10-38-37' -> '808054_2025-09-02'."""
    ses = str(session_id)
    if ses.startswith("behavior_"):
        ses = ses[len("behavior_"):]
    return ses.rsplit("_", 1)[0]


def _to_nwb_suffix(session_id):
    """'behavior_808054_2025-09-02_10-38-37' -> 103837, matching df_sess's nwb_suffix."""
    return int(str(session_id).rsplit("_", 1)[-1].replace("-", ""))


def prepare_curation(curation):
    """
    Key a curation table for the analysis, whatever its source (a results CSV or
    a table built in code): add ses_idx and nwb_suffix, and make keep a bool.

    Sessions recorded twice in a day share a ses_idx, so rows stay keyed by
    nwb_suffix too; get_nwb_processed narrows them to the recording it loaded.

    Returns a copy; the input's own columns are left unchanged so the table can be
    saved back out.
    """
    curation = curation.copy()
    curation["ses_idx"] = curation["session_id"].map(_to_ses_idx)
    curation["nwb_suffix"] = curation["session_id"].map(_to_nwb_suffix)
    curation["keep"] = curation["keep"].astype(bool)
    return curation


def load_curation(csv_path):
    """
    Read a fiber curation results CSV and prepare it (see prepare_curation).

    csv_path: path under CURATION_DATA_DIR, without the .csv extension, e.g.
        'bilateral_4_channels/curation_results/bilateral_4_channels_results'
        (the _G variant covers only G fibers)
    """
    return prepare_curation(pd.read_csv((CURATION_DATA_DIR / csv_path).with_suffix(".csv")))


def make_default_curation(session_ids, channels):
    """
    All-pass curation for QC without a curation CSV: one row per session and channel
    in `channels`, every fiber kept, target taken from the channels dict. Same
    decision columns and keys load_curation gives, so the rest of the pipeline cannot
    tell the two apart.

    session_ids: e.g. 'behavior_808054_2025-09-02_10-38-37'
    channels: parameters['channels'], patch cord -> intended measurement
    """
    # decision columns from curation-utils; dropped_by / drop_reason start NaN
    curation = pd.DataFrame(
        [{"subject": session_id.split("_")[1], "session_id": session_id, "fiber": fiber,
          "target": target, "keep": True}
         for session_id in session_ids for fiber, target in channels.items()],
        columns=generate_curation_results.CURATION_COLUMNS,
    )
    return prepare_curation(curation)


def drop_unchosen_recordings(curation, df_sess):
    """
    Drop curation rows for recordings that were not the one loaded.

    A day recorded more than once gives every fiber one row per nwb_suffix, which
    would make the (ses_idx, patch_cord) key apply_curation_df_fip looks up
    non-unique. The CSV carries both regardless of how many nwbs were dispatched,
    so this runs whether or not df_sess itself had duplicates.

    Rows for sessions absent from df_sess are left untouched.
    """
    chosen = df_sess[["ses_idx", "nwb_suffix"]].astype({"nwb_suffix": int})
    chosen_map = dict(chosen.values)
    ses = curation["ses_idx"]
    loaded = ses.isin(chosen_map)
    matches = ses.map(chosen_map).eq(curation["nwb_suffix"])

    return curation[~loaded | matches]


def drop_high_ignore_sessions(curation, df_sess, max_ignore_rate=IGNORE_RATE_THRESH):
    """
    Drop sessions whose df_sess['ignore_rate'] exceeds max_ignore_rate. Expects a
    curation; callers skip this when none is in use.

    Returns (curation, df_sess). Curation rows of the dropped sessions that were still
    kept get keep=False, dropped_by='high_ignore_rate', and the session's ignore rate as
    drop_reason; rows already dropped by an earlier step keep their own reason. Neither
    input is modified.
    """
    high = df_sess["ignore_rate"] > max_ignore_rate
    if not high.any():
        return curation, df_sess

    rates = df_sess.loc[high].set_index("ses_idx")["ignore_rate"]
    print(f"Dropping {len(rates)} session(s) with ignore_rate > {max_ignore_rate}: "
                 f"{rates.round(3).to_dict()}")

    curation = curation.astype({"dropped_by": object, "drop_reason": object})
    rows = curation["ses_idx"].isin(rates.index) & curation["keep"]
    curation.loc[rows, "keep"] = False
    curation.loc[rows, "dropped_by"] = "high_ignore_rate"
    curation.loc[rows, "drop_reason"] = (
        curation.loc[rows, "ses_idx"].map(rates).round(3).astype(str)
    )
    return curation, df_sess[~high]


def curate_poor_model_fits(curation, df_sess_fm, df_trials_enriched, enrich_cols,
                           min_qvalue_spread=MIN_QVALUE_SPREAD):
    """
    Remove the model-derived trial columns of sessions whose foraging model fit is
    poor: qvalue_spread below min_qvalue_spread, or no fit at all (NaN or missing).

    The session itself is kept, so analyses that need no model (PSTHs, Bayer &
    Glimcher) still use it, while RPE / Q-value binning and RPE slope fits skip it.

    curation: the run's curation table. Its fiber keep decisions are not changed;
        poor_model_fit and qvalue_spread are added per session (NaN for sessions that
        were not loaded).
    df_sess_fm: from get_foraging_model_info. poor_model_fit is added.
    df_trials_enriched: from enrich_df_trials_fm.
    enrich_cols: the columns enrich_df_trials_fm added (RPE_*, Q_chosen, P_*, K_*, ...).
        These are set to NaN for poor-fit trials. The raw fit outputs from
        get_foraging_model_info (Q_left/Q_right, L_prob/R_prob, kernels) are kept, so
        running enrich_df_trials_fm on the saved df_trials recovers every column.

    Returns (curation, df_sess_fm, df_trials_enriched).
    """
    df_sess_fm = df_sess_fm.copy()
    # absent when no session had a fit, since get_foraging_model_info merges it in per fit
    spread = df_sess_fm.get("qvalue_spread", pd.Series(np.nan, index=df_sess_fm.index))
    # written as not-(>=) so a NaN spread (no fit) is flagged too
    poor = ~(spread >= min_qvalue_spread)
    df_sess_fm[POOR_FIT_COLUMN] = poor

    curation = curation.copy()
    curation[POOR_FIT_COLUMN] = curation["ses_idx"].map(dict(zip(df_sess_fm["ses_idx"], poor)))
    curation["qvalue_spread"] = curation["ses_idx"].map(dict(zip(df_sess_fm["ses_idx"], spread)))

    if not poor.any():
        return curation, df_sess_fm, df_trials_enriched

    poor_sessions = df_sess_fm.loc[poor, "ses_idx"]
    print(f"Removing the model fit of {poor.sum()} session(s) (qvalue_spread < "
                 f"{min_qvalue_spread}; NaN = no fit); sessions kept for non-RPE analyses: "
                 f"{dict(zip(poor_sessions, spread[poor].round(3)))}")

    df_trials_enriched = df_trials_enriched.copy()
    rows = df_trials_enriched["ses_idx"].isin(poor_sessions)
    df_trials_enriched.loc[rows, list(enrich_cols)] = np.nan
    return curation, df_sess_fm, df_trials_enriched


def apply_curation_df_fip(df_fip, curation):
    """
    Set df_fip['event'] to the curated intended measurement and drop fibers that
    did not pass curation.

    Expects the 'patch_cord' column that nwb_utils.split_fiber adds. Call this
    BEFORE enrich_fip_in_df_trials so the df_trials columns are built from the
    target names and need no second rename.

    Raises if a loaded curatable fiber has no curation row -- curation is expected to
    cover everything requested, so absence is a mistake rather than a silent drop. Iso is
    not curatable and passes through with its patch cord as the event name.
    """
    patch_cord = df_fip["patch_cord"]
    curatable = patch_cord.str.fullmatch(CURATABLE_PATCH_CORD, na=False)

    # curation keeps rows for sessions that were never loaded, and those can still
    # hold a row per recording, so scope the lookup to what df_fip actually needs
    lookup = curation[curation["ses_idx"].isin(df_fip["ses_idx"].unique())]
    lookup = lookup.set_index(["ses_idx", "fiber"])
    keys = pd.MultiIndex.from_arrays([df_fip["ses_idx"], patch_cord.where(curatable)])

    # a curatable fiber whose data is entirely NaN carries no signal: drop it, and don't
    # require a curation row for it
    all_nan = (
        df_fip["data"].isna().groupby([df_fip["ses_idx"], patch_cord]).transform("all")
    )
    empty = curatable & all_nan
    if empty.any():
        pairs = sorted(set(zip(df_fip.loc[empty, "ses_idx"], patch_cord[empty])))
        print(f"Dropping {len(pairs)} curatable fiber(s) whose data is all NaN: {pairs}")

    absent = curatable & ~empty & ~pd.Series(keys.isin(lookup.index), index=df_fip.index)
    if absent.any():
        pairs = sorted(set(zip(df_fip.loc[absent, "ses_idx"], patch_cord[absent])))
        raise ValueError(
            f"Curation is missing {len(pairs)} loaded (ses_idx, patch_cord) pair(s), e.g. {pairs[:5]}. "
            "Curation must cover every channel requested in parameters['channels']."
        )

    # a fiber with no intended measurement -- NaN or the literal 'no_fiber', both
    # appear -- gets target NA, so it is dropped below
    target = pd.Series(lookup["target"].reindex(keys).to_numpy(), index=df_fip.index)
    target = target.where(target != NO_FIBER)
    keep = pd.Series(lookup["keep"].reindex(keys).to_numpy(), index=df_fip.index).eq(True)

    selected = ~empty & (~curatable | (keep & target.notna()))

    out = df_fip.loc[selected].copy()
    out["event"] = target[selected].fillna(patch_cord[selected])
    return out.reset_index(drop=True)


def add_square_pulse_pvalue(curation, square_pulse_pval, pvalue_thresh=PVALUE_DELAY_THRESH):
    """
    Add square_pulse_pvalue per fiber, and drop (keep=False, with dropped_by/drop_reason
    recorded) kept fibers below pvalue_thresh: a step in the signal at a fixed delay
    before the go cue is a square pulse artifact, not a neural signal.

    square_pulse_pval: qc_curation_helpers.expand_qc_metric(df, "Significant response
        Delay before Go Cue") -- one row per session (ses_idx, nwb_suffix, status), one
        column per fiber (G_0, R_2, Iso_3, ...). Iso columns are ignored: not curatable.
        Rows match the curation on ses_idx and nwb_suffix.

    Also adds square_pulse_qc_missing: True for sessions whose QC has no such metric
    (status 'missing', e.g. assets processed before the pipeline added it), so the
    check never ran for them; those sessions are printed. NaN for curation rows of
    sessions not in square_pulse_pval (not dispatched in this run).

    A fiber the CSV already dropped keeps its original reason. A missing p-value is
    NaN and leaves the fiber kept.

    Returns a copy; curation itself is not modified.
    """
    sessions = square_pulse_pval.assign(nwb_suffix=square_pulse_pval["nwb_suffix"].astype(int))
    fiber_cols = [c for c in sessions.columns if pd.Series([c]).str.fullmatch(CURATABLE_PATCH_CORD).iloc[0]]
    # one row per (session, fiber)
    pvals = sessions.melt(id_vars=["ses_idx", "nwb_suffix"], value_vars=fiber_cols,
                          var_name="fiber", value_name="square_pulse_pvalue")
    missing = sessions.assign(square_pulse_qc_missing=sessions["status"].eq("missing"))[
        ["ses_idx", "nwb_suffix", "square_pulse_qc_missing"]]

    # dropped_by/drop_reason read in as all-NaN floats when nothing was dropped
    curation = curation.astype({"dropped_by": object, "drop_reason": object})
    curation = curation.drop(columns=["square_pulse_pvalue", "square_pulse_qc_missing"], errors="ignore")
    curation = (curation.merge(pvals, on=["ses_idx", "nwb_suffix", "fiber"], how="left")
                        .merge(missing, on=["ses_idx", "nwb_suffix"], how="left"))

    no_qc = sessions.loc[sessions["status"].eq("missing"), "ses_idx"].tolist()
    if no_qc:
        print(f"No 'Significant response Delay before Go Cue' QC for {len(no_qc)} of "
              f"{len(sessions)} session(s); square pulse check skipped for: {no_qc}")

    square_pulse = curation["keep"] & (curation["square_pulse_pvalue"] < pvalue_thresh)
    if square_pulse.any():
        print(f"Dropping {square_pulse.sum()} fiber(s) as square pulses "
              f"(square_pulse_pvalue < {pvalue_thresh}): "
              f"{list(zip(curation.loc[square_pulse, 'session_id'], curation.loc[square_pulse, 'fiber']))}")
    curation.loc[square_pulse, "keep"] = False
    curation.loc[square_pulse, "dropped_by"] = "fiber_qc"
    curation.loc[square_pulse, "drop_reason"] = "square_pulse_pvalue"
    return curation
