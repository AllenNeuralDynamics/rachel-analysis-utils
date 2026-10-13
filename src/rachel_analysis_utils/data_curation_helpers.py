"""Load fiber curation CSVs and apply them to FIP data."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from aind_bwnm_fiber_data_curation_utils import data as _curation_data
from aind_dynamic_foraging_data_utils import nwb_utils

# the data/ subpackage ships an __init__.py, so this follows the install location
CURATION_DATA_DIR = Path(_curation_data.__file__).parent
# Iso is the isosbestic control channel and carries no intended measurement, so no
# curation CSV has rows for it. Leaving it out here passes it through untouched rather
# than tripping the missing-coverage check in apply_curation_df_fip.
CURATABLE_PATCH_CORD = r"(G|R)_\d+"
NO_FIBER = "no_fiber"
# a fiber whose signal steps at a fixed delay before the go cue is carrying a square
# pulse artifact rather than a neural signal
PVALUE_DELAY_THRESH = 1e-10
DELAY_DROPPED_BY = "delay_pvalue_curation"
DELAY_DROP_REASON = "square_pulse"
# sessions where the animal ignored more than this fraction of trials are too disengaged
IGNORE_RATE_THRESH = 0.40


def _to_ses_idx(session_id):
    """'behavior_808054_2025-09-02_10-38-37' -> '808054_2025-09-02'."""
    ses = str(session_id)
    if ses.startswith("behavior_"):
        ses = ses[len("behavior_"):]
    return ses.rsplit("_", 1)[0]


def _to_nwb_suffix(session_id):
    """'behavior_808054_2025-09-02_10-38-37' -> 103837, matching df_sess's nwb_suffix."""
    return int(str(session_id).rsplit("_", 1)[-1].replace("-", ""))


def load_curation(csv_path):
    """
    Read a fiber curation results CSV, keyed to ses_idx.

    csv_path: path under CURATION_DATA_DIR, without the .csv extension, e.g.
        'bilateral_4_channels/curation_results/bilateral_4_channels_results'
        (the _G variant covers only G fibers)

    Returns the CSV's own columns, unchanged so the table can be saved back out, plus
    ses_idx, nwb_suffix, and p_value_delay (NaN until add_delay_pvalue fills it).

    Sessions recorded twice in a day share a ses_idx, so rows stay keyed by
    nwb_suffix too; get_nwb_processed narrows them to the recording it loaded.
    """
    curation = pd.read_csv((CURATION_DATA_DIR / csv_path).with_suffix(".csv"))
    curation["ses_idx"] = curation["session_id"].map(_to_ses_idx)
    curation["nwb_suffix"] = curation["session_id"].map(_to_nwb_suffix)
    curation["keep"] = curation["keep"].astype(bool)
    curation["p_value_delay"] = np.nan
    return curation


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
    logging.info(f"Dropping {len(rates)} session(s) with ignore_rate > {max_ignore_rate}: "
                 f"{rates.round(3).to_dict()}")

    curation = curation.astype({"dropped_by": object, "drop_reason": object})
    rows = curation["ses_idx"].isin(rates.index) & curation["keep"]
    curation.loc[rows, "keep"] = False
    curation.loc[rows, "dropped_by"] = "high_ignore_rate"
    curation.loc[rows, "drop_reason"] = (
        curation.loc[rows, "ses_idx"].map(rates).round(3).astype(str)
    )
    return curation, df_sess[~high]


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

    absent = curatable & ~pd.Series(keys.isin(lookup.index), index=df_fip.index)
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

    selected = ~curatable | (keep & target.notna())

    out = df_fip.loc[selected].copy()
    out["event"] = target[selected].fillna(patch_cord[selected])
    return out.reset_index(drop=True)


def compute_p_values_delay(nwbfile, GoCueTime, patch_cords, pre=40, post=80, delay=0.5):
    """
    Wilcoxon p-value per patch cord ('G_0', 'R_2', ...) for a step in the signal
    `delay` s before the go cue.

    Assumes 20 Hz sampling, so pre/post are in frames. GoCueTime must be on the nwb's
    own clock (df_events 'raw_timestamps'). Returns a Series indexed by patch cord; one
    that cannot be computed is NaN.
    """
    p_values_delay = pd.Series(np.nan, index=list(patch_cords), dtype=float)
    delay_in_frames = int(np.round(delay * 20))
    for patch_cord in p_values_delay.index:
        try:
            ts = nwbfile.acquisition[patch_cord].timestamps[:]
            data = nwbfile.acquisition[patch_cord].data[:]
            inds = [
                np.argmin(np.abs(ts - go))
                for go in GoCueTime
                if ts[pre] <= go <= ts[-post]
            ]
            if not inds:
                raise Exception("No GoCue occurs within timestamps")
            snippets = np.array(
                [data[i - pre : i + post] for i in inds if i <= len(data) - post]
            )
            snippets -= snippets[:, pre - 10 : pre].mean(1)[:, None]
            snippets -= snippets[
                :, pre - delay_in_frames - 10 : pre - delay_in_frames
            ].mean(1)[:, None]
            p_values_delay[patch_cord] = wilcoxon(
                snippets[:, pre - delay_in_frames : pre - delay_in_frames + 10].mean(1)
            ).pvalue
        except Exception as e:
            logging.error(f"Error computing delay p-value for {patch_cord}: {e}")
    return p_values_delay


def add_delay_pvalue(curation, df_fip, df_events, df_sess, pvalue_thresh=PVALUE_DELAY_THRESH):
    """
    Fill p_value_delay for every curatable fiber left in df_fip, and drop (keep=False,
    with dropped_by/drop_reason recorded) those below pvalue_thresh as square pulses.

    Run on df_fip AFTER apply_curation_df_fip, so only fibers that passed the fiber
    metrics are tested and the nwbs reloaded here are limited to those sessions;
    re-apply apply_curation_df_fip afterwards to remove the new drops. A NaN p-value
    leaves the fiber kept.

    Go cues come from df_events' raw_timestamps, which share the nwb's own clock;
    the 'timestamps' column is shifted so the first go cue is t=0.

    Returns a copy; curation itself is not modified.
    """
    # dropped_by/drop_reason read in as all-NaN floats when nothing was dropped
    curation = curation.astype({"dropped_by": object, "drop_reason": object})
    fibers = df_fip.loc[
        df_fip["patch_cord"].str.fullmatch(CURATABLE_PATCH_CORD, na=False),
        ["ses_idx", "patch_cord"],
    ].drop_duplicates()

    for ses_idx, ses_fibers in fibers.groupby("ses_idx"):
        sess = df_sess[df_sess["ses_idx"] == ses_idx].iloc[0]
        nwb = nwb_utils.load_nwb_from_filename(sess["s3_location"])
        ses_events = df_events[df_events["ses_idx"] == ses_idx]
        go_cue = ses_events.query("event == 'goCue_start_time'")["raw_timestamps"].values

        p_values = compute_p_values_delay(nwb, go_cue, ses_fibers["patch_cord"])
        # match nwb_suffix too, in case curation still holds the day's other recording
        rows = (
            (curation["ses_idx"] == ses_idx)
            & (curation["nwb_suffix"] == int(sess["nwb_suffix"]))
            & curation["fiber"].isin(p_values.index)
        )
        curation.loc[rows, "p_value_delay"] = curation.loc[rows, "fiber"].map(p_values)

    square_pulse = curation["p_value_delay"] < pvalue_thresh
    curation.loc[square_pulse, "keep"] = False
    curation.loc[square_pulse, "dropped_by"] = DELAY_DROPPED_BY
    curation.loc[square_pulse, "drop_reason"] = DELAY_DROP_REASON
    return curation
