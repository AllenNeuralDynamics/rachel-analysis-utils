

import warnings
import glob
import pandas as pd
import numpy as np
from pathlib import Path

from aind_dynamic_foraging_data_utils import nwb_utils, enrich_dfs
from aind_dynamic_foraging_data_utils import code_ocean_utils as co_utils



import copy 
import re

def drop_na_measurments(nwb_list):
    for nwb in nwb_list:
        df = getattr(nwb, "df_fip", None)
        if isinstance(df, pd.DataFrame) and "intended_measurement" in df.columns:
            keep = df["intended_measurement"].notna() & (
                df["intended_measurement"].astype(str).str.strip().str.upper() != "N/A"
            )
            nwb.df_fip = df.loc[keep].reset_index(drop=True)
    return nwb_list

# --- helper utilities ---
def _parse_session_id(ses_idx):
    """Return (subject_id, date) from ses_idx like '781900_2025-06-24'"""
    ses = str(ses_idx)
    parts = ses.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return ses, ""

def _channels_for_date(mapping, target_date):
    # otherwise assume channel->list_of_dates (we only include the channel keys that are G_/R_ and have this date)
    chans = []
    for ch, dates in mapping.items():
        if not (isinstance(ch, str) and (ch.startswith("G_") or ch.startswith("R_"))):
            continue
        if isinstance(dates, (list, tuple)) and target_date in dates:
            chans.append(ch)
    return chans

def _actual_map_for_session(entry, date, correct_map):
    """
    Return mapping G_ -> region for this session date.
    Prefer explicit per-session misconnect_fixes/misconnections that contain G_ keys;
    otherwise return a shallow copy of correct_map.
    """
    misconnections = entry.get("misconnect_fixes", {}) or entry.get("misconnections", {}) or {}
    if date in misconnections and any(isinstance(k, str)for k in misconnections[date].keys()):
        return misconnections[date]
    return correct_map

def _map_event_to_intended_measurement(event_str, session_map, correct_map):
    """
    Map an event string to intended area(s).
    - Extract G_* tokens from the event (e.g. G_0, G_2).
    - Prefer session_map, fall back to correct_map.
    - Return colon-joined unique areas in order of appearance, or np.nan.
    """
    if not isinstance(event_str, str):
        return np.nan
    tokens = re.findall(r'(G_\d+|R_\d+)', event_str)
    if not tokens:
        return np.nan
    areas = []
    seen = set()
    for t in tokens:
        area = (session_map.get(t) if session_map else None) or (correct_map.get(t) if correct_map else None)
        if area and area not in seen:
            seen.add(area)
            areas.append(area)
    return ":".join(areas) if areas else np.nan

def _apply_channel_drops_to_nwb(nwb, entry, correct_map, drop_borderline=False):
    """
    Return a deepcopy of nwb with channels/sessions removed per entry.

    drop_borderline: also drop the borderline sessions.

    Drops whole session if drop_all or date in drop_sessions.
    Removes rows from n.df_fip where the 'intended_measurement' matches any region
    listed in the subject/session drop lists. If intended_measurement is not present
    in df_fip, falls back to removing by G_/R_ tokens (using correct_map).
    """
    # tolerate non-dict entry
    if not isinstance(entry, dict):
        entry = {}

    # parse session date
    subject, date = _parse_session_id(getattr(nwb, "session_id", ""))

    # drop entire session if requested
    if entry.get("drop_all"):
        return None
    drop_sessions = entry.get("drop_sessions", []) or entry.get("drop_dates", []) or []
    if date in drop_sessions:
        return None

    # start with a deepcopy to avoid mutating input
    n = copy.deepcopy(nwb)


    # Build a set of intended_measurement names to drop (region names)
    subject_drop_raw = entry.get("drop_channels", []) or []
    intended_drop_set = set()
    for item in subject_drop_raw:
        if isinstance(item, str) and (item.startswith("G_") or item.startswith("R_")):
            region = correct_map.get(item)
            if region:
                intended_drop_set.add(region)
        elif isinstance(item, str):
            intended_drop_set.add(item)

    # per-session drops (either keyed by date or by channel)
    per_session_map = entry.get("drop_sessions_channels", {}) or {}
    session_specific_chs = _channels_for_date(per_session_map, date)  # returns G_/R_ keys
    for ch in session_specific_chs:
        region = correct_map.get(ch)
        if region:
            intended_drop_set.add(region)

    # optional borderline drops (support both naming variants) if requested
    borderline_map = entry.get("borderline_drop_sessions_channels", {}) or entry.get("borderline_drop_channels", {}) or {}
    borderline_specific_chs = _channels_for_date(borderline_map, date) if drop_borderline else []
    for ch in borderline_specific_chs:
        region = correct_map.get(ch)
        if region:
            intended_drop_set.add(region)

    # Now remove rows in df_fip corresponding to intended_measurements to drop.
    if hasattr(n, "df_fip") and isinstance(n.df_fip, pd.DataFrame) and intended_drop_set:
        df_fip = n.df_fip.copy()
        if "intended_measurement" in df_fip.columns:
            # keep rows whose intended_measurement is not in intended_drop_set
            keep_mask = ~df_fip["intended_measurement"].astype(str).isin({str(x) for x in intended_drop_set})
            n.df_fip = df_fip.loc[keep_mask].reset_index(drop=True)
        elif "event" in df_fip.columns:
            # fallback: map intended regions back to G_/R_ keys using correct_map
            region_to_G = {v: k for k, v in (correct_map or {}).items()}
            gkeys_to_drop = [region_to_G.get(r) for r in intended_drop_set if region_to_G.get(r)]
            # also include any raw G_/R_ keys that were provided originally
            gkeys_provided = [x for x in subject_drop_raw if isinstance(x, str) and (x.startswith("G_") or x.startswith("R_"))]
            gkeys_all = sorted({*(k for k in gkeys_to_drop if k), *gkeys_provided}, key=len, reverse=True)
            if gkeys_all:
                pattern = r'(?:' + r'|'.join(re.escape(s) for s in gkeys_all) + r')'
                keep_mask = ~df_fip["event"].astype(str).str.contains(pattern, regex=True, na=False)
                n.df_fip = df_fip.loc[keep_mask].reset_index(drop=True)
            else:
                # no viable fallback keys: leave df_fip unchanged
                n.df_fip = df_fip
        else:
            # no intended_measurement or event column -> cannot drop channel-specific rows
            n.df_fip = df_fip

    return n
def apply_curation_nwb_list(nwb_list, curation, drop_borderline = False):
    """
    Apply curation rules to a list (or list-of-lists) of nwb objects.

    nwb_list: list of nwbs
    curation: dataframe of curation notes from json. 
    drop_borderline: determines if we should drop borderline sessions as well

    Returns two lists:
      - curated: nwb objects with drop_channels removed and drop_sessions omitted
      - curated_with_borderline: same as above but also applying 'borderline_drop' variants

    Notes:
      - Annotates nwb.df_fip with 'intended_measurement' ONCE before applying drops.
      - intended_measurement uses per-session misconnect_fixes if available, otherwise correct_mapping.
    """
    

    correct_map = curation.get("correct_mapping", {}) or {}

    curated = []
    curated_with_borderline = []

    for nwb in nwb_list:
        ses_idx = getattr(nwb, "session_id", "")
        subject, date = _parse_session_id(ses_idx)
        entry = curation.get(str(subject), {}) or {}

        # compute session-specific mapping once
        session_map = _actual_map_for_session(entry, date, correct_map)

        # annotate intended_measurement ONCE on a deepcopy before any drops
        nwb_annot = copy.deepcopy(nwb)
        if hasattr(nwb_annot, "df_fip") and isinstance(nwb_annot.df_fip, pd.DataFrame):
            df_fip = nwb_annot.df_fip.copy()
            if "event" in df_fip.columns:
                df_fip["intended_measurement"] = df_fip["event"].apply(lambda ev: _map_event_to_intended_measurement(ev, session_map, correct_map))
            else:
                df_fip["intended_measurement"] = np.nan
            nwb_annot.df_fip = df_fip.reset_index(drop=True)

        # now apply drops (these return deepcopy-modified objects)
        n_cur = _apply_channel_drops_to_nwb(nwb_annot, entry, correct_map, drop_borderline=False)
        if n_cur is not None:
            curated.append(n_cur)

        if drop_borderline:
            n_cur_b = _apply_channel_drops_to_nwb(nwb_annot, entry, correct_map, drop_borderline=True)
            if n_cur_b is not None:
                curated_with_borderline.append(n_cur_b)

    return curated, curated_with_borderline


def apply_curation_by_subject_df_sess(df_sess, curation):
    """
    Subject-by-subject curation that applies per-session misconnect fixes by
    renaming (copying -> region-prefixed) the G_* metric columns into columns
    named by the actual fiber/region (e.g. "PL(L)_slope_pos"), then drops the
    original G_* columns. Sessions in drop_sessions / drop_dates or subjects
    with drop_all are returned in df_dropped. Regions listed in drop_channels
    are set to NaN.

    Parameters
    ----------
    df_sess : pd.DataFrame
        Session-level dataframe containing G_* metric columns and columns:
        - subject_id
        - session_date (YYYY-MM-DD or date-like)
    curation : dict
        Curation dict loaded from the JSON (contains correct_mapping, per-subject entries, etc.)

    Returns
    -------
    (df_curated, df_dropped)
    """
    df = df_sess.copy()
    correct_map = curation.get("correct_mapping", {})

    # normalize session_date column to YYYY-MM-DD strings if present
    if 'session_date' in df.columns:
        try:
            df['session_date'] = pd.to_datetime(df['session_date']).dt.strftime('%Y-%m-%d')
        except Exception:
            # leave as-is if conversion fails
            pass

    # discover G_ prefixes and metric columns for each G_
    all_cols = df.columns.tolist()
    G_prefixes = sorted({c.split('_slope')[0] for c in all_cols if c.startswith('G_')})
    metrics_by_G = {g: [c for c in all_cols if c.startswith(g + '_')] for g in G_prefixes}

    processed = []
    dropped = []

    for sid in df['subject_id'].dropna().unique():
        subj_df = df[df['subject_id'] == sid].copy()
        entry = curation.get(str(sid), {}) or {}

        # subject-level drops
        if isinstance(entry, dict) and entry.get('drop_all'):
            dropped.append(subj_df)
            continue

        # subject-level drop_channels: set corresponding G_* metric columns to NaN
        subject_drop_raw = entry.get('drop_channels', []) or []
        # build mapping region -> G_ for lookup
        region_to_G = {v: k for k, v in (correct_map or {}).items()}
        subject_drop = set()
        for item in subject_drop_raw:
            if isinstance(item, str) and item.startswith('G_'):
                subject_drop.add(item)
                region = (correct_map or {}).get(item)
                if region:
                    subject_drop.add(region)
            else:
                subject_drop.add(item)
                gkey = region_to_G.get(item)
                if gkey:
                    subject_drop.add(gkey)

        # pre-null any metric columns for G_ prefixes that are in subject_drop
        for g_prefix in list(metrics_by_G.keys()):
            if g_prefix in subject_drop:
                for col in metrics_by_G.get(g_prefix, []):
                    subj_df[col] = float(np.nan)

        # iterate sessions for this subject
        for idx, row in subj_df.iterrows():
            date = row.get('session_date')

            # check per-subject drop_sessions / drop_dates
            drop_dates = entry.get('drop_sessions', []) or entry.get('drop_dates', []) or []
            if date in drop_dates:
                dropped.append(row.to_frame().T)
                continue

            # choose actual map for this date: if misconnect_fixes exists for date use it
            misconnections = entry.get('misconnect_fixes', {}) or entry.get('misconnections', {}) or {}
            if date in misconnections and any(k.startswith('G_') for k in misconnections[date].keys()):
                actual_map = {k: v for k, v in misconnections[date].items() if k.startswith('G_')}
            else:
                actual_map = correct_map

            # apply actual_map: for each source G_ prefix, copy/assign metric columns
            # If the source G_ is flagged in subject_drop, ensure those metric cols are NaN.
            for source_G, target_region in actual_map.items():
                for gcol in metrics_by_G.get(source_G, []):
                    if source_G in subject_drop:
                        row[gcol] = float(np.nan)
                    else:
                        # keep the existing value (already in row under G_* columns)
                        row[gcol] = float(row.get(gcol, np.nan))

            # annotate what was applied for traceability
            row["_applied_actual_map"] = str(actual_map)
            row["_dropped_regions"] = str(sorted(list(subject_drop)))

            processed.append(row.to_frame().T)

    df_curated = pd.concat(processed, ignore_index=True) if processed else pd.DataFrame(columns=df.columns)
    df_dropped = pd.concat(dropped, ignore_index=True) if dropped else pd.DataFrame(columns=df.columns)

    # coerce metric columns to numeric floats
    metric_cols = sorted({c for cols in metrics_by_G.values() for c in cols} & set(df_curated.columns))
    if metric_cols:
        df_curated[metric_cols] = df_curated[metric_cols].apply(pd.to_numeric, errors='coerce').astype(float)

    return df_curated.reset_index(drop=True), df_dropped.reset_index(drop=True)

def pick_side(side_pos, seg_bounds, per_seg_min):
    if len(side_pos) == 0 or per_seg_min == 0:
        return np.array([]), []
    side_by_seg = []
    for lo, hi in seg_bounds:
        side_in = side_pos[(side_pos >= lo) & (side_pos < hi)]
        side_in_subset = np.random.choice(side_in, size = per_seg_min, replace = False)
        side_by_seg.append(np.sort(side_in_subset))
    all_selected = np.sort(np.concatenate([arr for arr in side_by_seg if arr.size]))
    return all_selected, side_by_seg
    
def subsample_lr_thirds(nwb, per_seg_min = 50):
    nwb_subset = copy.deepcopy(nwb)
    df_trials = nwb.df_trials
    n = len(df_trials)
    
    # thirds boundaries (integer division)
    t1 = n // 3
    t2 = 2 * n // 3

    positions = df_trials['trial'].values

    left_mask = df_trials['animal_response'].to_numpy() == 0
    right_mask = df_trials['animal_response'].to_numpy() == 1

    left_pos = positions[left_mask]
    right_pos = positions[right_mask]

    # compute per-third minimums
    seg_bounds = [(0, t1), (t1, t2), (t2, n)]
    for lo, hi in seg_bounds:
        left_in = left_pos[(left_pos >= lo) & (left_pos < hi)]
        right_in = right_pos[(right_pos >= lo) & (right_pos < hi)]
        per_seg_min = int(min(int(min(left_in.size, right_in.size)), per_seg_min))

    left_idx, left_by_seg = pick_side(left_pos, seg_bounds, per_seg_min)
    right_idx, right_by_seg = pick_side(right_pos, seg_bounds, per_seg_min)

    # ensure equal totals (they should be by construction)
    keep = min(left_idx.size, right_idx.size)
    left_idx = left_idx[:keep]
    right_idx = right_idx[:keep]

    sel_idx = np.sort(np.concatenate([left_idx, right_idx]))
    df_sub = df_trials[df_trials['trial'].isin(sel_idx)].copy()

    nwb_subset.df_trials = df_sub
    return nwb_subset

def split_nwb_by_choice(nwb):
    nwb_split = copy.deepcopy(nwb)
    nwb_split.df_trials_left = nwb.df_trials.query('animal_response == 0.0')
    nwb_split.df_trials_right = nwb.df_trials.query('animal_response == 1.0')
    nwb_split.df_trials_ignore = nwb.df_trials.query('animal_response == 2.0')
    return nwb_split

def split_nwb_by_time(nwb):
    num_trials = np.max(nwb.df_trials['trial'])
    mid_point = int(num_trials/2)
    nwb_split_early, nwb_split_late = copy.deepcopy(nwb), copy.deepcopy(nwb)
    nwb_split_early.df_trials = nwb.df_trials.query(f'trial < {mid_point}')
    nwb_split_late.df_trials = nwb.df_trials.query(f'trial >= {mid_point}')

    return (nwb_split_early, nwb_split_late)

class dummy_nwb:
    def __init__(self, df_trials, df_events, df_fip, ses_idx = None, df_licks = None, grouped = False) -> None:
        if grouped is True:
            self.df_events = df_events
            self.df_fip = df_fip
            self.df_trials = df_trials
            self.session_id = ', '.join(df_trials.ses_idx.unique())
            return
        if ses_idx is None and grouped is False:

            if len(df_trials.ses_idx.unique()) > 1 or \
                len(df_events.ses_idx.unique()) > 1 or \
                len(df_fip.ses_idx.unique()) > 1:

                warnings.warn('multiple sessions found, only one will be attached to this nwb')
            ses_idx = df_trials.ses_idx.unique()[0]
             
                
        assert df_fip[df_fip['ses_idx'] == ses_idx].shape[0] != 0 ,(
            "No session exists in the df_fip"
        )
        self.session_id = ses_idx
        self.df_events = df_events[df_events['ses_idx'] == ses_idx]
        self.df_fip = df_fip[df_fip['ses_idx'] == ses_idx].copy().reset_index(drop=True)
        self.df_trials = df_trials[df_trials['ses_idx'] == ses_idx]
        if df_licks:
            self.df_licks = df_licks[df_licks['ses_idx'] == ses_idx]

        nwb_file_name = glob.glob(f"/root/capsule/data/**{ses_idx}**/nwb/**.nwb")
        if len(nwb_file_name):
            self.nwb_file_loc = nwb_file_name[0]
        else:
            self.nwb_file_loc = None
        

    def __str__(self):
        return f"session {self.session_id}"

    def __repr__(self):
        return f"{self.session_id}"

    def save(self, plot_loc, df_sess = None):
        """
        Save dataframe attributes into:
            plot_loc / session_id / <attr>.parquet
        """

        session_folder = Path(plot_loc) / str(self.session_id)
        session_folder.mkdir(parents=True, exist_ok=True)

        for attr, val in self.__dict__.items():
            if isinstance(val, pd.DataFrame):
                # print(f"now saving {attr}")

                if attr == "df_events":
                    val["data"] = val["data"].astype(str)
                
                # df = self.convert_df_to_saveable_format(val)
                if attr == "df_trials" and "side_bias_confidence_interval" in val.columns:
                        val["side_bias_confidence_interval_low"] = val["side_bias_confidence_interval"].apply(lambda x: x[0])
                        val["side_bias_confidence_interval_high"] = val["side_bias_confidence_interval"].apply(lambda x: x[1])
                        val = val.drop(columns=["side_bias_confidence_interval"])
                val.to_parquet(session_folder / f"{attr}.parquet", index=False, engine="fastparquet")

        return session_folder

    @classmethod
    def load(cls, session_folder, load_fip = False):
        """
        Load object from a saved session folder.
        """

        session_folder = Path(session_folder)

        obj = cls.__new__(cls)
        obj.session_id = session_folder.name
        obj.nwb_file_loc = None

        for file in session_folder.glob("*.parquet"):
            if "df_fip" in file.name and load_fip == False:
                continue
            setattr(obj, file.stem, pd.read_parquet(file, engine="fastparquet"))

        return obj

def save_nwb_list(flat_dummy_nwbs, plot_loc, df_sess=None):
    """
    Save a list or list-of-lists of dummy_nwb objects.

    Folder structure:
        plot_loc/
            <subject_id>/
                df_sess.parquet (optional)
                <session_id>/
                    <attr>.parquet
    """


    subject_ids = set()

    for nwb in flat_dummy_nwbs:

        subject_id = str(nwb.session_id).split("_")[0]
        subject_ids.add(subject_id)

        subject_folder = Path(plot_loc) / subject_id
        subject_folder.mkdir(parents=True, exist_ok=True)

        # call the class save method
        print(f'now saving {nwb.session_id}')
        nwb.save(subject_folder)

    # save optional df_sess once per subject
    if df_sess is not None:
        print(f"now saving df_sess")
        subject_ids_sorted = sorted(subject_ids)
        suffix = "_".join(subject_ids_sorted)
        df_sess.to_csv(
            Path(plot_loc) / f"df_sess_{suffix}.csv"
        )


def load_nwb_list(plot_loc, load_fip = False):
    """
    Load dummy_nwb objects from:

        plot_loc/
            df_sess.csv (optional)
            <subject_id>/
                <session_id>/
                    df_events.parquet
                    df_fip.parquet
                    df_trials.parquet
    """

    plot_loc = Path(plot_loc)

    nwbs = []
    df_sess = None

    # load df_sess if present
    sess_files = sorted(plot_loc.glob("df_sess*.csv"))
    if len(sess_files):
        print(f"loading {len(sess_files)} df_sess file(s)")
        dfs = [pd.read_csv(f) for f in sess_files]
        # concatenate into a single dataframe
        df_sess = pd.concat(dfs, ignore_index=True)
    else:
        print("no df_sess found at plot location, skipping")
        df_sess = None

    # load df_slope if present
    slope_files = glob.glob(f"{plot_loc}/**/rpe_slope.csv")
    if slope_files:
        print("loading df_slope")
        df_slope = pd.concat([pd.read_csv(file) for file in slope_files], ignore_index=True)
    else:
        print("no rpe_slope.csv found at plot location, skipping")
        df_slope = None

    # load sessions
    for subject_folder in sorted(plot_loc.iterdir()):

        if not subject_folder.is_dir():
            continue

        for session_folder in sorted(subject_folder.iterdir()):

            if not session_folder.is_dir():
                continue

            print(f"loading {session_folder.name}")

            nwb = dummy_nwb.load(session_folder, load_fip = load_fip)
            nwbs.append(nwb)

    return nwbs, df_sess, df_slope 



def get_dummy_nwbs(df_trials, df_events, df_fip):
    ses_idx_list = df_trials.ses_idx.unique()
    dummy_nwbs_list = []
    ses_dates_order = np.argsort(pd.to_datetime([ses_idx.split('_')[1] for ses_idx in ses_idx_list]))

    for ses_idx in ses_idx_list[ses_dates_order]:
        # Check if ses_idx exists in all 3 dataframes
        if (
            ses_idx in df_events['ses_idx'].values and
            ses_idx in df_fip['ses_idx'].values and
            ses_idx in df_trials['ses_idx'].values
        ):
            df_trials_i = df_trials[df_trials['ses_idx'] == ses_idx]
            df_events_i = df_events[df_events['ses_idx'] == ses_idx]
            df_fip_i = df_fip[df_fip['ses_idx'] == ses_idx]

            dummy_nwbs_list.append(dummy_nwb(df_trials_i, df_events_i, df_fip_i))
        else:
            warnings.warn(f"Skipping {ses_idx}: not found in all input DataFrames.", UserWarning)

    return dummy_nwbs_list

def get_dummy_nwbs_by_subject(df_trials, df_events, df_fip):
    df_trials['subject_id'] =  df_trials['ses_idx'].str.split('_').str[0]
    df_events['subject_id'] =  df_events['ses_idx'].str.split('_').str[0]
    df_fip['subject_id'] =  df_fip['ses_idx'].str.split('_').str[0]
    subject_id_list = df_trials.subject_id.unique()
    dummy_nwbs_list = []
    for subject_id in subject_id_list:
        # Check if ses_idx exists in all 3 dataframes
        if (
            subject_id in df_events['subject_id'].values and
            subject_id in df_fip['subject_id'].values and
            subject_id in df_trials['subject_id'].values
        ):
            df_trials_i = df_trials[df_trials['subject_id'] == subject_id]
            df_events_i = df_events[df_events['subject_id'] == subject_id]
            df_fip_i = df_fip[df_fip['subject_id'] == subject_id]

            dummy_nwbs_list.append(get_dummy_nwbs(df_trials_i, df_events_i, df_fip_i))
        else:
            warnings.warn(f"Skipping {subject_id}: not found in all input DataFrames.", UserWarning)

    return dummy_nwbs_list

def get_date_and_week_interval(df, start_date):
    date_series = pd.to_datetime(df['ses_idx'].str.split('_').str[1], format='%Y-%m-%d')
    week_interval_series = ((date_series - start_date).dt.days // 7) + 1
    return week_interval_series

def split_nwbs_by_week(nwbs_all):
    nwbs_by_week = []
    nwb_week_i = []
    curr_week = 1
    for nwb in nwbs_all:
        week_interval = nwb.df_trials.week_interval.unique()[0]
        if week_interval == curr_week:
            nwb_week_i.append(nwb)
        else:
            nwbs_by_week.append(nwb_week_i)
            nwb_week_i = [nwb]
            curr_week = week_interval
    nwbs_by_week.append(nwb_week_i)

    return nwbs_by_week

def enrich_nwb_by_week(df_sess, df_trials, df_events, df_fip):
    start_date = pd.to_datetime(df_sess['session_date'].min())

    df_sess['week_interval'] = get_date_and_week_interval(df_sess, start_date)
    df_trials['week_interval'] = get_date_and_week_interval(df_trials, start_date)
    df_events['week_interval'] = get_date_and_week_interval(df_events, start_date)
    df_fip['week_interval'] = get_date_and_week_interval(df_fip, start_date)

    return (df_sess, df_trials, df_events, df_fip)

def get_dummy_nwbs_by_week(df_sess,df_trials, df_events, df_fip):

    if 'week_interval' not in df_sess:
        df_sess, df_trials, df_events, df_fip = enrich_nwb_by_week(df_sess, df_trials, df_events, df_fip)
    week_interval_list = df_trials.week_interval.unique()
    dummy_nwbs_list = []
    for week_interval in week_interval_list:
        # Check if ses_idx exists in all 3 dataframes
        if (
            week_interval in df_events['week_interval'].values and
            week_interval in df_fip['week_interval'].values and
            week_interval in df_trials['week_interval'].values
        ):
            df_trials_i = df_trials[df_trials['week_interval'] == week_interval]
            df_events_i = df_events[df_events['week_interval'] == week_interval]
            df_fip_i = df_fip[df_fip['week_interval'] == week_interval]

            dummy_nwbs_list.append(get_dummy_nwbs(df_trials_i, df_events_i, df_fip_i))
        else:
            warnings.warn(f"Skipping {week_interval}: not found in all input DataFrames.", UserWarning)

    return df_sess, dummy_nwbs_list



def combine_dummy_nwbs_to_dfs(dummy_nwbs_list):
    """
    Given a list of dummy_nwb objects, concatenate their df_trials, df_events, and df_fip
    into three large DataFrames.

    Parameters
    ----------
    dummy_nwbs : list of dummy_nwb

    Returns
    -------
    tuple of pd.DataFrame
        (df_trials_all, df_events_all, df_fip_all)
    """

    df_trials_list = []
    df_events_list = []
    df_fip_list = []

    for nwb in dummy_nwbs_list:
        df_trials_list.append(nwb.df_trials)
        df_events_list.append(nwb.df_events)
        df_fip_list.append(nwb.df_fip)

    df_trials_all = pd.concat(df_trials_list, ignore_index=True)
    df_events_all = pd.concat(df_events_list, ignore_index=True)
    df_fip_all = pd.concat(df_fip_list, ignore_index=True)

    return df_trials_all, df_events_all, df_fip_all

def attach_dfs(nwb_file):

    nwb_file.df_events = nwb_utils.create_df_events(nwb_file)
    nwb_file.df_fip = nwb_utils.create_df_fip(nwb_file)
    nwb_file.df_trials = nwb_utils.create_df_trials(nwb_file)

    return nwb_file


def get_nwb_processed(file_locations, **parameters) -> None:
    interested_channels = list(parameters["channels"].keys())
    if parameters['preprocessing'] != "raw":
        interested_channels = [channel + '_' + parameters['preprocessing'] for channel in interested_channels]
    df_sess = nwb_utils.create_df_session(file_locations)
    df_sess['s3_location'] = file_locations

    # check for multiple sessions on the same day
    dup_mask = df_sess.duplicated(subset=['ses_idx'], keep=False)
    if dup_mask.any():
        warnings.warn(f"Duplicate sessions found for ses_idx: {df_sess[dup_mask]['ses_idx'].tolist()}."
                        "Keeping the one with more finished trials.")
        df_sess = (df_sess.sort_values(by=['ses_idx','finished_trials'], ascending=False)
                         .drop_duplicates(subset=['ses_idx'], keep='first')
                  )
    # sort sessions
    df_sess = (df_sess.sort_values(by=['session_date']) 
                         .reset_index(drop=True)
              )
    # only read last N sessions unless daily, weekly plots are requested
    if parameters["plot_types"]=="avg_lastN_sess":
        df_sess = df_sess.tail(parameters["last_N_sess"])

    
    (df_trials, df_events, df_fip) = co_utils.get_all_df_for_nwb(filename_sessions=df_sess['s3_location'].values, interested_channels = interested_channels)

        
    df_trials_fm, df_sess_fm = co_utils.get_foraging_model_info(df_trials, df_sess, loc = None, model_name = parameters["fitted_model"])
    df_trials_enriched = enrich_dfs.enrich_df_trials_fm(df_trials_fm)
    if len(df_fip):
        [df_fip_all, df_trials_fip_enriched] = enrich_dfs.enrich_fip_in_df_trials(df_fip, df_trials_enriched)
        (df_fip_final, df_trials_final, df_trials_fip) = enrich_dfs.remove_tonic_df_fip(df_fip_all, df_trials_enriched, df_trials_fip_enriched)
    else:
        warnings.warn(f"channels {interested_channels} not found in df_fip.")
        df_fip_final = df_fip
        df_trials_final = df_trials 
    # add week intervals
    df_sess_fm, df_trials_final, df_events, df_fip_final = enrich_nwb_by_week(df_sess_fm, df_trials_final, df_events, df_fip_final)
    # return all dataframes
    return (df_sess_fm, df_trials_final, df_events, df_fip_final) 