

import pandas as pd
import numpy as np


import copy 

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
        df_fip = nwb_annot.df_fip.copy()
        df_fip["intended_measurement"] = df_fip["event"].apply(lambda ev: _map_event_to_intended_measurement(ev, session_map, correct_map))
        nwb_annot.df_fip = df_fip.reset_index(drop=True)

        df_trials = nwb_annot.df_trials.copy()
        df_trials_col_map = _get_df_trials_col_mapping(session_map, df_trials.columns)
        nwb_annot.df_trials = df_trials.rename(columns=df_trials_col_map)

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