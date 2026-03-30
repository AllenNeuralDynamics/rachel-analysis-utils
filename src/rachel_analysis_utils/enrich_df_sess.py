
import pandas as pd
import numpy as np









def add_slope_to_df_sess(df_sess, df_slope, slope_col_name, channel_name,
                      session_date_col='session_date', channel_col='channel',
                      new_col_name=None):
    """
    Add a slope column from df_slope into df_sess by matching session_date.
    - df_sess: sessions dataframe
    - df_slope: slopes dataframe (must contain channel_col and session_date_col)
    - slope_col_name: name of the slope column in df_slope to pull (e.g. 'slope (RPE >= 0)')
    - channel_name: channel string to filter df_slope by
    - new_col_name: optional name for the added column in df_sess (defaults to slope_col_name)
    Returns a new dataframe (does not modify inputs).
    """
    if new_col_name is None:
        new_col_name = channel_name.split('dff')[0] + slope_col_name

    df_slope = df_slope.rename(columns={'date': session_date_col})
    # filter slope table to the requested channel and keep only date + slope column
    slope_filtered = (df_slope
                      .loc[df_slope[channel_col] == channel_name, [session_date_col, 'subject_id', slope_col_name]]
                      .copy())

    # merge into sessions on session date
    merged = df_sess.merge(slope_filtered, on=[session_date_col, 'subject_id'], how='left')

    # if original slope column name collides with other columns, rename to new_col_name
    if slope_col_name != new_col_name and slope_col_name in merged.columns:
        merged = merged.rename(columns={slope_col_name: new_col_name})

    return merged

def add_all_slopes_to_df_sess(df_sess, df_slope, slope_type,
                            session_date_col='session_date',
                            channel_col='channel'):

    
    df_sess_slope = df_sess.copy()

    if isinstance(slope_type, str):
        slope_type = [slope_type]
    slope_cols = []
    slope_col_names = {'slope (RPE >= 0)':'slope_pos', 'slope (RPE < 0)':'slope_neg'}
    if 'pos' in slope_type or 'positive' in slope_type or'slope (RPE >= 0)' in slope_type:
      slope_cols.append('slope (RPE >= 0)')
    if 'neg' in slope_type or 'negative' in slope_type or'slope (RPE < 0)' in slope_type:
      slope_cols.append('slope (RPE < 0)')
    if 'both' in slope_type or 'all' in slope_type:
      slope_cols.append(['slope (RPE >= 0)', 'slope (RPE < 0)'])

    for channel in df_slope[channel_col].unique():
        for slope_col in slope_cols:
            df_sess_slope = add_slope_to_df_sess(df_sess_slope, df_slope, slope_col, channel,
                                            session_date_col=session_date_col,
                                            channel_col=channel_col, 
                                            new_col_name=f'{channel.split("dff")[0]}{slope_col_names[slope_col]}')
    return df_sess_slope

def enrich_df_sess_from_nwbs(nwb_list, df_sess, extractor_dict: dict[str, callable]):
    """
    Call extractor_func for each nwb in nwb_list and merge results into df_sess.
    extractor_func returns a scalar value. 
    The merged column will be named new_col_name.
    Returns a new dataframe (does not modify inputs).
    """
    rows = []
    for nwb in nwb_list:
        extracted_res = {}
        for new_col_name, extractor_func in extractor_dict.items():
            res = extractor_func(nwb)
            extracted_res[new_col_name] = res

        session_date = nwb.session_id.split('_')[1]
        subject_id = nwb.session_id.split('_')[0]
        rows.append({'session_date': str(session_date) if session_date is not None else None, 
                      'subject_id': int(subject_id) if subject_id is not None else None,
                      **extracted_res})

    df_new = pd.DataFrame(rows).dropna(subset=['session_date'])

    # align column name for merge
    merged = df_sess.merge(df_new, on=['subject_id', 'session_date'], how='left')
    return merged

# a bunch of extractor functions


def get_max_side_bias(nwb):

  return nwb.df_trials['side_bias'].max()

def get_mean_side_bias(nwb):

  return nwb.df_trials['side_bias'].mean()

def get_baited_rate(nwb):
  df_trials = nwb.df_trials
  mask = ((df_trials['bait_left'] == True) & (df_trials['animal_response'] == 0.0)) | ((df_trials['bait_right'] == True) & (df_trials['animal_response'] == 1.0))
  return float(mask.sum()) / float(len(df_trials))

def get_left_choice_rate(nwb):
  df_trials = nwb.df_trials
  left_count = (df_trials['animal_response'] == 0.0).sum()
  return float(left_count) / float(len(df_trials))

def get_ignore_choice_rate(nwb):
  df_trials = nwb.df_trials
  ignore_count = (df_trials['animal_response'] == 2.0).sum()
  return float(ignore_count) / float(len(df_trials))

def get_left_right_diff(nwb):
    df = getattr(nwb, 'df_trials', None)
    if df is None or len(df) == 0:
        return float('nan')
    left = (df['animal_response'] == 0.0).sum()
    right = (df['animal_response'] == 1.0).sum()
    return float(left - right) / float(len(df))   # signed difference normalized by total trials

def get_left_right_abs_diff(nwb):
    v = get_left_right_diff(nwb)
    return abs(v) if not np.isnan(v) else v       # magnitude of bias (0..1)

def enrich_df_sess_with_all_getters(nwb_list, df_sess):
    """
    Enrich df_sess by calling enrich_df_sess_from_nwbs for each get_* extractor
    defined in this module. Returns a new dataframe with added columns:

      - 'max_side_bias'
      - 'mean_side_bias'
      - 'baited_rate'
      - 'left_choice_rate'
      - 'ignore_choice_rate'
      - 'left_right_diff'
      - 'left_right_abs_diff'

    Parameters:
      - nwb_list: iterable of nwb objects
      - df_sess: sessions dataframe

    This function does not modify the inputs; it returns an enriched copy.
    """
    enrichments = {
        'max_side_bias': get_max_side_bias,
        'mean_side_bias': get_mean_side_bias,
        'baited_rate': get_baited_rate,
        'left_choice_rate': get_left_choice_rate,
        'ignore_choice_rate': get_ignore_choice_rate,
        'left_right_diff': get_left_right_diff,
        'left_right_abs_diff': get_left_right_abs_diff,
    }

    df_out = df_sess.copy()
    
    df_out = enrich_df_sess_from_nwbs(nwb_list, df_out, enrichments)

    return df_out