import numpy as np
import pandas as pd
import copy 
from aind_dynamic_foraging_basic_analysis.metrics import trial_metrics
from scipy import stats



def get_RPE_by_avg_signal_fit(data, avg_signal_col):


    x = data['RPE_earned'].values
    y = data[avg_signal_col].values
    try:
        lr = stats.linregress(x, y)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = lr.intercept + lr.slope * x_fit
        slope = lr.slope
    except ValueError as e:
        print(f"Error in linear regression: {e}")
        x_fit = np.nan * np.arange(100)
        y_fit = np.nan * np.arange(100)
        slope = np.nan
    return (x_fit, y_fit, slope)

output_col_name = lambda channel, data_column, alignment_event: f"avg_{data_column}_{channel.split('_dff')[0]}_{alignment_event.split('_in_')[0]}"

def add_AUC_and_rpe_slope(nwbs_by_week, all_channels, save_dfs, data_column = 'data_z_norm', 
                            alignment_event = 'choice_time_in_session',offsets = [0.33,1]):
    """
    Enrich NWB weeks with average signal windows and compute RPE slopes per session for each channel.
    Fixes previous bug where only the last channel was saved.
    """
    nwbs_by_week_enriched = []

    # Enrich each week with average signals for every channel
    for nwb_week in nwbs_by_week:
        nwb_week_enriched = copy.deepcopy(nwb_week)
        for channel in all_channels:
            avg_signal_col = output_col_name(channel, data_column, alignment_event)

            nwb_week_enriched = trial_metrics.get_average_signal_window_multi(
                nwb_week_enriched,
                alignment_event=alignment_event,
                offsets=offsets,
                channel=channel,
                data_column=data_column,
                output_col=avg_signal_col
            )
        nwbs_by_week_enriched.append(nwb_week_enriched)

    # After enriching all weeks, compute RPE slopes per session for each channel
    df_trials_all = pd.concat([nwb.df_trials for nwb_week in nwbs_by_week_enriched for nwb in nwb_week])
    rpe_rows = []
    subject_id = str(nwbs_by_week_enriched[0][0]).split(' ')[1].split('_')[0]

    for channel in all_channels:

        avg_signal_col = output_col_name(channel, data_column, alignment_event)
        if avg_signal_col not in df_trials_all.columns:
            continue
        
        for ses_idx in sorted(df_trials_all['ses_idx'].unique()):
            data = df_trials_all[df_trials_all['ses_idx'] == ses_idx]
            data = data.dropna(subset=[avg_signal_col, 'RPE_earned'])
            if len(data) == 0:
                continue

            data_neg = data[data['RPE_earned'] < 0]
            data_pos = data[data['RPE_earned'] >= 0]

            ses_date = pd.to_datetime(ses_idx.split('_')[1])
            (_, _, slope_pos) = get_RPE_by_avg_signal_fit(data_pos, avg_signal_col)
            (_, _, slope_neg) = get_RPE_by_avg_signal_fit(data_neg, avg_signal_col)
            rpe_rows.append([subject_id, ses_date, channel, slope_pos, slope_neg])

    # Combine per-channel dataframes into one table with a channel column

    combined_rpe_slope = pd.DataFrame(rpe_rows, columns=['subject_id', 'date', 'channel', 'slope (RPE >= 0)', 'slope (RPE < 0)'])



    if save_dfs == True:
        combined_rpe_slope.to_csv(f"/results/data/{subject_id}/rpe_slope.csv")

    return nwbs_by_week_enriched, combined_rpe_slope

def enrich_df_trials(df_trials):

##### PART I: REWARD #######
    df_trials['reward_all'] = df_trials['earned_reward'] + df_trials['extra_reward']
    # Compute num_reward_past and num_no_reward_past
    df_trials['rewarded_prev'] = df_trials.groupby('ses_idx')['reward_all'].shift(1)  # Shift to look at past values

    df_trials['num_reward_past'] = df_trials.groupby(
                            (df_trials['rewarded_prev'] != df_trials['reward_all']).cumsum()).cumcount() + 1

    # Set 'NA' for mismatched reward types
    df_trials.loc[df_trials['reward_all'] == 0, 'num_reward_past'] = df_trials.loc[df_trials['reward_all'] == 0, 'num_reward_past']* -1 

    ##### PART II: BINNING RPE #######
    # get RPE binned columns. 
    RPE_binned3_label_names = [str(np.round(i,2)) for i in np.arange(-1,0.99,1/3)]

    bins = np.arange(-1,1.01,1/3)
    bins[-1] = 1.001

    df_trials['RPE-binned3'] = pd.cut(df_trials['RPE_earned'],# all versus earned not a huge difference
                        bins = bins, right = True, labels=RPE_binned3_label_names)

    ##### PART III: BINNING QCHOSEN #######
    bins = [0.0, 1/3, 2/3, 1.01]
    q_labels = ["Qch 0", "Qch 0.33", "Qch 0.66"]

    q_bin = pd.cut(df_trials['Q_chosen'], bins=bins, labels=q_labels, include_lowest=True, right=True)
    reward_label = df_trials['earned_reward'].map({True: "R+", False: "R-"})

    # build combined label series (None where q_bin is NA)
    reward_Qcat_series = pd.Series(
        np.where(q_bin.isna(), None, reward_label.astype(str) + " (" + q_bin.astype(str) + ")"),
        index=df_trials.index
    )

    # ordered categories you requested
    Qch_binned3_label_names = [
        "R- (Qch 0)", "R- (Qch 0.33)", "R- (Qch 0.66)",
        "R+ (Qch 0)", "R+ (Qch 0.33)", "R+ (Qch 0.66)"
    ]

    # assign final ordered categorical to dataframe (no intermediate column left behind)
    df_trials['Qch-binned3'] = pd.Categorical(reward_Qcat_series, categories=Qch_binned3_label_names, ordered=True)

    
    ##### PART IV: GETTING STAY/LEAVE #######
    
    _choice_shifted = df_trials.groupby('ses_idx')['animal_response'].shift(1)
    df_trials['stay'] = df_trials['animal_response'] == _choice_shifted
    df_trials['switch'] = df_trials['animal_response'] != _choice_shifted
    df_trials['response_time'] = df_trials['choice_time_in_trial'] -  df_trials['goCue_start_time_in_trial']


    return df_trials

def interp_to_uniform(x_old, y_old, x_new):
    mask = np.isfinite(y_old) & np.isfinite(x_old)
    if np.sum(mask) < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    return np.interp(x_new, x_old[mask], y_old[mask])
def resample_pair_to_uniform(df_sel, signal1name, signal2name, value_col, fs, min_len=10):
    """
    Build separate per-signal timeseries, find overlap, and interpolate both onto
    a common uniform timebase. Returns (t, s1, s2) or None if resampling is not possible.
    """
    s1_df = (
        df_sel.loc[df_sel['event'] == signal1name, ['timestamps', value_col]]
              .dropna()
              .groupby('timestamps', as_index=False)
              .first()
              .sort_values('timestamps')
    )
    s2_df = (
        df_sel.loc[df_sel['event'] == signal2name, ['timestamps', value_col]]
              .dropna()
              .groupby('timestamps', as_index=False)
              .first()
              .sort_values('timestamps')
    )

    if s1_df.empty or s2_df.empty:
        return None

    t1 = s1_df['timestamps'].to_numpy(dtype=float)
    v1 = s1_df[value_col].to_numpy(dtype=float)
    t2 = s2_df['timestamps'].to_numpy(dtype=float)
    v2 = s2_df[value_col].to_numpy(dtype=float)

    # overlapping interval intersection
    start = max(t1.min(), t2.min())
    end = min(t1.max(), t2.max())
    if end <= start:
        return None

    t_uniform = np.arange(start, end + 1.0 / fs / 2, 1.0 / fs)
    if len(t_uniform) < min_len:
        return None

    s1 = interp_to_uniform(t1, v1, t_uniform)
    s2 = interp_to_uniform(t2, v2, t_uniform)
    t = t_uniform - t_uniform[0]
    return (t, s1, s2)

def add_sliding_window_corr(
    nwb,
    signal1name: str,
    signal2name: str,
    value_col: str = "data_z",
    fs: float = 20.0,          # Hz
    window_sec: float = 1.0,
    step_sec: float = 0.1,
    min_valid_frac: float = 0.8,
):
    """
    df_fip is long-form: rows are (timestamp, value, event==signal_name).
    Computes sliding-window Pearson correlation across TIME samples.
    fs : float, default 20.0
        Sampling rate of the signals in Hz.
        This is used to convert window size and step size from seconds
        to number of samples. For 20 Hz photometry, fs=20.0.

    window_sec : float, default 1.0
        Length of the sliding window in seconds.
        Correlation at each time point is computed using samples within
        this window. For photometry, values between 0.75–1.5 s are typical.

    step_sec : float, default 0.1
        Step size between successive sliding windows in seconds.
        Smaller values give smoother correlation traces at higher cost.

    min_valid_frac : float, default 0.8
        Minimum fraction of finite (non-NaN) samples required within
        a window for correlation to be computed. Windows with too many
        NaNs are skipped.
    

    Returns a dict with:
      nwb : updated nwb with pearsonR included
    """
    if hasattr(nwb, "df_fip"):
        df_fip = getattr(nwb, "df_fip")
    else:
        raise ValueError("Input 'nwb' must have a 'df_fip' attribute or be a DataFrame with fip data.")

    if signal1name not in df_fip.event.unique() or signal2name not in df_fip.event.unique():
        print("One or both signals not found in df_fip events. Skipping correlation.")
        return nwb
   # Select and pivot the two signals directly (align by timestamp)
    df_sel = (
        df_fip.loc[df_fip['event'].isin([signal1name, signal2name]),
                   ['timestamps', 'event', value_col]]
              .dropna()
    )

    if df_sel.empty:
        raise ValueError("No data found for the requested signals in df_fip.")

    merged = df_sel.pivot_table(index='timestamps', columns='event', values=value_col, aggfunc='first').reset_index()

    if signal1name not in merged.columns or signal2name not in merged.columns:
        raise ValueError("One or both signals not present after pivoting (maybe all NaN).")

    t_abs = merged['timestamps'].to_numpy(dtype=float)
    s1 = merged[signal1name].to_numpy(dtype=float)
    s2 = merged[signal2name].to_numpy(dtype=float)
    
    if len(t_abs) < 10:
        raise ValueError("Too few aligned samples after merging on timestamps.")
    # If there are very few timepoints where both signals are finite, resample
    # onto a uniform timebase at frequency `fs` and interpolate both signals.
    overlap_count = np.sum(np.isfinite(s1) & np.isfinite(s2))
    if overlap_count < max(10, int(len(t_abs) * 0.1)):
        # build uniform timebase spanning available timestamps
        try:
            t, s1, s2 = resample_pair_to_uniform(df_sel, signal1name, signal2name, value_col, fs, min_len=10)
        except:
            print("cannot calculate pearsonr")
            return nwb
    else:
        # Time axis for plotting / window centers (no trel): seconds from start
        t = t_abs - t_abs[0]
    # Sliding window params (in samples)
    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    if win < 5:
        raise ValueError("window_sec too small; need at least ~5 samples.")
    if step < 1:
        raise ValueError("step_sec too small; must be >= 1/fs.")

        # Use pandas rolling correlation (centered). Use integer index so window size is number of samples.
    s1s = pd.Series(s1)
    s2s = pd.Series(s2)

    min_periods = max(5, int(np.ceil(win * min_valid_frac)))

    # rolling .corr handles pairwise NaNs and produces NaN where insufficient valid pairs
    r_full = s1s.rolling(window=win, center=True, min_periods=min_periods).corr(s2s)

    # sample the centered rolling correlation at the same centers as before
    centers = np.arange(win // 2, len(t) - win // 2, step)
    t_centers = t[centers]

    r = r_full.to_numpy(dtype=float)[centers]

    df_corr = pd.DataFrame({'data_z':r, 'timestamps':t_centers})
    df_corr['event'] = f'{signal1name.split('_dff')[0]}:{signal2name.split('_dff')[0]}_pearsonR'

    # If the entire correlation vector is NaN, nothing to merge — return unchanged.
    if np.all(np.isnan(r)):
        return nwb

    nwb.df_fip = nwb.df_fip.merge(df_corr, how = 'outer')
    nwb.df_fip = nwb.df_fip.sort_values( by = 'timestamps')

    return nwb
