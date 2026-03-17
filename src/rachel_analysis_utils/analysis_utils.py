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

output_col_name = lambda channel, data_column, alignment_event: f"avg_{data_column}_{channel[:3]}_{alignment_event.split("_in_")[0]}"

# ...existing code...
def add_AUC_and_rpe_slope(nwbs_by_week, parameters, data_column = 'data_z_norm', 
                            alignment_event = 'choice_time_in_session',offsets = [0.33,1]):
    """
    Enrich NWB weeks with average signal windows and compute RPE slopes per session for each channel.
    Fixes previous bug where only the last channel was saved.
    """
    nwbs_by_week_enriched = []

    # Enrich each week with average signals for every channel
    for nwb_week in nwbs_by_week:
        nwb_week_enriched = copy.deepcopy(nwb_week)
        for ch in list(parameters["channels"].keys()):
            # build the channel name used for processing (append preprocessing suffix if present)
            channel = ch
            if parameters.get('preprocessing', 'raw') != 'raw':
                channel = channel + '_' + parameters['preprocessing']

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

    for ch in list(parameters["channels"].keys()):
        channel = ch
        if parameters.get('preprocessing', 'raw') != 'raw':
            channel = channel + '_' + parameters['preprocessing']

        avg_signal_col = output_col_name(channel, data_column, alignment_event)

        
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



    if parameters.get("save_dfs", False) == True:
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
    _choice_shifted = df_trials.groupby('ses_idx')['choice'].shift(1)
    df_trials['stay'] = df_trials['choice'] == _choice_shifted
    df_trials['switch'] = df_trials['choice'] != _choice_shifted
    df_trials['response_time'] = df_trials['choice_time_in_trial'] -  df_trials['goCue_start_time_in_trial']


    return df_trials