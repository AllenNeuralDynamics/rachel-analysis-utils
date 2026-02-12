import numpy as np
import pandas as pd 
from analysis_wrapper.plots import summary_plots
from aind_dynamic_foraging_basic_analysis.metrics import trial_metrics


def add_AUC_and_rpe_slope(nwbs_by_week, parameters, data_column = 'data_z_norm', 
                            alignment_event = 'choice_time_in_session',offsets = [0.33,1]):
    rpe_slope_dict = {}
    nwbs_by_week_enriched = []
    for channel in list(parameters["channels"].keys()):
        if parameters['preprocessing'] is not 'raw':
            channel = channel +  '_' + parameters['preprocessing'] 

        avg_signal_col = summary_plots.output_col_name(channel, data_column, alignment_event)
        for nwb_week in nwbs_by_week:
        
            nwb_week_enriched = trial_metrics.get_average_signal_window_multi(
                            nwb_week,
                            alignment_event=alignment_event,
                            offsets=offsets,
                            channel=channel,
                            data_column=data_column,
                            output_col = avg_signal_col
                        )
            nwbs_by_week_enriched.append(nwb_week_enriched)
        
        # get rpe slope per session 

        df_trials_all = pd.concat([nwb.df_trials for nwb_week in nwbs_by_week_enriched for nwb in nwb_week])
        rpe_slope = []
        for ses_idx in sorted(df_trials_all['ses_idx'].unique()):
            
            data = df_trials_all[df_trials_all['ses_idx'] == ses_idx]
            data = data.dropna(subset = [avg_signal_col, 'RPE_earned'])
            if len(data) == 0:
                continue
            data_neg = data[data['RPE_earned'] < 0]
            data_pos = data[data['RPE_earned'] >= 0]

            ses_date = pd.to_datetime(ses_idx.split('_')[1])
            (_,_, slope_pos) = summary_plots.get_RPE_by_avg_signal_fit(data_pos, avg_signal_col)
            (_,_, slope_neg) = summary_plots.get_RPE_by_avg_signal_fit(data_neg, avg_signal_col)
            rpe_slope.append([ses_date, slope_pos, slope_neg])
        rpe_slope = pd.DataFrame(rpe_slope, columns=['date', 'slope (RPE >= 0)', 'slope (RPE < 0)'])
        rpe_slope_dict[channel] = rpe_slope

    subject_id = str(nwbs_by_week_enriched[0][0]).split(' ')[1].split('_')[0]
    # Concatenate with keys, turning dict keys into an index
    combined_rpe_slope = pd.concat(rpe_slope_dict, names=["channel"])
    combined_rpe_slope = combined_rpe_slope.reset_index(level="channel").reset_index(drop=True)

    combined_rpe_slope.to_csv(f"/results/{subject_id}_rpe_slope.csv")

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