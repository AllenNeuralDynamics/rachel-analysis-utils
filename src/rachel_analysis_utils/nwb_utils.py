

import warnings
import glob
import pandas as pd
import numpy as np

from aind_dynamic_foraging_data_utils import nwb_utils, enrich_dfs
from aind_dynamic_foraging_data_utils import code_ocean_utils as co_utils


import copy 

def split_nwb(nwb):
    nwb_split = copy.deepcopy(nwb)
    nwb_split.df_trials_left = nwb.df_trials.query('choice == 0.0')
    nwb_split.df_trials_right = nwb.df_trials.query('choice == 1.0')
    nwb_split.df_trials_ignore = nwb.df_trials.query('choice == 2.0')
    return nwb_split


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

def get_dummy_nwbs_by_week(df_sess,df_trials, df_events, df_fip):
    start_date = pd.to_datetime(df_sess['session_date'].min())

    df_sess['week_interval'] = get_date_and_week_interval(df_sess, start_date)
    df_trials['week_interval'] = get_date_and_week_interval(df_trials, start_date)
    df_events['week_interval'] = get_date_and_week_interval(df_events, start_date)
    df_fip['week_interval'] = get_date_and_week_interval(df_fip, start_date)

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
    
    # return all dataframes
    return (df_sess, df_trials_final, df_events, df_fip_final) 