

import warnings

from aind_dynamic_foraging_data_utils import nwb_utils, enrich_dfs
from aind_dynamic_foraging_data_utils import code_ocean_utils as co_utils


def attach_dfs(nwb_file):

    nwb_file.df_events = nwb_utils.create_df_events(nwb_file)
    nwb_file.df_fip = nwb_utils.create_df_fip(nwb_file)
    nwb_file.df_trials = nwb_utils.create_df_trials(nwb_file)

    return nwb_file


def get_nwb_processed(file_locations, **parameters) -> None:
    interested_channels = list(parameters["channels"].keys())
    if parameters['preprocessing'] is not "raw":
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