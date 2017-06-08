from __future__ import print_function
import os
import json
import datetime
import numpy as np
import WVProfileFunctions as wv
import LidarProfileFunctions as lp

def get_data (delta_t_int, hour_start_int, hour_end_int, ucar_dial_data_dirP_str, cal_path_str):
    """
    Parameters
    ----------
    delta_t_int : int
        The time resolution for the photon counting data.
    hour_start_int : int
        The starting hour of the data.
    hour_end_int : int
        The last hour of the data.
    ucar_dial_data_dirP_str : str
        The file path the directory that contains the UCARDial data. E.g. UCARDial/Data/
    cal_path_str : str
        The directory that contains the json calibration data.
    
    Returns
    -------
    A dictionary with the following fields:
        on_cnts_arr : A 2-D numpy array of the online-channel photon counts.
        off_cnts_arr : A 2-D numpy array of the offline-channel photon counts.
        on_nr_profiles_arr : A 1-D row-vector numpy array which indicate how many laser shots were accumulated in 
            the corresponding photon counting arrays for the online-channel.
        offline_nr_profiles_arr : A 1-D row-vector numpy array which indicate how many laser shots were accumulated in 
            the corresponding photon counting arrays for the offline-channel."""
    
    # Check whether ucar_dial_data_dirP_str ends with '/'
    if ucar_dial_data_dirP_str [-1] != os.path.sep:
        ucar_dial_data_dirP_str += os.path.sep
    
    # Field labels
    FieldLabel_WV = 'FF'
    ON_FileBase = 'Online_Raw_Data.dat'
    OFF_FileBase = 'Offline_Raw_Data.dat'

    # Set the profile bin size
    tres_wv = delta_t_int * 1.0    # HSRL bin time resolution in seconds (2 sec typical base)
    
    # Define time grid for lidar signal processing to occur
    # Years, Months, Days, Hours = lp.generate_WVDIAL_day_list (2017, 5, 24, startHr = hour_start_int,
    #     duration = hour_end_int - hour_start_int)
    Years, Months, Days, Hours = lp.generate_WVDIAL_day_list (2015, 5, 30, startHr = hour_start_int, 
        duration = hour_end_int - hour_start_int)

    ProcStart_dt = datetime.datetime (Years [0], Months [0], Days [0], 
        np.int (Hours [0] [0]), np.int (np.remainder (Hours [0][0], 1.0) * 60))
    
    cal_file_str = os.path.join (cal_path_str, 'dlb_calvals_msu.json')
    with open (cal_file_str, 'r') as f_obj:
        cal_jdata_str = json.loads (f_obj.read ())
    f_obj.close()

    MCSbins = lp.get_calval (ProcStart_dt, cal_jdata_str, 'MCS bins') [0]
    dt = lp.get_calval (ProcStart_dt, cal_jdata_str, 'Data Time Resolution') [0]
    BinWidth = lp.get_calval (ProcStart_dt, cal_jdata_str, 'Bin Width') [0]
    LaserPulseWidth = lp.get_calval (ProcStart_dt, cal_jdata_str, 'Laser Pulse Width') [0]

    Roffset = ((1.25 + 0.5) - 0.5 / 2) * lp.c * LaserPulseWidth / 2  # offset in range
    
    print ('[stage 0] Loading photon counting data for Far-field observations')
    MasterTimeWV = np.arange (Hours [0,0] * 3600, Days.size * 24 * 3600 - (24 - Hours [-1, -1]) * 3600, tres_wv)
    [OnLine, OffLine], [[lambda_on, lambda_off], [surf_temp], [surf_pres], [surf_humid]], HourLim = \
        wv.Load_DLB_Data (ucar_dial_data_dirP_str, FieldLabel_WV, [ON_FileBase, OFF_FileBase], 
            MasterTimeWV, Years, Months, Days, Hours, MCSbins, 
            lidar = 'WV-DIAL', dt = dt, Roffset = Roffset, BinWidth = BinWidth)
    
    # Get the online and offline photon counts
    stage0_data_dct = dict ()
    stage0_data_dct ['on_cnts_arr'] = (OnLine.profile.T * OnLine.NumProfList [np.newaxis]).astype (np.int)
    stage0_data_dct ['off_cnts_arr'] = (OffLine.profile.T * OffLine.NumProfList [np.newaxis]).astype (np.int)
    stage0_data_dct ['orignal_on_cnts_arr'] = OnLine.profile.T
    stage0_data_dct ['orignal_off_cnts_arr'] = OffLine.profile.T
    stage0_data_dct ['on_nr_profiles_arr'] = OnLine.NumProfList [np.newaxis]
    stage0_data_dct ['off_nr_profiles_arr'] = OffLine.NumProfList [np.newaxis]
    stage0_data_dct ['range_arr'] = OnLine.range_array [np.newaxis].T
    
    return stage0_data_dct
    