from __future__ import print_function
import os
import sys
import json
import datetime
import numpy as np
import cPickle as pickle
import WVProfileFunctions as wv
import LidarProfileFunctions as lp
from ptv.utilities.utilities import bin_image

def get_geoO_delR_120m ():
    fileP_str = os.path.join (os.path.dirname (__file__), '..', '..', 'calibrations', 'geo_DLB_20170524.npz')
    data_dct = np.load (fileP_str)
    
    # Get the geometric overlap and normalize it
    geoO_arr = 1 / data_dct ['geo_prof'] [:, 1:2]
    geoO_arr /= geoO_arr.max ()
    
    geo_range_arr = data_dct ['geo_prof'] [:, 0:1]
    
    # Bin the overlap and the corresponding range array
    binned_geoO_arr = bin_image (geoO_arr, 4, 1) / 4.0
    binned_geo_range_arr = bin_image (geo_range_arr, 4, 1) / 4.0
    
    return binned_geo_range_arr, binned_geoO_arr

def get_data_delR_120m_delT_120s (recompute_bl = False):
    """Return photon counting data with a range resolution of 120m, and time resolution of 120s. The dataset that is 
    used is from 2017-05-24, from hour 0 (UTC?) to hour 18."""
    
    data_fileP_str = os.path.join (os.path.dirname (__file__), 'data', 'data_delR_120m_delT_120s.p')
    if (os.path.exists (data_fileP_str) is True) and (recompute_bl is False):
        data_dct = pickle.load (open (data_fileP_str, 'rb'))
        
        return data_dct
    
    # Set the paths of the data
    basepath = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/Research_local/UCARDial/Data/')

    # Set the private data directory
    priv_dataP_str = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/GitLab/NCAR-LidarProcessing/data')

    # path to calibration files
    cal_path_str = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/GitLab/NCAR-LidarProcessing/calibrations')
    cal_file_str = os.path.join (cal_path_str, 'dlb_calvals_msu.json')

    # Field labels
    FieldLabel_WV = 'FF'
    ON_FileBase = 'Online_Raw_Data.dat'
    OFF_FileBase = 'Offline_Raw_Data.dat'

    # Set the bin size
    # QUESTION: 60 seconds?
    # QUESTION: Are these the profile duration?
    tres_wv = 1.0 * 1    # HSRL bin time resolution in seconds (2 sec typical base)
    zres = 37.5  # bin range resolution in m (37.5 m typical base)

    # parameters for WV Channels smoothing
    tsmooth_wv = 5 * 60 # convolution kernal time (HW sigma) in seconds
    zsmooth_wv = 1 #150  # convolution kernal range (HW sigma) in meters

    # Set maximum altitude
    MaxAlt = 6e3 #12e3

    # Define time grid for lidar signal processing to occur
    Years, Months, Days, Hours = lp.generate_WVDIAL_day_list (2017, 5, 24, startHr = 0, duration = 18)

    ProcStart_dt = datetime.datetime (Years [0], Months [0], Days [0], 
        np.int (Hours [0] [0]), np.int (np.remainder (Hours [0][0], 1.0) * 60))

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
        wv.Load_DLB_Data (basepath, FieldLabel_WV, [ON_FileBase, OFF_FileBase], 
            MasterTimeWV, Years, Months, Days, Hours, MCSbins, 
            lidar = 'WV-DIAL', dt = dt, Roffset = Roffset, BinWidth = BinWidth)

    # WILLEM MARAIS PLAY - START

    # Remove the profiles that do not have any data
    tmp_arr = OnLine.profile.T.sum (axis = 0)
    on_smpl_idx = np.where (tmp_arr > 0) [0]

    tmp_arr = OffLine.profile.T.sum (axis = 0)
    off_smpl_idx = np.where (tmp_arr > 0) [0]

    on_cnts_arr = (OnLine.profile.T.copy () * OnLine.NumProfList [np.newaxis]) [:, on_smpl_idx]
    off_cnts_arr = (OffLine.profile.T.copy () * OffLine.NumProfList [np.newaxis]) [:, off_smpl_idx]
    N, K = on_cnts_arr.shape

    # Do a greedy accumulation over the temporal axis. Assuming that each accumulated shot is 2s. 
    # What time resolution do we want?
    del_Tshot_flt = 2.0 # Seconds
    del_T_flt = 120.0 # Seconds
    nr_accum_prfl_int = int (del_T_flt / del_Tshot_flt)

    # Truncate the photon counting images so that the number of profiles is divisable with the number of profiles that 
    # will be used to do the accumulation
    new_K = int (np.floor (K / nr_accum_prfl_int) * nr_accum_prfl_int)
    on_cnts_arr = bin_image (on_cnts_arr [:, :new_K], 4, nr_accum_prfl_int)
    off_cnts_arr = bin_image (off_cnts_arr [:, :new_K], 4, nr_accum_prfl_int)
    # Get the range axis
    pre_bin_range_arr = OnLine.range_array [np.newaxis].T
    range_arr = bin_image (OnLine.range_array [np.newaxis].T, 4, 1) / 4.0
    
    # Readjust the nr of profiles and bins
    N, K = on_cnts_arr.shape

    # Compute the background radiation
    on_bg_arr = on_cnts_arr [(N - 12):, :].mean (axis = 0) [np.newaxis]
    off_bg_arr = off_cnts_arr [(N - 12):, :].mean (axis = 0) [np.newaxis]

    # on_cnts_bs_arr = on_cnts_arr - on_bg_arr
    # off_cnts_bs_arr = off_cnts_arr - off_bg_arr
    #
    # phi_arr = np.log (on_cnts_bs_arr * off_cnts_bs_arr)
    # psi_arr = np.log (on_cnts_bs_arr / off_cnts_bs_arr)

    # Get the molecular backscatter
    print ('[stage 0] Loading H20 differential absorption coefficient')
    tres_wv = 120.0
    MasterTimeWV = np.arange (Hours [0,0] * 3600, Days.size * 24 * 3600 - (24 - Hours [-1, -1]) * 3600, tres_wv)
    [OnLine, OffLine], [[lambda_on, lambda_off], [surf_temp], [surf_pres], [surf_humid]], _ = \
        wv.Load_DLB_Data (basepath, FieldLabel_WV, [ON_FileBase, OFF_FileBase], 
            MasterTimeWV, Years, Months, Days, Hours, MCSbins, 
            lidar = 'WV-DIAL', dt = dt, Roffset = Roffset, BinWidth = BinWidth)

    beta_mol_sonde, temp, pres = lp.get_beta_m_model (OffLine, surf_temp, surf_pres, returnTP = True)
    pres.descript = 'Ideal Atmosphere Pressure in atm'
    # convert pressure from Pa to atm.
    pres.gain_scale (9.86923e-6)
    pres.profile_type = '$atm.$'

    range_diff = OnLine.range_array [1:] - OnLine.mean_dR / 2.0  # range grid for diffentiated signals
    # create the water vapor profile
    nWV = OnLine.copy()
    nWV.label = 'Water Vapor Number Density'
    nWV.descript = 'Water Vapor Number Density'
    nWV.profile_type = '$m^{-3}$'
    nWV.range_array = range_diff

    wv_abs_cs_fileP_str = os.path.join (priv_dataP_str, 'wv_absorption_cross_section_2017_05_24_00h00_18h00_resol_120s.p')
    if os.path.exists (wv_abs_cs_fileP_str) is False:
        print ('[stage 0] (WARNING) The file {:s} does not exist, recomputing H20 differential absorption coefficient.')
        dsig = np.zeros ((nWV.time.size, nWV.range_array.size))
        for ai in range (OnLine.time.size):
            # compute frequencies from wavelength terms
            nuOff = lp.c / lambda_off [ai]  # Offline laser frequency
            nuOn = lp.c / lambda_on [ai]   # Online laser frequency
            sigWV0 = lp.WV_ExtinctionFromHITRAN (np.array ([nuOn, nuOff]), temp.profile [ai, :], pres.profile [ai, :], 
                nuLim = np.array ([lp.c / 828.5e-9, lp.c / 828e-9]), freqnorm = True)
            sigOn = sigWV0 [:, 0]
            sigOff = sigWV0 [:, 1]
    
            # interpolate difference in absorption to range_diff grid points
            dsig[ai,:] = np.interp (range_diff, OnLine.range_array, sigOn - sigOff)  
        
            with open (wv_abs_cs_fileP_str, 'wb') as file_obj:
                pickle.dump (dsig, file_obj, protocol = 2)
    else:
        dsig = pickle.load (open (wv_abs_cs_fileP_str, 'rb'))
    
    # Remove the first bin from the photon counting observations so to remove laser energy pulse scattering
    on_cnts_arr = on_cnts_arr [1:, :]
    off_cnts_arr = off_cnts_arr [1:, :]    
    range_arr = range_arr [1:, :]
    pre_bin_range_arr = pre_bin_range_arr [1:, :]
    
    # Reduce the resolution of the difference absorption coefficient
    binned_dsig_arr = bin_image (dsig.T, 4, 1) / 4.0
    
    # Truncate dsig over the temporal axis. This is a hack; need to find a way to properly align photon counting images
    # and difference absorption coefficient.
    binned_dsig_arr = binned_dsig_arr [:, 28:][:, :-28]
    
    data_dct = dict (range_arr = range_arr,
        pre_bin_range_arr = pre_bin_range_arr,
        on_cnts_arr = on_cnts_arr,
        off_cnts_arr = off_cnts_arr,
        binned_dsig_arr = binned_dsig_arr)
    
    data_fileP_str = os.path.join (os.path.dirname (__file__),  'data', 'data_delR_120m_delT_120s.p')
    with open (data_fileP_str, 'wb') as file_obj:
        pickle.dump (data_dct, file_obj, protocol = 2)
    
    return data_dct
    