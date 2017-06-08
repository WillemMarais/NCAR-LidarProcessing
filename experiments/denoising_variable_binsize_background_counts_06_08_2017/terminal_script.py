# The conent of the script is copy pasted into the Python/IPython terminal

import os
import sys
import socket

# ---------------------------------------------------------------------------------------------------------------------
# Remove conflicting Python modules from sys.path
# ---------------------------------------------------------------------------------------------------------------------
hostname_str = socket.gethostname ()
if (hostname_str == 'poisson.local') or (hostname_str == 'poisson.ssec.wisc.edu'):
    tmp_fileP_str = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/GitLab/poisson_inference_tools')
    if tmp_fileP_str in sys.path:
        sys.path.remove (tmp_fileP_str)

    tmp_fileP_str = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/Research_local/willem.marais.phd/source/python')
    if tmp_fileP_str in sys.path:
        sys.path.remove (tmp_fileP_str)

# ---------------------------------------------------------------------------------------------------------------------
# Add desired Python modules to sys.path
# ---------------------------------------------------------------------------------------------------------------------
hostname_str = socket.gethostname ()
if (hostname_str == 'poisson.local') or (hostname_str == 'poisson.ssec.wisc.edu'):
    base_dirP_str = 'ProjectsSSEC/GitLab'
else:
    base_dirP_str = 'PythonScripts'
        
tmp_fileP_str = os.path.join (os.environ ['HOME'], base_dirP_str, 'NCAR-LidarProcessing/libraries')
if tmp_fileP_str not in sys.path:
    sys.path.append (tmp_fileP_str)

tmp_fileP_str = os.path.join (os.environ ['HOME'], base_dirP_str, 
    'NCAR-LidarProcessing/experiments/denoising_variable_binsize_background_counts_06_08_2017')
if tmp_fileP_str not in sys.path:
    sys.path.append (tmp_fileP_str)

tmp_fileP_str = os.path.join (os.environ ['HOME'], base_dirP_str, 'ptvv1_python/python')
if tmp_fileP_str not in sys.path:
    sys.path.append (tmp_fileP_str)
    
# ---------------------------------------------------------------------------------------------------------------------
# Import modules
# ---------------------------------------------------------------------------------------------------------------------
import inference
import stage0_prepare_data
import ptv.hsrl.denoise as denoise

# ---------------------------------------------------------------------------------------------------------------------
# Do the experiment
# ---------------------------------------------------------------------------------------------------------------------

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stage 0 - prepare data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print ('(INFO) Stage 0 - Preparing data')

# Get data using Matt Hayman's code
delta_t_int = 5
hour_start_int = 0
hour_end_int = 24
# TODO: Set the directoru where the UCAR data are located 
ucar_dial_data_dirP_str = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/Research_local/UCARDial/Data')
cal_path_str = os.path.join (os.environ ['HOME'], base_dirP_str, 'NCAR-LidarProcessing/calibrations')
stage0_data_dct = stage0_prepare_data.get_data (delta_t_int, hour_start_int, hour_end_int, ucar_dial_data_dirP_str, 
    cal_path_str)

# Get the number of bins
N, _ = stage0_data_dct ['on_cnts_arr'].shape
# Set the bin start and end of the range bin region where the photon counts are denoised
del_R_flt = stage0_data_dct ['range_arr'][1] - stage0_data_dct ['range_arr'][0]
# Compute along the way how many bins should be used for the background count accumulation
bin_start_idx = N - int (50 * (37.47405725 / del_R_flt)) 
bin_end_idx = N

on_bg_arr, on_denoiser_obj = inference.denoise_background (stage0_data_dct ['on_cnts_arr'],
    stage0_data_dct ['on_nr_profiles_arr'], bin_start_idx, bin_end_idx)
noisy_on_bg_arr = stage0_data_dct ['orignal_on_cnts_arr'][bin_start_idx:bin_end_idx, :].mean (axis = 0)

# Plot the result
figure (1)
plot (noisy_on_bg_arr, label = 'Background counts via averaging')
plot (on_bg_arr.T, label = 'Denoised background counts')
title ('Online channel background counts')
xlabel ('Profile number')
ylabel ('Background counts')

# Plot the tuning parameters versus the validation error
log10_reg_flt_arr, vld_err_arr = on_denoiser_obj.get_validation_loss ()
figure (2)
plot (10**log10_reg_flt_arr, vld_err_arr / float (on_bg_arr.size), label = 'Validation error')
plot (10**log10_reg_flt_arr [vld_err_arr.argmin ()], vld_err_arr.min ()/float (on_bg_arr.size), 'rx', label = 'Minimum')
legend ()
semilogx ()
xlabel ('Number of profiles')
ylabel ('Normalized validation error')
title ('Validation error versus tuning parameter\nvalues for online background denoising')

off_bg_arr, off_denoiser_obj = inference.denoise_background (stage0_data_dct ['off_cnts_arr'],
    stage0_data_dct ['off_nr_profiles_arr'], bin_start_idx, bin_end_idx)
noisy_off_bg_arr = stage0_data_dct ['orignal_off_cnts_arr'][bin_start_idx:bin_end_idx, :].mean (axis = 0)

figure (3)
plot (noisy_off_bg_arr, label = 'Background counts via averaging')
plot (off_bg_arr.T, label = 'Denoised background counts')
title ('Offline channel background counts')
xlabel ('Profile number')
ylabel ('Background counts')

# Plot the tuning parameters versus the validation error
log10_reg_flt_arr, vld_err_arr = off_denoiser_obj.get_validation_loss ()
figure (4)
plot (10**log10_reg_flt_arr, vld_err_arr / float (off_bg_arr.size), label = 'Validation error')
plot (10**log10_reg_flt_arr [vld_err_arr.argmin ()], vld_err_arr.min ()/float (off_bg_arr.size), 'rx', label = 'Minimum')
legend ()
semilogx ()
xlabel ('Number of profiles')
ylabel ('Normalized validation error')
title ('Validation error versus tuning parameter\nvalues for offline background denoising')
