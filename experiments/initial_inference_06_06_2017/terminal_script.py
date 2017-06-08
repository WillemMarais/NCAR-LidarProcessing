# The conent of the script is copy pasted into the Python/IPython terminal

import os
import sys
import socket
import matplotlib.pyplot as plt

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
    'NCAR-LidarProcessing/experiments/initial_inference_06_06_2017')
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

## Define different profiles indices that select different regions of the DIAL image
# # 1. Night time data without clouds
# prfl_start_idx = 78
# prfl_end_idx = 300
# 2. Night time data with clouds in the begining and partly day time
prfl_start_idx = 40
prfl_end_idx = 300

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stage 0 - prepare data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print ('(INFO) Stage 0 - Preparing data')

# Get data using Matt Hayman's code
stage0_data_dct = stage0_prepare_data.get_data_delR_120m_delT_120s (recompute_bl = False, recompute_sig_bl = False)
# Select the portion of the data we want to work with
stage0_data_dct ['on_cnts_arr'] = stage0_data_dct ['on_cnts_arr'][:, prfl_start_idx:prfl_end_idx]
stage0_data_dct ['off_cnts_arr'] = stage0_data_dct ['off_cnts_arr'][:, prfl_start_idx:prfl_end_idx]
stage0_data_dct ['binned_dsig_arr'] = stage0_data_dct ['binned_dsig_arr'][:, prfl_start_idx:prfl_end_idx]

# Get geometric overlap function
geo_range_arr, geoO_arr = stage0_prepare_data.get_geoO_delR_120m ()
# Safe the geometric overlap function to the stage0 dictionary
stage0_data_dct ['geoO_arr'] = geoO_arr

# Denoise the background; set the bin numbers which select the region from which the background counts are computed
bin_start_idx = 127
bin_end_idx = 139

on_bg_arr = inference.denoise_background (stage0_data_dct ['on_cnts_arr'], bin_start_idx, bin_end_idx)
off_bg_arr = inference.denoise_background (stage0_data_dct ['off_cnts_arr'], bin_start_idx, bin_end_idx)

# Save the background counts to the stage0 dictionary
stage0_data_dct ['on_bg_arr'] = on_bg_arr
stage0_data_dct ['off_bg_arr'] = off_bg_arr

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reduce the size of the image to make it more computational feasible
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# TODO: There must be a better way of doing this; figure out something...
stage0_reduced_data_dct = dict ()
stage0_reduced_data_dct ['range_arr'] = stage0_data_dct ['range_arr'][0:80, :]
stage0_reduced_data_dct ['geoO_arr'] = stage0_data_dct ['geoO_arr'][0:80, :]
stage0_reduced_data_dct ['pre_bin_range_arr'] = stage0_data_dct ['pre_bin_range_arr']
stage0_reduced_data_dct ['on_cnts_arr'] = stage0_data_dct ['on_cnts_arr'].copy () [0:80, 0:1]
stage0_reduced_data_dct ['off_cnts_arr'] = stage0_data_dct ['off_cnts_arr'].copy () [0:80, 0:1]
stage0_reduced_data_dct ['on_bg_arr'] = stage0_data_dct ['on_bg_arr'].copy () [:, 0:1]
stage0_reduced_data_dct ['off_bg_arr'] = stage0_data_dct ['off_bg_arr'].copy () [:, 0:1]
stage0_reduced_data_dct ['binned_dsig_arr'] = stage0_data_dct ['binned_dsig_arr'].copy () [0:80, 0:1]
stage0_reduced_data_dct ['binned_on_sig_arr'] = stage0_data_dct ['binned_on_sig_arr'].copy () [0:80, 0:1]
stage0_reduced_data_dct ['binned_off_sig_arr'] = stage0_data_dct ['binned_off_sig_arr'].copy () [0:80, 0:1]
stage0_reduced_data_dct ['scale_to_H2O_den_flt'] = stage0_data_dct ['scale_to_H2O_den_flt']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stage 1 - get intial estimate of the attenuated backscatter cross-section, which I call \chi. In this stage we 
# assume that the water vapor optical depth is zero, which is obviously not true. But, we get some estimate of the 
# attenuated backscatter cross-section which is useful. 
# 
# The forward model in stage1 is 
# F(\chi) = A\exp(\chi) + b_{off}, 
# where A is the geometric overlap function divided by the squared range and b_{off} is the background counts of the 
# offline channel. So basically \chi is the log of some of the the calibration parameters time the attenduated 
# backscatter cross-section.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print ('(INFO) Stage 1 - Get initial estimate of the attenuated backscatter cross-section')

sparsa_cfg_obj = denoise.sparsaconf (eps_flt = 1e-6)
# Set the number of tuning parameters to be 24, from 10**-2 to 10**1
kwargs_denoiseconf_dct = dict (log10_reg_lst = [-2, 1], nr_reg_int = 24)
# Get the estimate of the attenuated backscatter cross-section
hat_chi_arr, _ = inference.get_denoiser_atten_backscatter_chi (stage0_reduced_data_dct, sparsa_cfg_obj, 
    kwargs_denoiseconf_dct)
# Let's recreate the backscattered energy of the offline channel

geoO_arr = stage0_reduced_data_dct ['geoO_arr']
range_arr = stage0_reduced_data_dct ['range_arr']
off_bg_arr = stage0_reduced_data_dct ['off_bg_arr']
A_arr = geoO_arr / (range_arr**2)
# TODO: Find a better way to scale A_arr. Maybe use the time resolution. The reason why I scale it is to make the 
# values of \chi reasonable. E.g., if A_arr is small, the values of \chi will be like 6 or more. If it not really a 
# big deal, I just want to prevent overflows (i.e. if \chi is too large, \exp(\chi) might give a floating point 
# overflow error).
A_arr = A_arr / A_arr.max () * 1000
denoised_off_y_arr = A_arr * np.exp (hat_chi_arr) + off_bg_arr

# Plot the noisy and denoised offline channel observations. They will look very similar for high SNR data. For low 
# SNR data there will be a significant difference
off_y_arr = stage0_reduced_data_dct ['off_cnts_arr'] # The noisy observations of the offline channel
plt.figure (1);
plt.plot (range_arr * 1e-3, off_y_arr, label = 'Noisy offline channel')
plt.plot (range_arr * 1e-3, denoised_off_y_arr, label = 'Denoised offline channel')
plt.legend ()
plt.xlabel ('Range [km]')
plt.ylabel ('Backscattered energy photon counts')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stage 2 - now estimate both the water vapor and the attenuated backscatter cross-section. The water vapor is 
# denoted by \varphi, and the \chi denotes the attenuated backscatter cross-section.
# 
# The forward models are as follows. Let \sigma_{on} and \sigma_{off} be the H20 absorption coefficients for the 
# online and offline channels, respectively. Let A be matrix as defined in stage 1; i.e. A is the product of the
# geometric overlap function and one over the squared range. 
# 
# The forward model for the online channel is 
# G_{on}(\varphi, \chi) = A\exp(\chi)\exp(-2Q[\sigma_{on}\cdot\varphi]) + b_{on},
# and for the offline channel it is 
# G_{off}(\varphi, \chi) = A\exp(\chi)\exp(-2Q[\sigma_{off}\cdot\varphi]) + b_{off},
# where Q is the integrator matrix. 
# 
# We estimate \varphi and \chi directy using these forwards models.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print ('(INFO) Stage 2 - Estimating water vapor and attenuated backscatter cross-section')

prev_hat_chi_arr = hat_chi_arr.copy ()
max_iter_int = 100
epsilon_flt = 1e-5
verbose_bl = False
tau_varphi_flt = 1
tau_chi_flt = 1
# Create SpaRSA configuration objects
sparsa_cfg_chi_obj = denoise.sparsaconf (max_iter_int = 1e5, M_int = 0, verbose_int = 1e6)
sparsa_cfg_varphi_obj = denoise.sparsaconf (max_iter_int = 1e5, M_int = 0, verbose_int = 1e6)

# Try different tuning parmeters. Set the chi and varphi tuning parameters equal to each other
# tau_flt_arr = np.logspace (-1, 1, 12)
tau_flt_arr = np.logspace (0, 0, 1)
# Record the validation errors
vld_err_arr = np.zeros_like (tau_flt_arr)

for tau_idx in range (tau_flt_arr.size):
    tau_varphi_flt = 1 * tau_flt_arr [tau_idx]
    tau_chi_flt = tau_flt_arr [tau_idx]
    
    message_str = '[{:d}/{:d}] tau_flt = {:.2e}'.format (tau_idx + 1, tau_flt_arr.size, tau_chi_flt)
    print (message_str)
    
    hat_varphi_arr, hat_chi_arr, j_idx, objF_arr, _vld_err_arr, re_step_avg_arr, re_step_varphi_arr, re_step_chi_arr = \
        inference.estimate_water_vapor_varphi (stage0_reduced_data_dct, prev_hat_chi_arr, tau_chi_flt, tau_varphi_flt, 
            max_iter_int, epsilon_flt, verbose_bl, sparsa_cfg_chi_obj, sparsa_cfg_varphi_obj)
    
    vld_err_arr [tau_idx] = _vld_err_arr [j_idx]

opt_tau_flt = tau_flt_arr [vld_err_arr.argmin ()]
tau_varphi_flt = opt_tau_flt
tau_chi_flt = opt_tau_flt
hat_varphi_arr, hat_chi_arr, j_idx, objF_arr, _vld_err_arr, re_step_avg_arr, re_step_varphi_arr, re_step_chi_arr = \
    inference.estimate_water_vapor_varphi (stage0_reduced_data_dct, prev_hat_chi_arr, tau_chi_flt, tau_varphi_flt, 
        max_iter_int, epsilon_flt, verbose_bl, sparsa_cfg_chi_obj, sparsa_cfg_varphi_obj)

# Reconstruct the online and offline photon counts
on_sigma_arr = stage0_reduced_data_dct ['binned_on_sig_arr'] / stage0_reduced_data_dct ['scale_to_H2O_den_flt']
off_sigma_arr = stage0_reduced_data_dct ['binned_off_sig_arr'] / stage0_reduced_data_dct ['scale_to_H2O_den_flt']
range_arr = stage0_reduced_data_dct ['range_arr']
geoO_arr = stage0_reduced_data_dct ['geoO_arr']
on_bg_arr = stage0_reduced_data_dct ['on_bg_arr']
off_bg_arr = stage0_reduced_data_dct ['off_bg_arr']
del_R_flt = np.mean (np.diff (stage0_reduced_data_dct ['range_arr'].ravel ()))
# This is used in both the forwards models for the estimating chi and varphi
chi_A_arr = geoO_arr / (range_arr**2)
# TODO: Find a better way to scale A_arr. Maybe use the time resolution.
chi_A_arr = chi_A_arr / chi_A_arr.max () * 1000

rcnstrct_on_arr = chi_A_arr * np.exp (hat_chi_arr) \
    * np.exp (-2 * del_R_flt * np.cumsum (on_sigma_arr * hat_varphi_arr, axis = 0)) + on_bg_arr
rcnstrct_off_arr = chi_A_arr * np.exp (hat_chi_arr) \
    * np.exp (-2 * del_R_flt * np.cumsum (off_sigma_arr * hat_varphi_arr, axis = 0)) + off_bg_arr

plt.figure (2)
plt.plot (stage0_reduced_data_dct ['on_cnts_arr'])
plt.plot (rcnstrct_on_arr)
plt.xlabel ('Bin number')
plt.ylabel ('Backscattered energy')
plt.title ('Online channel backscattered energy')

plt.figure (3)
plt.plot (stage0_reduced_data_dct ['off_cnts_arr'])
plt.plot (rcnstrct_off_arr)
plt.xlabel ('Bin number')
plt.ylabel ('Backscattered energy')
plt.title ('Offline channel backscattered energy')
