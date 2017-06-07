%pylab

# The conent of the script is copy pasted into the Python/IPython terminal

import os
import sys

# ---------------------------------------------------------------------------------------------------------------------
# Remove conflicting Python modules from sys.path
# ---------------------------------------------------------------------------------------------------------------------
tmp_fileP_str = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/GitLab/poisson_inference_tools')
if tmp_fileP_str in sys.path:
    sys.path.remove (tmp_fileP_str)

tmp_fileP_str = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/Research_local/willem.marais.phd/source/python')
if tmp_fileP_str in sys.path:
    sys.path.remove (tmp_fileP_str)

# ---------------------------------------------------------------------------------------------------------------------
# Add desired Python modules to sys.path
# ---------------------------------------------------------------------------------------------------------------------
tmp_fileP_str = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/GitLab/NCAR-LidarProcessing/libraries')
if tmp_fileP_str not in sys.path:
    sys.path.append (tmp_fileP_str)

tmp_fileP_str = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/GitLab/NCAR-LidarProcessing/experiments/initial_inference_06_06_2017')
if tmp_fileP_str not in sys.path:
    sys.path.append (tmp_fileP_str)
    
tmp_fileP_str = os.path.join (os.environ ['HOME'], 'ProjectsSSEC/GitLab/ptvv1_python/python')
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
# 2. Night time data with clouds in the begining and parly day time
prfl_start_idx = 40
prfl_end_idx = 300

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stage 0 - prepare data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print ('(INFO) Stage 0 - Preparing data')

# Get data
stage0_data_dct = stage0_prepare_data.get_data_delR_120m_delT_120s (recompute_bl = False, recompute_sig_bl = False)
# Select data that corresponds to night time
stage0_data_dct ['on_cnts_arr'] = stage0_data_dct ['on_cnts_arr'][:, prfl_start_idx:prfl_end_idx]
stage0_data_dct ['off_cnts_arr'] = stage0_data_dct ['off_cnts_arr'][:, prfl_start_idx:prfl_end_idx]
stage0_data_dct ['binned_dsig_arr'] = stage0_data_dct ['binned_dsig_arr'][:, prfl_start_idx:prfl_end_idx]

# Get geometric overlap
geo_range_arr, geoO_arr = stage0_prepare_data.get_geoO_delR_120m ()

# Denoise the background
bin_start_idx = 127
bin_end_idx = 139

on_bg_arr = inference.denoise_background (stage0_data_dct ['on_cnts_arr'], bin_start_idx, bin_end_idx)
off_bg_arr = inference.denoise_background (stage0_data_dct ['off_cnts_arr'], bin_start_idx, bin_end_idx)

stage0_data_dct ['on_bg_arr'] = on_bg_arr
stage0_data_dct ['off_bg_arr'] = off_bg_arr
stage0_data_dct ['geoO_arr'] = geoO_arr

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reduce the size of the image to make it more computational feasible
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# subs_on_y_arr = stage0_data_dct ['on_cnts_arr'].copy ()[:, 0:1]
# subs_off_y_arr = stage0_data_dct ['off_cnts_arr'].copy ()[:, 0:1]
# subs_on_bg_arr = on_bg_arr.copy () [:, 0:1]
# subs_off_bg_arr = on_bg_arr.copy () [:, 0:1]

stage0_reduced_data_dct = dict ()
stage0_reduced_data_dct ['range_arr'] = stage0_data_dct ['range_arr'][:80, :]
stage0_reduced_data_dct ['geoO_arr'] = stage0_data_dct ['geoO_arr'][:80, :]
stage0_reduced_data_dct ['pre_bin_range_arr'] = stage0_data_dct ['pre_bin_range_arr']
stage0_reduced_data_dct ['on_cnts_arr'] = stage0_data_dct ['on_cnts_arr'].copy () [:80, 0:36]
stage0_reduced_data_dct ['off_cnts_arr'] = stage0_data_dct ['off_cnts_arr'].copy () [:80, 0:36]
stage0_reduced_data_dct ['on_bg_arr'] = stage0_data_dct ['on_bg_arr'].copy () [:80, 0:36]
stage0_reduced_data_dct ['off_bg_arr'] = stage0_data_dct ['off_bg_arr'].copy () [:80, 0:36]
stage0_reduced_data_dct ['binned_dsig_arr'] = stage0_data_dct ['binned_dsig_arr'].copy () [:80, 0:36]
stage0_reduced_data_dct ['binned_on_sig_arr'] = stage0_data_dct ['binned_on_sig_arr'].copy () [:80, 0:36]
stage0_reduced_data_dct ['binned_off_sig_arr'] = stage0_data_dct ['binned_off_sig_arr'].copy () [:80, 0:36]
stage0_reduced_data_dct ['scale_to_H2O_den_flt'] = stage0_data_dct ['scale_to_H2O_den_flt']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stage 1 - get intial estimate of the attenuated backscatter cross-section, which I call \chi.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print ('(INFO) Stage 1 - Get initial estimate of the attenuated backscatter cross-section')

sparsa_cfg_obj = denoise.sparsaconf (eps_flt = 1e-5)
hat_chi_arr, chi_denoiser_obj = inference.get_denoiser_atten_backscatter_chi (stage0_reduced_data_dct, sparsa_cfg_obj)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stage 2 - now estimate both the water vapor and the attenuated backscatter cross-section. The water vapor is 
# denoted by \varphi, and the \chi denotes the attenuated backscatter cross-section.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print ('(INFO) Stage 2 - Estimating water vapor and attenuated backscatter cross-section')

# Create SpaRSA configuration objects
prev_hat_chi_arr = hat_chi_arr.copy ()
max_iter_int = 20
epsilon_flt = 1e-5
verbose_int = 1
tau_varphi_flt = 10
tau_chi_flt = 10
sparsa_cfg_chi_obj = denoise.sparsaconf (max_iter_int = 1e2, M_int = 0)
sparsa_cfg_varphi_obj = denoise.sparsaconf (max_iter_int = 1e2, M_int = 0)

hat_varphi_arr, hat_chi_arr, j_idx, objF_arr, re_step_avg_arr, re_step_varphi_arr, re_step_chi_arr = \
    inference.estimate_water_vapor_varphi (stage0_reduced_data_dct, prev_hat_chi_arr, tau_chi_flt, tau_varphi_flt, 
        max_iter_int, epsilon_flt, verbose_int, sparsa_cfg_chi_obj, sparsa_cfg_varphi_obj)

figure (1)
plot (objF_arr - objF_arr.min () + 1)
semilogy ()
title ('Relative objective function')

figure (2)
plot (re_step_avg_arr)
semilogy ()
title ('Relative step size')

# Get calibration parameters
range_arr = stage0_reduced_data_dct ['range_arr']
geoO_arr = stage0_reduced_data_dct ['geoO_arr']

# Get the background counts
on_bg_arr = stage0_reduced_data_dct ['on_bg_arr']
off_bg_arr = stage0_reduced_data_dct ['off_bg_arr']

on_sigma_arr = stage0_reduced_data_dct ['binned_on_sig_arr'] / stage0_reduced_data_dct ['scale_to_H2O_den_flt']
off_sigma_arr = stage0_reduced_data_dct ['binned_off_sig_arr'] / stage0_reduced_data_dct ['scale_to_H2O_den_flt']

# Compute delta range
del_R_flt = np.mean (np.diff (stage0_data_dct ['range_arr'].ravel ()))

on_reconstruct_arr = range_arr * geoO_arr * np.exp (hat_chi_arr) \
    * np.exp (-2 * del_R_flt * np.cumsum (on_sigma_arr * hat_varphi_arr, axis = 0)) + on_bg_arr

off_reconstruct_arr = range_arr * geoO_arr * np.exp (hat_chi_arr) \
    * np.exp (-2 * del_R_flt * np.cumsum (off_sigma_arr * hat_varphi_arr, axis = 0)) + off_bg_arr

on_cnts_arr = stage0_reduced_data_dct ['on_cnts_arr']
off_cnts_arr = stage0_reduced_data_dct ['off_cnts_arr']

figure (3)
plot (on_cnts_arr [:, 12] - on_reconstruct_arr [:, 12])

figure (4)
plot (off_cnts_arr [:, 12] - off_reconstruct_arr [:, 12])
