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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stage 0 - prepare data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get data
stage0_data_dct = stage0_prepare_data.get_data_delR_120m_delT_120s (recompute_bl = False, recompute_sig_bl = False)
# Select data that corresponds to night time
stage0_data_dct ['on_cnts_arr'] = stage0_data_dct ['on_cnts_arr'][:, 78:300]
stage0_data_dct ['off_cnts_arr'] = stage0_data_dct ['off_cnts_arr'][:, 78:300]
stage0_data_dct ['binned_dsig_arr'] = stage0_data_dct ['binned_dsig_arr'][:, 78:300]

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

stage0_reduced_data_dct = stage0_data_dct
stage0_reduced_data_dct ['on_cnts_arr'] = stage0_data_dct ['on_cnts_arr'].copy () [:, 0:12]
stage0_reduced_data_dct ['off_cnts_arr'] = stage0_data_dct ['off_cnts_arr'].copy () [:, 0:12]
stage0_reduced_data_dct ['on_bg_arr'] = stage0_data_dct ['on_bg_arr'].copy () [:, 0:12]
stage0_reduced_data_dct ['off_bg_arr'] = stage0_data_dct ['off_bg_arr'].copy () [:, 0:12]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stage 1 - get intial estimate of the attenuated backscatter cross-section, which I call \chi.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
sparsa_cfg_obj = denoise.sparsaconf ()
hat_chi_arr, chi_denoiser_obj = inference.get_denoiser_atten_backscatter_chi (stage0_reduced_data_dct, sparsa_cfg_obj)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stage 2 - now estimate both the water vapor and the attenuated backscatter cross-section. The water vapor is 
# denoted by \varphi, and the \chi denotes the attenuated backscatter cross-section.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create SpaRSA configuration objects
sparsa_cfg_varphi_obj = denoise.sparsaconf (max_iter_int = 1e2, M_int = 1)
sparsa_cfg_chi_obj = denoise.sparsaconf (max_iter_int = 1e2, M_int = 1)
