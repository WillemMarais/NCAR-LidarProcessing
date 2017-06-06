import numpy as np
import ptv.hsrl.denoise as denoise
from ptv.estimators.poissonnoise import poissonmodel0

def denoise_background (y_arr, bin_start_idx, bin_end_idx):
    bg_y_arr = y_arr [bin_start_idx:bin_end_idx, :]
    N, _ = bg_y_arr.shape
    bg_y_arr = bg_y_arr.sum (axis = 0)[np.newaxis].T
    
    # Create the Poisson thin object
    poisson_thn_obj = denoise.poissonthin (bg_y_arr, p_trn_flt = 0.5, p_vld_flt = 0.5)
    # Create the estimator object
    est_obj = poissonmodel0 (poisson_thn_obj, log_model_bl = True, penalty_str = 'condatTV')
    # Create the denoiser object
    denoise_cnf_obj = denoise.denoiseconf (log10_reg_lst = [1, 4], pen_type_str = 'condatTV', verbose_bl = True)
    denoiser_obj = denoise.denoisepoisson (est_obj, denoise_cnf_obj)
    # Start the denoising
    denoiser_obj.denoise ()
    
    return denoiser_obj.getdenoised ().T / float (N)

def get_denoiser_atten_backscatter_chi (off_y_arr, off_bg_arr, range_arr, geoO_arr):
    """
    Parameters
    ----------
    range_arr : np.array
        Must be a column vector.
    geoO_arr : np.array
        Must be a column vector."""
    
    # Create system matrix
    A_arr = range_arr * geoO_arr
    # Create the Poisson thin object
    poisson_thn_obj = denoise.poissonthin (off_y_arr, p_trn_flt = 0.5, p_vld_flt = 0.5)
    # Create SpaRSA configuration object
    sparsa_cfg_obj = denoise.sparsaconf ()
    # Create the estimator object
    est_obj = poissonmodel0 (poisson_thn_obj, b_arr = off_bg_arr, A_arr = A_arr, log_model_bl = True, 
        sparsaconf_obj = sparsa_cfg_obj, penalty_str = 'TV')
    # Create the denoiser object
    denoise_cnf_obj = denoise.denoiseconf (log10_reg_lst = [-2, 2], pen_type_str = 'TV', verbose_bl = True)
    denoiser_obj = denoise.denoisepoisson (est_obj, denoise_cnf_obj)
    
    return denoiser_obj
    