import numpy as np
from scipy.special import gammaln
import ptv.hsrl.denoise as denoise
from ptv.estimators.poissonnoise import poissonmodel0, poissonmodelLogLinearTwoChannel, poissonmodel5

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

def get_denoiser_atten_backscatter_chi (stage0_data_dct, sparsa_cfg_chi_obj):
    """
    Parameters
    ----------
    range_arr : np.array
        Must be a column vector.
    geoO_arr : np.array
        Must be a column vector."""
    
    off_y_arr = stage0_data_dct ['off_cnts_arr']
    off_bg_arr = stage0_data_dct ['off_bg_arr']
    range_arr = stage0_data_dct ['range_arr']
    geoO_arr = stage0_data_dct ['geoO_arr']
    
    N, K = off_y_arr.shape
    if K == 1:
        penalty_str = 'condatTV'
    else:
        penalty_str = 'TV'
    
    # Create system matrix
    A_arr = geoO_arr / (range_arr**2)
    # Create the Poisson thin object
    poisson_thn_obj = denoise.poissonthin (off_y_arr, p_trn_flt = 0.5, p_vld_flt = 0.5)
    # Create lower and upper bounds
    lb_arr = np.zeros (shape = off_y_arr.shape) - np.inf
    ub_arr = np.zeros (shape = off_y_arr.shape) + np.inf
    # Create the estimator object
    est_obj = poissonmodel0 (poisson_thn_obj, b_arr = off_bg_arr, A_arr = A_arr, log_model_bl = True, 
        lb_arr = lb_arr, ub_arr = ub_arr, sparsaconf_obj = sparsa_cfg_chi_obj, penalty_str = penalty_str)
    # Create the denoiser object
    denoise_cnf_obj = denoise.denoiseconf (log10_reg_lst = [-1.5, 1.5], pen_type_str = penalty_str, verbose_bl = True)
    denoiser_obj = denoise.denoisepoisson (est_obj, denoise_cnf_obj)
    
    denoiser_obj.denoise ()
    hat_chi_arr = denoiser_obj.getestimate ()
    
    return hat_chi_arr, denoiser_obj

def estimate_water_vapor_varphi (stage0_data_dct, prev_hat_chi_arr, tau_chi_flt, tau_varphi_flt, 
    max_iter_int, epsilon_flt, verbose_int, sparsa_cfg_chi_obj, sparsa_cfg_varphi_obj):
    
    # Create the Poisson thinning objects
    on_poisson_thn_obj = denoise.poissonthin (stage0_data_dct ['on_cnts_arr'], 
        p_trn_flt = 0.5, p_vld_flt = 0.5)
    off_poisson_thn_obj = denoise.poissonthin (stage0_data_dct ['off_cnts_arr'], 
        p_trn_flt = 0.5, p_vld_flt = 0.5)
    
    # Get calibration parameters
    range_arr = stage0_data_dct ['range_arr']
    geoO_arr = stage0_data_dct ['geoO_arr']
    
    # Get the background counts
    on_bg_arr = stage0_data_dct ['on_bg_arr']
    off_bg_arr = stage0_data_dct ['off_bg_arr']
    
    # Record the objective function in the following array
    objF_arr = np.zeros (shape = (max_iter_int * 2, ))
    
    # Record the relative step size in the following array
    re_step_chi_arr = np.zeros (shape = (max_iter_int, ))
    re_step_varphi_arr = np.zeros (shape = (max_iter_int, ))
    re_step_avg_arr = np.zeros (shape = (max_iter_int, ))
    
    # Create an initial estimate of the water vapor
    prev_hat_varphi_arr = np.ones_like (prev_hat_chi_arr)
    
    # Scale the H2O absorption coefficient for the online and offline data
    on_sigma_arr = stage0_data_dct ['binned_on_sig_arr'] / stage0_data_dct ['scale_to_H2O_den_flt']
    off_sigma_arr = stage0_data_dct ['binned_off_sig_arr'] / stage0_data_dct ['scale_to_H2O_den_flt']
    
    # Compute delta range
    del_R_flt = np.mean (np.diff (stage0_data_dct ['range_arr'].ravel ()))
    
    # Set the water vapor lower and upper bounds
    varphi_lb_arr = np.zeros_like (prev_hat_varphi_arr)
    varphi_ub_arr = np.zeros_like (prev_hat_varphi_arr) + np.inf
    
    # Set the attenuated backscatter lower and upper bounds
    chi_lb_arr = np.zeros_like (prev_hat_chi_arr) - np.inf
    chi_ub_arr = np.zeros_like (prev_hat_chi_arr) + np.inf
    
    for j_idx in range (max_iter_int):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Estimate water vapor
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        print ('[{:d}/{:d}] varphi_tau = {:.2e}; estimating water vapor'.format (j_idx, 
            max_iter_int, tau_varphi_flt))
        # Create water vapor estimator
        A_arr = (geoO_arr / (range_arr**2)) * np.exp (prev_hat_chi_arr)
        est_varphi_obj = poissonmodel5 (on_poisson_thn_obj, off_poisson_thn_obj, 
            on_sigma_arr, off_sigma_arr, on_bg_arr, A_arr, off_bg_arr, A_arr, 
            sparsa_cfg_varphi_obj, varphi_lb_arr, varphi_ub_arr, delta_R_flt = del_R_flt)
        
        # Do the estimation
        hat_varphi_arr, _, status_msg_str, varphi_sparsa_obj = est_varphi_obj.estimate (prev_hat_varphi_arr, tau_varphi_flt)
        print ('\t{:s}'.format (status_msg_str)) 
        
        # Record the objective function; first get the forward model and then the subproblem
        fw_model_obj, _, _ = est_varphi_obj.getphymodel ()
        subp_obj = est_varphi_obj.getsubproblem ()
        objF_arr [j_idx * 2] = fw_model_obj.lossfunction (hat_varphi_arr) \
            + tau_chi_flt * subp_obj.penalty (prev_hat_chi_arr) \
            + tau_varphi_flt * subp_obj.penalty (hat_varphi_arr)
        
        # Record relative step size for varphi
        re_step_varphi_arr [j_idx] = np.linalg.norm (prev_hat_varphi_arr - hat_varphi_arr, 'fro') \
            / np.linalg.norm (hat_varphi_arr, 'fro')
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Create the attenuated backscatter
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        print ('[{:d}/{:d}] chi_tau = {:.2e}; estimating attenuated backscatter'.format (j_idx, 
            max_iter_int, tau_chi_flt))
        # Create attenuated backscatter estimator
        on_A_arr = (geoO_arr / (range_arr**2)) * np.exp (-2 * del_R_flt * np.cumsum (on_sigma_arr * hat_varphi_arr, axis=0))
        on_B_arr = np.ones_like (on_A_arr)
        off_A_arr = (geoO_arr / (range_arr**2)) * np.exp (-2 * del_R_flt * np.cumsum (off_sigma_arr * hat_varphi_arr, axis=0))
        off_B_arr = np.ones_like (off_A_arr)
        
        est_chi_obj = poissonmodelLogLinearTwoChannel (on_poisson_thn_obj, off_poisson_thn_obj, 
            on_bg_arr, on_A_arr, True, on_B_arr, True, 
            off_bg_arr, off_A_arr, True, off_B_arr, True, 
            sparsa_cfg_varphi_obj, chi_lb_arr, chi_ub_arr)
        
        # Do the estimation
        hat_chi_arr, _, status_msg_str, chi_sparsa_obj = est_chi_obj.estimate (prev_hat_chi_arr, tau_chi_flt)
        print ('\t{:s}'.format (status_msg_str))
        
        # Record the objective function; first get the forward model and then the subproblem
        fw_model_obj, _, _ = est_chi_obj.getphymodel ()
        subp_obj = est_chi_obj.getsubproblem ()
        objF_arr [j_idx * 2 + 1] = fw_model_obj.lossfunction (hat_chi_arr) \
            + tau_chi_flt * subp_obj.penalty (hat_chi_arr) \
            + tau_varphi_flt * subp_obj.penalty (hat_varphi_arr)
        
        # Record relative step size for chi
        re_step_chi_arr [j_idx] = np.linalg.norm (prev_hat_chi_arr - hat_chi_arr, 'fro') \
            / np.linalg.norm (hat_chi_arr, 'fro')
        
        # Record relative step size of both varphi and chi
        avg_re_step_num_flt = np.sqrt (np.linalg.norm (prev_hat_varphi_arr - hat_varphi_arr, 'fro')**2 \
            + np.linalg.norm (prev_hat_chi_arr - hat_chi_arr, 'fro')**2)
        avg_re_step_dem_flt = np.sqrt (np.linalg.norm (hat_varphi_arr, 'fro')**2 + np.linalg.norm (hat_chi_arr, 'fro')**2)
        avg_re_step_flt = avg_re_step_num_flt / avg_re_step_dem_flt
        re_step_avg_arr [j_idx] = avg_re_step_flt
        
        # Check if the average relative step size is less than the given tolerance
        if avg_re_step_flt < epsilon_flt:
            break
        
        prev_hat_chi_arr = hat_chi_arr.copy ()
        prev_hat_varphi_arr = hat_varphi_arr.copy ()
    
    return hat_varphi_arr, hat_chi_arr, j_idx, objF_arr, re_step_avg_arr, re_step_varphi_arr, re_step_chi_arr
    