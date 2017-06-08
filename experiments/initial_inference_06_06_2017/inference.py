import numpy as np
from scipy.special import gammaln
import ptv.hsrl.denoise as denoise
from ptv.estimators.poissonnoise import poissonmodel0, poissonmodelLogLinearTwoChannel, poissonmodel5

def denoise_background (y_arr, bin_start_idx, bin_end_idx):
    """
    Denoise the background counts. 
    
    Parameters
    ----------
    y_arr : np.array
        The photon counting image as 2-D numpy array, where the column axis (axis-0 in numpy) of the image is a photon 
        counting profile.
    bin_start_idx : int
        The starting bin number of where the background counts are acquired. 
    bin_end_idx : int
        The last bin number of where the background counts are acquired.
    
    Notes
    -----
    The forward model is F(x) = \exp(x). So the recovered background is then exp(x). The forward model could have been 
    x, but in this case then x has to be constrained to be non-negative. If we use the log-linear model, then x does 
    not have to be constrained. 
    
    Returns
    -------
    The estimated background counts."""
    
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

def get_denoiser_atten_backscatter_chi (stage0_data_dct, sparsa_cfg_chi_obj, kwargs_denoiseconf_dct = None):
    """
    Get an initial estimate of the log of the attenuated backscatter cross-section from the offline photon counts. It 
    is an initial estimate because we basically assume that the water vapor optical depth is zero, which is obviously 
    not true. Nonetheless the assumption is blatantly false, we do get an estimate of the attenuated backscatter 
    cross-section which is scaled by some calibration parameters. The estimate will become more inaccurate as the 
    water vapor optical depth increases.
    
    The forward model is F(x) = Aexp(x) + b, where A is the geometric overlap function divided by the squared range and 
    b is the background energy. 
    
    Parameters
    ----------
    stage0_data_dct : dictionary
        The dictionary created by the function stage0_prepare_data.get_data_delR_120m_delT_120s.
    sparsa_cfg_chi_obj : ptv.hsrl.denoise.sparsaconf
        The SpaRSA configuration object.
    kwargs_denoiseconf_dct : dictionary, optional
        Keyword arguments that can be passed to the denoiser class. Refer to ptv.hsrl.denoise.denoiseconf for more 
        information.
    
    Returns
    -------
    A two element tuple: 1) The estimate of \chi and 2) the denoiser object. Suppose denoiser_obj is the denoiser 
    object; the denoised offline photon counts can be accessed via denoiser_obj.getdenoised ()"""
    
    off_y_arr = stage0_data_dct ['off_cnts_arr']
    off_bg_arr = stage0_data_dct ['off_bg_arr']
    range_arr = stage0_data_dct ['range_arr']
    geoO_arr = stage0_data_dct ['geoO_arr']
    
    N, K = off_y_arr.shape
    if K == 1:
        penalty_str = 'condatTV'
    else:
        penalty_str = 'TV'
    
    if kwargs_denoiseconf_dct is None:
        kwargs_denoiseconf_dct = dict (
            log10_reg_lst = [-1.5, 1.5],
            pen_type_str = pen_type_str,
            verbose_bl = True)
    
    # Create system matrix and scale it so that \chi is not too large or too small (which could cause numerical
    # computational problems).
    A_arr = geoO_arr / (range_arr**2)
    # TODO: Find a better way to scale A_arr. Maybe use the time resolution.
    A_arr = A_arr / A_arr.max () * 1000
    # Create the Poisson thin object
    poisson_thn_obj = denoise.poissonthin (off_y_arr, p_trn_flt = 0.5, p_vld_flt = 0.5)
    # Create lower and upper bounds
    lb_arr = np.zeros (shape = off_y_arr.shape) - np.inf
    ub_arr = np.zeros (shape = off_y_arr.shape) + np.inf
    # Create the estimator object
    est_obj = poissonmodel0 (poisson_thn_obj, b_arr = off_bg_arr, A_arr = A_arr, log_model_bl = True, 
        lb_arr = lb_arr, ub_arr = ub_arr, sparsaconf_obj = sparsa_cfg_chi_obj, penalty_str = penalty_str)
    # Create the denoiser object
    denoise_cnf_obj = denoise.denoiseconf (**kwargs_denoiseconf_dct)
    denoiser_obj = denoise.denoisepoisson (est_obj, denoise_cnf_obj)
    
    denoiser_obj.denoise ()
    hat_chi_arr = denoiser_obj.getestimate ()
    
    return hat_chi_arr, denoiser_obj

def estimate_water_vapor_varphi (stage0_data_dct, prev_hat_chi_arr, tau_chi_flt, tau_varphi_flt, 
    max_iter_int, epsilon_flt, verbose_bl, sparsa_cfg_chi_obj, sparsa_cfg_varphi_obj):
    """
    Estimate the water vapor from the online and offline channels. Let \chi represent the log of the attenuated 
    backscatter cross-section; see the documentation of the function get_denoiser_atten_backscatter_chi. Let \varphi 
    represent the water vapor density that units of [g / m^3]. The forward for the online and offline channels are 
    G_{on}(\varphi, \chi) = A\exp(\chi)\exp(-2Q[\sigma_{on}\cdot\varphi]) + b_{on}
    and
    G_{off}(\varphi, \chi) = A\exp(\chi)\exp(-2Q[\sigma_{off}\cdot\varphi]) + b_{off},
    where A is the geometric overlap function divided by the squared range and Q is the integrator matrix.
    
    This function accepts an initial estimate of \chi, which is computed via the function 
    get_denoiser_atten_backscatter_chi. The algorithm then performs the following pseudo code:
    1. Denote given \chi estimate as the previous \chi estimate
    2. Repeat until a stopping criteria is met:
        3. Given the previous estimate of \chi, estimate estimate \varphi.
        4. Given the estimate of \varphi from step 3, re-estimate \chi.
        5. Let the \chi estimate of 3 denote the previous \chi estimate.
    
    The stopping criteria is the relative step size of both \chi and \varphi.
    
    Notes
    -----
    The one drawback of the this estimator is that it requires a geometric overlap function. The forward model will be 
    adapted to abolish the requirement of the geometric overlap function. A second drawback is that it is assumed that 
    the lidar range sampling interval is equivalent to the laser pulse length, which is not true for the UCAR DIAL; the 
    DIAL transmits a 150 meters pulse, and the sampling is done at 150/4 meters. To take in account the sampling range 
    interval and laser pulse length difference, we need to introduce a deblurring operator. 
    
    Parameters
    ----------
    stage0_data_dct : dictionary
        The dictionary created by the function stage0_prepare_data.get_data_delR_120m_delT_120s.
    prev_hat_chi_arr : np.array
        An initial estimate of \chi; refer to the function get_denoiser_atten_backscatter_chi.
    tau_chi_flt : float
        The tuning parameter for estimating \chi.
    tau_varphi_flt : float
        The tuning parameter for estimating \varphi.
    max_iter_int : int
        The maximum number of iterations of the inversion algorithm.
    epsilon_flt : float
        The relative step size tolerance limit that is used for the stopping criteria. 
    verbose_bl : bool
        Set to true if verbose message are to be printed. 
    sparsa_cfg_chi_obj : ptv.hsrl.denoise.sparsaconf
        The SpaRSA configuration object for estimating chi. The most important parameters that have to be set in the 
        configuration is max_iter_int and eps_flt; max_iter_int should be relative small, like 100. 
    sparsa_cfg_varphi_obj : ptv.hsrl.denoise.sparsaconf
        The SpaRSA configuration object for estimating varphi. The most important parameters that have to be set in the 
        configuration is max_iter_int and eps_flt; max_iter_int should be relative small, like 100. 
    
    Returns
    -------
    An six element tuple is returned:
    1. The water vapor estimate in units of [g / m^3].
    2. The log of the attenuated backscatter cross-section. 
    3. The number iteration that the algorithm went through either before the stopping criteria has been met or 
        the maximum number of iterations have been reached. 
    4. The objective function; it is suppose to be a monotonically decreasing function. If it is not, then something 
        is very wrong and the code needs to be debugged. 
    5. The validation error as a function of number of iterations.
    6. The relative step size as a function of number of iterations."""
    
    
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
    
    # Record the objective function in the following array, which is computed from the training data
    objF_arr = np.zeros (shape = (max_iter_int * 2, ))
    # Record the validation errors
    vld_err_arr = np.zeros (shape = (max_iter_int * 2, ))
    
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
    
    # This is used in both the forwards models for the estimating chi and varphi
    chi_A_arr = geoO_arr / (range_arr**2)
    # TODO: Find a better way to scale A_arr. Maybe use the time resolution.
    chi_A_arr = chi_A_arr / chi_A_arr.max () * 1000
    
    for j_idx in range (max_iter_int):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Estimate water vapor
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose_bl is True:
            print ('[{:d}/{:d}] varphi_tau = {:.2e}; estimating water vapor'.format (j_idx + 1, 
                max_iter_int, tau_varphi_flt))
        # Create water vapor estimator
        A_arr = chi_A_arr * np.exp (prev_hat_chi_arr)
        est_varphi_obj = poissonmodel5 (on_poisson_thn_obj, off_poisson_thn_obj, 
            on_sigma_arr, off_sigma_arr, on_bg_arr, A_arr, off_bg_arr, A_arr, 
            sparsa_cfg_varphi_obj, varphi_lb_arr, varphi_ub_arr, delta_R_flt = del_R_flt)
        
        # Do the estimation
        hat_varphi_arr, _, status_msg_str, varphi_sparsa_obj = est_varphi_obj.estimate (prev_hat_varphi_arr, tau_varphi_flt)
        if verbose_bl is True:
            print ('\t{:s}'.format (status_msg_str)) 
        
        # Record the objective function; first get the forward model and then the subproblem
        trn_fw_model_obj, vld_fw_model_obj, _ = est_varphi_obj.getphymodel ()
        subp_obj = est_varphi_obj.getsubproblem ()
        objF_arr [j_idx * 2] = trn_fw_model_obj.lossfunction (hat_varphi_arr) \
            + tau_chi_flt * subp_obj.penalty (prev_hat_chi_arr) \
            + tau_varphi_flt * subp_obj.penalty (hat_varphi_arr)
        
        # Record the validation error
        vld_err_arr [j_idx * 2] = vld_fw_model_obj.lossfunction (hat_varphi_arr)
        
        # Record relative step size for varphi
        re_step_varphi_arr [j_idx] = np.linalg.norm (prev_hat_varphi_arr - hat_varphi_arr, 'fro') \
            / np.linalg.norm (hat_varphi_arr, 'fro')
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Create the attenuated backscatter
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if verbose_bl is True:
            print ('[{:d}/{:d}] chi_tau = {:.2e}; estimating attenuated backscatter'.format (j_idx + 1, 
                max_iter_int, tau_chi_flt))
        # Create attenuated backscatter estimator
        on_A_arr = chi_A_arr * np.exp (-2 * del_R_flt * np.cumsum (on_sigma_arr * hat_varphi_arr, axis=0))
        on_B_arr = np.ones_like (on_A_arr)
        off_A_arr = chi_A_arr * np.exp (-2 * del_R_flt * np.cumsum (off_sigma_arr * hat_varphi_arr, axis=0))
        off_B_arr = np.ones_like (off_A_arr)
        
        est_chi_obj = poissonmodelLogLinearTwoChannel (on_poisson_thn_obj, off_poisson_thn_obj, 
            on_bg_arr, on_A_arr, True, on_B_arr, True, 
            off_bg_arr, off_A_arr, True, off_B_arr, True, 
            sparsa_cfg_varphi_obj, chi_lb_arr, chi_ub_arr)
        
        # Do the estimation
        hat_chi_arr, _, status_msg_str, chi_sparsa_obj = est_chi_obj.estimate (prev_hat_chi_arr, tau_chi_flt)
        if verbose_bl is True:
            print ('\t{:s}'.format (status_msg_str))
        
        # Record the objective function; first get the forward model and then the subproblem
        trn_fw_model_obj, vld_fw_model_obj, _ = est_chi_obj.getphymodel ()
        subp_obj = est_chi_obj.getsubproblem ()
        objF_arr [j_idx * 2 + 1] = trn_fw_model_obj.lossfunction (hat_chi_arr) \
            + tau_chi_flt * subp_obj.penalty (hat_chi_arr) \
            + tau_varphi_flt * subp_obj.penalty (hat_varphi_arr)
        
        # Record the validation error
        vld_err_arr [j_idx * 2 + 1] = vld_fw_model_obj.lossfunction (hat_chi_arr)
        
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
    
    # Offset the validation error array so that it is non-negative
    vld_err_arr += np.sum (gammaln (on_poisson_thn_obj._y_vld_arr + 1) + gammaln (off_poisson_thn_obj._y_vld_arr + 1))
    
    return hat_varphi_arr, hat_chi_arr, j_idx, objF_arr, vld_err_arr, re_step_avg_arr
    