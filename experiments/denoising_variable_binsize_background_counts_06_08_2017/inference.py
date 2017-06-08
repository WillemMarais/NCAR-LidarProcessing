import numpy as np
import ptv.hsrl.denoise as denoise
from ptv.estimators.poissonnoise import poissonmodel0

def denoise_background (y_arr, nr_profiles_arr, bin_start_idx, bin_end_idx):
    """
    Denoise the background counts. 
    
    Parameters
    ----------
    y_arr : np.array
        The photon counting image as 2-D numpy array, where the column axis (axis-0 in numpy) of the image is a photon 
        counting profile.
    nr_profiles_arr : np.array
        A 1-D row vector numpy array that indicates how many laser shots were accumulated in the corresponding photon 
        counting array.
    
    Notes
    -----
    The forward model is F(x) = A\exp(x), where A is a row vector that indicates the number of laser shots that were 
    fired per profile bin. So the recovered background is then exp(x). The forward model could have been Ax, but in this 
    case then x has to be constrained to be non-negative. If we use the log-linear model, then x does not have to be 
    constrained. 
    
    Returns
    -------
    A two element tuple: 1) the estimated background count and 2) the denoiser object."""
    
    # Accumulate the photon counts for the range region where there is mostly dark counts and solar background radation
    bg_y_arr = y_arr [bin_start_idx:bin_end_idx, :]
    # Check how many bin numbers the accumulated photon counts corresponds to
    N, K = bg_y_arr.shape
    # The denoising software in this specific case expect a column vector. TODO: Adapt the PTV software to be invariant 
    # of they type of vector.
    bg_y_arr = bg_y_arr.sum (axis = 0)[np.newaxis].T
    
    # Create the Poisson thin object so that we can do cross-validation
    poisson_thn_obj = denoise.poissonthin (bg_y_arr, p_trn_flt = 0.5, p_vld_flt = 0.5)
    # Create the estimator object by creating the system matrix which describes the number of laser shots be profile.
    # The reason why I add 1e-6 to the number of profiles, is because when the gradient is computed A_arr is used in a 
    # division. So 1e-6 prevent divided by zero values. 
    A_arr = nr_profiles_arr.T.astype (np.float) + 1e-6
    sparsa_cfg_obj = denoise.sparsaconf (eps_flt = 1e-6, verbose_int = 1e6)
    est_obj = poissonmodel0 (poisson_thn_obj, A_arr = A_arr, log_model_bl = True, penalty_str = 'condatTV', 
        sparsaconf_obj = sparsa_cfg_obj)
    # Create the denoiser object
    denoise_cnf_obj = denoise.denoiseconf (log10_reg_lst = [3 - np.log10 (K), 7 - np.log10 (K)], nr_reg_int = 48, 
        pen_type_str = 'condatTV', verbose_bl = False)
    denoiser_obj = denoise.denoisepoisson (est_obj, denoise_cnf_obj)
    # Start the denoising
    denoiser_obj.denoise ()
    
    return np.exp (denoiser_obj.getestimate ().T) / float (N), denoiser_obj
    