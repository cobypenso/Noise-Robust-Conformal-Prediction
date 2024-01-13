import os
import json
import torch
import numpy as np

# ---- imports ---- #
from HPS import hps
from APS import aps, aps_randomized
from RAPS import raps
from NoiseRobustAPS import noise_robust_aps, noise_robust_aps_randomized
from NoiseRobustRAPS import noise_robust_raps
from NoiseRobustHPS import noise_robust_hps


def calc_baseline_mets(val_probs, val_labels, test_probs, test_labels, n_calib=0, alpha=0.1,
                       model_names=['aps', 'noise_robust_aps'],
                       k_raps=5, noise_rate = 0.2):
    mets = {}

    ##########################################################################
    ######################### Original Scores ################################
    ##########################################################################


    if 'hps' in model_names:
        (sets, labels), qhat = hps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha)
        mets['hps'] = calc_conformal_mets(sets, labels)
        mets['hps_qhat'] = qhat
        print ('hps: ----> qhat:', qhat)
        print ('hps: ----> mean_set_size:', mets['hps']['size_mean'])

    if 'aps' in model_names:
        (sets, labels), qhat = aps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha)
        mets['aps'] = calc_conformal_mets(sets, labels)
        mets['aps_qhat'] = qhat
        print ('aps: ----> qhat:', qhat)
        print ('aps: ----> mean_set_size:', mets['aps']['size_mean'])

    if 'aps_randomized' in model_names:
        (sets, labels), qhat = aps_randomized(val_probs, val_labels, test_probs, test_labels, n_calib, alpha)
        mets['aps_randomized'] = calc_conformal_mets(sets, labels)
        mets['aps_randomized_qhat'] = qhat
        print ('aps_randomized: ----> qhat:', qhat)
        print ('aps_randomized: ----> mean_set_size:', mets['aps_randomized']['size_mean'])

    if 'raps' in model_names:
        (sets, labels), qhat = raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, rand=False, k_reg=k_raps)
        mets['raps'] = calc_conformal_mets(sets, labels)
        mets['raps_qhat'] = qhat
        print ('raps: ----> qhat:', qhat)
        print ('raps: ----> mean_set_size:', mets['raps']['size_mean'])
    
    if 'raps_randomized' in model_names:
        (sets, labels), qhat = raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, rand=True, k_reg=k_raps)
        mets['raps_randomized'] = calc_conformal_mets(sets, labels)
        mets['raps_randomized_qhat'] = qhat

    ##########################################################################
    ####################### Noise Robust Scores ##############################
    ##########################################################################

    if 'noise_robust_hps' in model_names:
        (sets, sets_with_fixed_qhat, labels), qhat, fixed_qhat = noise_robust_hps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, noise_rate = noise_rate)
        mets['noise_robust_hps'] = calc_conformal_mets(sets, labels)
        mets['noise_robust_hps_with_fixed_qhat'] = calc_conformal_mets(sets_with_fixed_qhat, labels)
        mets['noise_robust_hps_qhat'] = qhat
        mets['noise_robust_hps_with_fixed_qhat_qhat'] = fixed_qhat
        print ('noise_robust_hps: ----> qhat:', qhat)
        print ('noise_robust_hps: ----> mean_set_size:', mets['noise_robust_hps']['size_mean'])

    if 'noise_robust_aps' in model_names:
        (sets, sets_with_fixed_qhat, labels), qhat, fixed_qhat = noise_robust_aps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, noise_rate = noise_rate)
        mets['noise_robust_aps'] = calc_conformal_mets(sets, labels)
        mets['noise_robust_aps_qhat'] = qhat
        mets['noise_robust_aps_with_fixed_qhat'] = calc_conformal_mets(sets_with_fixed_qhat, labels)
        mets['noise_robust_aps_with_fixed_qhat_qhat'] = fixed_qhat
        print ('noise_robust_aps: ----> qhat:', qhat)
        print ('noise_robust_aps: ----> mean_set_size:', mets['noise_robust_aps']['size_mean'])
    

    if 'noise_robust_aps_randomized' in model_names:
        (sets, sets_with_fixed_qhat, labels), qhat, fixed_qhat = noise_robust_aps_randomized(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, noise_rate = noise_rate)
        mets['noise_robust_aps_randomized'] = calc_conformal_mets(sets, labels)
        mets['noise_robust_aps_randomized_qhat'] = qhat
        mets['noise_robust_aps_randomized_with_fixed_qhat'] = calc_conformal_mets(sets_with_fixed_qhat, labels)
        mets['noise_robust_aps_randomized_with_fixed_qhat_qhat'] = fixed_qhat
        print ('noise_robust_aps_randomized: ----> qhat:', qhat)
        print ('noise_robust_aps_randomized: ----> mean_set_size:', mets['noise_robust_aps_randomized']['size_mean'])
    
    if 'noise_robust_raps' in model_names:
        (sets, sets_with_fixed_qhat, labels), qhat, fixed_qhat = noise_robust_raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, rand=False, k_reg=k_raps, noise_rate = noise_rate)
        mets['noise_robust_raps'] = calc_conformal_mets(sets, labels)
        mets['noise_robust_raps_qhat'] = qhat
        mets['noise_robust_raps_with_fixed_qhat'] = calc_conformal_mets(sets_with_fixed_qhat, labels)
        mets['noise_robust_raps_with_fixed_qhat_qhat'] = fixed_qhat
        print ('noise_robust_raps: ----> qhat:', qhat)
        print ('noise_robust_raps: ----> mean_set_size:', mets['noise_robust_raps']['size_mean'])

    if 'noise_robust_raps_randomized' in model_names:
        (sets, sets_with_fixed_qhat, labels), qhat, fixed_qhat = noise_robust_raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, rand=True, k_reg=k_raps, noise_rate = noise_rate)
        mets['noise_robust_raps_randomized'] = calc_conformal_mets(sets, labels)
        mets['noise_robust_raps_randomized_qhat'] = qhat
        mets['noise_robust_raps_randomized_with_fixed_qhat'] = calc_conformal_mets(sets_with_fixed_qhat, labels)
        mets['noise_robust_raps_randomized_with_fixed_qhat_qhat'] = fixed_qhat
        print ('noise_robust_raps_randomized: ----> qhat:', qhat)
        print ('noise_robust_raps_randomized: ----> mean_set_size:', mets['noise_robust_raps_randomized']['size_mean'])

    return mets