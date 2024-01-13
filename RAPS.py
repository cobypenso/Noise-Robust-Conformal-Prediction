import os
import json
import torch
import numpy as np

def raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, lam_reg=0.01, k_reg=5, disallow_zero_sets=False, rand=False, noisy_labels=False):
    cal_probs, cal_labels, val_probs, val_labels = val_probs, val_labels, test_probs, test_labels
    n = len(cal_labels)

    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    reg_vec = np.array(k_reg*[0,] + (cal_probs.shape[1]-k_reg)*[lam_reg,])[None,:]
    cal_pi = cal_probs.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_probs,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]

    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

    # Deploy
    n_val = val_probs.shape[0]
    val_pi = val_probs.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_probs,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    return (sets, val_labels), qhat


