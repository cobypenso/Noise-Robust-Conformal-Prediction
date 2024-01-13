import os
import json
import torch
import numpy as np

def noise_robust_raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, lam_reg=0.01, k_reg=5, disallow_zero_sets=False, rand=False, noisy_labels=False, noise_rate=0.2):
    cal_probs, cal_labels, val_probs, val_labels = val_probs, val_labels, test_probs, test_labels
    n_classes = cal_probs.shape[-1]

    cal_probs_resampled = np.zeros((cal_probs.shape[0] * cal_probs.shape[1], cal_probs.shape[1]))
    cal_labels_resampled = np.zeros((cal_probs.shape[0] * cal_probs.shape[1],), dtype=int)
    weights = np.zeros((cal_probs.shape[0] * cal_probs.shape[1],))
    for idx, (prob, label) in enumerate(zip(cal_probs, cal_labels)):
        for j in range(cal_probs.shape[1]):
            cal_probs_resampled[(idx * n_classes) + j, :] = prob
            cal_labels_resampled[(idx * n_classes) + j] = j
            if j == label:
                weights[(idx * n_classes) + j] = (1 - noise_rate)
            else:
                weights[(idx * n_classes) + j] = noise_rate / (n_classes - 1)

    n = len(cal_labels_resampled)

    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    reg_vec = np.array(k_reg*[0,] + (cal_probs_resampled.shape[1]-k_reg)*[lam_reg,])[None,:]
    cal_pi = cal_probs_resampled.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_probs_resampled,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels_resampled[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]

    cal_scores_weighted =  cal_scores * weights
    res = cal_scores_weighted.reshape(-1,n_classes).sum(axis = 1)
    # Get the score quantile
    qhat = np.quantile(res, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

    # Deploy
    n_val = val_probs.shape[0]
    val_pi = val_probs.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_probs,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
        
    fixed_qhat = (qhat - noise_rate * val_srt.mean(axis=-1)) / (1 - noise_rate)
    indicators_with_fixed_qhat = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= fixed_qhat[:, np.newaxis] if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= fixed_qhat[:, np.newaxis]
    if disallow_zero_sets: indicators_with_fixed_qhat[:,0] = True
    prediction_sets_with_fixed_qhat = np.take_along_axis(indicators_with_fixed_qhat,val_pi.argsort(axis=1),axis=1)

    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))

    sets_with_fixed_qhat = []
    for i in range(len(prediction_sets)):
        sets_with_fixed_qhat.append(tuple(np.where(prediction_sets_with_fixed_qhat[i, :] != 0)[0]))

    return (sets, sets_with_fixed_qhat, val_labels), qhat, fixed_qhat