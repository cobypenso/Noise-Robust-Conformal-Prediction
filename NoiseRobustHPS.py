import os
import json
import torch
import numpy as np

def noise_robust_hps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, noise_rate = 0.1):
    cal_probs, cal_labels, val_probs, val_labels = val_probs, val_labels, test_probs, test_labels
    n_classes = cal_probs.shape[-1]
    n_examples = cal_probs.shape[0]
    
    # 0: compute noise-robust scores - weighted average based on noise rate
    cal_probs_resampled = np.zeros((n_examples * n_classes, n_classes))
    cal_labels_resampled = np.zeros((n_examples * n_classes,), dtype=int)
    weights = np.zeros((n_examples * n_classes,))
    for idx, (prob, label) in enumerate(zip(cal_probs, cal_labels)):
        for j in range(n_classes):
            cal_probs_resampled[(idx * n_classes) + j, :] = prob
            cal_labels_resampled[(idx * n_classes) + j] = j
            if j == label:
                weights[(idx * n_classes) + j] = (1 - noise_rate)
            else:
                weights[(idx * n_classes) + j] = (noise_rate / (n_classes - 1))

    n = len(cal_labels_resampled)
    
    # 1: get conformal scores. n = calib_Y.shape[0]
    cal_scores = 1 - cal_probs_resampled[np.arange(n), cal_labels_resampled]
    # 2: get adjusted quantile - based on original n
    q_level = np.ceil((n + 1) * (1 - alpha)) / n

    # weight and average based on noise level
    cal_scores_weighted =  cal_scores * weights
    res = cal_scores_weighted.reshape(-1,n_classes).sum(axis = 1)

    # compute qhat
    qhat = np.quantile(res, q_level, interpolation="higher")

    prediction_sets = val_probs >= (1 - qhat)
    
    fixed_qhat = (qhat - noise_rate * (1-val_probs).mean(axis=-1)) / (1 - noise_rate)
    prediction_sets_with_fixed_qhat = val_probs >= (1 - fixed_qhat[:, np.newaxis])

    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    
    sets_with_fixed_qhat = []
    for i in range(len(prediction_sets)):
        sets_with_fixed_qhat.append(tuple(np.where(prediction_sets_with_fixed_qhat[i, :] != 0)[0]))

    return (sets, sets_with_fixed_qhat, val_labels), qhat, fixed_qhat