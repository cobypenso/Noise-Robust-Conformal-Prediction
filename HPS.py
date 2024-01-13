import os
import json
import torch
import numpy as np

def hps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1):
    cal_probs, cal_labels, val_probs, val_labels = val_probs, val_labels, test_probs, test_labels
    n = len(cal_labels)

    # 1: get conformal scores. n = calib_Y.shape[0]
    cal_scores = 1 - cal_probs[np.arange(n), cal_labels]
    # 2: get adjusted quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, interpolation='higher')
    prediction_sets = val_probs >= (1 - qhat)
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    
    return (sets, val_labels), qhat