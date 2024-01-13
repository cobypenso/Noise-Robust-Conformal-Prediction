import os
import json
import torch
import numpy as np


def noise_robust_aps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, noise_rate = 0.1, fig_path=None):
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

    # sort cal_probs based on axis = 1, and then reverse the order on axis = 1
    cal_pi = cal_probs_resampled.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_probs_resampled, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels_resampled
    ]
    cal_scores_weighted =  cal_scores * weights
    res = cal_scores_weighted.reshape(-1,n_classes).sum(axis = 1)
    qhat = np.quantile(
        res, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )

    # Deploy (output=list of length n, each element is tensor of classes)
    val_pi = val_probs.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_probs, val_pi, axis=1).cumsum(axis=1)

    fixed_qhat = (qhat - noise_rate * val_srt.mean(axis=-1)) / (1 - noise_rate)
    prediction_sets_with_fixed_qhat = np.take_along_axis(val_srt <= fixed_qhat[:, np.newaxis], val_pi.argsort(axis=1), axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    

    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))

    sets_with_fixed_qhat = []
    for i in range(len(prediction_sets_with_fixed_qhat)):
        sets_with_fixed_qhat.append(tuple(np.where(prediction_sets_with_fixed_qhat[i, :] != 0)[0]))

    return (sets, sets_with_fixed_qhat, val_labels), qhat, fixed_qhat




def noise_robust_aps_randomized(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, randomized=True, no_zero_size_sets=True, noise_rate = 0.1):
    cal_probs, cal_labels, val_probs, val_labels = val_probs, val_labels, test_probs, test_labels

    n_classes = cal_probs.shape[-1]
    n_examples = cal_probs.shape[0] 

    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    cal_probs = cal_probs.astype(np.float64)

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

    cal_pi = cal_probs_resampled.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_probs_resampled, cal_pi, axis=1).cumsum(axis=1)
    cal_softmax_correct_class = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels_resampled
    ]
    if not randomized:
        cal_scores = cal_softmax_correct_class
    else:
        cumsum_index = np.where(cal_srt == cal_softmax_correct_class[:,None])[1]
        if cumsum_index.shape[0] != cal_srt.shape[0]:
            _, unique_indices = np.unique(np.where(
                cal_srt == cal_softmax_correct_class[:,None])[0], return_index=True)
            cumsum_index = cumsum_index[unique_indices]

        high = cal_softmax_correct_class
        low = np.zeros_like(high)
        low[cumsum_index != 0] = cal_srt[np.where(cumsum_index != 0)[0], cumsum_index[cumsum_index != 0]-1]
        cal_scores = np.random.uniform(low=low, high=high)

    cal_scores_weighted =  cal_scores * weights
    res = cal_scores_weighted.reshape(-1,n_classes).sum(axis = 1)

    # Get the score quantile
    qhat = np.quantile(
        res, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )

    # Deploy (output=list of length n, each element is tensor of classes)
    val_pi = val_probs.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_probs, val_pi, axis=1).cumsum(axis=1)
    if not randomized:
        fixed_qhat = (qhat - noise_rate * val_srt.mean(axis=-1)) / (1 - noise_rate)
        prediction_sets_with_fixed_qhat = np.take_along_axis(val_srt <= fixed_qhat[:, np.newaxis], val_pi.argsort(axis=1), axis=1)
        prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    else:
        n_val = val_srt.shape[0]
        cumsum_index = np.sum(val_srt <= qhat, axis=1)
        high = val_srt[np.arange(n_val), cumsum_index]
        low = np.zeros_like(high)
        low[cumsum_index > 0] = val_srt[np.arange(n_val), cumsum_index-1][cumsum_index > 0]
        prob = (qhat - low)/(high - low)
        rv = np.random.binomial(1,prob,size=(n_val))
        randomized_threshold = low
        randomized_threshold[rv == 1] = high[rv == 1]
        if no_zero_size_sets:
            randomized_threshold = np.maximum(randomized_threshold, val_srt[:,0])
        
        fixed_qhat = (randomized_threshold - noise_rate * val_srt.mean(axis=-1)) / (1 - noise_rate)
        prediction_sets_with_fixed_qhat = np.take_along_axis(val_srt <= fixed_qhat[:, np.newaxis], val_pi.argsort(axis=1), axis=1)
        prediction_sets = np.take_along_axis(val_srt <= randomized_threshold[:,None], val_pi.argsort(axis=1), axis=1)
    
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    
    sets_with_fixed_qhat = []
    for i in range(len(prediction_sets_with_fixed_qhat)):
        sets_with_fixed_qhat.append(tuple(np.where(prediction_sets_with_fixed_qhat[i, :] != 0)[0]))

    return (sets, sets_with_fixed_qhat, val_labels), qhat, fixed_qhat