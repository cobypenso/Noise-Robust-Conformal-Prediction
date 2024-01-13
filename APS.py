import os
import json
import torch
import numpy as np

def aps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1):

    cal_probs, cal_labels, val_probs, val_labels = val_probs, val_labels, test_probs, test_labels
    n = len(cal_labels)

    # sort cal_probs based on axis = 1, and then reverse the order on axis = 1
    cal_pi = cal_probs.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_probs, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    # Get the score quantile
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )
    # Deploy (output=list of length n, each element is tensor of classes)
    val_pi = val_probs.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_probs, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)

    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))

    # Calculate empirical coverage
    empirical_coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]), val_labels
    ].mean()
    print(f"The empirical coverage is: {empirical_coverage}")

    return (sets, val_labels), qhat


def aps_randomized(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, randomized=True, no_zero_size_sets=True):
    cal_probs, cal_labels, val_probs, val_labels = val_probs, val_labels, test_probs, test_labels
    n = len(cal_labels)

    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    cal_probs = cal_probs.astype(np.float64)
    cal_pi = cal_probs.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_probs, cal_pi, axis=1).cumsum(axis=1)
    cal_softmax_correct_class = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
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

    # Get the score quantile
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )
    # Deploy (output=list of length n, each element is tensor of classes)
    val_pi = val_probs.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_probs, val_pi, axis=1).cumsum(axis=1)
    if not randomized:
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
        prediction_sets = np.take_along_axis(val_srt <= randomized_threshold[:,None], val_pi.argsort(axis=1), axis=1)
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    return (sets, val_labels), qhat