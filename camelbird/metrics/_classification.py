"""Metrics to assess fairness on classification task given class prediction and group membership

"""
# Authors: Hilde Weerts
# License: BSD 3 clause

import numpy as np

from sklearn.utils import column_or_1d
from sklearn.utils.validation import _check_sample_weight

from sklearn.metrics import recall_score


def _check_binary(x):
    return len((np.unique(x))) == 2


def _diff(scores):
    """Compute difference between scores.

    Parameters
    ----------
    scores : array of size 2

    Returns
    -------
    diff : float
    """
    if scores.size != 2:
        raise ValueError("Non-binary sensitive feature is currently not supported.")
    diff = scores[0] - scores[1]
    return diff


def _ratio(scores):
    """Compute ratio of scores.

    Parameters
    ----------
    scores : array of size 2

    Returns
    -------
    diff : float
    """
    if scores.size != 2:
        raise ValueError("Non-binary sensitive feature is currently not supported.")
    ratio = scores[1] / scores[0]
    return ratio


def score_subgroups(y_true, y_pred, a, metric, sample_weight=None, **metric_params):
    """

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.

    y_pred : 1d array-like
        Estimated targets as returned by a classifier.

    a : 1d array-like
        Sensitive group membership.

    metric : callable
        Metric that must be computed for each group. Callable of form metric(y_true, y_pred, a, sample_weight).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    scores : array of shape (n_subgroups,)
        Computed scores for each group.
    """
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    a = column_or_1d(a)
    sample_weight = _check_sample_weight(sample_weight, y_true)

    if not _check_binary(y_true) or not _check_binary(y_pred):
        raise NotImplementedError("Non-binary target is currently not supported.")
    if not _check_binary(a):
        raise NotImplementedError("Non-binary sensitive feature is currently not supported.")

    subgroups = np.unique(a)
    n_subgroups = len(subgroups)
    scores = np.empty((n_subgroups,))
    for subgroup, i in zip(subgroups, range(n_subgroups)):
        cond = a == subgroup
        scores[i] = metric(y_true=y_true[cond], y_pred=y_pred[cond], sample_weight=sample_weight[cond],
                           **metric_params)
    return scores


def equal_opportunity(y_true, y_pred, a, aggregate=None, sample_weight=None):
    """Equal opportunity.

    Equal true positive rates (a.k.a. recall).


    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels as integers.

    y_pred : array-like of shape (n_samples,)
        Predicted labels as integers.

    a : array-like of shape (n_samples,)
        Sensitive group membership

    aggregate : string, [None, 'diff', 'ratio']

        If ``None``, the true positive rate (also known as recall) for each group is returned. Otherwise, this
        determines the type of aggregation performed on the scores:

        ``'diff'``:
            Report the difference.
        ``'ratio'``:
            Report the ratio.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    equalopp : array, or float
        Equal opportunity scores.
    """
    equalopp = score_subgroups(y_true, y_pred, a, recall_score, sample_weight=sample_weight)
    if aggregate not in (None, 'diff', 'ratio'):
        raise ValueError("'aggregate' must be in [None, 'diff', 'ratio'].")
    if aggregate is None:
        return equalopp
    elif aggregate == 'diff':
        return _diff(equalopp)
    elif aggregate == 'ratio':
        return _ratio(equalopp)


if __name__ == "__main__":
    y_true = [1, 1, 1, 1, 1, 0, 0]
    y_pred = [0, 1, 1, 1, 1, 0, 0]
    a = [1, 1, 1, 0, 0, 1, 0]
    sample_weight = [1, 1, 2, 2, 1, 1, 1]

    print(equal_opportunity(y_true, y_pred, a, aggregate=None, sample_weight=None))
    print(equal_opportunity(y_true, y_pred, a, aggregate='ratio', sample_weight=None))
    print(equal_opportunity(y_true, y_pred, a, aggregate='diff', sample_weight=None))
    print(equal_opportunity(y_true, y_pred, a, aggregate=None, sample_weight=sample_weight))