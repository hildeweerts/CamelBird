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
    """Compute scores for each subgroup according to a (scikit-learn style) metric.

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.

    y_pred : 1d array-like
        Estimated targets as returned by a classifier.

    a : 1d array-like
        Sensitive group membership.

    metric : callable
        Metric that must be computed for each group. Callable of form metric(y_true, y_pred, sample_weight, **kwargs).

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


def _base_rate_score(y_true, y_pred, sample_weight):
    """Compute base rate of predictions.

    Parameters
    ----------
    y_true  : ignored
        Not used, present here for API consistency by convention.

    y_pred : array-like of shape (n_samples,)
        Predicted labels as integers, ``1`` for positive class, ``0`` for negative class.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    base_rate : float
        Base rate of predictions.

    """
    y_pred = column_or_1d(y_pred)
    a = column_or_1d(a)
    sample_weight = _check_sample_weight(sample_weight, y_pred)

    if not _check_binary(y_pred):
        raise NotImplementedError("Non-binary target is currently not supported.")

    return (y_pred*sample_weight).mean()


def equal_opportunity(y_true, y_pred, a, aggregate=None, sample_weight=None):
    """
    Equal opportunity fairness metric.

    Equal opportunity requires that the true positive rates, ``tp / (tp + fn)`` of all subgroups are equal, where ``tp``
    is the number of true positives and ``fn`` the number of false negatives. The true positive rate is also known as
    recall. In other words, the classifier should predict the *preferred* class equally well, regardless of
    sensitive group membership. Note that this fairness metric does not take into account correct classification of true
    negatives.

    For example, consider a job hiring scenario in which we assume that getting hired is the preferred target outcome
    and gender is the sensitive feature of interest (which we will assume to be binary for ease of presentation). Equal
    opportunity requires that women who possess the abilities to do well on a job (i.e. `positives`) are just as often
    classified correctly as men who possess the ability to do well on a job.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels as integers.

    y_pred : array-like of shape (n_samples,)
        Predicted labels as integers.

    a : array-like of shape (n_samples,)
        Sensitive group membership

    aggregate : string, [None, 'diff', 'ratio']
        The type of aggregation performed on the true positive rates.

        ``None``:
            Report the score for each group.
        ``'diff'``:
            Report the difference.
        ``'ratio'``:
            Report the ratio.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    tpr_scores : array of shape (n_groups,)
        Raw true positive rates, if ``aggregate=None``.

    equalopp : float
        Fairness metric score, if ``aggregate='diff'`` or ``aggregate="ratio"``.

    See Also
    --------
    camelbird.metrics.equal_odds: Fairness metric that also requires equal true negative rates.
    camelbird.metrics.demographic_parity: Fairness metric that requires equal base rates.

    Notes
    -----

    See [1]_ for a more detailed discussion on equal opportunity.

    ..  [1] M. Hardt, E. Price, and N. Srebro.  Equality of opportunity in supervised learning.  In Advances in  Neural
        Information  Processing  Systems  29, pages 3315–3323. 2016

    """
    tpr_scores = score_subgroups(y_true, y_pred, a, recall_score, sample_weight=sample_weight, pos_label=1)
    if aggregate is None:
        return tpr_scores
    elif aggregate == 'diff':
        return _diff(tpr_scores)
    elif aggregate == 'ratio':
        return _ratio(tpr_scores)
    else:
        raise ValueError("'aggregate' must be in [None, 'diff', 'ratio'].")


def demographic_parity(y_true, y_pred, a, aggregate=None, sample_weight=None):
    """Demographic parity fairness metric.

    Demographic parity requires that the base rates (i.e. percentage of predicted positives), are equal for all
    subgroups.

    For example, consider a job hiring scenario in which we assume that getting hired is the preferred target outcome
    and gender is the sensitive feature of interest (which we will assume to be binary for ease of presentation).
    Demographic parity requires that the percentage of women that is hired is the same as the percentage of men,
    irrespective of whether one group is, on average, more qualified.

    In contrast to equal odds and equal opportunity, demographic parity does not take into account error rates;
    i.e. the fairness metric does have any requirements regarding 'how wrong' the classifier is.

    Parameters
    ----------
    y_true  : ignored
        Not used, present here for API consistency by convention.

    y_pred : array-like of shape (n_samples,)
        Predicted labels as integers, ``1`` for preferred class, ``0`` for undesired class.

    a : array-like of shape (n_samples,)
        Sensitive group membership, ``1`` for unprivileged group, ``0`` for privileged group.

    aggregate : string, [None, 'diff', 'ratio']
        The type of aggregation performed on the base rates.

        ``None``:
            Report the score for each group.
        ``'diff'``:
            Report the difference.
        ``'ratio'``:
            Report the ratio.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    base_rates : array of shape (n_groups,)
        Raw base rates if ``aggregate=None``

    demopar : float
        Fairness metric score if ``aggregate='diff'`` or ``aggregate="ratio"``

    See Also
    --------
    camelbird.metrics.equal_opportunity : Fairness metric that requires equal true positive rates.
    camelbird.metrics.equal_odds : Fairness metric that requires equal true positive rates and
        equal true negative rates.

    Notes
    -----

    See [1]_ for a more detailed discussion on demographic parity.

    ..  [1] Something

    """
    base_rates = score_subgroups(y_true, y_pred, a, _base_rate_score, sample_weight=sample_weight)
    if aggregate is None:
        return base_rates
    elif aggregate == 'diff':
        return _diff(base_rates)
    elif aggregate == 'ratio':
        return _ratio(base_rates)
    else:
        raise ValueError("'aggregate' must be in [None, 'diff', 'ratio'].")


def equal_odds(y_true, y_pred, a, aggregate=None, sample_weight=None):
    """Equal odds fairness metric.

    Equal odds requires that the true positive rates, ``tp / (tp + fn)``, as well as the true negative rates,
     ``tn / (tn + fp)`` of all subgroups are equal, where ``tp`` is the number of true positives, ``fn`` the number of
     false negatives, ``tn`` the number of true negatives, and ``fp`` the number of false positives.

    For example, consider a job hiring scenario in which we assume that getting hired is the preferred target outcome
    and gender is the sensitive feature of interest (which we will assume to be binary for ease of presentation).
    Equal odds requires that

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels as integers.

    y_pred : array-like of shape (n_samples,)
        Predicted labels as integers.

    a : array-like of shape (n_samples,)
        Sensitive group membership

    aggregate : string, [None, 'diff', 'ratio']
        The type of aggregation performed on the true positive rates and true negative rates.

        ``None``:
            Report the tpr and tnr for each group.
        ``'diff'``:
            Report the maximum difference in tpr or tnr between groups.
        ``'ratio'``:
            Report the minimum ratio of tpr or tnr between groups.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    scores : array of shape (n_groups,2)
        Raw true positive rates and true negative rates, if ``aggregate=None``.

    equalodd : float
        If ``aggregate='diff'``, returns the maximum of the tpr difference and tnr difference.
        If ``aggregate="ratio"``, returns the minimum of the tpr ratio and the tnr ratio.

    See Also
    --------
    camelbird.metrics.equal_opportunity : Fairness metric that requires only equal true positive rates.
    camelbird.metrics.demographic_parity : Fairness metric that requires equal base rates.

    Notes
    -----

    See [1]_ for a more detailed discussion on equal opportunity.

    ..  [1] Something

    """
    tpr_scores = score_subgroups(y_true, y_pred, a, recall_score, sample_weight=sample_weight, pos_label=1)
    tnr_scores = score_subgroups(y_true, y_pred, a, recall_score, sample_weight=sample_weight, pos_label=0)
    if aggregate is None:
        return np.array([tpr_scores, tnr_scores])
    elif aggregate == 'diff':
        diff_tpr = _diff(tpr_scores)
        diff_tnr = _diff(tnr_scores)
        return max(diff_tpr, diff_tnr)
    elif aggregate == 'ratio':
        ratio_tpr = _ratio(tpr_scores)
        ratio_tnr = _ratio(tnr_scores)
        return min(ratio_tpr, ratio_tnr)
    else:
        raise ValueError("'aggregate' must be in [None, 'diff', 'ratio'].")
