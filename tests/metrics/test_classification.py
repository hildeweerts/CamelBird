"""Tests for classification metrics.

"""
# Authors: Hilde Weerts
# License: BSD 3 clause

import pytest
import numpy as np
from camelbird.metrics._classification import _diff, _ratio, _score_subgroups, _base_rate_score
from camelbird.metrics._classification import equal_opportunity, equal_odds, demographic_parity

from sklearn.metrics import accuracy_score


def test__diff():
    # test difference between scores
    assert _diff([1, 3]) == 1 - 3
    assert _diff([3, 1]) == 3 - 1
    with pytest.raises(ValueError):
        _diff([3, 0, 1])
    return


def test__ratio():
    # test ratio between scores
    assert _ratio([1, 3]) == 3 / 1
    assert _ratio([3, 1]) == 1 / 3
    with pytest.raises(ValueError):
        _ratio([0, 1, 3])
    return


def test__score_subgroups():
    # test score subgroups with scikit-learn metric accuracy_score
    y_true = [1, 1, 0, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 0, 1, 1, 1, 0]
    a = [1, 1, 1, 1, 0, 0, 0, 0]
    sample_weight = [2, 1, 1, 1, 1, 1, 1, 2]

    assert np.array_equal(_score_subgroups(y_true, y_pred, a, metric=accuracy_score, sample_weight=None),
                          [3 / 4, 2 / 4])
    assert np.array_equal(_score_subgroups(y_true, y_pred, a, metric=accuracy_score, sample_weight=sample_weight),
                          [4 / 5, 3 / 5])
    return


def test__base_rate_score():
    # test base rate computation
    y_pred = [0, 1, 1, 0]
    sample_weight = [1, 2, 1, 1]
    assert _base_rate_score(None, y_pred, sample_weight=None) == 2 / 4
    assert _base_rate_score(None, y_pred, sample_weight=sample_weight) == 3 / 5
    return


def test_equal_opportunity():
    # test equal opportunity fairness metric
    y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [0, 1, 1, 1, 1, 1, 0, 1, 0, 1]
    a = [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
    sample_weight = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    assert np.array_equal(equal_opportunity(y_true, y_pred, a, aggregate=None, sample_weight=None),
                          np.array([1, 3 / 4]))
    assert equal_opportunity(y_true, y_pred, a, aggregate='ratio', sample_weight=None) == 0.75
    assert equal_opportunity(y_true, y_pred, a, aggregate='diff', sample_weight=None) == 0.25
    assert equal_opportunity(y_true, y_pred, a, aggregate='diff', sample_weight=sample_weight) == 0.40
    return


def test_demographic_parity():
    # test demographic parity fairness metric
    y_pred = [1, 0, 1, 0, 1, 1, 1, 0]
    a = [1, 1, 1, 1, 0, 0, 0, 0]
    sample_weight = [2, 1, 1, 1, 1, 1, 1, 2]

    assert np.array_equal(demographic_parity(None, y_pred, a, aggregate=None, sample_weight=None),
                          np.array([3 / 4, 2 / 4]))
    assert demographic_parity(None, y_pred, a, aggregate='ratio', sample_weight=None) == 2/3
    assert demographic_parity(None, y_pred, a, aggregate='diff', sample_weight=None) == 1/4
    assert demographic_parity(None, y_pred, a, aggregate='diff', sample_weight=sample_weight) == 0
    return


def test_equal_odds():
    # test equal odds fairness metric
    a = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_true = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    sample_weight = [2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]

    assert np.array_equal(equal_odds(y_true, y_pred, a, aggregate=None, sample_weight=None),
                          np.array([[2 / 3, 0], [1, 1 / 3]]))
    assert np.array_equal(equal_odds(y_true, y_pred, a, aggregate=None, sample_weight=sample_weight),
                          np.array([[2 / 3, 0], [1, 1 / 2]]))
    assert equal_odds(y_true, y_pred, a, aggregate='diff', sample_weight=None) == pytest.approx(2/3)
    assert equal_odds(y_true, y_pred, a, aggregate='ratio', sample_weight=None) == pytest.approx(1/6)
    return
