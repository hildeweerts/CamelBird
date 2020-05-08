"""Tests for classification metrics.

"""
# Authors: Hilde Weerts
# License: BSD 3 clause

import numpy as np
from camelbird.metrics._classification import equal_opportunity


def test_equal_opportunity():
    # test equal opportunity score
    y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [0, 1, 1, 1, 1, 1, 0, 1, 0, 1]
    a = [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
    sample_weight = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    assert np.equal(equal_opportunity(y_true, y_pred, a, aggregate=None, sample_weight=None).all(),
                    np.array([1, 0.5]).all())
    assert equal_opportunity(y_true, y_pred, a, aggregate='ratio', sample_weight=None) == 0.75
    assert equal_opportunity(y_true, y_pred, a, aggregate='diff', sample_weight=None) == 0.25
    assert equal_opportunity(y_true, y_pred, a, aggregate='diff', sample_weight=sample_weight) == 0.40
    return
