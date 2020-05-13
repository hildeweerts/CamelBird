"""
The `camelbird.metrics` module includes fairness metrics.

"""


from ._classification import equal_opportunity, equal_odds, demographic_parity

__all__ = ['equal_opportunity',
           'equal_odds',
           'demographic_parity'
           ]