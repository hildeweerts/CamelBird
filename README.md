# CamelBird
CamelBird is a Python package for measuring and optimizing for fairness of machine learning models.

Find more information on https://camelbird.readthedocs.io.

## Installation
You can install `camelbird` from github. Simply open a terminal and run:

```
pip install git+https://github.com/hildeweerts/camelbird.git
```

## Usage
``` 
import camelbird as cb

# ground truth labels
y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
# predicted labels
y_pred = [0, 1, 1, 1, 1, 1, 0, 1, 0, 1]
# sensitive group membership
a = [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
cb.metrics.equal_opportunity(y_true, y_pred, a, aggregate='ratio')
```

## Roadmap
This library is still very much under development. 

### Fairness Metrics
* Group Fairness Metrics: equal opportunity, equal odds, demographic parity

##### Future Features
* Equal classification performance metric: evaluate classification performance per subgroup
* Metrics for classification problems: predictive parity
* Metrics for regression problems: demographic parity (i.e. mean difference)
* Support for polytomous sensitive features (>2 categories)
* Support multi-class classification
* Subgroup discrimination discovery

### Fairness Algorithms
... 

## Contributing
...

## Authors
* [Hilde Weerts](https://github.com/hildeweerts)

## What did you just call me?
Camel bird is an old word for ostrich. In ancient egypt, Ma'at, the goddess of justice, truth, wisdom and morality, used to wear an ostrich feather: the Feather of Truth. As opposed to common myth, ostriches actually do not bury their head in the sand.
