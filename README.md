### RRCT
Relevance, Redundancy, and Complementarity Trade-off, a robust feature selection algorithm (Python version).

****************************************
This algorithm is a computationally efficient, robust approach for feature selection. The algorithm can be thought of as a natural extension to the popular mRMR feature selection algorithm, given that RRCT explicitly takes into account relevance and redundancy (like mRMR), and also introduces an additional third term to account for conditional relevance (also known as complementarity).

The RRCT algorithm is computationally very efficient and can run within a few seconds including on massive datasets with thousands of features. Moreover, it can serve as a useful 'off-the-shelf' feature selection algorithm because it generalizes well on both regression and classification problems, also without needing further adjusting for mixed-type variables.

[R. Ibraheem](https://www.linkedin.com/in/rasheed-oyewole-ibraheem-768955246/) is the author of this implementation and also maintains this package; this implementation is based on the earlier Python implementation by [A. Tsanas](https://www.ed.ac.uk/profile/thanasis-tsanas) (associated to below mentioned publication) that can be found in https://github.com/ThanasisTsanas/RRCT.
****************************************


### Class description 
*RRCTFeatureSelection(K=None, scale_feature=False)*

1. ***Parameter:*** 
    - *K*, non-zero positive integer to specify the number of selected features. Default value is *K=None*, which means all features will be selected.
    - *scale_feature*, a boolean. Set to `False` if your features have been standardized (mean of 0 and standard deviation of 1), otherwise set to `True` and the features will be standardized before applying the algorithm. Default value is `False`.
2. ***Attributes:*** 
    - *selected_feature_indices_*, an array of indices corresponding to the indices of selected features
    - *rrct_values_*, a dictionary containing the relevance, redundancy, complementarity, and RRCT metrics of the selected features
3. ***Methods:***
    - *apply(X=X, y=y, verbose=0)*, apply the RRCT algorithm to a given training set *X, y* where *X* is an *n by m* numpy array of features, and *y* is an *n by 1* numpy array of   target values. *verbose*, a non-negative integer, controls  the verbosity of output
    - *select(X=X)*,  select features from a given design matrix *X* based on the results of the application of RRCT algorithm
    - *apply_select(X=X, y=y, verbose=0)*, apply RRCT algorithm to a given training set *X, y* and then select features from *X*.

### Installation
```
pip install rrct
```

### Example
```python
# import the RRCT algorithm
from rrct.algorithm import RRCTFeatureSelection

# RRCT with K=20
selector = RRCTFeatureSelection(K=20, scale_feature=False)

# Apply RRCT to a training set X, y
selector.apply(X=X, y=y)

# Select features from X
X_selected = selector.select(X=X)

# Alternatively, apply_select can be called, which applies RRCT and select features from  X
X_selected = selector.apply_select(X=X, y=y)

# Get the selected feature indices
selector.selected_feature_indices_

# Get the summary of the RRCT metrics 
selector.rrct_values_
```
****************************************

### Reference
A. Tsanas: "Relevance, redundancy and complementarity trade-off (RRCT): a generic, efficient, robust feature selection tool", _Patterns_, Vol. 3:100471, 2022
https://doi.org/10.1016/j.patter.2022.100471

*R. Ibraheem is a PhD student of EPSRC's MAC-MIGS Centre for Doctoral Training and he is hosted by the University of Edinburgh. MAC-MIGS is supported by the UK's Engineering and Physical Science Research Council (grant number EP/S023291/1). R. Ibraheem is supervised by [Dr G. dos Reis](https://www.maths.ed.ac.uk/~gdosrei/). R. Ibraheem further contact points: [LinkedIn](https://www.linkedin.com/in/rasheed-oyewole-ibraheem-768955246/), [ORCID](https://orcid.org/0000-0003-4862-5811), [GitHub](https://github.com/Rasheed19).*