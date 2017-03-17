# Linear-Regression

The first feature
is a constant feature, always equal to one: this is just here to simplify estimation of the
"intercept" term. The next 13 features are CRIM, ZN, . . . , LSTAT, as described in https:
//archive.ics.uci.edu/ml/datasets/Housing; note that standardization (based on the
training data) has been applied (to both the training data and test data). The output (label)
is MEDV, the median value of owner-occupied homes (in units of $1000).

This code first compute the ordinary least squares (OLS) estimator based on the training data (data and
labels). Then compute a sparse weight vector with at most three non-
zero entries (not including the "intercept") based on the training data using Lasso, LARS (which is actually an algorithm for solving the
Lasso optimization problem with some additional convenient properties),
orthogonal matching pursuit. Compute the average
squared loss of this sparse linear predictor on the test data (testdata and testlabels).
