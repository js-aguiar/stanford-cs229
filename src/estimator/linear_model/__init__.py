"""
    This module implements the following generalized linear models:
    Linear regression, Ridge regression and Logistic regression.
"""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .ridge_regression import RidgeRegression


__all__ = ['LinearRegression', 'RidgeRegression', 'LogisticRegression']