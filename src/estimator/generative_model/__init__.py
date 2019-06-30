"""
    This module implements the following generative models:
    Gaussian Discriminant Analysis and Multinomial Naive Bayes.

    Generative models try to model p(x|y) and p(y) to later derive
    p(y|x) trough the Bayes Theorem.
"""

from .gda import GaussianDiscriminantAnalysis
from .naive_bayes import MultinomialNaiveBayes


__all__ = ['GaussianDiscriminantAnalysis', 'MultinomialNaiveBayes']