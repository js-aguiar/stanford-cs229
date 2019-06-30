"""
Includes all custom warnings and error classes.
"""

class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting."""

class InvalidActivation(Exception):
    """Exception class to raise if the activation function was not implemented."""