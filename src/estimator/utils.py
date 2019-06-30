
import numpy as np

def check_features(x):
    """Verify if input x is in the right type and shape.
    
    The argument must a be two dimensional numeric numpy
    array with at least two items in the first dimension.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numpy array.")
    elif x.ndim != 2:
        raise ValueError("x must have two dimensions: (m_samples, n_features).")
    elif x.shape[0] < 2:
        raise ValueError("x must have shape (m_samples, n_features)"
                            " with at least two samples.")
    elif not np.issubdtype(x.dtype, np.number):
        raise ValueError("x must be a numeric ndarray.")


def check_target(y):
    """Verify if input y is in the right type and shape.
    
    The argument must be a one dimensional numpy array, also
    know as Rank 1 Array with numeric data.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy array.")
    if y.ndim == 2:
        if y.shape[1] > 1:
            raise ValueError("y must have shape (m_samples, 1) or (m_samples,)")
    elif y.ndim > 2:
        raise ValueError("y must have one or two dimensions.")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("y must be a numeric ndarray.")