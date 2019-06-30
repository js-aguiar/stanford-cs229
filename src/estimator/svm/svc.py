import numpy as np
from ..exceptions import NotFittedError


class SVC(object):
    """Support Vector Machine for classification.
    
    Using SMO algorithm for optimization

    To be implemented...
    """

    def __init__(self, regulatization):
        self.C = regularization