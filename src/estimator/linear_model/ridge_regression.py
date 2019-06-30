
import numpy as np
from ..utils import check_features, check_target
from .linear_regression import LinearRegression

class RidgeRegression(LinearRegression):
    """Linear regression with l2 regularization.

    Attributes:
        learning_rate: Rate of update for gradient descent at
        each iteration. Not used for other solving methods (float).
        num_iterations: Number of iterations for gradient descent (int).
        save_cost: Whether to save or not the cost of each iteration.
        Only used for gradient descent (boolean).
        grad_descent_cost: List with the cost for each iteration when
        using gradient descent (list of floats).
        parameters: nd array with the model parameters (theta).
    """

    def __init__(self, regularization=0.1, solving_method='normal_equation',
                 learning_rate=0.01, num_iter=1000, save_cost=False):
        self.regularization = regularization
        self.solving_method = solving_method
        self.learning_rate = learning_rate
        self.num_iterations = num_iter
        self.save_cost = save_cost

        solvers = ['normal_equation', 'gradient_descent']
        if self.solving_method not in solvers:
            raise ValueError("solving_method must be one of: ", solvers)

    def fit(self, x, y):
        """Fit a ridge regression model to features x and target y.

        Arguments:
            x: Features - nd array with shape (m_samples, n_features).
            If you have a pandas Dataframe use df.values.
            y: Target values. Can be a one or two dimensional numpy
            array with shape (m_samples,) or (m_samples, 1)
        """
        check_features(x)
        check_target(y)
        x = np.insert(x, 0, 1, axis=1)  # add bias
        y = y.reshape(-1, 1) if y.ndim == 1 else y
    
        if self.solving_method == 'normal_equation':
            self.normal_equation(x, y)
        else:
            self.parameters = np.zeros(shape=(x.shape[1], 1))
            self.batch_gradient_descent(x, y)
        return self

    def normal_equation(self, x, y):
        """Normal equation for ridge: inv(X'*X + lambda*L) * X' * Y."""
        # L is the identity matrix with shape (n+1) and first cell zero.
        L = np.identity(x.shape[1])
        L[0][0] = 0
        inverse = np.linalg.pinv(x.T @ x + self.regularization*L)
        self.parameters = inverse @ x.T @ y

    def hypothesis(self, x):
        """Hypothesis for linear regression (ax + b)."""
        return x @ self.parameters

    def gradient_function(self, x, y, predictions=None):
        """Compute the gradient with the current parameters."""
        if predictions is None:
            predictions = self.hypothesis(x)
        penalty = self.regularization * np.sum(self.parameters[1:])
        return x.T @ (predictions - y) + penalty

    def cost_function(self, x, y, predictions=None):
        """Return the cost for the current parameters using MSE."""
        if predictions is None:
            predictions = self.hypothesis(x)
        squared_error = np.sum(np.square(predictions - y))
        penalty = self.regularization * np.sum(np.square(self.parameters))
        return (squared_error + penalty) / (2 * x.shape[0])