
import numpy as np
from ..exceptions import NotFittedError
from ..utils import check_features, check_target

class LinearRegression(object):
    """Linear regression estimator.

    Can be solved in closed form using the normal equation or singular
    value decomposition. It's also possible to fit using a batch 
    gradient descent algorithm.

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

    def __init__(self, solving_method='svd', learning_rate=0.01,
                 num_iter=1000, save_cost=False):
        self.solving_method = solving_method
        self.learning_rate = learning_rate
        self.num_iterations = num_iter
        self.save_cost = save_cost

        solvers = ['svd', 'normal_equation', 'gradient_descent']
        if self.solving_method not in solvers:
            raise ValueError("solving_method must be one of: ", solvers)

    def fit(self, x, y):
        """Fit a linear regression model to features x and target y.

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
    
        if self.solving_method == 'svd':
            self.parameters = np.linalg.pinv(x) @ y
        elif self.solving_method == 'normal_equation':
            # Normal equation: inv(X'*X) * X' * Y
            self.parameters = np.linalg.pinv(x.T @ x) @ x.T @ y
        else:
            self.parameters = np.zeros(shape=(x.shape[1], 1))
            self.batch_gradient_descent(x, y)
        return self

    def predict(self, x):
        """Return predictions for ndarray with shape (m_samples, n_features)."""
        check_features(x)
        x = np.insert(x, 0, 1, axis=1)  # add bias
        try:
            return self.hypothesis(x)
        except AttributeError:
            raise NotFittedError("You must fit before making predictions.")

    def hypothesis(self, x):
        """Hypothesis for linear regression (ax + b)."""
        return x @ self.parameters

    def gradient_function(self, x, y, predictions=None):
        """Compute the gradient with the current parameters."""
        if predictions is None:
            predictions = self.hypothesis(x)
        return x.T @ (predictions - y)

    def cost_function(self, x, y, predictions=None):
        """Return the cost for the current parameters using MSE."""
        if predictions is None:
            predictions = self.hypothesis(x)
        squared_error = np.sum(np.square(predictions - y))
        return squared_error / (2 * x.shape[0])

    def batch_gradient_descent(self, x, y):
        """Batch gradient descent implementation (vectorized)."""

        self.grad_descent_cost = []
        for _ in range(self.num_iterations):
            # Make predictions and calculate gradient
            predictions = self.hypothesis(x)
            gradient = self.gradient_function(x, y, predictions)

            # Update current parameters
            self.parameters = self.parameters - self.learning_rate / x.shape[0] * gradient
            if self.save_cost:
                self.grad_descent_cost.append(self.cost_function(x, y, predictions))


lr = LinearRegression()
