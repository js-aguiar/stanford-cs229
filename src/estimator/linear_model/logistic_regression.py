
import numpy as np
from ..utils import check_features, check_target
from .linear_regression import LinearRegression

class LogisticRegression(LinearRegression):
    """Logistic regression estimator with l2 penalty.

    In this estimator we are assuming that the target is
    distributed according to a Bernoulli distribution. This
    implementation uses only gradient descent for fitting the
    model parameters.

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

    def __init__(self, regularization=0.1, learning_rate=0.01,
                 num_iter=1000, save_cost=False):
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.num_iterations = num_iter
        self.save_cost = save_cost

    def fit(self, x, y):
        """Fit a logistic regression model to features x and target y.

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

        self.parameters = np.zeros(shape=(x.shape[1], 1))
        self.batch_gradient_descent(x, y)
        return self

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, x):
        """Hypothesis for logistic regression."""
        return self.sigmoid(x @ self.parameters)

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
        cost = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
        penalty = self.regularization * np.sum(np.square(self.parameters))
        return cost.mean() + penalty / (2 * x.shape[0])