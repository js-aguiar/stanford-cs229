
import numpy as np
from ..exceptions import NotFittedError
from ..utils import check_features, check_target

class GaussianDiscriminantAnalysis(object):
    """This classifier models p(x|y) using a multivariate normal distribution.

    This implementation works only for binary classification.
    Model assumptions are:

    y ∼ Bernoulli(φ)
    x|y = 0 ∼ N (µ0, Σ)
    x|y = 1 ∼ N (µ1, Σ)

    The parameters for this model are φ, Σ, µ0 and µ1.

    Attributes:
        sigma: feature covariance matrix used for both classes.
        This is a numpy array with shape (n_features, n_features).
        mean0: Mean value for each feature for samples with target 0.
        mean0: Mean value for each feature for samples with target 1.
        phi: positive case rate (prior to our model)
    """
    def calculate_covariance(self, x, y):
        """Returns the covariance matrix sigma."""
        m = len(y)
        sigma = np.zeros(shape=(x.shape[1], x.shape[1]))
        for i in range(m):
            xi = x[i].reshape(-1, 1)
            diff = xi - self.mean0 if y[i] == 0 else xi - self.mean1
            sigma += np.dot(diff, diff.T)
        return sigma / m

    def fit(self, x, y):
        """Fit GDA model to features x and target y.

        Arguments:
            x: Features - nd array with shape (m_samples, n_features).
            If you have a pandas Dataframe use df.values.
            y: Target values. Can be a one or two dimensional numpy
            array with shape (m_samples,) or (m_samples, 1)
        """
        check_features(x)
        check_target(y)
        y = y.flatten() if y.ndim > 1 else y

        # parameter φ is the positive case ratio
        self.phi = len(y[y == 1]) / len(y)

        # calculate the mean vector for each class (µ0 and µ1)
        self.mean0 = np.mean(x[y == 0], axis=0).reshape(-1, 1)
        self.mean1 = np.mean(x[y == 1], axis=0).reshape(-1, 1)

        # covariance matrix
        self.sigma = self.calculate_covariance(x, y)
        return self

    def predict(self, x):
        """Return predictions for ndarray with shape (m_samples, n_features)."""
        check_features(x)

        try:
            # calculate the determinant and inverse for multivariate normal
            denom = (2*np.pi)**(x.shape[1] / 2) * np.linalg.det(self.sigma)**0.5
            inv = np.linalg.inv(self.sigma)
        except AttributeError:
            raise NotFittedError("You must fit before making predictions.")

        predictions = np.zeros(shape=(x.shape[0], 1))
        for i in range(len(x)):
            xi = x[i].reshape(-1, 1)
            p0 = np.exp(-0.5 * (xi - self.mean0).T @ inv @ (xi - self.mean0)) / denom
            p1 = np.exp(-0.5 * (xi - self.mean1).T @ inv @ (xi - self.mean1)) / denom
            # prediction is arg max y: p(x|y)p(y)
            predictions[i] = 1 if p1 * self.phi > p0 * (1 - self.phi) else 0
        return predictions