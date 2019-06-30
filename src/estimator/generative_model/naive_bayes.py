
import numpy as np
from ..exceptions import NotFittedError
from ..utils import check_features, check_target


class MultinomialNaiveBayes(object):
    """Naive Bayes for multinomial distributed features.

    This implementation is also based on the following reference:
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

    Attributes:
        alpha: A smoothing parameter. If alpha is one it's called
        Laplace smoothing, while alpha < 1 is called Lidstone smoothing.
        feature_log_prob: Log probability for each feature and class.
        class_log_prior: Prior probability for each class.
    """
    def __init__(self, alpha=1):
        if alpha > 1 or alpha < 0:
            raise ValueError("Alpha must be a number between 0 and 1.")
        self.alpha = alpha

    def fit(self, x, y):
        """Fit Naive Bayes to features x and target y.

        Arguments:
            x: Features - nd array with shape (m_samples, n_features).
            If you have a pandas Dataframe use df.values.
            y: Target values. Can be a one or two dimensional numpy
            array with shape (m_samples,) or (m_samples, 1)
        """
        check_features(x)
        check_target(y)
        y = y.flatten() if y.ndim > 1 else y

        num_classes = len(np.unique(y))
        n = x.shape[1]
        self.feature_log_prob = np.zeros(shape=(num_classes, n))
        self.class_log_prior = np.zeros(shape=(num_classes, 1))
        for c in range(num_classes):
            # count the number of times each feature appears in a class
            count = x[y == c].sum(axis=0)
            # count all features and add smoothing
            count_all = count.sum() + self.alpha * n
            count += self.alpha
            
            self.feature_log_prob[c] = np.log(count / count_all)
            self.class_log_prior[c] = np.log(len(y[y == c]) / len(y))
        return self

    def predict(self, x):
        """Return predictions for ndarray with shape (m_samples, n_features)."""
        check_features(x)
        try:
            return np.argmax(np.dot(x, self.feature_log_prob.T) + self.class_log_prior.T, axis=1)
        except AttributeError:
            raise NotFittedError("You must fit before making predictions.")