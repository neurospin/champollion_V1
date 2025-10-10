from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.metrics import pairwise_distances
import numpy as np
from numpy import power, cov, asanyarray

class KernelMetric(BaseEstimator):
    """
        This class provides an interface for fast kernel computation. It computes the similarity 
        matrix between input samples based on Kernel Density Estimation (KDE) weighting scheme [1, 2].
        Concretely, it computes the following similarity matrix between x1, ..., xn:

                        `Sij = K(H**(-1/2) (xi-xj))`
        
        with K a kernel such that (1) K(x) >= 0, (2) int K(x) dx = 1 and (3) K(x) = K(-x)
        and H is the bandwidth in the KDE estimation of p(X). H is symmetric definite positive and it 
        can be automatically computed based on Scott's method or Silverman's method. 

        [1] Rosenblatt, M. (1956). "Remarks on some nonparametric estimates of a density function". Annals of Mathematical Statistics.
        [2] Parzen, E. (1962). "On estimation of a probability density function and mode"
        
        Attributes:
            d_: data dimensionality
            n_: number of samples seen during fit
            sqr_bandwidth_: square root of the bandwidth computed during fit, shape (d_, d_)
            inv_sqr_bandwidth_: inverse of scaled square root of the bandwidth computed during fit, shape (d_, d_)

    """

    def __init__(self, kernel="gaussian", bw_method="scott"):
        """
        Parameters
        ----------
        kernel: {'gaussian', 'epanechnikov', 'exponential', 'linear', 'cosine'}, default='gaussian'
            The kernel applied to the distance between samples.

        bw_method: str or scalar, default="scott"
            The method used to calculate the estimator bandwidth:
            - If `bw_method` is str, must be 'scott' or 'silverman'. 
                Bandwidth is a scaled version of the data covariance matrix.
            - If `bw_method` is scalar (float or int), it sets the bandwidth to H=diag(scalar)
        """
        self.kernel = self._validate_kernel(kernel)
        self.bw_method = bw_method

        # Get covariance factor from bandwidth estimator
        if self.bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif self.bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif isinstance(self.bw_method, float) or isinstance(self.bw_method, int):
            self.covariance_factor = lambda: None
        else:
            raise ValueError("`bw_method` should be 'scott', 'silverman' or a scalar.")

    def fit(self, X):
        """ Computes the bandwidth in the kernel density estimator of p(X).

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Input data used to estimate the bandwidth (based on covariance matrix).

        Returns
        ----------
        self: KernelMetric
        """
        X = check_array(self.atleast_2d(X))
        self.n_, self.d_ = X.shape[0], X.shape[1]
        self.set_bandwidth(X)
        return self

    def set_bandwidth(self, X):
        """Compute the estimator bandwidth. Implementation from scipy.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluation of the estimated density.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Input data.
        """
        X = check_array(self.atleast_2d(X))

        if isinstance(self.bw_method, float) or isinstance(self.bw_method, int):
            self.sqr_bandwidth_ = np.sqrt(np.diag([self.bw_method for _ in range(X.shape[1])]))
            self.inv_sqr_bandwidth_ = np.divide(
                1., self.sqr_bandwidth_, 
                out=np.zeros_like(self.sqr_bandwidth_),
                where=self.sqr_bandwidth_!=0
            )
        else:
            factor = self.covariance_factor()
            covariance = self.atleast_2d(cov(X, rowvar=False, bias=False))
            # Removes non-diagonal term in covariance matrix to produce bandwidth estimator
            # Computes square root inverse of covar matrix (can be prone to error...)
            _data_sqr_cov = np.sqrt(np.diag(np.diag(covariance)))
            _data_inv_sqr_cov = np.divide(1., _data_sqr_cov,
                                        out=np.zeros_like(_data_sqr_cov),
                                        where=_data_sqr_cov!=0)
            self.sqr_bandwidth_ = _data_sqr_cov * factor
            self.inv_sqr_bandwidth_ = _data_inv_sqr_cov / factor

    def scotts_factor(self):
        """Compute Scott's factor.
        Returns
        -------
        s : float
            Scott's factor.
        """
        check_is_fitted(self, attributes=["n_", "d_"])
        return power(self.n_, -1./(self.d_+4))

    def silverman_factor(self):
        """Compute the Silverman factor.
        Returns
        -------
        s : float
            The silverman factor.
        """
        check_is_fitted(self, attributes=["n_", "d_"])
        return power(self.n_*(self.d_+2.0)/4.0, -1./(self.d_+4))

    def pairwise(self, X):
        """
        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Input data.

        Returns
        ----------
        S: array of shape (n_samples, n_samples)
            Similarity matrix between input data. 
        """
        check_is_fitted(self, attributes=["inv_sqr_bandwidth_"])
        X = check_array(self.atleast_2d(X))
        if X.shape[1] != self.d_:
            raise ValueError("Wrong data dimension, got %i (expected %i)"%(X.shape[1], self.d_))
        X_transformed = X @ self.inv_sqr_bandwidth_.T
        pdist = pairwise_distances(X_transformed, metric='euclidean')
        S = self.kernel(pdist)
        return S

    def fit_pairwise(self, X):
        self.fit(X)
        return self.pairwise(X)

    def atleast_2d(self, X):
        X = asanyarray(X)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X[:, np.newaxis]
        return X

    def _validate_kernel(self, kernel):
        if isinstance(kernel, str) and hasattr(self, kernel.capitalize()):
            return getattr(self, kernel.capitalize())()
        raise NotImplementedError("Unknown kernel: %s"%kernel)

    class Gaussian(object):
        def __call__(self, x):
            return np.exp(-x**2/2)

    class Epanechnikov(object):
        def __call__(self, x):
            return (1 - x**2) * (np.abs(x) < 1)

    class Exponential(object):
        def __call__(self, x):
            return np.exp(-x)

    class Linear(object):
        def __call__(self, x):
            return (1 - x) * (np.abs(x) < 1)

    class Cosine(object):
        def __call__(self, x):
            return np.cos(np.pi * x /2.0) * (np.abs(x) < 1)


def get_kernel_distance(K: np.ndarray):
    """ Estimate the kernel distance from a kernel matrix K
    :param K: array with shape (n_samples, n_samples)
    :return: distance matrix with shape (n_samples, n_samples)
    """
    diag_K = np.diag(K)
    squared_dist_K = np.maximum(diag_K[:, np.newaxis] + diag_K[np.newaxis, :] - 2 * K, 0)
    return np.sqrt(squared_dist_K)