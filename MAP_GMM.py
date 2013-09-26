__author__ = 'GongLi'

import GMM
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold


EPS = np.finfo(float).eps

class MAP_GMM:

    def __init__(self, GaussianMixtureModel, videoClip):

        self.globalGMM = GaussianMixtureModel
        self.videoClip = videoClip
        self.clipMean = np.copy(self.globalGMM.means_)
        self.clipWeights = np.copy(self.globalGMM.weights_)

    def Expectation(self):

        videoClip = np.asarray(self.videoClip)
        if videoClip.ndim == 1:
            videoClip = videoClip[:, np.newaxis]
        if videoClip.size == 0:
            return np.array([]), np.empty((0, self.globalGMM.n_components))
        if videoClip.shape[1] != self.videoClip.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')

        lpr = GMM.log_multivariate_normal_density(videoClip, self.clipMean, self.globalGMM.covars_, self.globalGMM.covariance_type) + np.log(self.clipWeights)
        logprob = GMM.logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])

        return logprob, responsibilities


    def Maximization(self, responsibilities):

        weights = responsibilities.sum(axis=0)
        weighed_X_sum = np.dot(responsibilities.T, self.videoClip)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        self.clipWeights = (weights / (weights.sum() + 10 * EPS) + EPS)
        E = weighed_X_sum * inverse_weights
        alpa = (weights) / ((weights + self.videoClip.shape[0]) +EPS)
        oneMinusAlpha = 1 - alpa
        self.clipMean = E * alpa.reshape((alpa.shape[0], 1)) + self.clipMean * oneMinusAlpha.reshape((alpa.shape[0], 1))

    def MAP(self):

        log_likehood = []

        for i in range(self.globalGMM.n_iter):

            # Expectation
            curr_log_likehood, responsibilities = self.Expectation()
            log_likehood.append(curr_log_likehood.sum())

            if i > 0 and abs(log_likehood[-1] - log_likehood[-2]) < 1e-2:
                break

            # Maximization
            self.Maximization(responsibilities)

        return self.clipMean

if __name__ == "__main__":

    iris = datasets.load_iris()

    # Break up the dataset into non-overlapping training (75%) and testing
    # (25%) sets.
    skf = StratifiedKFold(iris.target, n_folds=4)
    # Only take the first fold.
    train_index, test_index = next(iter(skf))


    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    X_test = iris.data[test_index]
    y_test = iris.target[test_index]

    globalGMM = GMM.GMM(n_components=3, covariance_type="spherical", init_params="wmc", n_iter=50)
    globalGMM.fit(X_train)
    mapGMM = MAP_GMM(globalGMM, X_test)
    result = mapGMM.MAP()

    print "Yes"
