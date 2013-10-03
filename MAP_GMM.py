__author__ = 'GongLi'

import GMM
import numpy as np
import os
import Utility as util
import time


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



            if i > 1:
                difference = abs(log_likehood[-1] - log_likehood[-2])
                print "Iteration: "+str(i) +"  "+ str(difference)

            # Maximization
            self.Maximization(responsibilities)

        return self.clipMean

if __name__ == "__main__":

    logFile = open("log", "w")
    logFile.write("Satrt reading Kodak videos: "+time.ctime() +"\n")

    path = "/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak"
    videoList = []
    pca64 = util.loadObject("/Users/GongLi/PycharmProjects/GaussianMixtureModel/ClusterSample50/pca64.pkl")

    for label in os.listdir(path):
        if label == ".DS_Store":
            continue

        classPath = path +"/"+ label
        for video in os.listdir(classPath):
            if video == ".DS_Store":
                continue

            videoPath = classPath +"/"+ video
            print videoPath

            videoData = util.readVideoData(videoPath, subSampling=50)
            videoData = pca64.transform(videoData)
            videoList.append(videoData)


    allKodakVideos = np.vstack(videoList)
    print "Kodak Videos Size: " + str(allKodakVideos.shape)
    logFile.write("Kodak Video Size: " +str(allKodakVideos.shape) +"\n")
    logFile.write("Finish reading Kodak videos: " +time.ctime()+ "\n")
    print "Finish reading Kodak videos: "+time.ctime()

    # Perform GMM
    globalGaussianMixture = GMM.GMM(n_components=500, covariance_type="full", init_params="wmc", n_iter=50)
    globalGaussianMixture.fit(allKodakVideos)

    util.storeObject("PCA64_FullCovariance_GlobalGaussianMixtureModel.pkl", globalGaussianMixture)
    logFile.write("Finishing building Global GMM:" +time.ctime()+"/n")
    print "Finishing building Global GMM" + time.ctime()
    logFile.close()








