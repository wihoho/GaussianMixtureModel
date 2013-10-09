__author__ = 'GongLi'

import Utility as util
from numpy.linalg import norm
import os
import numpy as np

def GMM_Distance(clipOne, clipTwo, globalGMM, covarianceType):


    if len(globalGMM.covars_.shape) == 3:
        numberGaussian, numberDimension, numberDimension = globalGMM.covars_.shape
    elif len(globalGMM.covars_.shape) == 2:
        numberGaussian, numberDimension = globalGMM.covars_.shape




    resultDistance = 0
    if covarianceType == "spherical":

        for i in range(numberGaussian):
            temp = norm(clipOne[i] - clipTwo[i]) ** 2

            resultDistance += globalGMM.weights_[i] * temp * globalGMM.covars_[i][0]

    if covarianceType == "full":

        for i in range(numberGaussian):
            temp = (clipOne[i] - clipTwo[i]).reshape((1, numberDimension))
            resultDistance += globalGMM.weights_[i] * abs(np.dot(np.dot(temp, 1.0 / globalGMM.covars_[i]), temp.T))

    return resultDistance / 2.0

if __name__ == "__main__":

    globalGMM = util.loadObject("/Users/GongLi/PycharmProjects/GaussianMixtureModel/ClusterSample50/PCA64_SphericalCovariance_GlobalGaussianMixtureModel.pkl")

    path = "/Users/GongLi/PycharmProjects/GaussianMixtureModel/MAP_n_iteration = 50_Spherical Covariance_PCA64"
    labels = []
    clips = []
    for label in os.listdir(path):
        if label == ".DS_Store":
            continue

        labelPath = path +"/"+label

        for video in os.listdir(labelPath):
            if video == ".DS_Store":
                continue

            c = util.loadObject(labelPath +"/"+ video)
            clips.append(c)
            labels.append(label)

    # calculate distances
    numberOfClips = len(labels)
    distanceMatrix = np.zeros((numberOfClips, numberOfClips))
    for i in range(numberOfClips):
        for j in range(i + 1, numberOfClips, 1):

            distanceMatrix[i][j] = GMM_Distance(clips[i], clips[j], globalGMM, "spherical")
            distanceMatrix[j][i] = distanceMatrix[i][j]

            print "(" +str(i) +"," +str(j) +"): " +str(distanceMatrix[i][j])

    # store
    # util.storeObject("KodakLabels.pkl", labels)
    util.storeObject("PCA64_Spherical_GMM_n_iteration50_KodakDistances.pkl", distanceMatrix)








