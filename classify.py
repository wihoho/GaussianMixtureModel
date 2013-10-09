__author__ = 'GongLi'

import SVM_T
import Utility as util
import numpy as np
import math
from sklearn.svm import SVC


def multiClassSVM(distances, trainingIndice, testingIndice, semanticLabels):

    trainDistance = distances[np.ix_(trainingIndice, trainingIndice)]
    testDistance = distances[np.ix_(testingIndice,trainingIndice)]

    meanTrainValue = np.mean(trainDistance)

    trainGramMatrix = math.e **(0 - trainDistance / meanTrainValue)
    testGramMatrix = math.e ** (0 - testDistance / meanTrainValue)
    trainLabels = [semanticLabels[i] for i in trainingIndice]
    testLabels = [semanticLabels[i] for i in testingIndice]

    clf = SVC(kernel = "precomputed")
    clf.fit(trainGramMatrix, trainLabels)
    SVMResults = clf.predict(testGramMatrix)

    correct = sum(1.0 * (SVMResults == testLabels))
    accuracy = correct / len(testLabels)
    # print "accuracy: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

    return accuracy



if __name__ == "__main__":

    GMM_distance = util.loadObject("/Users/GongLi/PycharmProjects/GaussianMixtureModel/Distances/Spherical 128/GMM_n_iteration500_KodakDistances.pkl")
    GMM_labels = util.loadObject("/Users/GongLi/PycharmProjects/GaussianMixtureModel/Distances/Full 36/PCA36_GMM_n_iteration50_KodakLabels.pkl")

    EMD_distance = util.loadObject("/Users/GongLi/PycharmProjects/GaussianMixtureModel/Distances/Spherical 64/PCA64_Spherical_GMM_n_iteration50_KodakDistances.pkl")
    EMD_labels = util.loadObject("/Users/GongLi/PycharmProjects/GaussianMixtureModel/Distances/EMD_KodakLabelsLevel0.pkl")


    GMM_distances = []
    GMM_distances.append(GMM_distance)

    EMD_distances = []
    EMD_distances.append(EMD_distance)

    if GMM_labels == EMD_labels:
        print "Labels are the same!"

    binaryLabels = util.binaryLabels(EMD_labels)
    allGMM = []
    allEMD = []

    for i in range(10):

        trainingIndice, testingIndice = util.randomTrainingIndice(GMM_labels, 2)
        GMM_aps = SVM_T.runSVM_T(GMM_distances, binaryLabels, trainingIndice, testingIndice)
        EMD_aps = SVM_T.runSVM_T(EMD_distances, binaryLabels, trainingIndice, testingIndice)

        # GMM_aps = multiClassSVM(GMM_distance, trainingIndice, testingIndice, EMD_labels)
        # EMD_aps = multiClassSVM(EMD_distance, trainingIndice, testingIndice, EMD_labels)

        allGMM.append(GMM_aps)
        allEMD.append(EMD_aps)

        print str(i) +" ----------------"
        print "GMM: " + str(GMM_aps)
        print "EMD: " + str(EMD_aps)
        print " "

    all_aps = np.array(allGMM)
    meanAP = np.mean(all_aps)

    rowMean = np.mean(all_aps, axis=1)
    sd = np.std(rowMean)

    print "GMM_meanAP: "+str(meanAP)
    print "GMM_standard deviation: "+str(sd)

    all_aps = np.array(allEMD)
    meanAP = np.mean(all_aps)

    rowMean = np.mean(all_aps, axis=1)
    sd = np.std(rowMean)

    print "EMD_meanAP: "+str(meanAP)
    print "EMD_standard deviation: "+str(sd)