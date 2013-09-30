__author__ = 'GongLi'

import SVM_T
import Utility as util
import numpy as np

if __name__ == "__main__":

    GMM_distance = util.loadObject("Distances/GMM_KodakDistances.pkl")
    GMM_labels = util.loadObject("Distances/GMM_KodakLabels.pkl")

    EMD_distance = util.loadObject("Distances/EMD_KodakDistanceMatrixLevel0.pkl")
    EMD_labels = util.loadObject("Distances/EMD_KodakLabelsLevel0.pkl")


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

        trainingIndice, testingIndice = util.randomTrainingIndice(GMM_labels)
        GMM_aps = SVM_T.runSVM_T(GMM_distances, binaryLabels, trainingIndice, testingIndice)
        EMD_aps = SVM_T.runSVM_T(EMD_distances, binaryLabels, trainingIndice, testingIndice)

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