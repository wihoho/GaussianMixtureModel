__author__ = 'GongLi'

import os
import numpy as np
from scipy.cluster.vq import *
import pickle
import random


def normalizeSIFT(descriptor):
    descriptor = np.array(descriptor)
    norm = np.linalg.norm(descriptor)

    if norm > 1.0:
        result = np.true_divide(descriptor, norm)
    else:
        result = None

    return result

# Read in video frames under a folder
def readVideoData(pathOfSingleVideo, subSampling = 5):
    frames = os.listdir(pathOfSingleVideo)

    stackOfSIFTFeatures = []
    for frame in frames:
        completePath = pathOfSingleVideo +"/"+ frame
        lines = open(completePath, "r").readlines()

        for line in lines[1::subSampling]:
            data = line.split(" ")
            feature = data[4:]
            for i in range(len(feature)):
                item = int(feature[i])
                feature[i] = item

            # normalize SIFT feature
            feature = normalizeSIFT(feature)
            stackOfSIFTFeatures.append(feature)

    return np.array(stackOfSIFTFeatures)

def normalize(X):
    row = X.shape[0]
    column = X.shape[1]

    maxValues = np.amax(X, axis=1)
    minValues = np.amin(X, axis=1)

    for i in range(row):
        for j in range(column):
            X[i][j] = (X[i][j] - minValues[i]) * 2.0 / (maxValues[i] - minValues[i])

def storeObject(fileName, obj):
    file = open(fileName, "wb")
    pickle.dump(obj, file)
    file.close()

def loadObject(fileName):
    file = open(fileName, "rb")
    obj = pickle.load(file)
    return obj

def randomTrainingIndice(labels):
    labelSize = len(labels)

    birthdayIndices = [i for i in range(labelSize) if labels[i] == "birthday"]
    paradeIndices = [i for i in range(labelSize) if labels[i] == "parade"]
    picnicIndices = [i for i in range(labelSize) if labels[i] == "picnic"]
    showIndices = [i for i in range(labelSize) if labels[i] == "show"]
    sportsIndices = [i for i in range(labelSize) if labels[i] == "sports"]
    weddingIndices = [i for i in range(labelSize) if labels[i] == "wedding"]

    trainingIndice = []

    # randomly choose 3 indices from each class
    trainingIndice += random.sample(birthdayIndices, 3)
    trainingIndice += random.sample(paradeIndices, 3)
    trainingIndice += random.sample(picnicIndices, 3)
    trainingIndice += random.sample(showIndices, 3)
    trainingIndice += random.sample(sportsIndices, 3)
    trainingIndice += random.sample(weddingIndices, 3)

    testIndice = [i for i in range(len(labels))]
    for potential in trainingIndice:
        testIndice.remove(potential)

    return trainingIndice, testIndice

def binaryLabels(semanticLabels):

    # construct binary labels
    setlabels = ["birthday", "picnic", "parade", "show", "sports", "wedding"]
    binaryLabels = np.zeros((len(semanticLabels))).reshape((len(semanticLabels), 1))

    for label in setlabels:

        tempLabel = np.zeros((len(semanticLabels))).reshape((len(semanticLabels), 1))
        for i in range(len(semanticLabels)):
            if semanticLabels[i] == label:
                tempLabel[i][0] = 1

        binaryLabels = np.concatenate((binaryLabels, tempLabel), axis=1)

    binaryLabels = binaryLabels[::, 1::]

    return binaryLabels

if __name__ == "__main__":
    path = "/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak/wedding/VTS_05_01_1318"
    for item in os.listdir(path):
        print path +"/"+item
