import os
import Utility as util
from MAP_GMM import MAP_GMM

path = "/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak"
globalGaussianMixture = util.loadObject("/Users/GongLi/PycharmProjects/GaussianMixtureModel/GlobalGaussianMixtureModel.pkl")

# Perform MAP GMM for each video clip
for label in os.listdir(path):
    if label == ".DS_Store":
        continue

    classPath = path +"/"+ label
    for video in os.listdir(classPath):
        if video == ".DS_Store":
            continue

        videoPath = classPath +"/"+ video
        print "MAP: "+videoPath

        videoData = util.readVideoData(videoPath, subSampling=1)
        mapGMM = MAP_GMM(globalGaussianMixture, videoData)
        mapMean = mapGMM.MAP()

        outputFileName = "MAP/"+label+"_"+video
        util.storeObject(outputFileName, mapMean)

