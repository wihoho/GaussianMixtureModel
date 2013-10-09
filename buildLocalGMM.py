import os
import Utility as util
from MAP_GMM import MAP_GMM
import time
from multiprocessing import Process

def GMM_Histogram(labelName):

    path = "/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak"
    globalGaussianMixture = util.loadObject("/Users/GongLi/PycharmProjects/GaussianMixtureModel/ClusterSample50/FullCovariance_GlobalGaussianMixtureModel.pkl")
    globalGaussianMixture.n_iter = 50

    classPath = path +"/"+ labelName
    pidName = os.getpid()

    for video in os.listdir(classPath):
        if video == ".DS_Store":
            continue

        existingFiles = os.listdir("Full_128_Iteration_50")
        currentFileName = labelName+"_"+video+".pkl"
        if currentFileName in existingFiles:
            continue


        videoPath = classPath +"/"+ video

        initalTime = time.time()
        videoData = util.readVideoData(videoPath, subSampling = 5)

        mapGMM = MAP_GMM(globalGaussianMixture, videoData)
        mapMean = mapGMM.MAP()
        afterTime = time.time()

        processTime = afterTime - initalTime

        print pidName+ ": MAP_Full_128"+labelName+"_"+video+" "+ str(videoData.shape)+ "   "+str(processTime)+"s"

        outputFileName = "Full_128_Iteration_50/" + currentFileName
        util.storeObject(outputFileName, mapMean)

        del videoData

    print time.ctime()
    print "Finished label :" +labelName


if __name__ == "__main__":

    p1 = Process(target= GMM_Histogram, args =  ("parade",))
    p2 = Process(target= GMM_Histogram, args = ("picnic",))
    p3 = Process(target= GMM_Histogram, args = ("show",))
    p4 = Process(target= GMM_Histogram, args = ("sports",))

    p1.start()
    p2.start()
    p3.start()
    p4.start()