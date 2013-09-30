import os
import Utility as util
from MAP_GMM import MAP_GMM
import threading
import time
import xlwt

class videoGMM(threading.Thread):

    def __init__(self, labelNames):
        super(videoGMM, self).__init__()
        self.labelNames = labelNames

    def run(self):


        runningTime = xlwt.Workbook()
        rt = runningTime.add_sheet("GMM Running Time")
        inddex = 0

        threadName = self.getName()

        path = "/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak"
        globalGaussianMixture = util.loadObject("ClusterSample50/GlobalGaussianMixtureModel.pkl")
        globalGaussianMixture.n_iter = 50

        for labelName in self.labelNames:
            classPath = path +"/"+ labelName
            for video in os.listdir(classPath):

                if video == ".DS_Store":
                    continue

                videoPath = classPath +"/"+ video

                initalTime = time.time()
                videoData = util.readVideoData(videoPath, subSampling = 5)
                mapGMM = MAP_GMM(globalGaussianMixture, videoData)
                mapMean = mapGMM.MAP()
                afterTime = time.time()

                processTime = afterTime - initalTime
                rt.write(inddex, 0, processTime)
                inddex += 1


                print threadName+": MAP--"+labelName+"_"+video+" "+ str(videoData.shape)+ "   "+str(processTime)+"s!"

                outputFileName = "MAP_n_iteration_50/" +labelName+"_"+video+".pkl"
                util.storeObject(outputFileName, mapMean)

                del videoData

        runningTime.save(threadName+"_RunningTime.xls")
        print time.ctime()


t1 = videoGMM(["birthday", "parade", "picnic", "show", "sports", "wedding"])
t1.start()
