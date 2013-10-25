import Utility as util
import numpy as np

file = open("tempDistances", "r").readlines()
distanceMatrix = np.zeros((195, 195))

for line in file:
    line = line.split(":")
    locations = line[0][1:-1]
    dis = line[1]

    if "\n" in dis:
        dis = dis[:-1]

    locations = locations.split(",")
    dis = float(dis)
    x = int(locations[0])
    y = int(locations[1])

    distanceMatrix[x][y] = dis

    print str(x) +","+ str(y) +": " +str(dis)

util.storeObject("Full_GMM_n_iteration50_KodakDistances.pkl", distanceMatrix)
