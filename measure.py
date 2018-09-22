#城市坐标转换成欧氏距离
import numpy as np
def getdistmat(numcity,coordinates):
    num = coordinates.shape[0]
    distmat = np.zeros((numcity, numcity))
    # 初始化生成10*10的矩阵
    for i in range(num):
        for j in range(i, num):
            distmat[i][j] =distmat[j][i] = round(np.linalg.norm((coordinates[i] - coordinates[j])),1)
    return distmat

def GetTimeMat(numcity,coordinates):
    num = coordinates.shape[0]
    timemat = np.zeros((numcity, numcity))
    # 初始化生成10*10的矩阵
    for i in range(num):
        for j in range(i, num):
            timemat[i][j] = timemat[j][i] = round(np.linalg.norm(((coordinates[i] - coordinates[j])/90)),1)
    return timemat
