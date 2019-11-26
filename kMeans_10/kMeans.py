'''10_1 K-均值聚类支持函数
2019_11_25
'''

import numpy as np
import math
import random
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    #返回值是一个包含许多其他列表的列表
    return dataMat

def distEclud(vecA, vecB):
    '''
    Function Description:
        数据向量计算欧式距离
    Parameters:
        vecA:数据向量A
        vecB:数据向量B
    Returns:
        两个向量之间的欧几里得距离
    Time:
        2019_11_25
    '''
    #计算两个向量的欧式距离
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    '''
    Function Description:
        随机初始化k个质心(质心满足数据边界之内)
    Parameters:
        dataSet:输入的数据集
        k:选取k个质心
    Returns:
        centroids: 返回初始化得到的k个质心向量
    Time:
        2019_11_25    
    '''
    #得到数据样本的维度
    n = np.shape(dataSet)[1]
    #初始化一个(k, n)的全零矩阵
    centroids = np.mat(np.zeros((k, n)))
    #遍历数据集的每一个维度
    for j in range(n):
        #得到该列数据的最小值和最大值
        minJ = np.min(dataSet[:, j])
        maxJ = np.max(dataSet[:, j])
        #得到该列数据的范围
        rangeJ = float(maxJ - minJ)
        #k个质心向量的第j维数据值随机位于(最小值和最大值)内的某一值
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

'''10_2 K-均值聚类算法
2019_11_25
'''

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    '''
    Function Description:
        K-means聚类算法
    Parameters:
        dataSet:用于聚类的数据集
        k:选取k个质心
        distMeas:距离计算方法，默认欧式距离distEclud()
        createCent:获取k个质心的方法，默认随机获取randCent()
    Returns:
        centroids: k个聚类的聚类结果
        clusterAssment:聚类误差
    Time:
        2019_11_25        
    '''
    #获取数据样本数
    m = np.shape(dataSet)[0]
    #初始化一个(m, 2)全零矩阵
    clusterAssment = np.mat(np.zeros((m, 2)))
    #创建初始的k个质心向量
    centroids = createCent(dataSet, k)
    #聚类结果是否发生变化的布尔类型
    clusterChanged = True
    #只要聚类结果一直发生变化,就一直执行聚类算法，直到所有数据点聚类结果不发生变化
    while clusterChanged:
        #聚类结果变化布尔类型置为False
        clusterChanged = False
        #遍历数据每一个样本向量
        for i in range(m):
            #初始化最小距离为正无穷，最小距离对应的索引为-1
            minDist = float('inf')
            minIndex = -1
            #循环k个类的质心
            for j in range(k):
                #计算数据集中的点分别到质心的欧拉距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                #如果距离小于当前最小距离
                if distJI < minDist:
                    #当前距离为最小距离，最小距离对应索引应为j(第j个类)
                    minDist = distJI
                    minIndex = j
            #当前聚类结果中第i个样本的聚类结果发生变化:布尔值置为Ture，继续聚类算法
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            #更新当前变化样本的聚类结果和平方误差
            clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            #将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            #计算这些数据的均值(axis=0:求列均值),作为该类质心向量
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    #返回K个聚类,聚类结果以及误差
    return centroids, clusterAssment

def plotDataSet(filename, k):
    '''
    Function Description:
        绘制数据集
    Parameters:
        filename:用于绘制的数据集
    Returns:
        NONE
    Time:
        2019_11_25        
    '''
    #导入数据
    datMat = np.mat(loadDataSet(filename))
    #进行k-means算法，其中k为4
    #myCentroids, clustAssing = Kmeans(datMat, 4)
    centList, clusterAssment = biKmeans(datMat, k)
    clusterAssment = clusterAssment.tolist()
    #clustAssing = clustAssing.tolist()
    #myCentroids = myCentroids.tolist()
    xcord = [[], [], []]
    ycord = [[], [], []]
    datMat = datMat.tolist()
    m = len(clusterAssment)
    for i in range(m):
        if int(clusterAssment[i][0]) == 0:
            xcord[0].append(datMat[i][0])
            ycord[0].append(datMat[i][1])
        elif int(clusterAssment[i][0]) == 1:
            xcord[1].append(datMat[i][0])
            ycord[1].append(datMat[i][1])
        elif int(clusterAssment[i][0]) == 2:
            xcord[2].append(datMat[i][0])
            ycord[2].append(datMat[i][1])
        '''
        elif int(clustAssing[i][0]) == 3:
            xcord[3].append(datMat[i][0])
            ycord[3].append(datMat[i][1])
        '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #绘制样本点
    ax.scatter(xcord[0], ycord[0], s=20, c='b', marker='*', alpha=.5)
    ax.scatter(xcord[1], ycord[1], s=20, c='r', marker='D', alpha=.5)
    ax.scatter(xcord[2], ycord[2], s=20, c='c', marker='>', alpha=.5)
    #ax.scatter(xcord[3], ycord[3], s=20, c='k', marker='o', alpha=.5)    
    #绘制质心
    for i in range(k):
        ax.scatter(centList[i].tolist()[0][0], cenList[i].tolist()[0][1], s=100, c='k', marker='+', alpha=.5)
    '''
    ax.scatter(myCentroids[0][0], myCentroids[0][1], s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[1][0], myCentroids[1][1], s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[2][0], myCentroids[2][1], s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[3][0], myCentroids[3][1], s=100, c='k', marker='+', alpha=.5)
    '''
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

'''10_3 二分k-均值聚类算法
2019_11_25
'''

def biKmeans(dataSet, k, distMeas=distEclud):
    '''
    Function Description:
        二分k-means聚类算法
    Parameters:
        dataSet:用于聚类的数据集
        k:选取k个质心
        distMeas:距离计算方法，默认欧式距离distEclud()
    Returns:
        centroids: k个聚类的聚类结果
        clusterAssment:聚类误差
    Time:
        2019_11_25            
    '''
    #获取数据集的样本数
    m = np.shape(dataSet)[0]
    #初始化一个元素均值为0的(m, 2)的矩阵
    clusterAssment = np.mat(np.zeros((m, 2)))
    #获取数据集每一列数据的均值,组成一个列表
    #tolist():将数组或者矩阵转换为列表
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    #当前聚类列表为将数据集聚为一类
    centList = [centroid0]
    #遍历每个数据集样本
    for j in range(m):
        #计算当前聚为一类时各个数据点距离质心的平方距离
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2
    #循环，直至二分k-Means值达到k类为止
    while (len(centList) < k):
        #将当前最小平方误差置为正无穷
        lowerSSE = float('inf')
        #遍历当前的每个聚类
        for i in range(len(centList)):
            #通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            #对该分类利用二分k-means算法进行划分，返回划分后的结果以及误差
            centroidMat, splitClusAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #计算该类划分后两个类的误差平方和
            sseSplit = np.sum(splitClusAss[:, 1])
            #计算数据集重不属于该类的数据的误差平方和
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            #打印这两项误差值
            print('sseSplit = %f, and notSplit = %f' % (sseSplit, sseNotSplit))
            #划分第i类后总误差小于当前最小总误差
            if (sseSplit + sseNotSplit) < lowerSSE:
                #第i类作为本次划分类
                bestCentToSplit = i
                #第i类划分后得到的两个质心向量
                bestNewCents = centroidMat
                #复制第i类中数据点的聚类结果即误差值
                bestClusAss = splitClusAss.copy()
                #将划分为第i类后的总误差作为当前最小误差
                lowerSSE = sseSplit + sseNotSplit
        #数组过滤选出本次2-means聚类划分后类编号为1的数据点，将这些数据点类编号变为当前类个数+1，作为一个新的聚类
        bestClusAss[np.nonzero(bestClusAss[:, 0].A == 1)[0], 0] = len(centList)
        #同理，将划分数据中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号连续不出现空缺
        bestClusAss[np.nonzero(bestClusAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        #打印本次执行2-means聚类算法的类
        print('the bestCentToSplit is %d' % bestCentToSplit)
        #打印被划分的类的数据个数
        print('the len of bestClusAss is %d' % len(bestClusAss))
        #更新质心列表中变化后的质心向量
        centList[bestCentToSplit] = bestNewCents[0, :]
        #添加新的类的质心向量
        centList.append(bestNewCents[1, :])
        #更新clusterAssment列表中参与2-means聚类数据点变化后的分类编号，以及该类数据的误差平方
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClusAss
    #返回聚类结果
    return centList, clusterAssment

'''10_4 对地图上的点进行聚类
2019_11_26
'''

import urllib
import json
from time import sleep

def massPlaceFind(fileName):
    '''
    Function Description:
        具体文本数据批量地址经纬度获取
    Parameters:
        fileName:
    Returns:
        None
    Time:
        2019_11_26            
    '''
    #"wb+"以二进制写方式打开，可以读\写文件，如果文件不存在，创建该文家，如果文件已存在，
    #先清空，再打开文件，以写的方式打开，如果文件不存在，创建该文件，如果文件已存在，先清空，再打开文件
    fw = open('place.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        #获取相应的经度
        lat = float(lineArr[3])
        #获取相应的纬度
        lng = float(lineArr[4])
        #打印地名以及对应的经纬度信息
        print('%s\t%f\t%f' % (lineArr[0], lat, lng))
    fw.close()

def distSLC(vecA, vecB):
    '''
    Function Description:
        球面距离计算
    Parameters:
        vecA:数据向量
        vecB:数据向量
    Returns:
        球面距离
    Time:
        2019_11_26            
    '''
    a = math.sin(vecA[0, 1] * np.pi / 180) * math.sin(vecB[0, 1] * np.pi / 180)
    b = math.cos(vecA[0, 1] * np.pi / 180) * math.cos(vecB[0, 1] * np.pi / 180) * math.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return math.acos(a + b) * 6371.0

def clusterClubs(numClust=5):
    '''
    Function Description:
        使用k-means聚类解决问题
    Parameters:
        numClust:聚类个数
    Returns:
        None
    Time:
        2019_11_26            
    '''
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    #利用2-means聚类算法聚类
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0],\
                    ptsInCurrCluster[:, 1].flatten().A[0],\
                    marker=markerStyle, s=90)
    for i in range(numClust):
        ax1.scatter(myCentroids[i].tolist()[0][0], myCentroids[i].tolist()[0][1], s=300, c='k', marker='+', alpha=.5)
    plt.show()

if __name__ == '__main__':
    '''
    datMat = np.mat(loadDataSet('testSet.txt'))
    myCentroids, clustAssing = kMeans(datMat, 4)
    print(myCentroids, clustAssing)
    '''

    #plotDataSet('testSet.txt')
    
    '''
    datMat = np.mat(loadDataSet('testSet2.txt'))
    cenList, myNewAssments = biKmeans(datMat, 3)
    plotDataSet('testSet2.txt', 3)
    '''

    clusterClubs()






