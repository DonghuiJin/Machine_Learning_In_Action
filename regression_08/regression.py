import numpy as np
import math
'''8_1 标准回归函数和数据导入函数
2019_11_18
'''

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i])) 
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

#计算最佳拟合直线
def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    #判断xTx的行列式是否为0，如果为0说明该式没有逆矩阵
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

'''8_2 局部加权线性回归函数
2019_11_18
'''

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    #创建对角矩阵
    weights = np.mat(np.eye((m)))
    for j in range(m):
        #权重值大小以指数级衰减
        diffMat = testPoint - xMat[j, :]
        #高斯核对应的权重，参数k控制衰减的速度
        weights[j, j] = math.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    #求出回归系数w
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

#为数据集中的每个点调用lwlr()，这有助与求解k的大小
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

#预测鲍鱼年龄    
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


'''8_3 岭回归
2019_11_20
'''

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, math.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat





'''
8_1 标准回归函数和数据导入函数
import regression
import numpy as np
xArr, yArr = regression.loadDataSet('ex0.txt')
ws = regression.standRegres(xArr, yArr) 
画出图像
xMat = np.mat(xArr)
yMat = np.mat(yArr)
yHat = xMat * ws
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
画出拟合的直线
xCopy = xMat.copy()
xCopy.sort(0) #对数据点进行排序
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()
计算相关系数
yHat = xMat * ws
np.corrcoef(yHat.T, yMat)

8_2 局部加权线性回归
import regression
import numpy as np
xArr, yArr = regression.loadDataSet('ex0.txt')
yArr[0]
regression.lwlr(xArr[0], xArr, yArr, 1.0)
regression.lwlr(xArr[0], xArr, yArr, 0.001)
yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)
xMat = np.mat(xArr)
srtInd = xMat[:, 1].argsort(0)
xSort = xMat[srtInd][:, 0, :]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[srtInd])
ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red)
plt.show()

'''









