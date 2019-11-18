import numpy as np
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
    ws = xTx. * (xMat.T * yMat)
    return ws

'''8_2 局部加权线性回归函数
2019_11_18
'''

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.T * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat







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
'''

