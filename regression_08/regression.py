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
#用于计算回归系数
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

#数据标准化，所有特征都减去鸽子的均值并除以方差
def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    #数据标准化
    yMat = yMat - yMean
    #计算平均值
    xMeans = np.mean(xMat, 0)
    #计算方差
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, math.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

'''8_4 前向逐步线性回归
2019_11_21
'''

def regularize(xMat, yMat):
    inxMat = xMat.copy()
    inyMat = yMat.copy()
    yMean = np.mean(yMat, 0)
    inyMat = yMat - yMean
    inMeans = np.mean(inxMat, 0)
    inVar = np.var(inxMat, 0)
    inxMat = (inxMat - inMeans) / inVar
    return inxMat, inyMat

#逐步线性回归算法的实现
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''
    输入变量:
    xArr:输入数据
    yArr:预测变量
    eps:每次迭代需要调整的步长
    numIt:迭代次数
    返回值:
    returnMat:权重矩阵
    '''
    
    #标准化处理
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat, yMat = regularize(xMat, yMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    #为了实现贪心算法建立了ws的两份副本
    wsTest = ws.copy()
    wsMax = ws.copy()
    #优化迭代
    for i in range(numIt):
        print(ws.T)
        lowestError = float('inf')
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                #分别计算增加或减少该特征对误差的影响
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                #预测鲍鱼年龄的例子
                #.A 将矩阵转化为数组
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

'''8_5 购物信息的获取函数
2019_11_22
'''
from bs4 import BeautifulSoup
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    '''
    输入参数:
        retX - 数据X
        retY - 数据Y
        inFile - HTML文件
        yr - 年份
        numPce - 乐高部件数据
        origPrc - 原价
    '''
    #打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    #根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r = "%d" % i)
    while(len(currentRow) != 0):
        currentRow = soup.find_all('table', r = "%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        #查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        #查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            #解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            #去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r = "%d" % i)

def setDataCollect(retX, retY):
    '''
    函数说明:
        依次读取六种乐高套装的数据，并生成数据矩阵
    '''
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)                #2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)                #2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)                #2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)                #2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)                #2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)                #2009年的乐高10196,部件数目3263,原价249.99

def useStandRegres():
    '''
    函数说明:
        使用简单的线性回归
    '''
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    data_num, features_num = np.shape(lgX)
    lgX1 = np.mat(np.ones((data_num, features_num + 1)))
    lgX1[:, 1:5] = np.mat(lgX)
    ws = standRegres(lgX1, lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (ws[0],ws[1],ws[2],ws[3],ws[4]))

'''8_6 交叉验证测试岭回归
2019_11_22
'''
import random
def crossValidation(xArr, yArr, numVal = 10):
    '''
    函数说明:
        交叉验证岭回归
    输入参数:
        xArr:x数据集
        yArr:y数据集
        numVal:交叉验证次数
    返回值:
        wMat:回归系数矩阵
    '''
    #统计样本个数
    m = len(yArr)
    #生成索引值列表
    indexList = list(range(m))
    #生成30个错误矩阵
    errorMat = np.zeros((numVal, 30))
    #交叉验证30次
    for i in range(numVal):
        #训练集
        trainX = []
        trainY = []
        #测试集
        testX = []
        testY = []
        #打乱次序
        random.shuffle(indexList)
        #划分数据集:90%训练集 10%的测试集
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        #获得30个不同lambda下的岭回归
        wMat = ridgeTest(trainX, trainY)
        #遍历所有岭回归系数
        for k in range(30):
            #测试集矩阵
            matTestX = np.mat(testX)
            #训练集矩阵
            matTrainX = np.mat(trainX)
            #训练集矩阵平均值
            meanTrain = np.mean(matTrainX, 0)
            #训练集矩阵方差
            varTrain = np.var(matTrainX, 0)
            #测试集标准化
            matTestX = (matTestX - meanTrain) / varTrain
            #根据ws预测y值
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)
            #统计误差
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))
    #计算每次交叉验证的平均误差
    meanErrors = np.mean(errorMat, 0)
    #找到最小误差
    minMean = float(min(meanErrors))
    #找到最佳回归系数
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    #数据经过标准化，所以需要还原
    unReg = bestWeights / varX
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % ((-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat)), unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]))   







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

8_3 岭回归
import regression
import numpy as np
abX, abY = regression.loadDataSet('abalone.txt')
ridgeWeights = regression.ridgeTest(abX, abY)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()

8_4 前向逐步线性回归
import regression
import numpy as np
xArr, yArr = regression.loadDataSet('abalone.txt')
regression.stageWise(xArr, yArr, 0.01, 200)
regression.stageWise(xArr, yArr, 0.001, 5000)
xMat = np.mat(xArr)
yMat = np.mat(yArr).T
xMat, yMat = regression.regularize(xMat, yMat)
yM = np.mean(yMat, 0)
yMat = yMat - yM
weights = regression.standRegres(xMat, yMat.T)
weights.T

8_5 乐高玩具
import regression
import numpy as np
lgX = []
lgY = []
regression.setDataCollect(lgX, lgY)
regression.useStandRegres()

8_6 交叉验证集测试岭回归
import regression
import numpy as np
lgX = []
lgY = []
regression.setDataCollect(lgX, lgY)
regression.crossValidation(lgX, lgY)
'''






