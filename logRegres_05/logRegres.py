import numpy as np
import random

'''Logistic回归梯度上升优化算法
2019_11_7
'''

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    #一次性读取整个文件，自动将文件内容分析成一个行的列表
    for line in fr.readlines():
        lineArr = line.strip().split()
        #X0设置为1.0, a0+a1x1+a2x2,第一项x0的值就是1
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        #最后一列就是对应的标签
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

#sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

#计算梯度上升
def gradAscent(dataMatIn, classLabels):
    #1转换为numpy类型的矩阵
    #100*3的向量
    dataMatrix = np.mat(dataMatIn)
    #100*1的向量
    labelMat = np.mat(classLabels).transpose()
    #m=100, n=3
    m, n = np.shape(dataMatrix)
    #目标移动的步长
    alpha = 0.001
    #迭代次数
    maxCycles = 500
    #权重为n*1的向量
    weights = np.ones((n, 1))

    for k in range(maxCycles):
        #2矩阵相乘    100*3        3*1     =    100*1
        h = sigmoid(dataMatrix * weights)
        #100*1
        error = labelMat - h
          #3*1      3*1                    3*100            100*1
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

'''5_2画出数据集和Logistic回归最佳拟合直线的函数
2019_11_7
'''

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    #getA():将numpy矩阵转化为数组
    #这里对代码进行了改进，第一个梯度上升函数输出的权重的格式是numpy矩阵
    #第二个优化函数输出的权重格式是numpy数组，这样就不需要使用getA()函数进行转化
    if type(wei).__name__ == 'matrix':
        weights = wei.getA()
    else:
        weights = wei            
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    #在图中显示x属于(-3, 3)
    x =  np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

'''5_3随机梯度上升算法
2019_11_7
'''
#全是numpy数组的计算，没有numpy矩阵
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

'''5_4改进的随机梯度上升算法
2019_11_7
'''

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        dataIndex = list(dataIndex)
        for i in range(m):
            #1 alpha每次迭代时需要调整
            #alpha每次迭代时都会调整，会缓解数据波动
            #存在常数项0.01，所以alpha不会减小到0
            alpha = 4 / (1.0 + j + i) + 0.01
            #2 随机选取更新
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error *dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

'''5_5 Logistic回归分类函数
2019_11_9
'''

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    #读取训练集文件
    frTrain = open('horseColicTraining.txt')
    #读取测试集文件
    frTest = open('horseColicTest.txt')
    #建立训练集列表
    trainingSet = []
    #建立训练集标签列表
    trainingLabels = []
    #处理训练集文件
    for line in frTrain.readlines():
        #strip():移除字符串首尾的空格和制表符
        #split():返回一个列表，'\t'是分开字符串的标志，分为一个个列表元素
        currLine = line.strip().split('\t')
        lineArr = []
        #数据的前21个都是对应的特征
        for i in range(21):
            lineArr.append(float(currLine[i]))
        #将这一行的特征加入到训练集中的一行
        trainingSet.append(lineArr)
        #将这一行对应的标签加入训练标签列表
        trainingLabels.append(float(currLine[21]))
    #求出训练权重
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    #一样的方式对测试集进行处理
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        #如果分类结果和对应的标签不一致，错误数目加一
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    #调用函数colicTest()10次，并求出结果的平均值
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    #Logistic回归梯度上升优化算法测试
    dataArr, labelMat = loadDataSet()
    print("Logistic回归梯度上升优化算法")
    weights = gradAscent(dataArr, labelMat)
    print(weights)    
    #5_2画出数据集和Logistic回归最佳拟合直线的函数测试
    print("5_2画出数据集和Logistic回归最佳拟合直线的函数")
    plotBestFit(weights)
    #5_3随机梯度上升算法测试
    print("5_3随机梯度上升算法")
    data1 = np.array(dataArr)
    weights = stocGradAscent0(data1, labelMat)
    plotBestFit(weights)
    #5_4改进的随机梯度上升算法测试
    print("5_4改进的随机梯度上升算法")
    weights = stocGradAscent1(data1, labelMat)
    plotBestFit(weights)
    #5_5Logistic回归分类函数
    print("5_5Logistic回归分类函数测试")
    multiTest()

