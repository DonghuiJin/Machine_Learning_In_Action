import numpy as np
import boost
import math
def loadSimpData():
    datMat = np.matrix(
        [[ 1. , 2.1],
         [ 2. , 1.1],
         [ 1.3, 1. ],
         [ 1. , 1. ],
         [ 2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

'''7_2 基于单层决策树的AdaBoost训练过程
2019_11_16
'''

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    '''
    输入参数:
    dataArr:输入的数据集
    classLabels:类别标签
    numIt:迭代次数(需要自己指定)
    返回值:
    weakClassArr:单层决策树的数组
    '''
    #建立一个新的列表来保存单层决策树的数组
    weakClassArr = []
    #得到例子的个数
    m = np.shape(dataArr)[0]
    '''
    D包含了每个数据点的权重，开始都被赋予了相等的值
    在后续的迭代中，增加错分数据的权重同时降低正确分类数据的权重
    D的所有元素之和为1.0
    '''
    D = np.mat(np.ones((m, 1)) / m)
    #记录每个数据点的类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        '''
        返回值:
        bestStump:利用D而得到的具有最小错误率的单层决策树
        error:最小的错误率
        classEst:估计的类别向量
        '''
        bestStump, error, classEst = boost.buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        #书中用于调整权值的参数alpha，告诉总分类器本次单层决策树输出结果的权重
        alpha = float(0.5 * math.log((1.0 - error) / max(error, 1e-16)))
        #加入字典
        bestStump['alpha'] = alpha
        #加入列表
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        #1 为下一次迭代计算D
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        #2 错误率累加计算
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != \
                                np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")
        #如果错误率降低到0时，就停止迭代
        if errorRate == 0.0: break
    return weakClassArr

'''7_3 AdaBoost分类函数
2019_11_16
'''

def adaClassify(datToClass, classifierArray):
    '''
    输入参数:
    datToClass:一个或者多个待分类样例
    classifierArray:多个弱分类进行分类的函数
    '''
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    #记录每个数据点的类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    #遍历classifierArray中的所有弱分类器
    for i in range(len(classifierArray)):
        classEst = boost.stumpClassify(dataMatrix, classifierArray[i]['dim'], \
                                       classifierArray[i]['thresh'], \
                                       classifierArray[i]['ineq'])
        aggClassEst += classifierArray[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

'''7_4 自适应数据加载函数
2019_11_16
'''
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(float(curLine[-1]))
    return dataMat, labelMat

'''
import numpy as np
import adaboost
import boost
datMat, classLabels = adaboost.loadSimpData()
D = np.mat(np.ones((5, 1))/5) #创建的是平均加权
boost.buildStump(datMat, classLabels, D)
classifierArr = adaboost.adaBoostTrainDS(datMat, classLabels, 30)
adaboost.adaClassify([0, 0], classifierArr)
adaboost.adaClassify([[5, 5], [0, 0]], classifierArr)

7_4
datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray = adaboost.adaBoostTrainDS(datArr, labelArr, 10)
testArr, testLabelArr = adaboost.loadDataSet('gorseColicTest2.txt')
prediction10 = adaboost.adaClassify(testArr, classifierArray)
errArr = np.mat(np.ones((67, 1)))
errArr[prediction10 != np.mat(testLabelArr).T].sum()
'''



