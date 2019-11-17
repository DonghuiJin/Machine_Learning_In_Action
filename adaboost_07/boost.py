import numpy as np
from numpy import inf
'''7_1 单层决策树生成函数
2019_11_16
'''

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    #通过阈值分比较对数据进行分类的
    '''
    输入参数:
    dataMatrix:数据矩阵
    dimen:第几个特征
    threshVal:阈值
    threshIneq:标志
    返回值:
    retArray:分类结果
    '''
    #首先将数组中的所有元素均设置为1
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    #用于在特征的所有可能值上进行遍历
    numSteps = 10.0
    #创建字典，用于储存给定权重向量D时所得到的最佳单层决策树的相关信息
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    #初始化为无穷大, 之后用于寻找可能的最小错误率
    minError = float('inf')
    #第一层循环，对数据集中的每一个特征
    for i in range(n):
        #计算最小值和最大值来了解需要多大的步长
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        #第二层循环，对每个步长进行遍历
        for j in range(-1, int(numSteps) + 1):
            #第三层循环，对于每一个不等号
            for inequal in ['lt', 'gt']:
                #在大于和小于之间切换不等式
                threshVal = (rangeMin + float(j) * stepSize)
                #返回预测的分类结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                #1 计算加权错误率 
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh inequal: \
                    %s, the weighted error is %.3f" % \
                    (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


'''
import numpy as np
import adaboost
datMat, classLabels = adaboost.loadSimpData()
import boost
D = np.mat(np.ones((5, 1))/5) #创建的是平均加权
boost.buildStump(datMat, classLabels, D)
'''
