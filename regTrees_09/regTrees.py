'''9_0 根据特征切分数据集合
2019_11_22
'''

import numpy as np

def binSplitDataSet(dataSet, feature, value):
    '''
    Function Description:
        根据特征切分数据集合
    Parameters:
        dataSet:数据集合
        feature:带切分的特征
        value:该特征的值
    Returns:
        mat0:切分的数据集合0
        mat1:切分的数据集合1
    Time:
        2019_11_22
    '''
    #np.nonzero:得到数组中非零元素的位置
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

'''9_1&2 CART算法的实现代码以及回归树切分函数
2019_11_22
'''

import matplotlib.pyplot as plt

def loadDataSet(fileName):
    '''
    Function Description:
        load dataset
    Parameters:
        fileName
    Returns:
        dataMat
    Time:
        2019_11_22
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #转化为float类型
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def plotDataSet(filename):
    '''
    Function Description:
        draw a picture for dataset
    Parameters:
        filename
    Returns:
        None
    Time:
        2019_11_22
    '''
    dataMat = loadDataSet(filename)
    n = len(dataMat)
    xcord = []
    ycord = []
    for i in range(n):
        xcord.append(dataMat[i][0])
        ycord.append(dataMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s = 20, c = 'blue', alpha = .5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def regLeaf(dataSet):
    '''
    Function Description:
        generate leaf node
    Parameters:
        dataSet
    Returns:
        the average of goal
    Time:
        2019_11_22
    '''
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    '''
    Function Description:
        evalute the loss function
    Parameters:
        dataSet
    Returns:
        total variance of goal
    Time:
        2019_11_22
    '''
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType, errType = regErr, ops = (1, 4)):
    '''
    Function Description:
        找到数据的最佳二元切分方式函数
    Parameters:
        dataSet
        leafTypr: generate leaf node
        regErr: 误差估计函数
        ops: 用户定义的参数构成的元组
    Returns:
        bestIndex: 最佳的切分特征
        bestValue: 最佳的特征值
    Time:
        2019_11_22
    '''
    import types
    #tolS允许的误差下降值, tolN切分的最少样本数
    tolS = ops[0]
    tolN = ops[1]
    #如果当前所有值相等，则退出(根据set的特性)
    #tolist():将数组或者矩阵转换成列表
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    #统计数据集合的行m和列n
    m, n = np.shape(dataSet)
    #默认最后一个特征为最佳切分特征，计算其误差估计
    S = errType(dataSet)
    #分别为最佳误差，最佳特征切分的索引值，最佳特征值
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    #遍历所有特征列
    for featIndex in range(n - 1):
        #遍历所有的特征值
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            #根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            #如果数据少于tolN，则退出
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            #计算误差估计
            newS = errType(mat0) + errType(mat1)
            #如果误差估计更小，则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    #根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #如果切分出的数据集很小则退出
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    #返回最佳切分特征和特征值
    return bestIndex, bestValue

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    '''
    Function Description:
        树构建函数
    Parameters:
        dataSet
        leafType: 建立叶结点的函数
        errType: 误差计算函数
        ops: 包含书构建所有其他参数的元组
    Returns:
        retTree: 构建的回归树
    Time:
        2019_11_22
    '''
    #选择最佳切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    #r如果没有特征，则返回特征值
    if feat == None:
        return val
    #回归树
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    #创建左子树和右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

'''9_3 树剪枝(后剪枝)
2019_11_23
'''

def isTree(obj):
    '''
    Function Description:
        判断测试输入变量是否是一颗树
    Parameters:
        obj:测试对象
    Time:
        2019_11_23
    '''
    import types
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    '''
    Function Description:
        对树进行塌陷处理(即返回树平均值)
    Parameters:
        tree:树
    Returns:
        树的平均值
    Time:
        2019_11_23
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['tree'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    '''
    Function Description:
        后剪枝
    Parameters:
        tree: 树
        testData: 测试集
    Returns:
        树的平均值
    Time:
        2019_11_23
    '''
    #如果测试集为空，则对树进行塌陷处理
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    #如果有左子树或者右子树，则切分数据集
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    #处理左子树(剪枝)
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    #处理右子树(剪枝)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    #如果当前结点的左右结点为叶结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #计算没有合并的误差
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        #计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        #计算合并的误差
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        #如果合并的误差小于没有合并的误差，则合并
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
    '''
    testMat = np.mat(np.eye(4))
    #切分的特征是第二列的特征
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print('original set:\n', testMat)
    print('mat0:\n', mat0)
    print('mat1:\n', mat1)

    
    filename = 'ex00.txt'
    plotDataSet(filename)    

    myDat = loadDataSet('ex00.txt')
    myDat = np.mat(myDat)
    print(createTree(myDat))

    filename = 'ex0.txt'
    plotDataSet(filename)

    myDat = loadDataSet(filename)
    myMat = np.mat(myDat)
    print(createTree(myMat))

    filename = 'ex2.txt'
    plotDataSet(filename)
    myDat = loadDataSet(filename)
    myMat = np.mat(myDat)
    print(createTree(myMat, ops=(10000, 4)))
    '''
    print("剪枝前")
    train_filename = 'ex2.txt'
    train_Data = loadDataSet(train_filename)
    train_Mat = np.mat(train_Data)
    tree = createTree(train_Mat)
    print(tree)
    print("剪枝后")
    test_filename = 'ex2test.txt'
    test_Data = loadDataSet(test_filename)
    test_Mat = np.mat(test_Data)
    print(prune(tree, test_Mat))





