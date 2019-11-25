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
        判断测试输入变量是否是一颗树，判断当前处理的结点是否是叶结点
    Parameters:
        obj:测试对象
    Time:
        2019_11_23
    '''
    import types
    #判断返回类型是不是字典类型
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
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    '''
    Function Description:
        后剪枝
    Parameters:
        tree: 待剪枝的树
        testData: 剪枝所需的测试数据
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
    #判断处理完后的左子树是否还是子树，如果还是子树，递归
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    #同上一个的判断，针对右子树
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

'''9_4 模型树的叶结点生成函数
2019_11_24
'''

def linearSolve(dataSet):
    '''
    Function Description:
        数据集格式化为目标变量Y和自变量X
    Parameters:
        dataSet: 数据集
    Returns:
        计算线性回归之后的权重以及X和Y
    Time:
        2019_11_24
    '''
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    #判断矩阵的逆是不是存在
    if np.linalg.det(xTx) == 0:
        raise NameError('This matrix is singular, cannot do inverse, \n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    '''
    Function Description:
        调用linearSolve函数
    Parameters:
        dataSet: 数据集
    Returns:
        返回回归系数ws
    Time:
        2019_11_24
    '''
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    '''
    Function Description:
        可以在给定的数据集计算误差
    Parameters:
        dataSet: 数据集
    Returns:
        计算yHat和Y之间的平方误差
    Time:
        2019_11_24
    '''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))

'''9_5 用树回归进行预测的代码
2019_11_24
'''

def regTreeEval(model, inDat):
    '''
    Function Description:
        可以在给定的数据集计算误差
    Parameters:
        model: 已经建立好的决策树模型
        inDat: 输入数据
    Returns:
        返回树模型
    Time:
        2019_11_24
    '''
    return float(model)

def modelTreeEval(model, inDat):
    '''
    Function Description:
        对输入数据进行格式化处理
    Parameters:
        model: 已经建立好的决策树模型
        inDat: 输入的数据集
    Returns:
        返回预测值
    Time:
        2019_11_24
    '''
    n = np.shape(inDat)[1]
    #在第一列前加入一列1
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)

def treeForeCast(tree, inData, modelEval = regTreeEval):
    '''
    Function Description:
        对输入数据进行格式化处理
    Parameters:
        tree: 已经建立好的决策树模型
        inData: 输入的数据
        modelEval: 
    Returns:
        
    Time:
        2019_11_24
    '''
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    '''
    Function Description:
        多次调用createForCast()，以向量形式返回一组预测值
    Parameters:
        tree: 已经建立好的决策树模型
        testData: 测试数据集
        modelEval: 
    Returns:
        以向量形式返回一组预测值
    Time:
        2019_11_24
    '''
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat






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
    '''

    '''
    myMat2_filename = 'exp2.txt'
    myMat2 = np.mat(loadDataSet(myMat2_filename))
    print(createTree(myMat2, modelLeaf, modelErr, ops=(1, 10)))
    '''

    trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, ops = (1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    print(np.corrcoef(yHat, testMat[:, 1], rowvar = 0)[0, 1])
    
    myTree = createTree(trainMat, modelLeaf, modelErr, (1, 20))
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    print(np.corrcoef(yHat, testMat[:, 1], rowvar = 0)[0, 1])

    ws, X, Y = linearSolve(trainMat)
    print(ws)

    for i in range(np.shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]