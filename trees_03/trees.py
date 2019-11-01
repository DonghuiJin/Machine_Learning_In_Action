from math import log
import operator

'''3_1计算给定数据集的香农熵
2019_11_1
'''

def calcShannonEnt(dataSet):
    #计算出实例的总数
    numEntries = len(dataSet)
    #创建字典
    labelCounts = {}
    for featVec in dataSet:
        #特征的最后一列是对应的标签
        currentLabel = featVec[-1]
        #如果该键值不存在与这个字典中，就将其加入到字典之中
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        #如果存在，对应的样例增加
        labelCounts[currentLabel] += 1
    #计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        #计算出选择该分类的概率
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

'''3_2 按照给定特征划分数据集
2019_11_1
'''

#dataSet:待划分的数据集
#axis:划分数据集的特征
#value:需要返回的特征的值
def splitDataSet(dataSet, axis, value):
    #为了不修改原始数据集，创建一个新的列表对象
    retDataSet = []
    #
    for featVec in dataSet:
        #如果该列表中一个样例的第axis个项对应的值是value
        if featVec[axis] == value:
            #那么将这个样例的前几项放入reduceFeatvec
            reducedFeatVec = featVec[:axis]
            #把除了第axis这一项的后几项也加入进来
            reducedFeatVec.extend(featVec[axis + 1:])
            #将这一项加入retData的项中
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''3_3 选择最好的数据集划分方式
2019_11_1
'''

def chooseBestFeatureToSplit(dataSet):
    #特征的数量
    numFeatures = len(dataSet[0]) - 1
    #计算原始香农熵，保存最初的无序度量值
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    #遍历数据中的所有特征
    for i in range(numFeatures):
        
        '''创建唯一的分类标签列表'''
        
        #将dataSet中的数据先按行放到依次放入example中，
        #然后取得example中这一个特征的一列元素，放入列表featList中，变成一行
        #具体效果https://www.jianshu.com/p/0e2308acd559
        featList = [example[i] for example in dataSet]
        #set(集合)是python语言原生的集合数据类型
        #集合是一个无序的不重复元素序列
        #所以uniqueVals得到的特征没有重复值，如果集合中只有0，1出现
        #那么uniqueVals中只有两个元素0和1
        uniqueVals = set(featList)
        newEntropy = 0.0
        '''计算每种划分方式的信息熵'''
        
        #遍历当前特征中的所有唯一属性值
        for value in uniqueVals:
            #返回指定特征对应的样例，也就是整个数据集中的子集
            subDataSet = splitDataSet(dataSet, i, value)
            #计算每一个特征值对应的香农熵，并对其求和
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        '''计算最好的信息增益'''

        #baseEntropy是原始香农熵
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''投票表决
2019_11_1
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        #如果标签没有出现在字典中，加入这个新标签
        if vote not in classCount.keys():
            classCount[vote] = 0
        #标签有的话就将次数增加
        classCount[vote] += 1
    #将字典中的标签出现次数进行排序，最大的排在最前面
    sortedClassCount =sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回出现次数最多的标签
    return sortedClassCount[0][0]

'''3_4 创建树的函数代码
2019_11_1
'''
#dataSet:数据集
#labels:标签列表
def createTree(dataSet, labels):
    #把dataSet的最后一列拿出来放到列表中
    classList = [example[-1] for example in dataSet]
    #如果所有的类标签完全相同，那么直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有特征后返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #选择最好的数据集划分方式，然后将该分类标签存入变量
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #创建字典类型储存树
    myTree = {bestFeatLabel:{}}
    #删除labels列表中bestFeat下标的元素
    del(labels[bestFeat])
    #把特征对应的值集合化
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


'''生成测试数据
2019_11_1
'''
def createDataSet():
    dataSet = [[1, 1, 'yes'], 
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


if __name__ == '__main__':
    
    '''测试香农熵'''
    myDat, labels = createDataSet()
    print(calcShannonEnt(myDat))

    myDat[0][-1] = 'maybe'
    print(calcShannonEnt(myDat))

    '''测试按照给定特征划分数据集函数'''
    myDat, labels = createDataSet()
    print(splitDataSet(myDat, 0, 1))
    print(splitDataSet(myDat, 0, 0))

    '''测试最好的数据集划分方式'''
    print(myDat)
    print(chooseBestFeatureToSplit(myDat))