'''11_1 Apriori算法中的辅助函数
2019_11_26
'''

import numpy as np

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    '''
    Function Description:
        创建数据集中的所有单一元素组成的集合
    Parameters:
        dataSet:需要处理的数据集
    Returns:
        单一元素组成的集合
    Time:
        2019_11_26            
    '''
    C1 = []
    for transcation in dataSet:
        for item in transcation:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素
    return list(map(frozenset, C1))

def scanD(D, Ck, minSupport):
    '''
    Function Description:
        从C1生成L1
    Parameters:
        D:原始数据集
        Ck:上一步生成的单元素数据集
        minSupport:感兴趣项集的最小支持度
    Returns:
        retList:符合条件的元素
        supportData:符合条件的元素以及其支持率组成的字典
    Time:
        2019_11_26            
    '''
    ssCnt = {}
    #以下这一部分统计每一个元素出现的次数
    #遍历全体样本中的每一个元素
    for tid in D:
        #遍历单元素列表中的每一个元素
        for can in Ck:
            #s.issubest(X) 判断集合S是否是集合x的子集
            if can.issubset(tid):
                #算出数据集中所有单个元素出现的次数
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    #获取样本中的元素个数
    numItems = float(len(D))
    retList = []
    supportData = {}
    #遍历每一个元素
    for key in ssCnt:
        #计算每一个单元素的支持率
        support = ssCnt[key] / numItems
        #若支持率大于最小支持率
        if support >= minSupport:
            #insert()函数用于将指定对象插入列表的指定位置
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

'''11_2 Apriori算法
2019_11_26
'''

def aprioriGen(Lk, k):
    '''
    Function Description:
        组合向上合并
    Parameters:
        Lk:频繁项集的个数
        k:项集元素个数
    Returns:
        retList:符合条件的元素
    Time:
        2019_11_26            
    '''    
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        #两两组合遍历
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            #若两个组合的前k-2个项相同的时候，则将这两个集合合并
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    '''
    Function Description:
        apriori算法本身
    Parameters:
        dataSet:原始数据集
        minSupport:最小支持度
    Returns:
        L:符合条件的元素
        supportData:符合条件的元素及其支持率组成的字典
    Time:
        2019_11_26            
    '''    
    #创建数据集中所有单一元素组成的集合保存在C1中
    C1 = createC1(dataSet)
    #将数据集元素转为set集合然后将结果保存为列表
    D = list(map(set, dataSet))
    #从C1生成L1并返回符合条件的元素，符合条件的元素及其支持率组成的字典
    L1, supportData = scanD(D, C1, 0.5)
    #将符合条件的元素转换为列表保存在L中L会包含L1,L2,L3
    L = [L1]
    k = 2
    #L[m]就代表n+1元素集合,例如L[0]代表1个元素的集合
    #L[0]=[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        #dict.update(dict2)字典update()函数把字典dict的键值对更新到dict里
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


if __name__ == '__main__':
    suppData = loadDataSet()
    