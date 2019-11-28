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

'''11_3 关联规则生成函数
2019_11_27
'''

def generateRules(L, supportData, minConf=0.7):
    '''
    Function Description:
        生成关联规则
    Parameters:
        L:频繁项集列表
        supportData:包含那些频繁项集支持数据的字典
        minConf:最小可信度阈值
    Returns:
        bigRuleList:生成的规则列表
    Time:
        2019_11_27            
    '''        
    #储存所有的关联规则
    bigRuleList = []
    #只有获取两个或者更多集合的项目
    #两个以及以上才可能有关联一说，单个元素的项集不存在关联问题
    for i in range(1, len(L)):
        for freqSet in L[i]:
            #该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            H1 = [frozenset([item]) for item in freqSet]
            if(i > 1):
                #如果频繁项集元素数目超过2，那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                #第一层时，i为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    '''
    Function Description:
        如果频繁项目集的元素只有两个，使用该函数计算可信度值
    Parameters:
        freqSet:L中的某一个(i)频繁项集
        H:L中的某一个(i)频繁项集元素组成的列表
        supportData:包含那些频繁项集支持的数据的字典
        br1:关联规则
        minConf:最小可信度
    Returns:
        prunedH:返回满足最小可信度要求的项列表
    Time:
        2019_11_27            
    '''            
    #返回满足最小可信度要求的项列表
    prunedH = []
    #遍历L中的某一个(i)频繁项集的每个元素
    for conseq in H:
        #可信度计算，结合支持度数据
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            #如果某条规则满足最小可信度值，那么将这些规则输出到屏幕显示
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            #添加到规则里，br1是前面通过检查的bigRuleList
            br1.append((freqSet-conseq, conseq, conf))
            #通过检查的项进行保存
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    '''
    Function Description:
        如果频繁集的元素数目超过2，对其进行合并
    Parameters:
        freqSet:L中的某一个(i)频繁项集
        H:L中的某一个(i)频繁项集元素组成的列表
        supportData:包含那些频繁项集支持的数据的字典
        br1:关联规则
        minConf:最小可信度
    Returns:
        None
    Time:
        2019_11_27            
    '''        
    m = len(H[0])
    #频繁项集元素数目大于单个集合的元素数
    if(len(freqSet) > (m + 1)):
        #存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = aprioriGen(H, m+1)
        #计算可信度
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        #满足最小可信度要求的规则列表多于1，则递归来判断是否可以进一步组合这些规则
        if(len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


if __name__ == '__main__':
    '''
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet, 0.5)
    print(L)
    print(supportData)
    rules = generateRules(L, supportData, minConf=0.7)
    print(rules)
    '''
    
    mushuDataSet = [line.split() for line in open('mushroom.dat').readlines()]
    L, supportData = apriori(mushuDataSet, 0.3)
    for item in L[3]:
        if item.intersection('2'):
            print(item)


