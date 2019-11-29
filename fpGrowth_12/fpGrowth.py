'''12_1 FP树的类定义
2019_11_28
'''

#FP树的类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        #节点名称
        self.name = nameValue
        #节点出现次数
        self.count = numOccur
        #不同项集的相同项通过nodeLink连接在一起
        self.nodeLink = None
        #指向父节点
        self.parent = parentNode
        #储存叶子节点
        self.children = {}
    #节点出现次数累加
    def inc(self, numOccur):
        self.count += numOccur
    #将树以文本形式显示
    def disp(self, ind=1):
        print(' '*ind, self.name, ' ', self.count)
        #绘制子节点
        for child in self.children.values():
            #缩进处理
            child.disp(ind + 1)

'''12_2 Fp树构建函数
2019_11_28
'''


def createTree(dataSet, minSup=1):
    """
    函数说明：构建FP-tree
    Parameters:
        dataSet - 需要处理的数据集合
        minSup - 最少出现的次数（支持度）
    
    Returns:
        retTree - 树
        headerTable - 头指针表
    Modify:
        2019-11-28
    """
    headerTable = {}
    # 遍历数据表中的每一行数据
    for trans in dataSet:
        # 遍历每一行数据中的每一个数据元素
        # 统计每一个字母出现的次数，将次数保存在headerTable中
        for item in trans:
            # 字典get()函数返回指定键的值，如果值不在字典中返回0。
            # 由于dataSet里的每个列表均为frozenset所以每一个列表的值均为1即dataSet[trans]=1
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 遍历headerTable中的每一个字母
    # 若headerTable中的字母出现的次数小于minSup则把这个字母删除处理
    lessThanMinsup = list(filter(lambda k:headerTable[k] < minSup, headerTable.keys()))
    for k in lessThanMinsup: del(headerTable[k])
    for k in list(headerTable):
        if headerTable[k] < minSup:
            del(headerTable[k])
    # 将出现次数在minSup次以上的字母保存在freqItemSet中
    freqItemSet = set(headerTable.keys())
    # 如果没有达标的则返回None
    if len(freqItemSet) == 0:
        return None, None
    # 此时的headerTable中存放着出现次数在minSup以上的字母以及每个字母出现的次数
    # headerTable这个字典被称为头指针表
    for k in headerTable:
        # 保存计数值及指向每种类型第一个元素的指针
        headerTable[k] = [headerTable[k], None]
    # 初始化tree
    retTree = treeNode('Null Set', 1, None)
    # 遍历dataSet的每一组数据以及这组数据出现的次数
    for tranSet, count in dataSet.items():
        localD = {}
        # 遍历一组数据中的每一个字母
        for item in tranSet:
            # 如果这个字母出现在头指针表中
            if item in freqItemSet:
                # 将这个字母以及它在头指针表中出现的次数存储在localD中
                localD[item] = headerTable[item][0]
        # localD中存放的字母多于一个
        if len(localD) > 0:
            # 将字母按照出现的次数按降序排列
            ordereItems = [v[0] for v in sorted(localD.items(), key=lambda p:(p[1], p[0]), reverse=True)]
            # 对树进行更新
            updateTree(ordereItems, retTree, headerTable, count)
    # 返回树和头指针表
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    """
    函数说明：更新树
    Parameters:
        items - 将字母按照出现的次数按降序排列
        inTree - 树
        headerTable - 头指针表
        count - dataSet的每一组数据出现的次数，在本例中均为1
    
    Returns:
        None
    Modify:
        2019-11-28
    """
    # 首先查看是否存在该节点
    if items[0] in inTree.children:
        # 存在则计数增加
        inTree.children[items[0]].inc(count)
    # 不存在则新建该节点
    else:
        # 创建一个新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 若原来不存在该类别，更新头指针列表
        if headerTable[items[0]][1] == None:
            # 指向更新
            headerTable[items[0]][1] = inTree.children[items[0]]
        # 更新指向
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 仍有未分配完的树，迭代
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    """
    函数说明：更新树
    Parameters:
        nodeToTest - 需要插入的节点
        targetNode - 目标节点    
    Returns:
        None
    Modify:
        2019-11-28
    """
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


'''12_3 简单数据集以及数据包装器
2019_11_28
'''

def loadSimpDat():
    '''
    Function Description:
        创建数据集
    Parameters:
        None
    Returns:
        simDat；返回生成的数据集
    Time:
        2019_11_28            
    '''        
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def creatInitSet(dataSet):
    '''
    Function Description:
        将数据集数据项转换为frozenset并保存在字典中，其值均为1
    Parameters:
        dataSet:生成的数据集
    Returns:
        retDict:保存在字典中的数据集
    Time:
        2019_11_28            
    '''        
    retDict = {}
    for trans in dataSet:
        fset = frozenset(trans)
        retDict.setdefault(fset, 0)
        retDict[fset] += 1
    return retDict

'''12_4 发现以给定元素项结尾的所有路径的函数
2019_11_29
'''
def ascendTree(leafNode, prefixPath):
    '''
    Function Description：
        寻找当前非空节点的前缀
    Parameters:
        leafNode - 当前选定的节点
        prefixPath - 当前节点的前缀 
    Returns:
        None
    Time:
        2019_11_29
    '''
    #当前父节点不为空
    if leafNode.parent != None:
        #当前节点添加入前缀列表
        prefixPath.append(leafNode.name)
        #递归遍历所有前缀路线节点
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    '''
    Function Description：
        返回条件模式基
    Parameters:
        basePat:头指针列表中的元素
        treeNode:树中的节点
    Returns:
        condPats:返回条件模式基
    Time:
        2019_11_29
    '''  
    condPats = {}
    while treeNode != None:
        prefixPath = []
        #寻找当前非空节点的前缀
        ascendTree(treeNode, prefixPath)
        #如果遍历到前缀路线
        if len(prefixPath) > 1:
            #将前缀路线保存入字典
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        #到下一个频繁集出现的位置
        treeNode = treeNode.nodeLink
    #返回条件模式基
    return condPats

'''12_5 递归查找频繁项集的mineTree函数
2019_11_29
'''

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    '''
    Function Description：
        递归查找频繁项集
    Parameters:
        inTree:初始创建的FP树
        headerTable:头指针表
        minSup:最小支持度
        preFix:前缀
        freqItenList:条件树
    Returns:
        None
    Time:
        2019_11_29
    '''
    #从头指针列表的底端开始
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    for basePat in bigL:
        #加入频繁项表
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        #创造条件基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #从条件模式来构建条件树
        myContTree, myHead = createTree(condPattBases, minSup)
        #挖掘条件FP树，直到条件树中没有元素为止
        if myHead != None:
            print('conditional tree for: ', newFreqSet)
            myContTree.disp(1)
            mineTree(myContTree, myHead, minSup, newFreqSet, freqItemList)

'''
12_1测试代码
import fpGrowth
rootNode = fpGrowth.treeNode('pyramind', 9, None)
rootNode.children['eye']=fpGrowth.treeNode('eye', 13, None)
rootNode.disp()
'''

if __name__ == '__main__':
    simpDat = loadSimpDat()
    initSet = creatInitSet(simpDat)
    #print(initSet)
    #最小支持度为3
    myFPtree, myHeaderTab = createTree(initSet, 3)
    #myFPtree.disp()
    #print(findPrefixPath('r', myHeaderTab['r'][1]))
    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)

    #导入数据
    parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
    #数据初始化为字典
    initSet = creatInitSet(parsedDat)
    #创建FP树
    myFPtree, myHeaderTab = createTree(initSet, 100000)
    myFreqList = []
    #查找频繁项集，查找至少被10万人浏览过的新闻报道
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
    #打印频繁项集(哪些报道浏览量达到支持度)
    print(myFreqList)