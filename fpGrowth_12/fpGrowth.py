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
    '''
    Function Description:
        构建FP-tree
    Parameters:
        datSet:需要处理的数据集合
        minSup:最少出现的次数(支持度)
    Returns:
        retTree:树
        headrTabel:头指针表
    Time:
        2019_11_28            
    '''        
    headerTable = {}
    #遍历数据表中的每一行数据
    for trans in dataSet:
        #遍历每一行数据中出现的每一个数据元素
        #统计每一个字母出现的次数，将次数保存在headerTable中
        for item in trans:
            #字典get()函数返回指定键的值，如果值不在字典中返回0
            #由于dataSet里的每个列表均为frozenset，所以每一个列表的值均为1即dataSet[trans]=1
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    #遍历headerTable中的每一个字母
    #若headerTable中的字母出现的次数小于minSup，则把这个字母删除处理
    lessThanMinsup = list(filter(lambda k:headerTable[k] < minSup, headerTable.keys()))
    for k in lessThanMinsup:
        del(headerTable[k])
    for k in list(headerTable):
        if headerTable[k] < minSup:
            del(headerTable[k])
    #将出现次数在minSup次以上的字母保存在freqItemSet中
    freqItemSet = set(headerTable.keys())
    #如果没有达标的则返回None
    if len(freqItemSet) == 0:
        return None, None
    #此时的headerTable中存放着出现次数在minSup以上的字母以及每个字母出现的次数
    #headerTable这个字典被称为头指针表
    for k in headerTable:
        #保存计数值以及指向每种类型第一个元素的指针
        headerTable[k] = [headerTable[k], None]
    #初始化tree
    retTree = treeNode('Null Set', 1, None)
    #遍历dataSet的每一组数据以及这组数据出现的次数
    for tranSet, count in dataSet.items():
        localD = {}
        #遍历一组数据中的每一个字母
        for item in tranSet:
            #如果这个字母出现在头指针列表中
            for item in freqItemSet:
                #将这个字母以及它在头指针表中出现的次数储存在localD中
                localD[item] = headerTable[item][0]
        #localD中存放的字母多于一个
        if len(localD) > 0:
            #将字母按照出现的次数降序排列
            ordereItems = [v[0] for v in sorted(localD.items(), key=lambda p:(p[1], p[0]),reverse=True)]
            #对树进行更新
            updateTree(ordereItems, retTree, headerTable, count)
    #返回树和头指针列表
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    '''
    Function Description:
        更新树
    Parameters:
        items:将字母按照出现的次数降序排列
        inTree:树
        headerTable:头指针列表
        count:dataSet的每一组数据出现的次数，在本例中均为1
    Returns:
        None
    Time:
        2019_11_28            
    '''        
    #首先查看是否存在该节点
    if items[0] in inTree.children:
        #存在则计数增加
        inTree.children[items[0]].inc(count)
    #不存在则新建该节点
    else:
        #创建一个新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        #若原来不存在该类别，更新头指针列表
        if headerTable[items[0]][1] == None:
            #指向更新
            headerTable[items[0]][1] = inTree.children[items[0]]
        #更新指向
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    #仍有未分配完的树，迭代
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    '''
    Function Description:
        更新头指针列表
    Parameters:
        nodeToTest:需要插入的节点
        targetNode:目标节点
    Returns:
        None
    Time:
        2019_11_28            
    '''        
    while (nodeToTest.nodeLink != None):
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
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
    
