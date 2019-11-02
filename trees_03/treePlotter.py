import matplotlib.pyplot as plt

'''3_5使用文本注解绘制树节点
2019_11_1
'''

#定义文本框和箭头格式
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    #绘图区由全局变量定义
    createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords = 'axes fraction', \
        xytext = centerPt, textcoords = 'axes fraction', va = "center", \
        ha = "center", bbox = nodeType, arrowprops = arrow_args)

def createPlot(inTree):
    #创建了一个新图形
    fig = plt.figure(1, facecolor = 'white')
    #清空了绘图区
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    #plotTree.totalW储存树的宽度
    plotTree.totalW = float(getNumLeafs(inTree))
    #plotTree.totalD储存树的高度
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

'''3_6获取叶节点的数目和树的层数
2019_11_2
'''

def getNumLeafs(myTree):
    numLeafs = 0
    #python2的老代码，舍去
    #firstStr = list(myTree.keys()[0]) 
    
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #判断子节点是不是一个字典类型
        if type(secondDict[key]).__name__ == 'dict':
            #如果是字典类型，则该节点也是一个判断节点，需要继续调用自身
            numLeafs += getNumLeafs(secondDict[key])
        else:
            #如果不是，说明走到了叶子节点，加入
            numLeafs += 1
    return numLeafs

#实现的原理和统计子节点数目的差不多，只不过是找到一个字典类型的数据以后，
#将树的高度加1，
def getTreeDepth(myTree):
    maxDepth = 0
    #python2的老代码，舍去
    #firstStr = myTree.keys()[0]
    
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

#预先创建树
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}, \
                    {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

'''3_7 plotTree函数
2019_11_2
'''

def plotMidText(cntrPt, parentPt, txtString):
    #在父子节点之间填充文本信息
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    #计算树的宽
    numLeafs = getNumLeafs(myTree)
    #计算树的高
    depth = getTreeDepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    #plotTree.totalW储存树的宽度
    #plotTree.xOff和plotTree.yOff追踪已经绘制的节点位置，以及放置下一个节点的恰当位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, \
                            plotTree.yOff)
    #绘出子节点具有的特征值，或者沿此向分支向下的数据实例必须具有的特征值
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    #plotTree.totalD储存树的高度
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

if __name__ == '__main__':
    '''测试绘图函数'''
    #createPlot()

    '''测试树的节点数和树的高度'''
    myTree = retrieveTree(0)
    print(getNumLeafs(myTree), getTreeDepth(myTree))
    myTree = retrieveTree(1)
    print(getNumLeafs(myTree), getTreeDepth(myTree))

    '''测试创建的数的图'''
    myTree = retrieveTree(0)
    createPlot(myTree)

    myTree['no surfacing'][3] = 'maybe'
    createPlot(myTree)