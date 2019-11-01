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

def createPlot():
    #创建了一个新图形
    fig = plt.figure(1, facecolor = 'white')
    #清空了绘图区
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon = False)
    plotNode('decision code', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

if __name__ == '__main__':
    #测试绘图函数
    createPlot()


        