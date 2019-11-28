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


'''
12_1测试代码
import fpGrowth
rootNode = fpGrowth.treeNode('pyramind', 9, None)
rootNode.children['eye']=fpGrowth.treeNode('eye', 13, None)
rootNode.disp()
'''
