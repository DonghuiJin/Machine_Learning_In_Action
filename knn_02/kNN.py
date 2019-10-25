from numpy import *
import operator
import pylint

#初始数据以及标签分类
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''k-nn 算法实现'''
#inX : 用于分类的输入向量
#dataSet : 输入的样本训练集
#labels : 标签向量
#k : 选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    
    '''距离计算'''

    #求出的是输入样本训练集的行数，也就是样本的总数
    dataSetSize = dataSet.shape[0]
    

    #def tile(A, reps) Construct an array by repeating A the number of times given by reps.
    #把用于分类的输入向量按照(dataSetSize, 1)的形式进行复制
    #这一步就是把用于分类的输入向量减去每一个输入的样本，得到差的矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    
    #对矩阵中的每个差值求平方
    sqDiffMat = diffMat ** 2

    #(axis = 0)是普通相加 
    #(axis = 1)是将一个矩阵的每一行向量相加
    sqDistances = sqDiffMat.sum(axis = 1)

    #对求和后的值开根号，得到距离的矩阵 (dataSetSize, 1)为distance矩阵的尺寸大小
    distances = sqDistances ** 0.5

    '''选择距离最小的k个点'''

    #argsort函数返回的是数组值从小到大的索引值
    #这一步为了之后选取前k个元素做准备
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        #确定前k个距离最小元素所在的主要分类,这里返回的应该是大写字母A或者B
        voteIlabel = labels[sortedDistIndicies[i]]
        #计算前k个距离最小元素所在类别计数，这里就是A或者B的数目
        '''
        classCount字典的样子
        A:n(n个A)
        B:m(m个B)
        '''
        #get到的是对应类别累积的数目
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    '''排序'''
    
    #排序，最后的顺序是从大到小
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回的是对应的类别
    return sortedClassCount[0][0]

'''将文本记录到转换Numpy的解析程序'''
def file2matrix(filename):
    
    '''得到文件的行数'''

    #打开文件
    fr = open(filename)
    #readlines() 一次性读取整个文件，自动将文件内容分析成一个行的列表
    arrayOLines = fr.readlines()
    #列表的长度也就是文件的行数
    numberOfLines = len(arrayOLines)

    '''创建返回Numpy矩阵以及分类的列表'''

    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0

    '''解析文件数据到矩阵和列表'''

    for line in arrayOLines:
        #strip()方法用于移除字符串头尾指定的字符(默认为空格或换行符)或字符序列
        line = line.strip()
        #split()方法通过指定分隔符对字符串进行切片,如果参数 num 有指定值，则分隔 num+1 个子字符串
        #指定制表符对这一条字符串进行切割
        listFromLine = line.split('\t')
        #将这一行的三种特征这三个元素分别存入到矩阵
        returnMat[index, :] = listFromLine[0:3]
        #把最后的分类添加到标签列表中,这里需要将列表中储存的值的类型设置为整型
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    #返回三种特征的矩阵版本以及对应分类的列表
    return returnMat, classLabelVector

if __name__ == '__main__':
    #初始测试
    group, labels = createDataSet()
    print(classify0([0, 0], group, labels, 3))
    #改进约会网站的配对效果测试
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print(datingDataMat, datingLabels)

            