from numpy import *
import operator
import pylint
import matplotlib
import matplotlib.pyplot as plt
#可以列出给定目录的文件名
from os import listdir

#初始数据以及标签分类
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''k-nn 算法实现
2019.10.24
'''
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

'''将文本记录到转换Numpy的解析程序
2019.10.25
'''
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

'''归一化特征值 
2019.10.26
'''
def autoNorm(dataSet):
    #求出最大，最小值，参数0可以在列中选取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #求出分母
    ranges = maxVals - minVals
    #创建和数据一样大的零矩阵
    normDataSet = zeros(shape(dataSet))
    #求出数据的个数
    m = dataSet.shape[0]
    #求出分子
    #minVals和value都是1*3的矩阵，通过tile方法可以将其变为1000*3矩阵，和dataSet矩阵的规模一样
    normDataSet = dataSet - tile(minVals, (m, 1))
    #计算最后的式子
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

'''分类器针对约会网站的测试代码
2019.10.26
'''
def datingClassTest():
    #用于测试集的比率
    hoRatio = 0.10
    #读取数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    #将数据进行归一化处理
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #计算出用于测试集的样例数量numTestVecs
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    #计算错误率
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) : errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

'''约会网站测试函数
2019.10.26
'''
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent fliter miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minvals =autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minvals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])    

'''将图像转化为向量
2019_10_28
'''
def img2vector(filename):
    #创建图像转化为向量的矩阵
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        #读取一行32个数字
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


'''2_6手写数字识别系统的测试代码
2019_10_28
'''
def handwritingClassTest():
    hwLabels = []
    #listdir函数会获取目录的内容
    trainingFileList = listdir('trainingDigits')
    #计算目录中有多少个文件
    m = len(trainingFileList)
    #创建训练数据
    trainingMat = zeros((m, 1024))
    for i in range(m):

        '''从文件名解析数字'''

        #提取文件的名字
        fileNameStr = trainingFileList[i]
        #使用'.'对文件名进行切片,取第一部分
        fileStr = fileNameStr.split('.')[0]
        #使用'_'对第一部分切片，再取其中的第一部分，得到分类的数字，然后转化为int类型
        classNumStr = int(fileStr.split('_')[0])
        #将该数字加入标签列表中
        hwLabels.append(classNumStr)

        #使用img2vector函数载入图像
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    
    #获取测试数据的文件名
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):

        '''从文件名解析数字'''

        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        #将测试图片载入
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #将测试集，训练集放入分类器中
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    '''初始测试'''
    group, labels = createDataSet()
    print(classify0([0, 0], group, labels, 3))

    '''改进约会网站的配对效果测试'''
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print(datingDataMat, datingLabels)

    '''使用Matplotlib创建散点图'''
    #创建一个图像
    fig = plt.figure()
    #subplot(nrows, ncols, index)
    #创建nrows行， ncols列的网格，可以摆放nrows*ncols张子图，index为子图标号
    #add_subplot(111) = add_subplot(1, 1, 1)
    ax = fig.add_subplot(111)
    #关于scatter函数的相关信息
    #https://blog.csdn.net/AnneQiQi/article/details/64125186
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()

    '''测试归一化数值函数'''
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print(normMat, '\n', ranges, '\n', minVals)

    '''测试分类器错误率'''
    print(datingClassTest())

    '''约会网站测试函数的测试'''
    print(classifyPerson())
    
    '''测试手写数字识别'''
    print(handwritingClassTest())


            