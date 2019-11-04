from numpy import *

'''4_1词表到向量的转换函数
2019_11_3
'''
#创建一些实验样本
def loadDataSet():
    #进行词条切分后的文档集合
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'], \
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], \
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], \
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'], \
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], \
                   ['quit', 'buying', 'wortless', 'dog', 'food', 'stupid']]
    #1代表侮辱性文字，0代表正常言论，类别标签的集合
    classVec = [0, 1, 0, 1, 0, 1] 
    return postingList, classVec
#创建一个包含在所有文档中出现的不重复词的列表
def creatVocabList(dataSet):
    #1创建一个空集
    vocabSet = set([])
    for document in dataSet:
        #2创建两个集合的并集,将每个词汇加入集合，重复的就不再加入
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
#表示词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList, inputSet):
    #3创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        #如果出现词汇表中的单词，则将输出的文档向量中的对应值设为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

'''4_2朴素贝叶斯分类器训练函数
2019_11_3
'''
#trainMatrix: 输入参数为文档矩阵
#trainCategory: 每篇文档类别标签所构成的向量
def trainNB0(trainMatrix, trainCategory):
    #求出整个矩阵中有多少个实例(6)
    numTrainDocs = len(trainMatrix)
    #求出的词列表中词的总数(33)
    numWords = len(trainMatrix[0])
    #计算侮辱言论的例子在总的例子中比重
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    #1初始化概率
    #数字0表示非侮辱性文档，数字1表示侮辱性文档
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        #如果该条实例对应是侮辱性标签
        if trainCategory[i] == 1:
            #2向量相加
            #增加侮辱性词条的数值,向量加,维度是33
            p1Num += trainMatrix[i]
            #同时增加总的词条的数值
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #3对每个元素做除法
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

'''4_3朴素贝叶斯分类函数
2019_11_3
'''

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = creatVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

'''4_4朴素贝叶斯词袋模型
2019_11_4
'''

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

'''4_5文件解析及完整的垃圾邮件测试函数
2019_11_4
'''
#接收一个大写字符串并将其解析为字符串列表
def textParse(bigString):
    import re
    #将字符串中的一些标点符号去除
    listOfTokens = re.split(r'\W*', bigString)
    #将所有字符串转换为小写，去掉少于两个字符的字符串，转化为字符列表输出
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

#贝叶斯垃圾邮件分类器进行自动化处理
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        #1导入并解析文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #构建词列表
    vocabList = creatVocabList(docList)
    #构建测试集[0, 1, 2, ..., 49]
    trainingSet = range(50)
    #构建训练集
    testSet = []
    for i in range(10):
        #2随机构建测试集，测试集中有10项
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        #从训练集中将测试集去掉,这里trainingSet需要变为列表类型
        del(list(trainingSet)[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        #对每一封邮件基于词汇表建立词向量
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #计算分类所需的概率
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        #3对测试集分类，将其变为词向量
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        #使用分类函数，得到的分类结果和对应标签不同，错误数加一
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))

'''4_6 RSS源分类器及高频词去除函数
2019_11_4
'''

#1计算出现频率
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
        sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = creatVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen)
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


if __name__ == '__main__':
    '''测试词表转换为向量函数'''
    listOPosts, listClasses = loadDataSet()
    myVocabList = creatVocabList(listOPosts)
    print("词汇列表")
    print(myVocabList)
    print("\n转换第一行的词为向量")
    print(setOfWords2Vec(myVocabList, listOPosts[0]))
    
    '''测试朴素贝叶斯分类器训练函数'''
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print("pAb")
    print(pAb)
    print("\np0V")
    print(p0V)
    print('\np1V')
    print(p1V)

    '''测试朴素贝叶斯分类函数'''
    print(testingNB())

    '''贝叶斯垃圾邮件分类测试'''
    print(spamTest())

    '''RSS源分类器及高频词去除函数测试'''
    #因为本书成书的时间距离现在比较久，所以给出的网址已经不可以使用了
    #现改用新的网址
    #源1 NASA Image of the Day：http://www.nasa.gov/rss/dyn/image_of_the_day.rss
    #源0 Yahoo Sports - NBA - Houston Rockets News：http://sports.yahoo.com/nba/teams/hou/rss.xml
    



