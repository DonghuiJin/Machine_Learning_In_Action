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
                   ['quit', 'buying', 'wortless', 'dog', 'fog', 'food', 'stupid']]
    #代表侮辱性文字，0代表正常言论，类别标签的集合
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
    
if __name__ == '__main__':
    '''测试词表转换为向量函数'''
    listOPosts, listClasses = loadDataSet()
    myVocabList = creatVocabList(listOPosts)
    print("词汇列表")
    print(myVocabList)
    print("\n转换第一行的词为向量")
    print(setOfWords2Vec(myVocabList, listOPosts[0]))
    
