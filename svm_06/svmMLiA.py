import random
import numpy as np

'''6_1SMO算法中的辅助函数
2019_11_9
'''

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        linArr = line.strip().split('\t')
        dataMat.append([float(linArr[0]), float(linArr[1])])
        labelMat.append(float(linArr[2]))
    return dataMat, labelMat

#i:第一个alpha的下标
#m:所有alphda的数目
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

#调整大于H或者小于L的alpha值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

'''6_2简化版SMO算法
2019_11_10
理解欠缺，还需回头继续看
2019_11_11
算法原型在《统计学习方法(第二版)》p143
'''

#dataMatIn: 数据集 100*2
#classLabels: 类别标签 100*1
#C: 常数C，惩罚系数
#toler: 容错率
#maxIter: 退出当前最大的循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    #100*2
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    #储存没有任何alpha改变的情况下遍历数据集的次数
    iter = 0
    while (iter < maxIter):
        #该变量用来记录alpha是否已经进行了优化
        alphaPairsChanged = 0
        for i in range(m):
            #计算出预测的类别
            #multiply:数组和矩阵对应位置相乘，输出与相乘数组/矩阵大小一致
            #*:对数组执行对应位置相乘，对矩阵执行矩阵乘法运算 
            # p145 7.104  
            fXi = float(np.multiply(alphas, labelMat).T*\
                (dataMatrix * dataMatrix[i, :].T)) + b
            #计算与标签之间的误差
            Ei = fXi - float(labelMat[i])
            #1 如果alpha可以更改进优化过程
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                ((labelMat[i] * Ei > toler) and \
                (alphas[i] > 0)):
                #2 随机选择第二个alpha
                j = selectJrand(i, m)
                #p145 公式7.104
                fXj = float(np.multiply(alphas, labelMat).T*\
                    (dataMatrix * dataMatrix[j, :].T)) + b
                #p145 公式7.105
                Ej = fXj - float(labelMat[j])
                #copy():https://www.runoob.com/python/att-dictionary-copy.html
                #dict2 = dict1          # 浅拷贝: 引用对象
                #dict3 = dict1.copy()   # 浅拷贝：深拷贝父对象（一级目录），子对象（二级目录）不拷贝，还是引用
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #3 保证alpha在0和C之间
                #p144最后一行，p145第一行 
                #分别是y1!=y2和y1=y2两种情况
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                #eta是alpha[j]的最优修改量
                #p145 7.107
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                #p145 7.106
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                #4 对i进行修改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * \
                    (alphaJold - alphas[j])
                #5 设置常数项
                #p148 7.115
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T
                #p148 7.116
                b2 = b - Ej -labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if  (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % \
                                    (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas
            
'''6_3完整版Platt SMO的支持函数
2019_11_11
'''

#建立一个数据结构来保存所有的重要值，使用对象
#数据可以作为一个对象进行传递
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        #1 误差缓存
        self.eCache = np.mat(np.zeros((self.m, 2)))

#能够计算E值，并且返回
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T *\
        (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk -float(oS.labelMat[k])
    return Ek

#2内循环的启发式方法，用于第二个alpha
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            #3选择具有最大步长的j
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

#计算误差值并存入缓存，对alpha值进行优化之后使用
def updataEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

'''6_4完整Platt SMO算法中的优化例程
2019_11_13
'''

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or\
        ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alpha)

            








if __name__ == '__main__':
    #6_1SMO算法中的辅助函数测试
    '''6_1SMO算法中的辅助函数'''
    #dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    #print(labelArr)
    #6_2简化版SMO算法测试
    '''6_2简化版SMO算法'''
    #b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    #alphas[alphas>0]
    #b
    #for i in range(100):
    #if alphas[i]>0.0:print(dataArr[i], labelArr[i])
