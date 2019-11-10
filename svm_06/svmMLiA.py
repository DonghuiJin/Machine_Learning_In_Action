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
'''

#dataMatIn: 数据集 100*2
#classLabels: 类别标签 100*1
#C: 常数C
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
                fXj = float(np.multiply(alphas, labelMat).T*\
                    (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #3 保证alpha在0和C之间
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
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                #4 对i进行修改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * \
                    (alphaJold - alphas[j])
                #5 设置常数项
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T
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
