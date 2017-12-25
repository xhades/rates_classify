# !/usr/bin/env python
# -*-coding:utf-8-*-

"""
@author: xhades
@Date: 2017/12/24

"""

# 决策树分类器


import numpy as np
from numpy import array, argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.tree import DecisionTreeClassifier as DT


np.set_printoptions(threshold=np.inf)


# 训练集测试集 3/7分割
def train(xFile, yFile):
    with open(xFile, "rb") as file_r:
        X = pickle.load(file_r)

    # 读取label数据，并且One-Hot Encoding
    with open(yFile, "r") as yFile_r:
        labelLines = [_.strip("\n") for _ in yFile_r.readlines()]
    values = array(labelLines)
    labelEncoder = LabelEncoder()
    integerEncoded = labelEncoder.fit_transform(values)
    integerEncoded = integerEncoded.reshape(len(integerEncoded), 1)
    # print(integerEncoded)

    # 获得label one hot 编码
    Y = integerEncoded.reshape(212841, )
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # 决策树分类器
    clf = DT(criterion="entropy", splitter="random")
    # criterion 可以使用"gini"或者"entropy"，前者代表基尼系数，后者代表信息增益。一般说使用默认的基尼系数"gini"就可以了，即CART算法。除非你更喜欢类似ID3, C4.5的最优特征选择方法。
    # splitter 可以使用"best"或者"random"。前者在特征的所有划分点中找出最优的划分点。后者是随机的在部分划分点中找局部最优的划分点。默认的"best"适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐"random"

    clf.fit(X_train, Y_train)

    # 测试数据
    predict = clf.predict(X_test)
    count = 0
    for p, t in zip(predict, Y_test):
        if p == t:
            count += 1
    print("Decison tree Accuracy is:", count/len(Y_test))


if __name__ == "__main__":
    xFile = "Res/char_embedded.pkl"
    yFile = "data/label.txt"

    train(xFile, yFile)