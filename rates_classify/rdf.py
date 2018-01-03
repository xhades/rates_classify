# !/usr/bin/env python
# -*-coding:utf-8-*-

"""
@author: xhades
@Date: 2017/12/28

"""

# 随机森林分类器

import numpy as np
from numpy import *
from numpy import array, argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.ensemble import RandomForestClassifier as RDF


np.set_printoptions(threshold=np.inf)


# 训练集测试集 3/7分割
def train(xFile, yFile):
    with open(xFile, "rb") as file_r:
        X = pickle.load(file_r)

    X = reshape(X, (212841, -1))  # reshape一下 （212841, 30*128）

    # 读取label数据，并且encodig
    with open(yFile, "r") as yFile_r:
        labelLines = [_.strip("\n") for _ in yFile_r.readlines()]
    values = array(labelLines)
    labelEncoder = LabelEncoder()
    integerEncoded = labelEncoder.fit_transform(values)
    integerEncoded = integerEncoded.reshape(len(integerEncoded), 1)
    # print(integerEncoded)

    # 获得label  编码
    Y = integerEncoded.reshape(212841, )
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # 随机森林分类器
    clf = RDF(criterion="gini")
    # criterion 可以使用"gini"或者"entropy"，前者代表基尼系数，后者代表信息增益。一般说使用默认的基尼系数"gini"就可以了，即CART算法。除非你更喜欢类似ID3, C4.5的最优特征选择方法。

    clf.fit(X_train, Y_train)

    # 测试数据
    predict = clf.predict(X_test)
    count = 0
    for p, t in zip(predict, Y_test):
        if p == t:
            count += 1
    print("RandomForest Accuracy is:", count/len(Y_test))


if __name__ == "__main__":
    xFile = "Res/char_embedded.pkl"
    yFile = "data/label.txt"
    print("Start Training.....")
    train(xFile, yFile)
    print("End.....")
