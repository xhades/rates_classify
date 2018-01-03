# !/usr/bin/env python
# -*-coding:utf-8-*-

"""
@author: xhades
@Date: 2017/12/28

"""


import numpy as np
from numpy import array, argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.ensemble import GradientBoostingClassifier as GBC

np.set_printoptions(threshold=np.inf)


# 训练集测试集 3/7分割
def train(xFile, yFile):
    with open(xFile, "rb") as file_r:
        X = pickle.load(file_r)

    # 读取label数据，并且 Encoding
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

    # 梯度提升分类器
    clf = GBC(loss="deviance", subsample=0.8, criterion="friedman_mse")

    clf.fit(X_train, Y_train)

    # 测试数据
    predict = clf.predict(X_test)
    count = 0
    for p, t in zip(predict, Y_test):
        if p == t:
            count += 1
    print("GradientBoosting  Accuracy is:", count/len(Y_test))


if __name__ == "__main__":
    xFile = "Res/char_embedded.pkl"
    yFile = "data/label.txt"
    print("Start Training.....")
    train(xFile, yFile)
    print("End.....")