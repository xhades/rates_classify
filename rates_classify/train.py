# !/usr/bin/env python
# -*-coding:utf-8-*-

"""
@author: xhades
@Date: 2017/12/19

"""
import pandas as pd
import numpy as np
from numpy import array, argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import BernoulliNB
import pickle
from sklearn import svm

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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.001, random_state=42)
    print(X_test[1].shape)
    # # 朴素贝叶斯训练数据
    # clf = BernoulliNB()
    # clf.fit(X_train, Y_train)
    #
    # # 测试数据
    # predict = clf.predict(X_test)
    # count = 0
    # for p, t in zip(predict, Y_test):
    #     if p == t:
    #         count += 1
    # print("Bayes Accuracy is:", count/len(Y_test))

    # SVM
    print("-->")
    # svmClf = svm.SVC(kernel="rbf")
    # svmClf.fit(X_train, Y_train)
    # svmpredict = svmClf.predict(X_test)
    # svmcount = 0
    # for p, t in zip(svmpredict, Y_test):
    #     if p == t:
    #         svmcount += 1
    # print("SVM Accuracy is:", svmcount / len(Y_test))

if __name__ == "__main__":
    xFile = "Res/char_embedded.pkl"
    yFile = "data/label.txt"

    train(xFile, yFile)