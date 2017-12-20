#!/usr/bin/env python
# -*-coding:utf-8-*-
"""
@Time: 17-12-15 
@author: xhades
@version: v0.1
"""
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import random


def cleanTrianSet(filepath):
    """
    清洗句子中空行、空格
    目前采用将所有数据读取到内存，后续思考其他高效方式
    """
    # 删除评论上面的 \n
    fileDf = pd.read_csv(filepath, keep_default_na=False)
    fileDf["rate"] = fileDf["rate"].apply(lambda x: x.replace("\n", ""))
    linelist = fileDf.values.tolist()
    filelines = [ _[0] + "," + _[-1] for _ in linelist]
    cleaned_lines = map(lambda x: x.translate({ord('\u3000'): '', ord('\r'): '', ord('\xa0'): None,
                                                    ord(' '): None}), filelines[1:])  # 更加优雅的方式 在这个问题中是比较快的方式
    # print(list(cleaned_lines))
    return cleaned_lines  # 返回一个map对象


def embedding_lookup(voc, embDict):
    embedding = embDict.get(voc, [random.uniform(-0.5, 0.5) for i in range(128)])
    return embedding

if __name__ == "__main__":
    filepath = "../data/trainset.csv"
    # cleaned_lines = cleanTrianSet(filepath)

    # 词向量形式转变成字典
    with open("../data/embedding.txt") as embFile:
        embLines = embFile.readlines()
    embDict = {_.strip("\n").split(" ")[0]: _.strip("\n").split(" ")[1: ] for _ in embLines[1: ]}
    vocs = "孩子"
    embedding = embedding_lookup(vocs, embDict)
    print(embedding)
