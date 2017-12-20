# !/usr/bin/env python
# -*-coding:utf-8-*-

"""
@author: xhades
@Date: 2017/12/19

"""
import codecs
import numpy as np
import pickle
from tools.utils import embedding_lookup

np.set_printoptions(threshold=np.inf)


# 将训练文本数据转换成embedding词矩阵
def build_embedding():
    # 词向量形式转变成字典
    with open("data/embedding.txt") as embFile:
        embLines = embFile.readlines()
    embDict = {_.strip("\n").split(" ")[0]: _.strip("\n").split(" ")[1:] for _ in embLines[1:]}

    # 加载splited  word文件
    fileData = codecs.open("data/splited_words.txt", "r", encoding="utf-8")

    # embedding文件
    embeddingMtx = np.zeros((212841, 128), dtype='float32')
    count = 0
    fileLine = fileData.readline()

    while fileLine:
        fileLine = fileLine.strip()
        if fileLine :
            words = fileLine.split(" ")
            # 对应词向量列表
            wordsEmbed = map(lambda word: embedding_lookup(word, embDict), words)
            # 列表转成矩阵, 序列化写入文件
            wordEmbeddingMtx = np.matrix(list(wordsEmbed))
            embeddingMtx[count] = wordEmbeddingMtx[0]

            fileLine = fileData.readline()
            count += 1
            continue

        fileLine = fileData.readline()
    fileData.close()
    print("End.....")
    # print(embeddingMtx)
    with open("Res/char_embedded.pkl", "wb") as file_w:
        pickle.dump(embeddingMtx, file_w)


if __name__ == "__main__":
    build_embedding()