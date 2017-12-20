#!/usr/bin/env python
# -*-coding:utf-8-*-
"""
@Time: 17-12-15
@author: xhades
@version: v0.1
"""
import jieba
import pandas as pd
from utils import cleanTrianSet

jieba.load_userdict("../data/user_dict.txt")


# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def build_splitwords(csvfile):
    rates = list(cleanTrianSet(csvfile))
    stopwords = stopwordslist('../data/stop_words.txt')  # 这里加载停用词的路径

    for rate in rates:
        # 分割平论（好评 差评）
        # print(rate)
        rate = rate.split(",")
        content = "".join(rate[: -1])
        if len(content) <1 :
            continue

        words = jieba.cut(content)
        outstr = ''
        for word in words:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        # 写入文件
        with open("../data/splited_words.txt", "a+") as f:
            f.write(outstr + '\n')
        # 将label写入文件
        with open("../data/label.txt", "a+") as f:
            f.write("".join(rate[-1]) + '\n')
    print("Split words Done!!")
    print("Write label Done!!")


def split_sentence(sentence_file):
    sentences = pd.read_csv(sentence_file, sep='\n').get_values()
    print(sentences)
    for sen in sentences:
        # print(rate)
        text = str(sen.tolist())
        words = jieba.cut(text)
        stopwords = stopwordslist('../data/stop_words.txt')  # 这里加载停用词的路径

        outstr = ''
        for word in words:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        print(outstr)

if __name__ == "__main__":
    jieba.load_userdict("../data/user_dict.txt")
    csvfile = "../data/trainset.csv"
    build_splitwords(csvfile)
    # split_sentence("../data/sentences.txt")



