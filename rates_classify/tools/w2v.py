#!/usr/bin/env python
# -*-coding:utf-8-*-
"""
@Time: 17-10-11 
@author: xhades
@version: v0.1
"""

from gensim.models import word2vec

sentence = word2vec.Text8Corpus('../data/splited_words.txt')
model = word2vec.Word2Vec(sentence, size=128, min_count=10, sg=1, window=12, workers=8)
model.wv.save_word2vec_format("../data/embedding.txt", binary=False, )
model.save("../Model/word2vec.model")

