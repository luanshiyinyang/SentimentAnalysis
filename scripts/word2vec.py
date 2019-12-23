"""
Author: Zhou Chen
Date: 2019/12/22
Desc: 词向量模型的生成
"""
import os
from gensim.models.word2vec import Word2Vec
import codecs


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in codecs.open(os.path.join(self.dirname, fname), "r", encoding="utf8"):
                yield line.strip().split()


# 生成的word2vec模型的地址
model_path = "../model/word2vec.pkl"
sentences = MySentences('../data/vocab/')

# 此处min_count=5代表5元模型，size=100代表词向量维度，worker=15表示15个线程
model = Word2Vec(sentences, min_count=5, size=100, workers=15)
print(model.most_similar(['interest']))

# 保存模型
model.save(model_path)
