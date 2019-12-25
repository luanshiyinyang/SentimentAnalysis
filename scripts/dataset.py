"""
Author: Zhou Chen
Date: 2019/12/22
Desc: 数据集处理模块
1.文本数据统一转化为utf8格式编码
2.删除超链接邮箱账号等无用信息
3.对标点等进行分词
"""
import numpy as np
import gensim


def load_stop_words():
    """
    加载停用词表
    """
    stop_words = []
    for line in open('../data/stop_words.txt', encoding='utf8').readlines():
        line = line.strip()
        stop_words.append(line)
    stop_words.append(" ")
    return stop_words


def match_word2vec(model, sentence, stop_list):
    """
    为每个句子生成词向量，求和平均法得到的应该是300维向量
    """
    result = np.zeros(300)
    num = 0
    for word in sentence.split(" "):

        try:
            if word in stop_list:
                result += np.zeros(300)
            else:
                result += model[word]
                num += 1
        except:
            result += np.zeros(300)
    if num == 0:
        num = 1.0
    return result / num


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


def generate_feature_vector():
    """
    从语料中根据预训练词向量生成每篇邮件的特征向量并保存本地npz文件
    """
    # 加载模型
    # 生成特征及标签

    x, y = [], []
    with open('../data/RT/rt-polarity.neg', 'r', encoding="utf8") as f:
        line = f.readline()
        while line:
            feature = match_word2vec(model, line.strip(), load_stop_words())
            x.append(feature)
            y.append(0)
            line = f.readline()
    with open('../data/RT/rt-polarity.pos', 'r', encoding="utf8") as f:
        line = f.readline()
        while line:
            feature = match_word2vec(model, line.strip(), load_stop_words())
            x.append(feature)
            y.append(1)
            line = f.readline()

    x = np.array(x)
    print(x.shape)
    y = np.array(y)
    print(y.shape)
    np.savez('data.npz', x=x, y=y)


if __name__ == '__main__':
    generate_feature_vector()