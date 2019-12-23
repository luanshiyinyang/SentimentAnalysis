"""
Author: Zhou Chen
Date: 2019/12/23
Desc: 加载特征向量化的数据，用于模型的训练
"""

import numpy as np


class RTPolarity(object):

    def __init__(self):
        data = np.load("../data/data.npz")
        self.x_train, self.y_train = data['x'], data['y']

    def load_data(self):
        return self.x_train, self.y_train


if __name__ == '__main__':
    x, y = RTPolarity().load_data()
    print(x.shape, y.shape)