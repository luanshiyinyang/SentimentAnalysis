"""
Author: Zhou Chen
Date: 2019/12/25
Desc: 工具模块
"""
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def plot_roc(roc):
    """
    绘制ROC曲线
    :return:
    """
    roc_auc_lr = auc(roc[0][0], roc[0][1])
    roc_auc_svm = auc(roc[1][0], roc[1][1])
    roc_auc_nn = auc(roc[2][0], roc[2][1])

    plt.plot(roc[0][0], roc[0][1], label="lr area is {:.2f}".format(roc_auc_lr))
    plt.plot(roc[1][0], roc[1][1], label="svm area is {:.2f}".format(roc_auc_svm))
    plt.plot(roc[2][0], roc[2][1], label="nn area is {:.2f}".format(roc_auc_nn))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc=0)
    plt.savefig('roc.png')
    plt.show()
