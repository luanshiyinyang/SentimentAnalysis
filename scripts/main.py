"""
Author: Zhou Chen
Date: 2019/12/23
Desc: 主模块，进行多模型实验
"""
from data import IMDB
from utils import plot_roc

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_curve, auc


def load_data():
    return IMDB().load_data()


def build_model():
    classifier_lr = LogisticRegression(solver='lbfgs')
    classifier_svm = SVC(gamma='scale', kernel='rbf', C=1)
    classifier_nn = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(256, 128, 32), shuffle=False)
    return [classifier_lr, classifier_svm, classifier_nn]


if __name__ == '__main__':
    k = 5
    x, y = load_data()
    kf = KFold(n_splits=k, shuffle=True, random_state=2019)  # 打乱后划分为5组
    acc_lr = []
    acc_svm = []
    acc_nn = []
    roc_lr = []
    roc_svm = []
    roc_nn = []
    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        lr, svm, nn = build_model()

        lr.fit(x_train, y_train)
        y_pred_lr = lr.predict(x_test)
        acc = accuracy_score(y_test, y_pred_lr)
        acc_lr.append(acc)
        print(acc)
        fpr, tpr, _ = roc_curve(y_test, lr.predict_proba(x_test)[:, 1])
        roc_lr = [fpr, tpr]

        svm.fit(x_train, y_train)
        y_pred_svm = svm.predict(x_test)
        acc = accuracy_score(y_test, y_pred_svm)
        acc_svm.append(acc)
        print(acc)
        fpr, tpr, _ = roc_curve(y_test, svm.predict_proba(x_test)[:, 1])
        roc_svm = [fpr, tpr]

        nn.fit(x_train, y_train)
        y_pred_nn = nn.predict(x_test)
        acc = accuracy_score(y_test, y_pred_nn)
        acc_nn.append(acc)
        print(acc)
        fpr, tpr, _ = roc_curve(y_test, nn.predict_proba(x_test)[:, 1])
        roc_nn = [fpr, tpr]
    print(acc_lr, acc_svm, acc_nn)
    plot_roc([roc_lr, roc_svm, roc_nn])
