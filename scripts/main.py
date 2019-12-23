"""
Author: Zhou Chen
Date: 2019/12/23
Desc: 主模块，进行多模型实验
"""
from data import RTPolarity

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data():
    return RTPolarity().load_data()


def build_model():
    classifier_lr = LogisticRegression(solver='lbfgs')
    classifier_svm = SVC(gamma='scale', kernel='rbf', C=1)
    classifier_nn = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(256, 128, 32), shuffle=False)
    return [classifier_nn]


if __name__ == '__main__':
    models = build_model()
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2019)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    for model in models:
        model.fit(x_train, y_train)
        print(accuracy_score(y_test, model.predict(x_test)))
