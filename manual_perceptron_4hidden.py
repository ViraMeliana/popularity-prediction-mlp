import os
import pickle

from sklearn.model_selection import KFold, cross_val_score

import engine.utils.vectorizer as vector
import numpy as np
from matplotlib import pyplot as plt

from engine.manual.mlp_keras_like.mlp_manual_fit import ManualMultiPerceptron
from engine.utils.utilities import load_parsed_csv, basic_clean


def main():
    train = load_parsed_csv(datasets_path='datasets/all/training.csv')
    test = load_parsed_csv(datasets_path='datasets/all/testing.csv')

    X_train = train.title
    y_train = train.is_popular
    X_test = test.title
    y_test = test.is_popular

    # MLP
    clf = ManualMultiPerceptron(n_iterations=200,
                                n_hidden=256, n_classes=2, vectorized="datasets/models/vectorized.pickle")

    train_err, val_err = clf.fit(X_train, y_train, X_test, y_test)

    n = len(train_err)

    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")

    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()


def predict():
    predict = ManualMultiPerceptron(n_iterations=200,
                                    n_hidden=256, n_classes=2, vectorized="datasets/models/vectorized.pickle").predict()
    return predict

def confusion_matrix():
    predict = ManualMultiPerceptron(n_iterations=200,
                                    n_hidden=256, n_classes=2, vectorized="datasets/models/vectorized.pickle").confusion_matrix()
    return predict

def classification_report():
    predict = ManualMultiPerceptron(n_iterations=200,
                                    n_hidden=256, n_classes=2, vectorized="datasets/models/vectorized.pickle").classification_report()
    return predict

if __name__ == "__main__":
    classification_report()
    confusion_matrix()
