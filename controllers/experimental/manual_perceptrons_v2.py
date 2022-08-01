import json
import os
import shutil
from urllib.request import urlopen

from engine.manual.mlp_keras_like.mlp_manual_fit import ManualMultiPerceptron
from engine.utils.utilities import load_parsed_csv


class ManualTrainTest:
    def __init__(self):
        self.vectorized = 'resources/models/vectorized.pickle'
        self.neural_net = 'resources/models/model_neural_net.pickle'
        self.layers = 'resources/models/model_layer.pickle'

    def train(self):
        train = load_parsed_csv(datasets_path='../../resources/datasets/all/training.csv')
        test = load_parsed_csv(datasets_path='../../resources/datasets/all/testing.csv')

        X_train = train.title
        y_train = train.is_popular
        X_test = test.title
        y_test = test.is_popular

        clf = ManualMultiPerceptron(n_iterations=200,
                                    n_hidden=256, n_classes=2, vectorized=self.vectorized)

        clf.fit(X_train, y_train, X_test, y_test)

    def predict(self, title):
        y_pred, title_cleaned = ManualMultiPerceptron(n_iterations=200,
                                                      n_hidden=256, n_classes=2, vectorized=self.vectorized).predict(
            title)
        return json.dumps({
            'result': y_pred,
            'description': json.dumps(
                {
                    'title_clean': title_cleaned
                }
            )
        })

    def confusion_matrix(self):
        return ManualMultiPerceptron(n_iterations=200,
                                     n_hidden=256, n_classes=2,
                                     vectorized=self.vectorized).confusion_matrix()

    def classification_report(self):
        return ManualMultiPerceptron(n_iterations=200,
                                     n_hidden=256, n_classes=2,
                                     vectorized=self.vectorized).classification_report()
