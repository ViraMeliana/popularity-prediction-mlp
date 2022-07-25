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
        if not os.path.exists(self.vectorized):
            print("Downloading pretrained model to " + self.vectorized + " ...")
            with urlopen('https://datasets.open-projects.org/vectorized.pickle') as resp, \
                    open(self.vectorized, 'wb') as out:
                shutil.copyfileobj(resp, out)

        if not os.path.exists(self.neural_net):
            print("Downloading pretrained model to " + self.neural_net + " ...")
            with urlopen('https://datasets.open-projects.org/model_neural_net.pickle') as resp, \
                    open(self.neural_net, 'wb') as out:
                shutil.copyfileobj(resp, out)

        if not os.path.exists(self.layers):
            print("Downloading pretrained model to " + self.layers + " ...")
            with urlopen('https://datasets.open-projects.org/model_layer.pickle') as resp, \
                    open(self.layers, 'wb') as out:
                shutil.copyfileobj(resp, out)

        return ManualMultiPerceptron(n_iterations=200,
                                     n_hidden=256, n_classes=2, vectorized=self.vectorized).predict(title)

    def confusion_matrix(self):
        return ManualMultiPerceptron(n_iterations=200,
                                     n_hidden=256, n_classes=2,
                                     vectorized=self.vectorized).confusion_matrix()

    def classification_report(self):
        return ManualMultiPerceptron(n_iterations=200,
                                     n_hidden=256, n_classes=2,
                                     vectorized=self.vectorized).classification_report()
