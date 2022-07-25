import sys

import engine.utils.vectorizer as vector
import pickle
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


class Training:

    def __init__(self):
        self.vectors = vector.Vectorizer()
        self.vectorize = self.vectors.vectorized
        self.train_data_features = self.vectors.victories()
        self.val_data_features = self.vectors.victories_val()

    def mlp_library(self):
        mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(4,), solver='adam', tol=0.001, verbose=True)
        neural_net = mlp.fit(self.train_data_features, np.nan_to_num(self.vectors.train["is_popular"]))

        pickle.dump(self.vectorize, open("datasets/models/vectorized.pickle", "wb"))
        pickle.dump(neural_net, open("datasets/models/model_mlp_library.pickle", "wb"))

    def validation(self):
        kf = KFold(shuffle=True, n_splits=5)
        mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(4,), solver='adam', tol=0.001, verbose=True)
        cv_results = cross_val_score(mlp, self.val_data_features, np.nan_to_num(self.vectors.val["is_popular"]),
                                     cv=kf, scoring='accuracy')
        print(cv_results)

        return cv_results

    def weight_extraction(self):
        weight = pickle.load(open("datasets/models/model_mlp_library.pickle", "rb"))
        return weight.coefs_
