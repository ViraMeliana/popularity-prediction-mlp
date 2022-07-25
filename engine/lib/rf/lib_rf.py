import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import engine.utils.vectorizer as vector
import pickle


class Training:

    def __init__(self):
        self.vectors = vector.Vectorizer()
        self.vectorize = self.vectors.vectorized
        self.train_data_features = self.vectors.victories()
        self.val_data_features = self.vectors.victories_val()

    def random_forest(self):
        clf = RandomForestClassifier(n_estimators=100)
        rf = clf.fit(self.train_data_features, np.nan_to_num(self.vectors.train["is_popular"]))
        pickle.dump(rf, open("datasets/models/model_rf.pickle", "wb"))

    def validation(self):
        kf = KFold(shuffle=True, n_splits=5)
        clf = RandomForestClassifier(n_estimators=100)
        cv_results = cross_val_score(clf, self.val_data_features, np.nan_to_num(self.vectors.val["is_popular"]), cv=kf, scoring='accuracy')
        print(cv_results)
        return cv_results