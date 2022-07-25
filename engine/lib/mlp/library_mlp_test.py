import pickle
import os
import engine.utils.utilities as util
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class Testing:
    def __init__(self):
        self.vectorized = None
        self.model = None

        self.test = util.load_parsed_csv(datasets_path='datasets/all/testing.csv')

        self.pickle_parser()

    def pickle_parser(self):
        if os.path.exists("datasets/models/vectorized.pickle") & os.path.exists(
                "datasets/models/model_mlp_library.pickle"):
            print("model is exist")
            self.vectorized = pickle.load(open("datasets/models/vectorized.pickle", 'rb'))
            self.model = pickle.load(open("datasets/models/model_mlp_library.pickle", 'rb'))
        else:
            print("model doesn't exist")

    def title_cleaner(self, title):
        if self.vectorized is not None:
            print("title cleaner")
            title_clean = title
            title_clean = np.append(title_clean, util.basic_clean(title_clean))

            test_data_features = self.vectorized.transform(title_clean)
            np.asarray(test_data_features)

            return test_data_features
        else:
            return None

    def predict(self, title):
        if self.model is not None:
            predict = self.model.predict(self.title_cleaner(title))
            print(predict[0])
            return predict[0]
        else:
            return None

    def classification_report(self):
        desc = np.asarray(self.test['title'].values.astype('U'))
        predict = self.model.predict(self.title_cleaner(desc))
        report = classification_report(np.nan_to_num(self.test['is_popular']), predict[:-1], labels=[1, 0])
        print(report)
        return report

    def confusion_matrix(self):
        desc = np.asarray(self.test['title'].values.astype('U'))
        predict = self.model.predict(self.title_cleaner(desc))
        cm = confusion_matrix(self.test['is_popular'], predict[:-1])
        ac = accuracy_score(self.test['is_popular'], predict[:-1])
        print(cm)
        print(ac)
        return ac,cm
