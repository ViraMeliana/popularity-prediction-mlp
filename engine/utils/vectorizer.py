import engine.utils.utilities as util
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
import engine.utils.utilities as utils

class Vectorizer:
    def __init__(self):
        self.vectorized = None
        self.train_data_features = None
        self.val_data_features = None

        self.train = utils.load_parsed_csv(datasets_path='datasets/all/training.csv')
        self.test = utils.load_parsed_csv(datasets_path='datasets/all/testing.csv')
        self.val =utils.load_parsed_csv(datasets_path='datasets/all/validation.csv')

        self.clean_train_news = []
        self.clean_val_news = []

        self.prepare_models()
        self.victories()

    def prepare_models(self):
        for key in self.train['title']:
            self.clean_train_news.append(util.basic_clean(key))
        for key in self.val['title']:
            self.clean_val_news.append(util.basic_clean(key))

    def victories(self):
        print("Creating the bag of words for training...\n")
        self.vectorized = TfidfVectorizer()
        train_data_features = self.vectorized.fit_transform(self.clean_train_news)
        return train_data_features

    def victories_val(self):
        print("Creating the bag of words for val...\n")
        self.vectorized = TfidfVectorizer()
        val_data_features = self.vectorized.fit_transform(self.clean_val_news)
        return val_data_features

