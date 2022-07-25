from __future__ import print_function
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import engine.utils.utilities as util
from engine.manual.mlp_keras_like.neural_network import NeuralNetwork
from engine.manual.utils.util_functions import to_categorical, accuracy_score
from engine.manual.mlp_keras_like.optimizers import Adam
from engine.manual.utils.loss_functions import CrossEntropy
from engine.manual.mlp_keras_like.layers import Hidden, Activation


class ManualMultiPerceptron:
    def __init__(self, n_iterations, n_hidden, n_classes, vectorized):
        self.n_hidden = n_hidden
        self.vectorized = vectorized
        self.n_iterations = n_iterations
        self.n_classes = n_classes
        self.optimizer = Adam()
        self.clf = None

    def fit(self, X, y, X1, y1):
        y = to_categorical(y.astype("int"))
        y1 = to_categorical(y1.astype("int"))

        print("title cleaner")
        X_train_clean = np.append(X, util.basic_clean(X))
        X_test_clean = np.append(X1, util.basic_clean(X1))

        vector = pickle.load(open(self.vectorized, 'rb'))
        X_train = vector.transform(X_train_clean[:-1]).toarray()
        y_train = y
        print(X_train.shape)

        X_test = vector.transform(X_test_clean[:-1]).toarray()
        y_test = y1
        print(X_test.shape)

        n_samples, n_features = X_train.shape
        print(n_features)

        self.clf = NeuralNetwork(optimizer=self.optimizer,
                                 loss=CrossEntropy,
                                 validation_data=(X_test, y_test))

        self.clf.add(Hidden(self.n_hidden, input_shape=(n_features,)))
        self.clf.add(Activation('relu'))
        self.clf.add(Hidden(self.n_hidden))
        self.clf.add(Activation('relu'))
        self.clf.add(Hidden(self.n_hidden))
        self.clf.add(Activation('relu'))
        self.clf.add(Hidden(self.n_hidden))
        self.clf.add(Activation('relu'))
        self.clf.add(Hidden(self.n_classes))
        self.clf.add(Activation('softmax'))

        print()
        self.clf.summary(name="MLP")
        train_err, val_err = self.clf.fit(X_train, y_train, n_epochs=self.n_iterations, batch_size=256)
        # savemodel = self.clf.saveModel()

        _, accuracy = self.clf.test_on_batch(X_test, y_test)
        print("Accuracy:", accuracy)

        y_pred = np.argmax(self.clf.predict(X_test), axis=1)
        return train_err, val_err

    def title_cleaner(self, title):
        print("title cleaner")
        title_clean = title
        title_clean = np.append(title_clean, util.basic_clean(title_clean))
        vectorized = pickle.load(open("resources/models/vectorized.pickle", 'rb'))
        val_data_features = vectorized.transform(title_clean)
        np.asarray(val_data_features)

        return val_data_features

    def predict(self, title):
        self.clf = NeuralNetwork(optimizer=self.optimizer,
                                 loss=CrossEntropy)
        y_pred = np.argmax(self.clf.predict(self.title_cleaner(title)[:-1]), axis=1)
        return y_pred

    def confusion_matrix(self):
        self.clf = NeuralNetwork(optimizer=self.optimizer,
                                 loss=CrossEntropy)
        test = util.load_parsed_csv(datasets_path='resources/datasets/all/testing.csv')
        y_pred = np.argmax(self.clf.predict(self.title_cleaner(test['title'])[
                                            :-1]), axis=1)
        cm = confusion_matrix(test['is_popular'], y_pred)
        y1 = to_categorical(test['is_popular'].astype("int"))
        ac = accuracy_score(test['is_popular'], y_pred)
        print(y_pred)
        print(cm)
        print(ac)
        return ac

    def classification_report(self):
        self.clf = NeuralNetwork(optimizer=self.optimizer,
                                 loss=CrossEntropy)
        test = util.load_parsed_csv(datasets_path='resources/datasets/all/testing.csv')
        y_pred = np.argmax(self.clf.predict(self.title_cleaner(test['title'])[
                                            :-1]), axis=1)
        report = classification_report(test['is_popular'], y_pred, labels=[0, 1])
        print(report)
        return report
