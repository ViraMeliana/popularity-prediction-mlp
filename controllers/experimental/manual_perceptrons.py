import pickle
import numpy as np

from engine.manual.mlp.manual_mlp import MultilayerPerceptron
from engine.manual.utils.util_functions import train_test_split, to_categorical, accuracy_score
from engine.utils.utilities import load_parsed_csv, basic_clean


def main():
    data = load_parsed_csv(datasets_path='../../resources/datasets/all/records_4k.csv')
    X = data.title
    y = data.is_popular

    # Convert the nominal y values to binary
    y = to_categorical(y.astype("int"))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=1)

    clean_train_news = []
    clean_test_news = []

    for key in X_train:
        clean_train_news.append(basic_clean(key))

    for key in X_test:
        clean_test_news.append(basic_clean(key))

    vector = pickle.load(open("resources/datasets/models/vectorized.pickle", 'rb'))
    X_train = vector.transform(clean_train_news).toarray()
    X_test = vector.transform(clean_test_news).toarray()

    # MLP
    clf = MultilayerPerceptron(n_hidden=256,
                               n_iterations=200,
                               learning_rate=0.001)

    mlp = clf.fit(X_train, y_train)
    pickle.dump(mlp, open("resources/datasets/models/model_mlp_manual.pickle", "wb"))

    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
