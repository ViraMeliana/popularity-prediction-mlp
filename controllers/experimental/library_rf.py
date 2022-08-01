import json

from engine.lib.rf import lib_rf
from engine.lib.rf import lib_rf_test


def main_rf():
    train = lib_rf.Training().random_forest()
    return train


def validation_rf():
    val = lib_rf.Training().validation()
    return val


def test_rf(title):
    tester = lib_rf_test.Testing()
    y_pred, title_cleaned = tester.predict(title)

    return json.dumps({
        'result': str(y_pred),
        'description':
            {
                'title_clean': title_cleaned
            }
    })


def confusion_matrix_rf():
    test = lib_rf_test.Testing()
    cm = test.confusion_matrix()
    return cm


def classification_report_rf():
    test = lib_rf_test.Testing()
    cr = test.classification_report()
    return cr
