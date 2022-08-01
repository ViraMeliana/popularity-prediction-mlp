import json

from engine.lib.mlp import lib_mlp
from engine.lib.mlp import library_mlp_test


def main_mlp():
    train = lib_mlp.Training().mlp_library()
    return train


def test_mlp(title):
    tester = library_mlp_test.Testing()
    y_pred, title_cleaned = tester.predict(title)

    return json.dumps({
        'result': str(y_pred),
        'description':
            {
                'title_clean': title_cleaned
            }
    })


def validation_mlp():
    val = lib_mlp.Training().validation()
    return val


def confusion_matrix_mlp():
    test = library_mlp_test.Testing()
    cm = test.confusion_matrix()
    return cm


def classification_report_mlp():
    test = library_mlp_test.Testing()
    cr = test.classification_report()
    return cr
