from engine.lib.mlp import lib_mlp
from engine.lib.mlp import library_mlp_test


def main():
    train = lib_mlp.Training().mlp_library()
    return train


def validation():
    val = lib_mlp.Training().validation()
    return val


def confusion_matrix():
    test = library_mlp_test.Testing()
    cm = test.confusion_matrix()
    return cm


def classification_report():
    test = library_mlp_test.Testing()
    cr = test.classification_report()
    return cr


if __name__ == "__main__":
    # main()
    validation()
    confusion_matrix()
    classification_report()
