from engine.lib.rf import lib_rf
from engine.lib.rf import lib_rf_test


def main():
    train = lib_rf.Training().random_forest()
    return train

def validation():
    val = lib_rf.Training().validation()
    return val


def confusion_matrix():
    test = lib_rf_test.Testing()
    cm = test.confusion_matrix()
    return cm


def classification_report():
    test = lib_rf_test.Testing()
    cr = test.classification_report()
    return cr

if __name__ == "__main__":
    # main()
    validation()
    confusion_matrix()
    classification_report()
