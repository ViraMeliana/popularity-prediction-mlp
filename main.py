from setup import core

# from controllers.experimental import library_perceptrons
# from controllers.experimental import library_rf
# from controllers.experimental.manual_perceptrons_v2 import ManualTrainTest


# def main():
    # library_perceptrons.confusion_matrix_mlp()
    # library_perceptrons.classification_report_mlp()
    # library_perceptrons.validation_mlp()

    # library_rf.validation_rf()
    # library_rf.confusion_matrix_rf()
    # library_rf.classification_report_rf()

    # manual_perceptrons_v2 = ManualTrainTest()
    # manual_perceptrons_v2.confusion_matrix()
    # manual_perceptrons_v2.classification_report()


if __name__ == '__main__':
    # main()
    core.run(host='0.0.0.0', port=5000, debug=True)
