
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def plot_roc(clf, X_test, y_test):
   
    RocCurveDisplay.from_estimator(clf, X_test, y_test)
    plt.title("ROC")
    plt.show()

def plot_pr(clf, X_test, y_test):
    
    PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
    plt.title("Precision-Recall")
    plt.show()

def plot_confusion_matrix(clf, X_test, y_test, labels=None):

    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=labels)
    plt.title("Confusion Matrix")
    plt.show()  