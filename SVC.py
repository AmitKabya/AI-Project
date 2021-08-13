from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
from Data import Data, MissingData
import PCA


def read_data(dr, md):
    if not md:
        data = Data(dr)
    else:
        data = MissingData(dr)
    mushrooms = data.dataF

    mushrooms = mushrooms.values.tolist()

    y = []

    for i in range(len(mushrooms)):
        y.append(mushrooms[i][0])
        mushrooms[i].pop(0)

    X = np.array(mushrooms)

    return X, y, data


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = LinearSVC()
    model.fit(X_train, y_train)
    return X_test, y_test, model


def tests_results(X_test, y_test, model):
    # accuracy:
    print(f"Accuracy {model.score(X_test, y_test)}")

    # confusion matrix:
    y_predicted = model.predict(X_test)
    cm = confusion_matrix(y_test, y_predicted)

    plt.figure(figsize=(10, 7))
    plt.suptitle("Confusion Matrix")
    sn.heatmap(cm, annot=True, fmt=".1f")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

    return y_predicted, model.score(X_test, y_test)


def missing_data_prediction(data, model):
    data.missingDataF.pop('classes')
    y_predicted = model.predict(data.missingDataF)
    missingMushrooms = data.missingDataF.copy()
    missingMushrooms.insert(0, 'classes', y_predicted)
    print(f"class prediction:\n{missingMushrooms.head(len(missingMushrooms))}")
    draw_results(data.missingDataF, y_predicted)


def draw_results(X_test, y_test, y_predicted=None):
    # test visualization:
    pca = PCA.pca_predict(X_test, y_test, 3)
    fig = plt.figure()
    fig.suptitle("Test Visualization")
    PCA.plot_pca(pca, fig)

    # original test data visualization:
    if y_predicted is not None:
        pca = PCA.pca_predict(X_test, y_predicted, 3)
        fig = plt.figure()
        fig.suptitle("Original Test Data Visualization")
        PCA.plot_pca(pca, fig)
    plt.show()


def main(dr=0, md=False):
    if not md:
        X, y = read_data(dr, md)[:2]
        X_test, y_test, model = train_model(X, y)
        y_predicted, acc = tests_results(X_test, y_test, model)
        draw_results(X_test, y_test, y_predicted)

    else:
        X, y, data = read_data(dr, md)
        X_test, y_test, model = train_model(X, y)
        y_predicted, acc = tests_results(X_test, y_test, model)
        draw_results(X_test, y_test, y_predicted)
        missing_data_prediction(data, model)

    return acc
