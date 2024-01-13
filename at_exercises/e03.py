import pickle

import pandas
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def get_data():
    with open("data/mnist.pkl", "rb") as f:
        mnist = pickle.load(f)
    return mnist


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    pyplot.imshow(image, cmap="binary")
    pyplot.axis("off")


def get_train_test_data(mnist):
    x = mnist.data
    y = mnist.target
    return x[:60000], x[60000:], y[:60000], y[60000:]


def q1():
    mnist = get_data()
    data, test_data, labels, test_labels = get_train_test_data(mnist)
    knn_clf = Pipeline([("knn", KNeighborsClassifier())])
    param_grid = [
        {
            'knn__weights': ["uniform", "distance"],
            "knn__n_neighbors": [1, 2, 3, 4],
         },
    ]
    search = GridSearchCV(knn_clf, param_grid, cv=3, scoring='accuracy')
    search.fit(data, labels)
    return pandas.DataFrame(search.cv_results_)


def q1_test():
    mnist = get_data()
    data, test_data, labels, test_labels = get_train_test_data(mnist)
    knn_clf = Pipeline([("knn", KNeighborsClassifier(n_neighbors=4, weights="distance"))])
    knn_clf.fit(data, labels)
    test_predictions = knn_clf.predict(test_data)
    return accuracy_score(test_predictions, test_labels)
